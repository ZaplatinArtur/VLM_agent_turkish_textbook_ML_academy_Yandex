from __future__ import annotations

import base64
import json
import mimetypes
import re
import shutil
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .backends import OpenAICompatibleBackend
from .ingest import read_records
from .prompts import JudgeRequest
from .sources import QUESTION_TYPE_MAP, _require_openpyxl


EXTRACTION_PROMPT_VERSION = "validation-transcription-v1"
IMAGE_ONLY_QUESTION = "(soru görselde)"
REFERENCE_IMAGE_ONLY = "[REFERENCE_IMAGE_ONLY]"

EXTRACTION_SYSTEM_PROMPT = """You transcribe Turkish school tasks from screenshots.

The first attachment is always the question. A second attachment, when present, is the
trusted answer or worked solution. Preserve all text, numbers, formulas, table contents,
labels, and answer choices needed to solve the task. Do not solve the question yourself.

Return exactly one JSON object with exactly these keys:
{"question_text":"complete transcription", "reference_answer":"short final trusted answer", "reference_solution":"transcription or concise summary of the trusted solution"}

If TRUSTED_REFERENCE_TEXT is supplied, copy it exactly into reference_answer. Otherwise,
extract the final answer from the trusted reference image. Never infer an answer from the
question image. A trusted reference image may contain a sentence, a completed diagram, or
several filled blanks instead of one answer token. In that case, put every visible completed
answer needed for grading into reference_answer and describe the diagram in text. Never return
an empty reference_answer when a trusted reference image is attached. Use an empty string for
reference_solution only when no reference image exists.
"""

REFERENCE_EXTRACTION_SYSTEM_PROMPT = """You transcribe a trusted completed answer image.
The attachment may be a short sentence, completed diagram, answer key, or exercise with several
filled blanks. Return exactly one JSON object with exactly these keys:
{"question_text":"[REFERENCE IMAGE]", "reference_answer":"all completed answers in grading-ready text", "reference_solution":"complete transcription or precise description of the completed diagram"}
reference_answer must never be empty. Do not solve or correct the image; transcribe what is shown.
"""

_ANSWER_TYPE_TO_MLA = {
    "multiple_choice": "choice",
    "multi_answer": "free_form",
    "short_text": "short_text",
    "open_ended": "free_form",
    "unknown": "free_form",
}


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _safe_subject(value: str) -> str:
    transliterated = value.casefold().translate(
        str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"})
    )
    normalized = re.sub(r"[^a-z0-9]+", "_", transliterated).strip("_")
    return normalized or "unknown"


def _safe_grade(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        grade = int(float(value))
    except (TypeError, ValueError):
        return None
    return grade if 1 <= grade <= 12 else None


def _is_local_image_ref(value: str) -> bool:
    return bool(re.fullmatch(r"images[\\/].+\.(?:png|jpe?g)", value, flags=re.I))


def _normalized_image_ref(value: str) -> str:
    return value.replace("\\", "/")


def _manifest_image_ref(
    record: dict[str, Any],
    *,
    canonical_key: str,
    merged_key: str,
    required: bool = False,
) -> str | None:
    """Read image paths from either the legacy or merged manifest contract."""
    value = _clean(record.get(canonical_key) or record.get(merged_key))
    if not value:
        if required:
            raise ValueError(
                f"task {record.get('task_id')!r} is missing "
                f"{canonical_key}/{merged_key}"
            )
        return None
    return _normalized_image_ref(value)


def _question_image_ref(record: dict[str, Any]) -> str:
    value = _manifest_image_ref(
        record,
        canonical_key="question_image_path",
        merged_key="question_image",
        required=True,
    )
    assert value is not None
    return value


def _reference_image_ref(record: dict[str, Any]) -> str | None:
    return _manifest_image_ref(
        record,
        canonical_key="reference_image_path",
        merged_key="reference_answer_image",
    )


def _manifest_asset(data_root: Path, value: str, *, label: str) -> Path:
    """Resolve a manifest asset while keeping it inside the merged dataset."""
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to data_root: {value}")
    root = data_root.resolve()
    candidate = (root / relative).resolve()
    if root not in candidate.parents:
        raise ValueError(f"{label} escapes data_root: {value}")
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    return candidate


def _validated_asset(root: Path, value: str) -> Path | None:
    if not _is_local_image_ref(value):
        return None
    candidate = (root / _normalized_image_ref(value)).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents or not candidate.is_file():
        return None
    return candidate


def extract_validation_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Safely extract the mentor validation ZIP without overwriting existing files."""
    if output_dir.exists() and any(output_dir.iterdir()):
        if not reuse_existing:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        root = output_dir.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (root / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"unsafe ZIP member path: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
    workbooks = sorted(output_dir.glob("*.xlsx"))
    images = sorted((output_dir / "images").glob("*")) if (output_dir / "images").is_dir() else []
    if len(workbooks) != 1:
        raise ValueError(f"expected one XLSX workbook, found {len(workbooks)}")
    return {
        "output_dir": str(output_dir),
        "workbook": str(workbooks[0]),
        "image_files": sum(path.is_file() for path in images),
    }


def build_validation_manifest(
    workbook_path: Path,
    data_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Resolve Sheet1 rows to local question/reference images from columns E/F and L/M."""
    load_workbook = _require_openpyxl()
    # read_only держит файл открытым до close(): на Windows незакрытый handle
    # не даёт удалить каталог, из-за чего падает уборка во временных тестах.
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    sheet = workbook["Sheet1"]
    records: list[dict[str, Any]] = []
    skipped_without_local_question = 0
    reference_kinds: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    workbook_rows = 0

    for row_number, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
        cells = list(row) + [None] * max(0, 13 - len(row))
        if not any(_clean(value) for value in cells[:10]):
            continue
        workbook_rows += 1
        visual = _clean(cells[4])
        original_reference = _clean(cells[5])
        mapped_question = _clean(cells[11])
        mapped_reference = _clean(cells[12])

        question_ref = next(
            (
                _normalized_image_ref(value)
                for value in (visual, mapped_question)
                if _validated_asset(data_root, value) is not None
            ),
            "",
        )
        if not question_ref:
            skipped_without_local_question += 1
            continue

        effective_reference = mapped_reference or original_reference
        reference_image = ""
        reference_answer = ""
        if _validated_asset(data_root, effective_reference) is not None:
            reference_image = _normalized_image_ref(effective_reference)
            reference_kind = "image"
        elif effective_reference:
            reference_answer = effective_reference
            reference_kind = "text"
        else:
            reference_kind = "missing"

        question_type_raw = _clean(cells[8])
        judge_answer_type = QUESTION_TYPE_MAP.get(question_type_raw, "unknown")
        mla_answer_type = _ANSWER_TYPE_TO_MLA[judge_answer_type]
        raw_subject = _clean(cells[1]) or "unknown"
        record = {
            "task_id": f"validation_sheet1_r{row_number:04d}",
            "source_row": row_number,
            "subject": _safe_subject(raw_subject),
            "subject_raw": raw_subject,
            "grade": _safe_grade(cells[2]),
            "question_image_path": question_ref,
            "reference_answer": reference_answer,
            "reference_image_path": reference_image or None,
            "reference_kind": reference_kind,
            "answer_type": mla_answer_type,
            "question_type_raw": question_type_raw,
            "question_format": _clean(cells[7]),
            "record_type": _clean(cells[0]),
            "source_url": _clean(cells[3]),
            "prompt_version": EXTRACTION_PROMPT_VERSION,
        }
        records.append(record)
        reference_kinds[reference_kind] += 1
        subject_counts[raw_subject] += 1
    workbook.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "workbook_rows": workbook_rows,
        "records": len(records),
        "skipped_without_local_question": skipped_without_local_question,
        "reference_kinds": dict(reference_kinds),
        "subjects": dict(subject_counts.most_common()),
        "output": str(output_path),
    }


def build_validation_seed_manifest(
    workbook_path: Path,
    data_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Resolve the eight historical Sheet6 failures to their packaged local images."""
    load_workbook = _require_openpyxl()
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    sheet = workbook["Sheet6"]
    rows = sheet.iter_rows(values_only=True)
    headers = [_clean(value) for value in next(rows)]
    image_files = {
        path.name.casefold(): path
        for path in (data_root / "images").iterdir()
        if path.is_file()
    }
    records = []
    missing_images = []
    for row_number, row in enumerate(rows, start=2):
        value = {
            headers[index]: row[index]
            for index in range(min(len(headers), len(row)))
            if headers[index]
        }
        candidate = _clean(value.get("result"))
        reference = _clean(value.get("answer"))
        image_url = _clean(value.get("image_url"))
        if not candidate or not reference or not image_url:
            continue
        basename = Path(unquote(urlparse(image_url).path)).name
        local_image = image_files.get(basename.casefold())
        task_id = f"legacy_failure_{row_number - 1:02d}"
        if local_image is None:
            missing_images.append(task_id)
            continue
        records.append(
            {
                "task_id": task_id,
                "source_row": row_number,
                "subject": _safe_subject(_clean(value.get("subject")) or "unknown"),
                "subject_raw": _clean(value.get("subject")) or "unknown",
                "grade": _safe_grade(value.get("class")),
                "question_image_path": local_image.relative_to(data_root).as_posix(),
                "reference_answer": reference,
                "reference_image_path": None,
                "reference_kind": "text",
                "answer_type": "choice",
                "question_type_raw": _clean(value.get("question_type")),
                "question_format": _clean(value.get("question_format")),
                "record_type": _clean(value.get("type")),
                "source_url": _clean(value.get("source")),
                "prompt_version": EXTRACTION_PROMPT_VERSION,
            }
        )
    workbook.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "records": len(records),
        "missing_images": missing_images,
        "output": str(output_path),
    }


def _data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _parse_extraction(raw: str) -> dict[str, str]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("extractor response must be a JSON object")
    expected = {"question_text", "reference_answer", "reference_solution"}
    if set(value) != expected or not all(isinstance(value[key], str) for key in expected):
        raise ValueError("extractor response has invalid fields")
    if not value["question_text"].strip():
        raise ValueError("question transcription is empty")
    return {key: value[key].strip() for key in expected}


class ValidationExtractor:
    def __init__(self, backend: OpenAICompatibleBackend, data_root: Path):
        self.backend = backend
        self.data_root = data_root

    def extract(self, record: dict[str, Any]) -> dict[str, Any]:
        question_ref = _question_image_ref(record)
        question_path = _manifest_asset(
            self.data_root,
            question_ref,
            label="question image",
        )
        image_urls = [_data_url(question_path)]
        image_labels = ["question image"]
        reference_image = _reference_image_ref(record)
        if reference_image:
            reference_path = _manifest_asset(
                self.data_root,
                reference_image,
                label="reference image",
            )
            image_urls.append(_data_url(reference_path))
            image_labels.append("trusted reference-answer image")
        trusted_text = str(record.get("reference_answer") or "")
        user_prompt = (
            f"TASK_ID: {record['task_id']}\n"
            f"TRUSTED_REFERENCE_TEXT: {trusted_text if trusted_text else '[NONE; use reference image]'}\n"
            "Transcribe the attachments according to the system instructions."
        )
        request = JudgeRequest(
            EXTRACTION_SYSTEM_PROMPT,
            user_prompt,
            tuple(image_urls),
            tuple(image_labels),
        )
        started = time.perf_counter()
        response = self.backend.complete(request)
        extracted = _parse_extraction(response.text)
        if trusted_text:
            extracted["reference_answer"] = trusted_text
        elif reference_image and not extracted["reference_answer"]:
            reference_request = JudgeRequest(
                REFERENCE_EXTRACTION_SYSTEM_PROMPT,
                "Transcribe this trusted completed answer image.",
                (_data_url(reference_path),),
                ("trusted completed answer image",),
            )
            reference_response = self.backend.complete(reference_request)
            reference_extraction = _parse_extraction(reference_response.text)
            extracted["reference_answer"] = reference_extraction["reference_answer"]
            extracted["reference_solution"] = reference_extraction["reference_solution"]
        if not extracted["reference_answer"]:
            raise ValueError("reference answer is empty")
        return {
            "task_id": record["task_id"],
            **extracted,
            "model": response.model,
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "latency_s": round(time.perf_counter() - started, 3),
            "metadata": response.metadata,
            "error": None,
        }


def extract_validation_text(
    manifest_path: Path,
    data_root: Path,
    output_path: Path,
    backend: OpenAICompatibleBackend,
    *,
    workers: int = 4,
    limit: int | None = None,
) -> dict[str, Any]:
    """Resume-safe multimodal transcription of questions and image references."""
    records = read_records(manifest_path)
    if limit is not None:
        records = records[:limit]
    done = (
        {
            str(record["task_id"])
            for record in read_records(output_path)
            if not record.get("error")
        }
        if output_path.exists()
        else set()
    )
    todo = [record for record in records if str(record["task_id"]) not in done]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extractor = ValidationExtractor(backend, data_root)
    lock = threading.Lock()
    failed = 0

    def run(record: dict[str, Any]) -> dict[str, Any]:
        try:
            return extractor.extract(record)
        except Exception as exc:
            return {
                "task_id": record["task_id"],
                "question_text": "",
                "reference_answer": str(record.get("reference_answer") or ""),
                "reference_solution": "",
                "model": backend.model,
                "prompt_version": EXTRACTION_PROMPT_VERSION,
                "latency_s": None,
                "metadata": {},
                "error": f"{type(exc).__name__}: {exc}",
            }

    with output_path.open("a", encoding="utf-8", newline="\n") as destination:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, record): record for record in todo}
            for future in as_completed(futures):
                result = future.result()
                if result["error"]:
                    failed += 1
                with lock:
                    destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                    destination.flush()
    return {
        "records": len(records),
        "already_done": len(records) - len(todo),
        "processed": len(todo),
        "failed": failed,
        "output": str(output_path),
    }


def build_validation_tasks(
    manifest_path: Path,
    extractions_path: Path,
    output_path: Path,
    *,
    require_all: bool = False,
) -> dict[str, Any]:
    manifest = read_records(manifest_path)
    extractions = {str(record["task_id"]): record for record in read_records(extractions_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = []
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for source in manifest:
            task_id = str(source["task_id"])
            extraction = extractions.get(task_id)
            if (
                extraction is None
                or extraction.get("error")
                or not str(extraction.get("question_text") or "").strip()
                or not str(extraction.get("reference_answer") or "").strip()
            ):
                skipped.append(task_id)
                continue
            image_path = _question_image_ref(source)
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
            task = {
                "task_id": task_id,
                "subject": source["subject"],
                "grade": source.get("grade"),
                "question": str(extraction["question_text"]).strip(),
                "question_images": [
                    {
                        "image_id": f"{task_id}_question",
                        "format": "file_path",
                        "data": image_path,
                        "mime_type": mime_type,
                        "caption": None,
                    }
                ],
                "reference_answer": str(extraction["reference_answer"]).strip(),
                "answer_type": source["answer_type"],
                "reference_solution": str(extraction.get("reference_solution") or "").strip() or None,
            }
            destination.write(json.dumps(task, ensure_ascii=False) + "\n")
            written += 1
    if require_all and skipped:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"missing successful extractions for {len(skipped)} tasks: {skipped[:10]}")
    return {
        "manifest_records": len(manifest),
        "written": written,
        "skipped": len(skipped),
        "skipped_task_ids": skipped,
        "output": str(output_path),
    }


def build_image_only_validation_tasks(
    manifest_path: Path,
    data_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build one MLA task per question screenshot without OCR or transcription.

    Text references remain in the task contract for deterministic metrics. When
    the trusted reference is an image, a sentinel satisfies the shared Task
    schema; the actual reference image is attached only to the judge.
    """
    manifest = read_records(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_task_ids: set[str] = set()
    seen_question_images: set[str] = set()
    reference_kinds: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for source in manifest:
            task_id = str(source.get("task_id") or "").strip()
            if not task_id:
                raise ValueError("validation manifest contains a record without task_id")
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate task_id in validation manifest: {task_id}")
            seen_task_ids.add(task_id)

            question_image = _question_image_ref(source)
            _manifest_asset(data_root, question_image, label=f"{task_id} question image")
            if question_image in seen_question_images:
                raise ValueError(
                    f"question image is assigned to multiple tasks: {question_image}"
                )
            seen_question_images.add(question_image)

            reference_answer = _clean(source.get("reference_answer"))
            reference_image = _reference_image_ref(source)
            if reference_image:
                _manifest_asset(data_root, reference_image, label=f"{task_id} reference image")
            if reference_answer:
                task_reference = reference_answer
                reference_kinds["text"] += 1
            elif reference_image:
                task_reference = REFERENCE_IMAGE_ONLY
                reference_kinds["image"] += 1
            else:
                raise ValueError(f"task {task_id} has no trusted reference")

            mime_type = mimetypes.guess_type(question_image)[0] or "image/png"
            task = {
                "task_id": task_id,
                "subject": str(source.get("subject") or "unknown"),
                "grade": source.get("grade"),
                "question": IMAGE_ONLY_QUESTION,
                "question_images": [
                    {
                        "image_id": f"{task_id}_question",
                        "format": "file_path",
                        "data": question_image,
                        "mime_type": mime_type,
                        "caption": None,
                    }
                ],
                "reference_answer": task_reference,
                "answer_type": str(source.get("answer_type") or "free_form"),
                "reference_solution": None,
            }
            destination.write(json.dumps(task, ensure_ascii=False) + "\n")

    return {
        "manifest_records": len(manifest),
        "written": len(seen_task_ids),
        "unique_question_images": len(seen_question_images),
        "reference_kinds": dict(reference_kinds),
        "uses_question_transcriptions": False,
        "output": str(output_path),
    }
