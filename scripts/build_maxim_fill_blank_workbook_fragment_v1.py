#!/usr/bin/env python3
"""Build task-ID-free source fragments for numbered fill-in activities.

Only a pinned public PDF and a source-native specification are read.  The
builder never opens benchmark references, solver answers, judge outputs,
scores, correctness labels, or task outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.fill_blank_answer_key import (  # noqa: E402
    FillBlankAnswerKeyError,
    attest_fill_blank_answer_key,
)
from evidence_os.official_ogm import canonical_json_bytes, sha256_file  # noqa: E402


SPEC_SCHEMA = "maxim-fill-blank-workbook-source-spec-v1"
OUTPUT_SCHEMA = "public-workbook-fill-blank-source-index-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")
_TASK_LIKE = re.compile(
    r"(?:^|[._-])(?:val|test|train|task|qid)[._-]?\d+(?:[._-]|$)",
    re.IGNORECASE,
)
_ROOT_KEYS = frozenset({"schema_version", "documents"})
_DOCUMENT_KEYS = frozenset(
    {
        "document_id",
        "locator",
        "pdf_sha256",
        "page_count",
        "content_page_ranges",
        "activities",
    }
)
_LOCATOR_KEYS = frozenset({"kind", "public_locator", "name"})
_ACTIVITY_KEYS = frozenset(
    {
        "record_id",
        "content_page_number",
        "key_page_number",
        "content_bbox",
        "word_bank_bbox",
        "key_bbox",
        "activity_title",
        "instruction_text",
        "expected_item_count",
        "expected_column_count",
        "answer",
        "visually_checked",
    }
)


class BuildError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise BuildError(f"{path}: expected object")
    return value


def _parse_documents(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        document_id, separator, raw_path = value.partition("=")
        document_id = document_id.strip()
        if not separator or not document_id or document_id in result:
            raise BuildError("--document must be a unique DOCUMENT_ID=PATH pair")
        result[document_id] = Path(raw_path).resolve()
    return result


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise BuildError(f"{label} must be a positive integer")
    return value


def _bbox(value: Any, label: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            not isinstance(item, (int, float)) or isinstance(item, bool)
            for item in value
        )
    ):
        raise BuildError(f"{label} must contain four numeric coordinates")
    result = [float(item) for item in value]
    if not (result[0] < result[2] and result[1] < result[3]):
        raise BuildError(f"{label} is empty")
    return result


def _validate_locator(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _LOCATOR_KEYS:
        raise BuildError("document locator fields are malformed")
    kind = value.get("kind")
    public_locator = value.get("public_locator")
    name = value.get("name")
    if (
        kind != "yandex_public"
        or not isinstance(public_locator, str)
        or not public_locator.startswith("ya-disk-public://")
        or any(character.isspace() for character in public_locator)
        or not isinstance(name, str)
        or name != name.strip()
        or not name.casefold().endswith(".pdf")
        or "/" in name
        or "\\" in name
    ):
        raise BuildError("document does not have a strict Yandex public identity")
    return {"kind": kind, "public_locator": public_locator, "name": name}


def _validate_ranges(value: Any, *, page_count: int) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise BuildError("content_page_ranges must be a non-empty list")
    result: list[list[int]] = []
    occupied: set[int] = set()
    for raw_range in value:
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise BuildError("content page range is malformed")
        start = _positive_integer(raw_range[0], "content range start")
        end = _positive_integer(raw_range[1], "content range end")
        pages = set(range(start, end + 1))
        if start > end or end > page_count or pages & occupied:
            raise BuildError("content page ranges overlap or leave the PDF")
        occupied.update(pages)
        result.append([start, end])
    return result


def build(
    spec_path: Path,
    paths: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise BuildError("fill-blank source builder requires pdfplumber") from exc
    spec = _load(spec_path)
    if set(spec) != _ROOT_KEYS or spec.get("schema_version") != SPEC_SCHEMA:
        raise BuildError("fill-blank source spec has an unsupported schema")
    raw_documents = spec.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise BuildError("fill-blank source spec has no documents")
    configured_ids = {
        str(item.get("document_id") or "")
        for item in raw_documents
        if isinstance(item, dict)
    }
    if len(configured_ids) != len(raw_documents) or configured_ids != set(paths):
        raise BuildError("document paths must exactly match unique spec document IDs")

    output_documents: list[dict[str, Any]] = []
    total_records = 0
    for raw_document in raw_documents:
        if not isinstance(raw_document, dict) or set(raw_document) != _DOCUMENT_KEYS:
            raise BuildError("fill-blank document fields are malformed")
        document_id = str(raw_document.get("document_id") or "")
        pdf_sha256 = str(raw_document.get("pdf_sha256") or "")
        if (
            _DOCUMENT_ID.fullmatch(document_id) is None
            or _TASK_LIKE.search(document_id) is not None
            or _HEX64.fullmatch(pdf_sha256) is None
            or not document_id.endswith(pdf_sha256[:12])
        ):
            raise BuildError("document_id is not source-derived from the pinned PDF")
        locator = _validate_locator(raw_document.get("locator"))
        page_count = _positive_integer(raw_document.get("page_count"), "page_count")
        ranges = _validate_ranges(
            raw_document.get("content_page_ranges"), page_count=page_count
        )
        pdf_path = paths[document_id]
        if sha256_file(pdf_path) != pdf_sha256:
            raise BuildError(f"PDF SHA-256 mismatch for {document_id}")
        raw_activities = raw_document.get("activities")
        if not isinstance(raw_activities, list) or not raw_activities:
            raise BuildError(f"{document_id}: activities must be non-empty")
        output_activities: list[dict[str, Any]] = []
        record_ids: set[str] = set()
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) != page_count:
                raise BuildError(f"{document_id}: page count changed")
            for raw_activity in raw_activities:
                if not isinstance(raw_activity, dict) or set(raw_activity) != _ACTIVITY_KEYS:
                    raise BuildError(f"{document_id}: activity fields are malformed")
                record_id = str(raw_activity.get("record_id") or "")
                content_page = _positive_integer(
                    raw_activity.get("content_page_number"), "content_page_number"
                )
                key_page = _positive_integer(
                    raw_activity.get("key_page_number"), "key_page_number"
                )
                if (
                    record_id != f"{document_id}:p{content_page}:fill_blank"
                    or record_id in record_ids
                    or _TASK_LIKE.search(record_id) is not None
                ):
                    raise BuildError("fill-blank record_id is not source-addressed")
                if (
                    content_page > page_count
                    or key_page > page_count
                    or not any(start <= content_page <= end for start, end in ranges)
                ):
                    raise BuildError("fill-blank activity page is outside its source ranges")
                if raw_activity.get("visually_checked") is not True:
                    raise BuildError("fill-blank activity lacks visual source review")
                content_bbox = _bbox(raw_activity.get("content_bbox"), "content_bbox")
                bank_bbox = _bbox(raw_activity.get("word_bank_bbox"), "word_bank_bbox")
                key_bbox = _bbox(raw_activity.get("key_bbox"), "key_bbox")
                title = str(raw_activity.get("activity_title") or "").strip()
                instruction = str(raw_activity.get("instruction_text") or "").strip()
                answer = str(raw_activity.get("answer") or "").strip()
                expected_items = _positive_integer(
                    raw_activity.get("expected_item_count"), "expected_item_count"
                )
                expected_columns = _positive_integer(
                    raw_activity.get("expected_column_count"), "expected_column_count"
                )
                if not title or not instruction or not answer or len(answer) > 512:
                    raise BuildError("fill-blank source text is empty or too long")
                verification = attest_fill_blank_answer_key(
                    pdf.pages[content_page - 1],
                    pdf.pages[key_page - 1],
                    pdf_sha256=pdf_sha256,
                    content_page_number=content_page,
                    key_page_number=key_page,
                    content_bbox=content_bbox,
                    word_bank_bbox=bank_bbox,
                    key_bbox=key_bbox,
                    activity_title=title,
                    instruction_text=instruction,
                    expected_item_count=expected_items,
                    expected_column_count=expected_columns,
                    expected_answer=answer,
                )
                output_activities.append(
                    {
                        "record_id": record_id,
                        "content_page_number": content_page,
                        "key_page_number": key_page,
                        "key_context_page_number": key_page,
                        "question_marker_kind": "numberless_page_activity",
                        "question_text": verification.content_text,
                        "answer": answer,
                        "answer_format": "short_text",
                        "source_answer_format": "numbered_short_text",
                        "key_binding_kind": "fill_blank_answer_key",
                        "activity_title": title,
                        "instruction_text": instruction,
                        "expected_item_count": expected_items,
                        "expected_column_count": expected_columns,
                        "key_crop_text": verification.key_text,
                        "key_projection_sha256": verification.key_projection_sha256,
                        "content_projection_sha256": (
                            verification.content_projection_sha256
                        ),
                        "binding_projection_sha256": verification.projection_sha256,
                        "content_bbox": content_bbox,
                        "word_bank_bbox": bank_bbox,
                        "key_bbox": key_bbox,
                        "visually_checked": True,
                    }
                )
                record_ids.add(record_id)
        output_documents.append(
            {
                "document_id": document_id,
                "locator": locator,
                "pdf_sha256": pdf_sha256,
                "page_count": page_count,
                "content_page_ranges": ranges,
                "activities": sorted(
                    output_activities, key=lambda value: str(value["record_id"])
                ),
            }
        )
        total_records += len(output_activities)

    output = {
        "schema_version": OUTPUT_SCHEMA,
        "documents": sorted(
            output_documents, key=lambda value: str(value["document_id"])
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(output) + b"\n")
    return {
        "schema_version": "maxim-fill-blank-workbook-build-v1",
        "source_spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "documents": len(output_documents),
        "records": total_records,
        "task_id_used": False,
        "benchmark_answer_reference_score_judge_or_outcome_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--document", action="append", default=[], metavar="DOCUMENT_ID=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        result = build(
            args.spec_json.resolve(),
            _parse_documents(args.document),
            args.output.resolve(),
        )
        if args.manifest is not None:
            manifest_path = args.manifest.resolve()
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_bytes(canonical_json_bytes(result) + b"\n")
    except (
        BuildError,
        FillBlankAnswerKeyError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
