#!/usr/bin/env python3
"""Build reviewed workbook records from inline ``Cevap: A-E`` solutions."""

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

from evidence_os.inline_solution import (  # noqa: E402
    InlineSolutionError,
    attest_inline_solution_content,
)
from evidence_os.official_ogm import (  # noqa: E402
    OfficialSourceError,
    canonical_json_bytes,
    sha256_file,
)
from evidence_os.official_workbook import (  # noqa: E402
    INDEX_SCHEMA,
    parse_workbook_index,
    reject_benchmark_metadata,
)


_QUESTION = re.compile(r"^(\d{1,3})\.$")
_ANSWER = re.compile(r"^[A-E]$", re.IGNORECASE)
SPEC_SCHEMA = "maxim-inline-solutions-workbook-spec-v1"
EXPECTED_PDFPLUMBER_VERSION = "0.11.9"
_ROOT_KEYS = frozenset({"schema_version", "documents"})
_DOCUMENT_KEYS = frozenset(
    {
        "document_id",
        "locator",
        "pdf_sha256",
        "page_count",
        "content_page_ranges",
        "reviewed_inline_pages",
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


def _is_question_margin(word: dict[str, Any], page_width: float) -> bool:
    x0 = float(word["x0"])
    midpoint = page_width / 2
    return x0 <= 80.0 or midpoint <= x0 <= midpoint + 80.0


def _column(word: dict[str, Any], page_width: float) -> int:
    return 0 if float(word["x0"]) < page_width / 2 else 1


def _expanded_bbox(word: dict[str, Any], width: float, height: float) -> list[float]:
    return [
        max(0.0, float(word["x0"]) - 0.75),
        max(0.0, float(word["top"]) - 0.75),
        min(width, float(word["x1"]) + 0.75),
        min(height, float(word["bottom"]) + 0.75),
    ]


def _records_for_page(
    page: Any,
    *,
    document_id: str,
    pdf_sha256: str,
    page_number: int,
) -> list[dict[str, Any]]:
    words = page.extract_words() or []
    width = float(page.width)
    height = float(page.height)
    markers = [
        (int(match.group(1)), word)
        for word in words
        if (match := _QUESTION.fullmatch(str(word.get("text") or "")))
        and _is_question_margin(word, width)
        and float(word["top"]) >= 45.0
    ]
    answers: list[tuple[str, dict[str, Any]]] = []
    for index, word in enumerate(words[:-1]):
        # Explanatory prose frequently contains a lower-case ``cevap`` token.
        # Only the source's explicit ``Cevap:`` footer is an answer marker.
        if re.fullmatch(
            r"cevap\s*:",
            str(word.get("text") or "").strip(),
            re.IGNORECASE,
        ) is None:
            continue
        answer_word = words[index + 1]
        answer = str(answer_word.get("text") or "").strip().upper()
        if (
            not _ANSWER.fullmatch(answer)
            or abs(float(answer_word["top"]) - float(word["top"])) > 3.0
            or _column(answer_word, width) != _column(word, width)
        ):
            raise BuildError(
                f"{document_id} page {page_number}: malformed inline Cevap token"
            )
        answers.append((answer, answer_word))
    records: list[dict[str, Any]] = []
    used_addresses: set[tuple[int, int]] = set()
    for answer, answer_word in answers:
        column = _column(answer_word, width)
        preceding = [
            (number, marker)
            for number, marker in markers
            if _column(marker, width) == column
            and float(marker["top"]) <= float(answer_word["top"])
        ]
        if not preceding:
            raise BuildError(
                f"{document_id} page {page_number}: answer has no preceding question"
            )
        question_number, marker = max(preceding, key=lambda item: float(item[1]["top"]))
        address = (page_number, question_number)
        if address in used_addresses:
            raise BuildError(f"{document_id}: duplicate inline source address {address}")
        used_addresses.add(address)
        later_markers = [
            other
            for _, other in markers
            if _column(other, width) == column
            and float(other["top"]) > float(marker["top"])
        ]
        content_top = max(0.0, float(marker["top"]) - 1.5)
        content_bottom = (
            min(float(other["top"]) for other in later_markers) - 1.5
            if later_markers
            else height
        )
        content_left = 0.0 if column == 0 else width / 2
        content_right = width / 2 if column == 0 else width
        content_bbox = [content_left, content_top, content_right, content_bottom]
        question_text = page.crop(tuple(content_bbox)).extract_text() or ""
        content_verification = attest_inline_solution_content(
            page,
            pdf_sha256=pdf_sha256,
            physical_page=page_number,
            bbox=content_bbox,
            question_number=question_number,
            question_text=question_text,
        )
        records.append(
            {
                "record_id": f"{document_id}:p{page_number}:q{question_number}",
                "content_page_number": page_number,
                "question_number": question_number,
                "question_text": content_verification.content_text,
                "answer": answer,
                "answer_format": "choice",
                "key_binding_kind": "inline_solution_projected",
                "key_page_number": page_number,
                "key_bbox": _expanded_bbox(answer_word, width, height),
                "content_bbox": content_bbox,
                "content_projection_sha256": (
                    content_verification.projection_sha256
                ),
                "visually_checked": True,
            }
        )
    return sorted(records, key=lambda item: int(item["question_number"]))


def build(spec_path: Path, paths: dict[str, Path], output_path: Path) -> dict[str, Any]:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise BuildError("inline source-index builder requires pdfplumber") from exc
    spec = _load(spec_path)
    if set(spec) != _ROOT_KEYS or spec.get("schema_version") != SPEC_SCHEMA:
        raise BuildError("inline source spec has an unsupported schema")
    reject_benchmark_metadata(spec, ("inline_source_spec",))
    if str(pdfplumber.__version__) != EXPECTED_PDFPLUMBER_VERSION:
        raise BuildError(
            "inline source builder requires pdfplumber "
            f"{EXPECTED_PDFPLUMBER_VERSION}"
        )
    raw_documents = spec.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise BuildError("document spec must contain a documents list")
    configured_ids = {
        str(item.get("document_id") or "") for item in raw_documents if isinstance(item, dict)
    }
    if len(configured_ids) != len(raw_documents) or configured_ids != set(paths):
        raise BuildError("document paths must exactly match unique spec document IDs")
    output_documents: list[dict[str, Any]] = []
    input_documents: list[dict[str, str]] = []
    for raw in raw_documents:
        if not isinstance(raw, dict) or set(raw) != _DOCUMENT_KEYS:
            raise BuildError("document spec fields are not on the strict allowlist")
        document_id = str(raw["document_id"])
        pdf_path = paths[document_id]
        expected_sha = str(raw["pdf_sha256"])
        if sha256_file(pdf_path) != expected_sha:
            raise BuildError(f"PDF SHA-256 mismatch for {document_id}")
        reviewed_pages = raw.get("reviewed_inline_pages")
        if (
            not isinstance(reviewed_pages, list)
            or not reviewed_pages
            or any(
                not isinstance(page, int) or isinstance(page, bool) or page < 1
                for page in reviewed_pages
            )
            or len(set(reviewed_pages)) != len(reviewed_pages)
        ):
            raise BuildError(f"{document_id}: reviewed_inline_pages are required")
        records: list[dict[str, Any]] = []
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) != int(raw["page_count"]):
                raise BuildError(f"{document_id}: page count changed")
            for page_number in reviewed_pages:
                page_number = int(page_number)
                if not 1 <= page_number <= len(pdf.pages):
                    raise BuildError(f"{document_id}: reviewed page is outside PDF")
                page_records = _records_for_page(
                    pdf.pages[page_number - 1],
                    document_id=document_id,
                    pdf_sha256=expected_sha,
                    page_number=page_number,
                )
                if not page_records:
                    raise BuildError(f"{document_id} page {page_number}: no inline answers")
                records.extend(page_records)
        input_documents.append(
            {
                "document_id": document_id,
                "path": str(pdf_path),
                "sha256": expected_sha,
            }
        )
        output_documents.append(
            {
                "document_id": document_id,
                "locator": raw["locator"],
                "pdf_sha256": expected_sha,
                "page_count": int(raw["page_count"]),
                "content_page_ranges": raw["content_page_ranges"],
                "questions": records,
            }
        )
    output = {"schema_version": INDEX_SCHEMA, "documents": output_documents}
    validated = parse_workbook_index(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(output) + b"\n")
    return {
        "schema_version": "maxim-inline-solutions-workbook-build-v1",
        "inputs": {
            "source_spec": {
                "path": str(spec_path),
                "sha256": sha256_file(spec_path),
            },
            "documents": sorted(
                input_documents, key=lambda item: item["document_id"]
            ),
        },
        "runtime": {"pdfplumber_version": str(pdfplumber.__version__)},
        "tools": {
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "inline_projection_sha256": sha256_file(
                REPO_ROOT / "src/evidence_os/inline_solution.py"
            ),
            "workbook_verifier_sha256": sha256_file(
                REPO_ROOT / "src/evidence_os/official_workbook.py"
            ),
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "documents": len(validated.documents),
        "records": sum(len(document.questions) for document in validated.documents),
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
        InlineSolutionError,
        OfficialSourceError,
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
