#!/usr/bin/env python3
"""Freeze source-only MEB-DEF coordinate-table answer attestations.

The input is task-ID-free and contains only a pinned public PDF source index.
This script verifies content markers, question crops, table geometry, numbered
headers, answers, and section context.  It never reads benchmark rows,
candidates, outcomes, judges, or scores.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.coordinate_table_answer_key import (  # noqa: E402
    CONTENT_PROJECTION_SCHEMA,
    CoordinateTableAnswerKeyError,
    PROJECTION_SCHEMA,
    attest_content_question_marker,
    attest_coordinate_table_answer_key,
)
from evidence_os.official_ogm import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)
from evidence_os.official_workbook import (  # noqa: E402
    INDEX_SCHEMA,
    reject_benchmark_metadata,
)


INPUT_SCHEMA = "public-workbook-coordinate-table-source-fragment-v1"
OUTPUT_SCHEMA = INDEX_SCHEMA
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_KEYS = frozenset(
    {
        "document_id",
        "locator",
        "pdf_sha256",
        "page_count",
        "content_page_ranges",
        "questions",
    }
)
_QUESTION_KEYS = frozenset(
    {
        "record_id",
        "content_page_number",
        "question_number",
        "question_marker_kind",
        "question_text",
        "answer",
        "answer_format",
        "content_bbox",
        "key_page_number",
        "key_context_page_number",
        "key_bbox",
        "key_binding_kind",
        "section",
        "test_variant",
        "visually_checked",
    }
)


class MebDefTableAttestationError(RuntimeError):
    pass


def _bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise MebDefTableAttestationError(f"{label} is malformed")
    result = tuple(float(item) for item in value)
    if not (result[0] < result[2] and result[1] < result[3]):
        raise MebDefTableAttestationError(f"{label} is empty")
    return result  # type: ignore[return-value]


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise MebDefTableAttestationError(f"{label} must be a positive integer")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MebDefTableAttestationError(f"{label} must be nonempty text")
    return value


def _load_input(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema_version") != INPUT_SCHEMA:
        raise MebDefTableAttestationError(f"{path}: expected {INPUT_SCHEMA}")
    reject_benchmark_metadata(payload)
    documents = payload.get("documents")
    if not isinstance(documents, list) or len(documents) != 1:
        raise MebDefTableAttestationError("source fragment must contain one document")
    return payload


def attest(
    input_path: Path,
    pdf_path: Path,
    output_path: Path,
    manifest_path: Path,
    expected_records: int | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(_load_input(input_path))
    document = payload["documents"][0]
    if not isinstance(document, dict) or set(document) != _DOCUMENT_KEYS:
        raise MebDefTableAttestationError("source document fields are malformed")
    expected_sha = str(document.get("pdf_sha256") or "")
    if _HEX64.fullmatch(expected_sha) is None or sha256_file(pdf_path) != expected_sha:
        raise MebDefTableAttestationError("source PDF SHA-256 differs from fragment")
    expected_pages = _positive_integer(document.get("page_count"), "page_count")
    questions = document.get("questions")
    if not isinstance(questions, list) or not questions:
        raise MebDefTableAttestationError("source fragment has no questions")
    if expected_records is not None and len(questions) != expected_records:
        raise MebDefTableAttestationError(
            f"expected {expected_records} records, found {len(questions)}"
        )

    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise MebDefTableAttestationError("table attestation requires pdfplumber") from exc

    attestations: list[dict[str, Any]] = []
    seen_addresses: set[tuple[int, int]] = set()
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != expected_pages:
            raise MebDefTableAttestationError("source PDF page count differs from fragment")
        for question in questions:
            if not isinstance(question, dict) or set(question) != _QUESTION_KEYS:
                raise MebDefTableAttestationError("source question fields are malformed")
            content_page = _positive_integer(
                question.get("content_page_number"), "content_page_number"
            )
            question_number = _positive_integer(
                question.get("question_number"), "question_number"
            )
            key_page = _positive_integer(
                question.get("key_page_number"), "key_page_number"
            )
            context_page = _positive_integer(
                question.get("key_context_page_number"), "key_context_page_number"
            )
            address = (content_page, question_number)
            if (
                min(content_page, question_number, key_page, context_page) < 1
                or max(content_page, key_page, context_page) > len(pdf.pages)
                or address in seen_addresses
            ):
                raise MebDefTableAttestationError("source question address is malformed")
            expected_record_id = (
                f"{_nonempty_string(document.get('document_id'), 'document_id')}:"
                f"p{content_page}:q{question_number}"
            )
            if question.get("record_id") != expected_record_id:
                raise MebDefTableAttestationError("record_id is not source-addressed")
            if (
                question.get("answer_format") != "short_text"
                or question.get("key_binding_kind") != "coordinate_table_answer_key"
                or question.get("visually_checked") is not True
            ):
                raise MebDefTableAttestationError("source question binding is malformed")
            content_verification = attest_content_question_marker(
                pdf.pages[content_page - 1],
                pdf_sha256=expected_sha,
                physical_page=content_page,
                bbox=_bbox(question.get("content_bbox"), "content_bbox"),
                question_number=question_number,
                marker_kind=_nonempty_string(
                    question.get("question_marker_kind"), "question_marker_kind"
                ),
                question_text=_nonempty_string(
                    question.get("question_text"), "question_text"
                ),
            )
            verification = attest_coordinate_table_answer_key(
                pdf.pages[key_page - 1],
                pdf_sha256=expected_sha,
                physical_page=key_page,
                bbox=_bbox(question.get("key_bbox"), "key_bbox"),
                question_number=question_number,
                expected_answer=_nonempty_string(question.get("answer"), "answer"),
                expected_section=_nonempty_string(question.get("section"), "section"),
                expected_test_variant=_nonempty_string(
                    question.get("test_variant"), "test_variant"
                ),
                section_page=pdf.pages[context_page - 1],
                section_physical_page=context_page,
            )
            key_crop_text = " ".join(
                (pdf.pages[key_page - 1].crop(
                    _bbox(question.get("key_bbox"), "key_bbox")
                ).extract_text() or "").split()
            )
            question["content_projection_sha256"] = (
                content_verification.projection_sha256
            )
            question["key_projection_sha256"] = verification.projection_sha256
            question["key_crop_text"] = key_crop_text
            attestations.append(
                {
                    "record_id": expected_record_id,
                    "content_page": content_page,
                    "content_marker_kind": question["question_marker_kind"],
                    "content_marker_count": content_verification.marker_count,
                    "content_projection_sha256": (
                        content_verification.projection_sha256
                    ),
                    "key_page": key_page,
                    "key_context_page": context_page,
                    "key_projection_sha256": verification.projection_sha256,
                    "component_matches": dict(verification.component_matches),
                }
            )
            seen_addresses.add(address)

    payload["schema_version"] = OUTPUT_SCHEMA
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    manifest = {
        "schema_version": "maxim-meb-def-coordinate-table-attestation-v1",
        "projection_schema": PROJECTION_SCHEMA,
        "content_projection_schema": CONTENT_PROJECTION_SCHEMA,
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "pdf": {
            "path": str(pdf_path),
            "sha256": expected_sha,
            "page_count": expected_pages,
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "records": len(attestations),
        "attestations": sorted(attestations, key=lambda item: item["record_id"]),
        "integration_status": {
            "official_workbook_binding_allowlisted": True,
            "required_change": None,
        },
        "task_id_present": False,
        "benchmark_candidate_or_outcome_access": False,
        "pdfplumber_version": str(pdfplumber.__version__),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-records", type=int)
    args = parser.parse_args()
    try:
        result = attest(
            args.input.resolve(),
            args.pdf.resolve(),
            args.output.resolve(),
            args.manifest.resolve(),
            args.expected_records,
        )
    except (
        MebDefTableAttestationError,
        CoordinateTableAnswerKeyError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
