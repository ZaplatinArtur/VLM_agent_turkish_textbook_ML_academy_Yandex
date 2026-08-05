#!/usr/bin/env python3
"""Adapt the source-only MEB Sociology wave to strict coordinate-choice records.

The adapter reads only a reviewed public-source fragment and the immutable PDF.
It never accepts or opens benchmark rows, task IDs, model candidates, scores,
or outcomes.  Every emitted answer is re-derived from PDF geometry and paired
with an independently attested content crop before the canonical fragment is
written.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_os.coordinate_choice_answer_key import (  # noqa: E402
    CoordinateChoiceAnswerKeyError,
    attest_coordinate_choice_answer_key,
    attest_coordinate_choice_content,
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


RAW_SCHEMA = "blind-public-source-index-fragment-v1"
MANIFEST_SCHEMA = "maxim-sociology-coordinate-choice-fragment-adaptation-v1"
PDFPLUMBER_VERSION = "0.11.9"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_ID = re.compile(r"^sha256:([0-9a-f]{64})$")
_CANONICAL_DOCUMENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")
_RAW_ROOT_KEYS = frozenset({"schema_version", "records"})
_RAW_RECORD_KEYS = frozenset(
    {
        "document_id",
        "public_locator",
        "public_name",
        "pdf_sha256",
        "page_count",
        "content_page_ranges",
        "content_page_number",
        "question_number",
        "answer",
        "answer_format",
        "key_binding_kind",
        "key_page_number",
        "key_bbox",
        "content_bbox",
        "section",
        "test_variant",
        "question_text",
        "visually_checked",
    }
)


class AdaptSociologyError(RuntimeError):
    """The reviewed Sociology wave cannot produce a strict source fragment."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise AdaptSociologyError(f"{path}: expected one JSON object")
    return value


def _positive(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AdaptSociologyError(f"{label} must be a positive integer")
    return value


def _literal(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AdaptSociologyError(f"{label} must be non-empty canonical text")
    return value


def _bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 4
        or any(
            not isinstance(item, (int, float)) or isinstance(item, bool)
            for item in value
        )
    ):
        raise AdaptSociologyError(f"{label} is malformed")
    result = tuple(float(item) for item in value)
    if not (result[0] < result[2] and result[1] < result[3]):
        raise AdaptSociologyError(f"{label} is empty")
    return result  # type: ignore[return-value]


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not result:
        raise AdaptSociologyError("public PDF name has no canonical slug")
    return result[:40]


def _canonical_document_id(name: str, pdf_sha256: str) -> str:
    document_id = f"yandex_{_slug(Path(name).stem)}_{pdf_sha256[:12]}"
    if _CANONICAL_DOCUMENT_ID.fullmatch(document_id) is None:
        raise AdaptSociologyError("canonical Sociology document_id is malformed")
    return document_id


def _identity(record: Mapping[str, Any]) -> tuple[str, str, str, int]:
    raw_document_id = _literal(record.get("document_id"), "raw document_id")
    match = _DOCUMENT_ID.fullmatch(raw_document_id)
    pdf_sha256 = _literal(record.get("pdf_sha256"), "raw PDF sha256")
    if match is None or _HEX64.fullmatch(pdf_sha256) is None or match.group(1) != pdf_sha256:
        raise AdaptSociologyError("raw document_id is not its exact PDF SHA-256")
    locator = _literal(record.get("public_locator"), "public locator")
    name = _literal(record.get("public_name"), "public PDF name")
    if (
        not locator.startswith("ya-disk-public://")
        or any(character.isspace() for character in locator)
        or not locator.endswith(f":/{name}")
        or Path(name).name != name
        or not name.casefold().endswith(".pdf")
    ):
        raise AdaptSociologyError("raw public PDF identity is malformed")
    return locator, name, pdf_sha256, _positive(record.get("page_count"), "page_count")


def _strict_records(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(raw) != _RAW_ROOT_KEYS or raw.get("schema_version") != RAW_SCHEMA:
        raise AdaptSociologyError("raw Sociology fragment has an unsupported schema")
    reject_benchmark_metadata(raw, ("raw_sociology_fragment",))
    records = raw.get("records")
    if (
        not isinstance(records, list)
        or not records
        or not all(isinstance(record, dict) for record in records)
    ):
        raise AdaptSociologyError("raw Sociology fragment has no source records")
    strict: list[dict[str, Any]] = []
    for record in records:
        if set(record) != _RAW_RECORD_KEYS:
            raise AdaptSociologyError(
                "raw Sociology record fields are not on the strict source allowlist"
            )
        strict.append(record)
    return strict


def adapt(
    raw_path: Path,
    pdf_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    raw = _load(raw_path)
    records = _strict_records(raw)
    identities = {_identity(record) for record in records}
    if len(identities) != 1:
        raise AdaptSociologyError("raw Sociology records disagree on document identity")
    locator, name, pdf_sha256, page_count = next(iter(identities))
    if sha256_file(pdf_path) != pdf_sha256:
        raise AdaptSociologyError("Sociology PDF hash differs from the source fragment")
    document_id = _canonical_document_id(name, pdf_sha256)

    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise AdaptSociologyError("Sociology adaptation requires pdfplumber") from exc
    if str(pdfplumber.__version__) != PDFPLUMBER_VERSION:
        raise AdaptSociologyError(
            f"Sociology adaptation requires pdfplumber {PDFPLUMBER_VERSION}"
        )

    canonical_questions: list[dict[str, Any]] = []
    pins: dict[str, dict[str, str]] = {}
    seen_addresses: set[tuple[int, int]] = set()
    content_pages: set[int] = set()
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != page_count:
            raise AdaptSociologyError("Sociology PDF page count changed")
        for record in records:
            content_page = _positive(
                record.get("content_page_number"), "content page number"
            )
            question_number = _positive(
                record.get("question_number"), "question number"
            )
            key_page = _positive(record.get("key_page_number"), "key page number")
            if content_page > page_count or key_page > page_count:
                raise AdaptSociologyError("source address is outside the Sociology PDF")
            address = (content_page, question_number)
            if address in seen_addresses:
                raise AdaptSociologyError("raw Sociology source address is duplicated")
            seen_addresses.add(address)
            content_pages.add(content_page)
            if record.get("content_page_ranges") != [[content_page, content_page]]:
                raise AdaptSociologyError(
                    "raw content range is not its exact reviewed physical page"
                )
            answer = _literal(record.get("answer"), "source answer").upper()
            if (
                answer not in frozenset("ABCDE")
                or record.get("answer_format") != "choice"
                or record.get("key_binding_kind") != "answer_key_table"
                or record.get("visually_checked") is not True
            ):
                raise AdaptSociologyError(
                    "raw Sociology record lacks reviewed choice/key gates"
                )
            section = _literal(record.get("section"), "key section")
            test_variant = _literal(record.get("test_variant"), "key test variant")
            question_text = _literal(record.get("question_text"), "question text")
            key_bbox = _bbox(record.get("key_bbox"), "key bbox")
            content_bbox = _bbox(record.get("content_bbox"), "content bbox")
            try:
                key_verification = attest_coordinate_choice_answer_key(
                    pdf.pages[key_page - 1],
                    pdf_sha256=pdf_sha256,
                    physical_page=key_page,
                    bbox=key_bbox,
                    question_number=question_number,
                    expected_answer=answer,
                    expected_section=section,
                    expected_test_variant=test_variant,
                )
                content_verification = attest_coordinate_choice_content(
                    pdf.pages[content_page - 1],
                    pdf_sha256=pdf_sha256,
                    physical_page=content_page,
                    bbox=content_bbox,
                    question_number=question_number,
                    question_text=question_text,
                    expected_content_unit=test_variant,
                    expected_test_variant=test_variant,
                )
            except CoordinateChoiceAnswerKeyError as exc:
                raise AdaptSociologyError(
                    "Sociology PDF proof failed for "
                    f"p{content_page}:q{question_number}: {exc}"
                ) from exc
            if key_verification.unit_number != content_verification.unit_number:
                raise AdaptSociologyError(
                    "Sociology key and content attest different units"
                )
            key_crop_text = (
                pdf.pages[key_page - 1].crop(key_bbox).extract_text() or ""
            ).strip()
            if not key_crop_text:
                raise AdaptSociologyError("Sociology key crop is empty")
            canonical_record_id = (
                f"{document_id}:p{content_page}:q{question_number}"
            )
            pins[canonical_record_id] = {
                "key_projection_sha256": key_verification.projection_sha256,
                "content_projection_sha256": (
                    content_verification.projection_sha256
                ),
            }
            canonical_questions.append(
                {
                    "record_id": canonical_record_id,
                    "content_page_number": content_page,
                    "question_number": question_number,
                    "question_marker_kind": "numbered_item",
                    "question_text": question_text,
                    "answer": answer,
                    "answer_format": "choice",
                    "key_crop_text": key_crop_text,
                    "key_projection_sha256": key_verification.projection_sha256,
                    "content_projection_sha256": (
                        content_verification.projection_sha256
                    ),
                    "key_binding_kind": "coordinate_choice_answer_key",
                    "key_page_number": key_page,
                    "key_context_page_number": key_page,
                    "key_bbox": [round(value, 3) for value in key_bbox],
                    "content_bbox": [round(value, 3) for value in content_bbox],
                    "section": section,
                    "test_variant": test_variant,
                    "content_section": test_variant,
                    "visually_checked": True,
                }
            )

    canonical_questions.sort(
        key=lambda question: (
            question["content_page_number"],
            question["question_number"],
        )
    )
    page_ranges = [[page, page] for page in sorted(content_pages)]
    output = {
        "schema_version": INDEX_SCHEMA,
        "documents": [
            {
                "document_id": document_id,
                "locator": {
                    "kind": "yandex_public",
                    "public_locator": locator,
                    "name": name,
                },
                "pdf_sha256": pdf_sha256,
                "page_count": page_count,
                "content_page_ranges": page_ranges,
                "questions": canonical_questions,
            }
        ],
    }
    validated = parse_workbook_index(output)
    if len(validated.documents) != 1:
        raise AdaptSociologyError("canonical Sociology fragment is not one document")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(output) + b"\n")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source_only": True,
        "inputs": {
            "raw_fragment": {
                "path": str(raw_path),
                "sha256": sha256_file(raw_path),
            },
            "pdf": {"path": str(pdf_path), "sha256": sha256_file(pdf_path)},
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "pdfplumber_version": str(pdfplumber.__version__),
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "document_id": document_id,
        "records": sum(
            len(document.questions) for document in validated.documents
        ),
        "projection_sha256": dict(sorted(pins.items())),
        "task_id_present": False,
        "benchmark_candidate_or_outcome_access": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = adapt(
            args.raw_json.resolve(),
            args.pdf.resolve(),
            args.output.resolve(),
            args.manifest.resolve(),
        )
    except (
        AdaptSociologyError,
        CoordinateChoiceAnswerKeyError,
        OfficialSourceError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
