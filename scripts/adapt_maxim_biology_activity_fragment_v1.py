#!/usr/bin/env python3
"""Adapt a pinned MEB activity source wave to the canonical workbook index.

The adapter is deliberately source-only.  It validates the raw projection
recipe against the immutable PDF, copies document metadata from an existing
canonical source index, and emits only the newly attested activity records.
It never reads benchmark rows, candidates, scores, or outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_os.activity_answer_key import (  # noqa: E402
    ActivityAnswerKeyError,
    attest_activity_answer_key,
)
from evidence_os.official_ogm import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)
from evidence_os.official_workbook import (  # noqa: E402
    INDEX_SCHEMA,
    parse_workbook_index,
    reject_benchmark_metadata,
)


RAW_SCHEMA = "public-workbook-biology-activity-source-fragment-v1"
MANIFEST_SCHEMA = "maxim-biology-activity-fragment-adaptation-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AdaptActivityError(RuntimeError):
    """The raw source wave cannot produce a canonical activity fragment."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise AdaptActivityError(f"{path}: expected a JSON object")
    return value


def _one(values: Any, label: str) -> dict[str, Any]:
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise AdaptActivityError(f"{label} must contain exactly one object")
    return values[0]


def _positive(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AdaptActivityError(f"{label} must be a positive integer")
    return value


def adapt(
    raw_path: Path,
    base_index_path: Path,
    pdf_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    raw = _load(raw_path)
    if raw.get("schema_version") != RAW_SCHEMA or raw.get("source_only") is not True:
        raise AdaptActivityError("raw activity source wave has an unsupported schema")
    reject_benchmark_metadata(raw, ("raw_activity_fragment",))

    base_payload = _load(base_index_path)
    base_index = parse_workbook_index(base_payload)
    raw_document = _one(raw.get("documents"), "raw documents")
    document_id = str(raw_document.get("document_id") or "")
    base_documents = {
        document.document_id: document for document in base_index.documents
    }
    base_document = base_documents.get(document_id)
    if base_document is None:
        raise AdaptActivityError("raw activity document is absent from the base index")
    base_raw_documents = {
        str(document.get("document_id") or ""): document
        for document in base_payload.get("documents", [])
        if isinstance(document, dict)
    }
    base_raw_document = base_raw_documents.get(document_id)
    if base_raw_document is None:
        raise AdaptActivityError("base document metadata is unavailable")

    expected_identity = {
        "source_file_name": base_document.identity.name,
        "public_locator": base_document.identity.public_locator,
        "pdf_sha256": base_document.pdf_sha256,
        "page_count": base_document.page_count,
    }
    if any(raw_document.get(key) != value for key, value in expected_identity.items()):
        raise AdaptActivityError("raw and canonical document identities differ")
    if sha256_file(pdf_path) != base_document.pdf_sha256:
        raise AdaptActivityError("activity PDF hash differs from the canonical document")

    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise AdaptActivityError("activity adaptation requires pdfplumber") from exc
    if str(pdfplumber.__version__) != "0.11.9":
        raise AdaptActivityError("activity source wave requires pdfplumber 0.11.9")

    raw_units = raw_document.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise AdaptActivityError("raw activity document has no unit inventory")
    units: dict[int, dict[str, Any]] = {}
    for unit in raw_units:
        if not isinstance(unit, dict):
            raise AdaptActivityError("raw activity unit is malformed")
        unit_number = _positive(unit.get("unit_number"), "unit_number")
        if unit_number in units:
            raise AdaptActivityError("raw activity unit is duplicated")
        units[unit_number] = unit

    activities = raw_document.get("activities")
    if not isinstance(activities, list) or not activities:
        raise AdaptActivityError("raw activity document has no activities")

    questions: list[dict[str, Any]] = []
    joint_pins: dict[str, str] = {}
    seen_addresses: set[tuple[int, int, int]] = set()
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != base_document.page_count:
            raise AdaptActivityError("activity PDF page count changed")
        for raw_activity in activities:
            if not isinstance(raw_activity, dict):
                raise AdaptActivityError("raw activity record is malformed")
            unit_number = _positive(
                raw_activity.get("source_address", {}).get("unit_number")
                if isinstance(raw_activity.get("source_address"), dict)
                else None,
                "source unit number",
            )
            content_page = _positive(
                raw_activity.get("content_physical_page"), "content physical page"
            )
            printed_page = _positive(
                raw_activity.get("printed_page_number"), "printed page number"
            )
            activity_number = _positive(
                raw_activity.get("activity_number"), "activity number"
            )
            key_page = _positive(
                raw_activity.get("key_physical_page"), "key physical page"
            )
            address = (unit_number, printed_page, activity_number)
            if address in seen_addresses:
                raise AdaptActivityError("raw activity source address is duplicated")
            seen_addresses.add(address)
            unit = units.get(unit_number)
            if unit is None:
                raise AdaptActivityError("activity names a unit outside the inventory")
            source_address = raw_activity.get("source_address")
            expected_source_address = {
                "pdf_sha256": base_document.pdf_sha256,
                "unit_number": unit_number,
                "printed_page_number": printed_page,
                "activity_number": activity_number,
            }
            if source_address != expected_source_address or content_page != printed_page:
                raise AdaptActivityError("activity source address is internally inconsistent")
            expected_raw_id = (
                f"{document_id}:u{unit_number}:p{printed_page}:activity{activity_number}"
            )
            if raw_activity.get("record_id") != expected_raw_id:
                raise AdaptActivityError("raw activity record_id is not source-addressed")
            content_range = unit.get("physical_content_page_range")
            if (
                not isinstance(content_range, list)
                or len(content_range) != 2
                or not all(isinstance(item, int) and not isinstance(item, bool) for item in content_range)
                or not content_range[0] <= content_page <= content_range[1]
                or unit.get("key_physical_page") != key_page
            ):
                raise AdaptActivityError("activity is outside its pinned unit range")
            if (
                raw_activity.get("marker_kind") != "activity_label"
                or raw_activity.get("key_binding_kind") != "activity_answer_key"
                or raw_activity.get("text_visible_activity_marker") is not True
                or raw_activity.get("visually_checked") is not True
            ):
                raise AdaptActivityError("activity lacks source-visible review gates")
            canonical_answer = str(raw_activity.get("canonical_answer") or "").strip()
            source_answer_format = str(raw_activity.get("answer_format") or "").strip()
            try:
                verification = attest_activity_answer_key(
                    pdf.pages[content_page - 1],
                    pdf.pages[key_page - 1],
                    pdf_sha256=base_document.pdf_sha256,
                    content_physical_page=content_page,
                    key_physical_page=key_page,
                    unit_number=unit_number,
                    activity_number=activity_number,
                    activity_page_number=printed_page,
                    content_bbox=raw_activity.get("content_bbox"),
                    key_bbox=raw_activity.get("key_bbox"),
                    answer_format=source_answer_format,
                    canonical_answer=canonical_answer,
                )
            except ActivityAnswerKeyError as exc:
                raise AdaptActivityError(
                    f"activity PDF proof failed for {expected_raw_id}: {exc}"
                ) from exc
            if (
                verification.content_projection_sha256
                != raw_activity.get("content_projection_sha256")
                or verification.key_projection_sha256
                != raw_activity.get("key_projection_sha256")
            ):
                raise AdaptActivityError(
                    f"raw activity projection pin differs for {expected_raw_id}"
                )
            if _HEX64.fullmatch(verification.projection_sha256) is None:
                raise AdaptActivityError("activity joint projection is malformed")

            canonical_record_id = (
                f"{document_id}:p{content_page}:q{activity_number}"
            )
            joint_pins[canonical_record_id] = verification.projection_sha256
            unit_title = str(unit.get("unit_title") or "").strip()
            if not unit_title or raw_activity.get("unit_title") != unit_title:
                raise AdaptActivityError("activity unit title differs from its inventory")
            content_crop_text = " ".join(
                (
                    pdf.pages[content_page - 1]
                    .crop(tuple(raw_activity.get("content_bbox")))
                    .extract_text()
                    or ""
                ).split()
            )
            key_crop_text = " ".join(
                (
                    pdf.pages[key_page - 1]
                    .crop(tuple(raw_activity.get("key_bbox")))
                    .extract_text()
                    or ""
                ).split()
            )
            if not content_crop_text or not key_crop_text:
                raise AdaptActivityError("activity canonical crop text is empty")
            questions.append(
                {
                    "record_id": canonical_record_id,
                    "content_page_number": content_page,
                    "question_number": activity_number,
                    "question_marker_kind": "activity_label",
                    "question_text": content_crop_text,
                    "answer": canonical_answer,
                    "answer_format": "short_text",
                    "key_crop_text": key_crop_text,
                    "key_projection_sha256": verification.key_projection_sha256,
                    "content_projection_sha256": verification.content_projection_sha256,
                    "binding_projection_sha256": verification.projection_sha256,
                    "key_page_number": key_page,
                    "key_context_page_number": key_page,
                    "key_bbox": raw_activity.get("key_bbox"),
                    "content_bbox": raw_activity.get("content_bbox"),
                    "key_binding_kind": "activity_answer_key",
                    "test_variant": f"{unit_number}. ÜNİTE",
                    "source_unit_number": unit_number,
                    "source_answer_format": source_answer_format,
                    "visually_checked": True,
                }
            )

    fragment_document = {
        key: value for key, value in base_raw_document.items() if key != "questions"
    }
    fragment_document["questions"] = sorted(
        questions,
        key=lambda question: (
            question["content_page_number"],
            question["question_number"],
        ),
    )
    output = {"schema_version": INDEX_SCHEMA, "documents": [fragment_document]}
    validated = parse_workbook_index(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(output) + b"\n")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "inputs": {
            "raw_fragment": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
            "base_index": {
                "path": str(base_index_path),
                "sha256": sha256_file(base_index_path),
            },
            "pdf": {"path": str(pdf_path), "sha256": sha256_file(pdf_path)},
        },
        "runtime": {"pdfplumber_version": str(pdfplumber.__version__)},
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "document_id": document_id,
        "records": sum(len(document.questions) for document in validated.documents),
        "joint_projection_sha256": dict(sorted(joint_pins.items())),
        "raw_projection_hashes_reproduced": len(questions) * 2,
        "task_id_present": False,
        "benchmark_candidate_or_outcome_access": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = adapt(
            args.raw_json.resolve(),
            args.base_index.resolve(),
            args.pdf.resolve(),
            args.output.resolve(),
            args.manifest.resolve(),
        )
    except (
        AdaptActivityError,
        ActivityAnswerKeyError,
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
