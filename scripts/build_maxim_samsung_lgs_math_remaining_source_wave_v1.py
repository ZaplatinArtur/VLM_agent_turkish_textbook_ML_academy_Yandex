#!/usr/bin/env python3
"""Build the task-ID-free Samsung LGS math source expansion.

This source-only wave adds four reviewed physical source addresses from the
pinned Samsung Il MEM LGS-1 booklet.  Selection and verification use only the
official PDF and the already-frozen workbook index; benchmark answers, solver
outputs, judge outcomes, evaluations, and scores are neither inputs nor policy
features.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.official_ogm import canonical_json_bytes, sha256_file  # noqa: E402
from evidence_os.official_workbook import (  # noqa: E402
    parse_workbook_index,
    reject_benchmark_metadata,
    verify_workbook_index_pdf,
)


REPORT_ROOT = ROOT / "reports" / "maxim_official_exact_source_v2_20260805"
FROZEN = REPORT_ROOT / "frozen"
BASE_INDEX = (
    FROZEN
    / (
        "public_workbook_source_index_meb3a_samsungis_mebdef10_sociology_"
        "biology_strict_candidate_v6.json"
    )
)
PDF = ROOT / "tmp" / "pdfs" / "portfolio_official_sources" / "samsung_lgs1.pdf"
FRAGMENT = (
    FROZEN
    / "public_workbook_source_fragment_samsungis_lgs1_math_remaining_candidate_v1.json"
)
OUTPUT_INDEX = (
    FROZEN
    / "public_workbook_source_index_samsungis_lgs1_math_remaining_candidate_v7.json"
)
MANIFEST = (
    FROZEN
    / "public_workbook_source_index_samsungis_lgs1_math_remaining_candidate_v7.manifest.json"
)

DOCUMENT_ID = "samsungis_lgs1_f88c9f40e3c6"
PDF_SHA256 = "f88c9f40e3c6f3a2494090ee7635d1f7254b736da561b0f0b98125cb25ad5997"
SECTION = "MATEMATİK"
TEST_VARIANT = "Samsungis LGS Denemeleri - 1"

# Physical PDF source addresses, frozen only after visual review of content
# pages 16-18 and the embedded answer list on physical page 25.  These records
# intentionally contain no benchmark row identifier.
NEW_QUESTIONS: tuple[dict[str, Any], ...] = (
    {
        "record_id": f"{DOCUMENT_ID}:p16:q12",
        "content_page_number": 16,
        "question_number": 12,
        "answer": "C",
        "key_binding_kind": "answer_key_list",
        "section": SECTION,
        "test_variant": TEST_VARIANT,
        "key_page_number": 25,
        "key_bbox": [402.664, 205.191, 421.672, 214.191],
        "visually_checked": True,
    },
    {
        "record_id": f"{DOCUMENT_ID}:p17:q15",
        "content_page_number": 17,
        "question_number": 15,
        "answer": "C",
        "key_binding_kind": "answer_key_list",
        "section": SECTION,
        "test_variant": TEST_VARIANT,
        "key_page_number": 25,
        "key_bbox": [402.664, 237.591, 421.672, 246.591],
        "visually_checked": True,
    },
    {
        "record_id": f"{DOCUMENT_ID}:p17:q16",
        "content_page_number": 17,
        "question_number": 16,
        "answer": "B",
        "key_binding_kind": "answer_key_list",
        "section": SECTION,
        "test_variant": TEST_VARIANT,
        "key_page_number": 25,
        "key_bbox": [402.912, 248.391, 421.425, 257.391],
        "visually_checked": True,
    },
    {
        "record_id": f"{DOCUMENT_ID}:p18:q19",
        "content_page_number": 18,
        "question_number": 19,
        "answer": "B",
        "key_binding_kind": "answer_key_list",
        "section": SECTION,
        "test_variant": TEST_VARIANT,
        "key_page_number": 25,
        "key_bbox": [402.912, 280.791, 421.425, 289.791],
        "visually_checked": True,
    },
)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("/", "\\")


def _write_canonical(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _document(payload: dict[str, Any], document_id: str) -> dict[str, Any]:
    matches = [
        document
        for document in payload.get("documents", [])
        if isinstance(document, dict) and document.get("document_id") == document_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one source document {document_id}")
    return matches[0]


def build() -> dict[str, Any]:
    base_payload = json.loads(BASE_INDEX.read_text(encoding="utf-8-sig"))
    reject_benchmark_metadata(base_payload)
    base_index = parse_workbook_index(base_payload)
    base_document = _document(base_payload, DOCUMENT_ID)
    if str(base_document.get("pdf_sha256") or "") != PDF_SHA256:
        raise RuntimeError("frozen Samsung document has an unexpected PDF identity")
    actual_pdf_sha256 = sha256_file(PDF)
    if actual_pdf_sha256 != PDF_SHA256:
        raise RuntimeError("official Samsung PDF SHA-256 changed")

    existing_addresses = {
        (int(question["content_page_number"]), int(question["question_number"]))
        for question in base_document.get("questions", [])
    }
    new_addresses = {
        (int(question["content_page_number"]), int(question["question_number"]))
        for question in NEW_QUESTIONS
    }
    if len(new_addresses) != len(NEW_QUESTIONS):
        raise RuntimeError("Samsung source expansion contains duplicate addresses")
    if existing_addresses & new_addresses:
        raise RuntimeError("Samsung source expansion overlaps the frozen base index")

    fragment_document = {
        key: deepcopy(value)
        for key, value in base_document.items()
        if key != "questions"
    }
    fragment_document["questions"] = [deepcopy(question) for question in NEW_QUESTIONS]
    fragment_payload = {
        "schema_version": "public-workbook-source-index-v1",
        "documents": [fragment_document],
    }
    reject_benchmark_metadata(fragment_payload)
    fragment_index = parse_workbook_index(fragment_payload)
    fragment_verification = verify_workbook_index_pdf(
        PDF, fragment_index.documents[0]
    )

    combined_payload = deepcopy(base_payload)
    combined_document = _document(combined_payload, DOCUMENT_ID)
    combined_document["questions"].extend(
        deepcopy(question) for question in NEW_QUESTIONS
    )
    combined_document["questions"] = sorted(
        combined_document["questions"],
        key=lambda question: (
            int(question["content_page_number"]),
            int(question["question_number"]),
            str(question["record_id"]),
        ),
    )
    reject_benchmark_metadata(combined_payload)
    combined_index = parse_workbook_index(combined_payload)
    combined_parsed_document = next(
        document
        for document in combined_index.documents
        if document.document_id == DOCUMENT_ID
    )
    combined_verification = verify_workbook_index_pdf(
        PDF, combined_parsed_document
    )

    _write_canonical(FRAGMENT, fragment_payload)
    _write_canonical(OUTPUT_INDEX, combined_payload)
    if b"val_" in FRAGMENT.read_bytes() or b"val_" in OUTPUT_INDEX.read_bytes():
        raise RuntimeError("benchmark row identifier leaked into a source-native artifact")

    manifest_payload = {
        "schema_version": "maxim-samsung-lgs-math-remaining-source-build-v1",
        "benchmark_answer_reference_score_or_outcome_access": False,
        "task_id_present_in_fragment_or_index": False,
        "task_id_used_for_policy": False,
        "selection_policy": {
            "official_source_only": True,
            "source_document": DOCUMENT_ID,
            "section": SECTION,
            "physical_content_pages": [16, 17, 18],
            "physical_key_page": 25,
            "binding_kind": "answer_key_list",
            "runtime_address": (
                "strict_document_identity_plus_content_page_plus_"
                "printed_question_number"
            ),
        },
        "inputs": {
            "builder": {
                "path": _relative(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "base_index": {
                "path": _relative(BASE_INDEX),
                "sha256": sha256_file(BASE_INDEX),
            },
            "official_pdf": {
                "path": _relative(PDF),
                "sha256": actual_pdf_sha256,
                "page_count": fragment_index.documents[0].page_count,
            },
        },
        "source_verification": {
            "fragment": fragment_verification,
            "combined_samsung_document": combined_verification,
        },
        "output": {
            "fragment": {
                "path": _relative(FRAGMENT),
                "sha256": sha256_file(FRAGMENT),
            },
            "index": {
                "path": _relative(OUTPUT_INDEX),
                "sha256": sha256_file(OUTPUT_INDEX),
            },
            "added_documents": 0,
            "added_records": len(NEW_QUESTIONS),
            "samsung_records_before": len(base_document["questions"]),
            "samsung_records_after": len(combined_document["questions"]),
            "total_documents": len(combined_index.documents),
            "total_records": sum(
                len(document.questions) for document in combined_index.documents
            ),
        },
    }
    _write_canonical(MANIFEST, manifest_payload)
    return manifest_payload


def main() -> int:
    print(json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
