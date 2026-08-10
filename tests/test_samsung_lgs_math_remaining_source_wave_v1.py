from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from evidence_os.official_ogm import (
    PageMatcher,
    parser_observation_primary_layout_number,
    sha256_file,
)
from evidence_os.official_workbook import (
    WorkbookThresholds,
    _question_marker_count,
    parse_workbook_index,
    resolve_workbook_question,
    verify_workbook_index_pdf,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "reports" / "maxim_official_exact_source_v2_20260805" / "frozen"
FRAGMENT = (
    FROZEN
    / "public_workbook_source_fragment_samsungis_lgs1_math_remaining_candidate_v1.json"
)
INDEX = (
    FROZEN
    / "public_workbook_source_index_samsungis_lgs1_math_remaining_candidate_v7.json"
)
MANIFEST = (
    FROZEN
    / "public_workbook_source_index_samsungis_lgs1_math_remaining_candidate_v7.manifest.json"
)
BASE_INDEX = (
    FROZEN
    / (
        "public_workbook_source_index_meb3a_samsungis_mebdef10_sociology_"
        "biology_strict_candidate_v6.json"
    )
)
PDF = ROOT / "tmp" / "pdfs" / "portfolio_official_sources" / "samsung_lgs1.pdf"
PARSER = (
    ROOT
    / "reports"
    / "maxim_document_parser_v1_20260803"
    / "parser_augmented_solver_v1"
    / "parser_artifacts"
    / "parser_results_274.jsonl"
)

SOURCE_URL = "https://samsungis.meb.gov.tr/storage/denemeler/lgs/lgs1.pdf"
DOCUMENT_ID = "samsungis_lgs1_f88c9f40e3c6"
PDF_SHA256 = "f88c9f40e3c6f3a2494090ee7635d1f7254b736da561b0f0b98125cb25ad5997"
EXPECTED_SOURCE_RECORDS = {
    (16, 12): "C",
    (17, 15): "C",
    (17, 16): "B",
    (18, 19): "B",
}
# These IDs are test-fixture join keys only.  They never enter the source
# artifacts or runtime selection policy; the test below also renames each key
# and proves that the content-derived source result is unchanged.
PARSER_ALIGNMENT = {
    "val_0042": (16, 12),
    "val_0043": (18, 19),
    "val_0044": (17, 16),
    "val_0046": (17, 15),
}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_parser_rows() -> dict[str, dict[str, object]]:
    wanted = set(PARSER_ALIGNMENT)
    rows = {}
    for line in PARSER.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_id") in wanted:
            rows[str(row["task_id"])] = row
    assert set(rows) == wanted
    return rows


def _strict_v6_thresholds() -> WorkbookThresholds:
    return WorkbookThresholds(
        min_page_coverage=0.65,
        min_page_matched_tokens=10,
        min_page_margin=0.12,
        min_numberless_question_coverage=1.0,
        min_numberless_question_matched_tokens=999,
        min_numberless_question_margin=1.0,
    )


def test_glued_pdf_question_marker_has_a_narrow_source_gate() -> None:
    assert _question_marker_count("12.Sefa starts the printed question", 12) == 1
    assert _question_marker_count("\n12)Sefa starts the printed question", 12) == 1

    for non_marker in (
        "x12.Sefa",
        "312.Sefa",
        "12.5 is a decimal",
        "prefix_12.Sefa",
        "12._field",
    ):
        assert _question_marker_count(non_marker, 12) == 0


def test_source_artifacts_are_task_id_free_and_add_only_reviewed_addresses() -> None:
    fragment_bytes = FRAGMENT.read_bytes()
    index_bytes = INDEX.read_bytes()
    assert b"val_" not in fragment_bytes
    assert b"val_" not in index_bytes

    fragment = parse_workbook_index(_load_json(FRAGMENT))
    combined = parse_workbook_index(_load_json(INDEX))
    base = parse_workbook_index(_load_json(BASE_INDEX))
    assert len(fragment.documents) == 1
    document = fragment.documents[0]
    assert document.document_id == DOCUMENT_ID
    assert {
        (question.content_page_number, question.question_number): question.answer
        for question in document.questions
    } == EXPECTED_SOURCE_RECORDS
    assert all(question.key_page_number == 25 for question in document.questions)
    assert all(question.key_binding_kind == "answer_key_list" for question in document.questions)
    assert all(question.section == "MATEMATİK" for question in document.questions)
    assert all(question.visually_checked for question in document.questions)

    assert len(combined.documents) == len(base.documents) == 11
    assert sum(len(document.questions) for document in combined.documents) == 139
    assert sum(len(document.questions) for document in base.documents) == 135
    combined_samsung = next(
        document for document in combined.documents if document.document_id == DOCUMENT_ID
    )
    assert len(combined_samsung.questions) == 22

    manifest = _load_json(MANIFEST)
    assert manifest["benchmark_answer_reference_score_or_outcome_access"] is False
    assert manifest["task_id_present_in_fragment_or_index"] is False
    assert manifest["task_id_used_for_policy"] is False
    output = manifest["output"]
    assert output["added_records"] == 4
    assert output["fragment"]["sha256"] == sha256_file(FRAGMENT)
    assert output["index"]["sha256"] == sha256_file(INDEX)


def test_official_pdf_reverifies_all_new_and_combined_samsung_records() -> None:
    pytest.importorskip("pdfplumber")
    if not PDF.exists():
        pytest.skip("official Samsung PDF is an external pinned source")
    assert sha256_file(PDF) == PDF_SHA256

    fragment_document = parse_workbook_index(_load_json(FRAGMENT)).documents[0]
    fragment_verification = verify_workbook_index_pdf(PDF, fragment_document)
    assert fragment_verification["records"] == 4
    assert fragment_verification["verified_records"] == 4
    assert set(fragment_verification["content_marker_counts"].values()) == {1}

    combined = parse_workbook_index(_load_json(INDEX))
    combined_document = next(
        document for document in combined.documents if document.document_id == DOCUMENT_ID
    )
    combined_verification = verify_workbook_index_pdf(PDF, combined_document)
    assert combined_verification["records"] == 22
    assert combined_verification["verified_records"] == 22
    assert set(combined_verification["content_marker_counts"].values()) == {1}


def test_permitted_parser_observations_bind_by_source_content_not_task_id() -> None:
    pytest.importorskip("pdfplumber")
    pypdf = pytest.importorskip("pypdf")
    if not PDF.exists():
        pytest.skip("official Samsung PDF is an external pinned source")

    document = parse_workbook_index(_load_json(FRAGMENT)).documents[0]
    verification = verify_workbook_index_pdf(PDF, document)
    page_texts = [page.extract_text() or "" for page in pypdf.PdfReader(str(PDF)).pages]
    matcher = PageMatcher(page_texts)
    parser_rows = _load_parser_rows()

    for task_id, source_address in PARSER_ALIGNMENT.items():
        observation = parser_observation_primary_layout_number(parser_rows[task_id])
        page_number, question_number = source_address
        assert observation.question_number == question_number
        result = resolve_workbook_question(
            observation,
            SOURCE_URL,
            document,
            matcher,
            page_texts,
            _strict_v6_thresholds(),
            verified_content_marker_counts=verification["content_marker_counts"],
        )
        expected_record_id = f"{DOCUMENT_ID}:p{page_number}:q{question_number}"
        assert result.accepted is True
        assert result.answer == EXPECTED_SOURCE_RECORDS[source_address]
        assert result.trace["source"]["record_id"] == expected_record_id
        assert all(passed for _, passed in result.checks)

        renamed = resolve_workbook_question(
            replace(observation, task_id=f"renamed-source-observation-{question_number}"),
            SOURCE_URL,
            document,
            matcher,
            page_texts,
            _strict_v6_thresholds(),
            verified_content_marker_counts=verification["content_marker_counts"],
        )
        assert renamed.accepted is True
        assert renamed.answer == result.answer
        assert renamed.trace["source"]["record_id"] == expected_record_id
