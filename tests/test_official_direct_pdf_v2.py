from __future__ import annotations

import pytest

from evidence_os.official_ogm import OcrObservation, OfficialSourceError, PageMatcher
from evidence_os.official_pdf import (
    DirectPdfThresholds,
    KeyEntry,
    resolve_direct_pdf_question,
)


SOURCE_URL = "https://odsgm.meb.gov.tr/synthetic/direct-source.pdf"
TARGET_TEXT = (
    "algebra triangle orchard compass lantern isotope fraction theorem polygon "
    "velocity cylinder matrix radius quotient symmetry tangent integer prism "
    "coordinate diagonal equation"
)
DISTRACTOR_TEXT = (
    "history empire treaty archive dynasty republic parliament chronology culture "
    "museum language geography migration monument citizenship reform document"
)


def _observation(*, question_number: int = 7) -> OcrObservation:
    statement = f"{question_number}. {TARGET_TEXT}"
    return OcrObservation(
        task_id="opaque-alignment-only",
        statement=statement,
        image_sha256="f" * 64,
        width=900,
        height=600,
        question_number=question_number,
        parser_identity="pipeline-v2/layout-v2/recognizer-v2",
        text_blocks=(statement,),
    )


def _adapter(*, sections: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "name": "synthetic-official-document",
        "source_url": SOURCE_URL,
        "pdf_sha256": "a" * 64,
        "key_page_number": 4,
        "sections": sections
        or [{"subject": "math", "start_page": 2, "end_page": 3}],
    }


def _page_texts(*, target_marker: str = "7.") -> list[str]:
    return [
        "official cover",
        f"{target_marker} {TARGET_TEXT}".strip(),
        f"8. {DISTRACTOR_TEXT}",
        "embedded answer key",
    ]


def _answer_key(*, question_number: int = 7) -> dict[str, dict[int, KeyEntry]]:
    return {
        "math": {
            question_number: KeyEntry(
                answer="C",
                key_page_number=4,
                bbox=(100.0, 200.0, 140.0, 215.0),
            )
        }
    }


def _resolve(
    *,
    observation: OcrObservation | None = None,
    adapter: dict[str, object] | None = None,
    page_texts: list[str] | None = None,
    answer_key: dict[str, dict[int, KeyEntry]] | None = None,
    source_url: str = SOURCE_URL,
):
    pages = page_texts if page_texts is not None else _page_texts()
    return resolve_direct_pdf_question(
        observation or _observation(),
        source_url,
        adapter or _adapter(),
        PageMatcher(pages),
        pages,
        _answer_key() if answer_key is None else answer_key,
        DirectPdfThresholds(),
    )


def test_strong_unique_page_subject_question_and_key_match_is_accepted() -> None:
    result = _resolve()

    assert result.accepted is True
    assert result.answer == "C"
    assert all(passed for _, passed in result.checks)
    assert result.trace["match"]["acceptance_path"] == "strong_page"
    assert result.trace["source"]["matched_page_number"] == 2
    assert result.trace["source"]["subject"] == "math"
    assert result.trace["source"]["printed_question_number"] == 7
    assert result.trace["source"]["key_page_number"] == 4


def test_equal_content_page_scores_are_ambiguous_and_fail_closed() -> None:
    pages = [
        "official cover",
        f"7. {TARGET_TEXT}",
        f"7. {TARGET_TEXT}",
        "embedded answer key",
    ]
    result = _resolve(page_texts=pages)
    checks = dict(result.checks)

    assert result.accepted is False
    assert result.answer is None
    assert checks["unique_content_page"] is False
    assert result.trace["match"]["page_margin"] == pytest.approx(0.0)
    assert result.trace["match"]["anchor_margin"] == pytest.approx(0.0)
    assert result.trace["match"]["acceptance_path"] == "none"


def test_page_in_overlapping_subject_sections_is_rejected() -> None:
    adapter = _adapter(
        sections=[
            {"subject": "math", "start_page": 2, "end_page": 3},
            {"subject": "science", "start_page": 2, "end_page": 2},
        ]
    )
    result = _resolve(adapter=adapter)
    checks = dict(result.checks)

    assert result.accepted is False
    assert result.answer is None
    assert checks["matched_page_in_one_subject_section"] is False
    assert result.trace["source"]["subject"] is None


def test_question_number_must_be_visibly_printed_on_matched_page() -> None:
    result = _resolve(page_texts=_page_texts(target_marker=""))
    checks = dict(result.checks)

    assert result.accepted is False
    assert result.answer is None
    assert checks["visible_question_number_on_matched_page"] is False
    assert checks["unique_embedded_key_entry"] is True


def test_missing_question_number_in_subject_key_is_rejected() -> None:
    result = _resolve(answer_key={"math": {}})
    checks = dict(result.checks)

    assert result.accepted is False
    assert result.answer is None
    assert checks["visible_question_number_on_matched_page"] is True
    assert checks["unique_embedded_key_entry"] is False


def test_invalid_choice_in_injected_key_is_rejected() -> None:
    result = _resolve(
        answer_key={
            "math": {
                7: KeyEntry(
                    answer="Z",
                    key_page_number=4,
                    bbox=(100.0, 200.0, 140.0, 215.0),
                )
            }
        }
    )
    checks = dict(result.checks)

    assert result.accepted is False
    assert result.answer is None
    assert checks["unique_embedded_key_entry"] is True
    assert checks["valid_choice_key"] is False


def test_matcher_and_page_projection_must_have_identical_length() -> None:
    pages = _page_texts()

    with pytest.raises(OfficialSourceError, match="do not align"):
        resolve_direct_pdf_question(
            _observation(),
            SOURCE_URL,
            _adapter(),
            PageMatcher(pages[:-1]),
            pages,
            _answer_key(),
            DirectPdfThresholds(),
        )


def test_non_exact_source_url_is_rejected_before_matching() -> None:
    with pytest.raises(OfficialSourceError, match="exactly match"):
        _resolve(source_url=f"{SOURCE_URL}?download=1")
