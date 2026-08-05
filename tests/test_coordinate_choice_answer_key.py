from __future__ import annotations

import copy
import inspect

import pytest

from evidence_os.coordinate_choice_answer_key import (
    CoordinateChoiceAnswerKeyError,
    attest_coordinate_choice_answer_key,
    attest_coordinate_choice_content,
    verify_coordinate_choice_answer_key,
    verify_coordinate_choice_content,
)


PDF_SHA256 = "a" * 64
Q6_BBOX = (225.0, 195.8, 245.0, 209.0)
Q8_BBOX = (225.0, 232.0, 246.0, 245.0)
CONTENT_BBOX = (50.0, 295.0, 300.0, 350.0)


def _word(
    text: str,
    x0: float,
    top: float,
    x1: float,
    bottom: float,
) -> dict[str, object]:
    return {"text": text, "x0": x0, "top": top, "x1": x1, "bottom": bottom}


class FakePage:
    width = 600.0
    height = 850.0

    def __init__(self, words: list[dict[str, object]]) -> None:
        self._words = words

    def extract_words(self, **_: object) -> list[dict[str, object]]:
        return copy.deepcopy(self._words)


def _entry(
    number: int,
    answer: str,
    x0: float,
    top: float,
) -> list[dict[str, object]]:
    marker_width = 8.0 if number < 10 else 13.0
    marker = _word(f"{number}.", x0, top, x0 + marker_width, top + 11.0)
    answer_x0 = x0 + marker_width + 6.0
    return [
        marker,
        _word(answer, answer_x0, top, answer_x0 + 6.0, top + 11.0),
    ]


def _key_page() -> FakePage:
    words = [
        _word("Cevap", 81.5, 130.0, 111.7, 141.0),
        _word("Anahtarı", 114.0, 130.0, 156.8, 141.0),
        _word("1.ÜNİTE", 466.9, 134.0, 511.9, 146.0),
        *_entry(5, "A", 100.0, 180.0),
        *_entry(6, "E", 225.0, 198.0),
        *_entry(7, "C", 350.0, 216.0),
        *_entry(8, "B", 225.0, 234.0),
        _word("Cevap", 81.5, 400.0, 111.7, 411.0),
        _word("Anahtarı", 114.0, 400.0, 156.8, 411.0),
        _word("2.ÜNİTE", 466.9, 404.0, 511.9, 416.0),
        *_entry(1, "D", 100.0, 450.0),
        *_entry(2, "A", 225.0, 468.0),
        *_entry(3, "E", 350.0, 486.0),
        _word("233", 288.0, 823.0, 308.0, 835.0),
    ]
    return FakePage(words)


def _attest_q6(page: FakePage | None = None):
    return attest_coordinate_choice_answer_key(
        page or _key_page(),
        pdf_sha256=PDF_SHA256,
        physical_page=234,
        bbox=Q6_BBOX,
        question_number=6,
        expected_answer="E",
        expected_section="Cevap Anahtarı",
        expected_test_variant="1. ÜNİTE",
    )


def _content_page() -> FakePage:
    return FakePage(
        [
            _word("1.ÜNİTE", 83.0, 41.0, 128.0, 54.0),
            _word("6.", 50.0, 300.0, 58.0, 311.0),
            _word("Toplum", 64.0, 300.0, 103.0, 311.0),
            _word("nasıl", 108.0, 300.0, 134.0, 311.0),
            _word("değişir?", 139.0, 300.0, 180.0, 311.0),
            _word("A)", 64.0, 320.0, 75.0, 331.0),
            _word("Yavaş", 79.0, 320.0, 110.0, 331.0),
            _word("sayfa", 400.0, 800.0, 430.0, 811.0),
        ]
    )


def _attest_content(page: FakePage | None = None):
    return attest_coordinate_choice_content(
        page or _content_page(),
        pdf_sha256=PDF_SHA256,
        physical_page=54,
        bbox=CONTENT_BBOX,
        question_number=6,
        question_text="Toplum nasıl değişir?",
        expected_content_unit="1.ÜNİTE",
        expected_test_variant="1. ÜNİTE",
    )


def test_attests_tight_choice_cell_full_descriptor_and_table_context() -> None:
    result = _attest_q6()

    assert result.derived_answer == "E"
    assert result.unit_number == 1
    assert result.table_entry_count == 4
    assert len(result.projection_sha256) == 64


def test_frozen_choice_projection_is_required() -> None:
    attested = _attest_q6()
    verified = verify_coordinate_choice_answer_key(
        _key_page(),
        pdf_sha256=PDF_SHA256,
        physical_page=234,
        bbox=Q6_BBOX,
        question_number=6,
        expected_answer="E",
        expected_section="Cevap Anahtarı",
        expected_test_variant="1. ÜNİTE",
        expected_projection_sha256=attested.projection_sha256,
    )

    assert verified == attested
    changed = _key_page()
    changed._words[-1]["text"] = "234"
    with pytest.raises(CoordinateChoiceAnswerKeyError, match="projection differs"):
        verify_coordinate_choice_answer_key(
            changed,
            pdf_sha256=PDF_SHA256,
            physical_page=234,
            bbox=Q6_BBOX,
            question_number=6,
            expected_answer="E",
            expected_section="Cevap Anahtarı",
            expected_test_variant="1. ÜNİTE",
            expected_projection_sha256=attested.projection_sha256,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"question_number": 8}, "number and answer"),
        ({"expected_answer": "A"}, "number and answer"),
        ({"expected_section": "Cevap"}, "full 'Cevap"),
        ({"expected_test_variant": "1"}, "full '<number>"),
        ({"expected_test_variant": "2. ÜNİTE"}, "different full unit"),
    ],
)
def test_wrong_claim_or_short_descriptor_fails_closed(
    overrides: dict[str, object], message: str
) -> None:
    kwargs: dict[str, object] = {
        "pdf_sha256": PDF_SHA256,
        "physical_page": 234,
        "bbox": Q6_BBOX,
        "question_number": 6,
        "expected_answer": "E",
        "expected_section": "Cevap Anahtarı",
        "expected_test_variant": "1. ÜNİTE",
    }
    kwargs.update(overrides)
    with pytest.raises(CoordinateChoiceAnswerKeyError, match=message):
        attest_coordinate_choice_answer_key(_key_page(), **kwargs)


def test_neighboring_q8_cell_cannot_be_relabelled_as_q6() -> None:
    with pytest.raises(CoordinateChoiceAnswerKeyError, match="number and answer"):
        attest_coordinate_choice_answer_key(
            _key_page(),
            pdf_sha256=PDF_SHA256,
            physical_page=234,
            bbox=Q8_BBOX,
            question_number=6,
            expected_answer="E",
            expected_section="Cevap Anahtarı",
            expected_test_variant="1. ÜNİTE",
        )


def test_broadened_key_bbox_fails_even_if_it_contains_the_same_pair() -> None:
    with pytest.raises(CoordinateChoiceAnswerKeyError, match="not tight"):
        attest_coordinate_choice_answer_key(
            _key_page(),
            pdf_sha256=PDF_SHA256,
            physical_page=234,
            bbox=(215.0, 190.0, 250.0, 215.0),
            question_number=6,
            expected_answer="E",
            expected_section="Cevap Anahtarı",
            expected_test_variant="1. ÜNİTE",
        )


def test_content_heading_can_be_outside_crop_and_binds_to_key_unit() -> None:
    result = _attest_content()

    assert result.marker_count == 1
    assert result.unit_number == 1
    assert len(result.projection_sha256) == 64


def test_content_and_key_cross_unit_mismatch_fails_closed() -> None:
    with pytest.raises(CoordinateChoiceAnswerKeyError, match="do not identify"):
        attest_coordinate_choice_content(
            _content_page(),
            pdf_sha256=PDF_SHA256,
            physical_page=54,
            bbox=CONTENT_BBOX,
            question_number=6,
            question_text="Toplum nasıl değişir?",
            expected_content_unit="1. ÜNİTE",
            expected_test_variant="3. ÜNİTE",
        )


def test_duplicate_preceding_unit_heading_is_rejected() -> None:
    page = _content_page()
    page._words.append(_word("1. ÜNİTE", 400.0, 100.0, 450.0, 112.0))

    with pytest.raises(CoordinateChoiceAnswerKeyError, match="no unique preceding"):
        _attest_content(page)


def test_frozen_content_projection_rejects_changed_bbox() -> None:
    attested = _attest_content()
    verified = verify_coordinate_choice_content(
        _content_page(),
        pdf_sha256=PDF_SHA256,
        physical_page=54,
        bbox=CONTENT_BBOX,
        question_number=6,
        question_text="Toplum nasıl değişir?",
        expected_content_unit="1.ÜNİTE",
        expected_test_variant="1. ÜNİTE",
        expected_projection_sha256=attested.projection_sha256,
    )
    assert verified == attested

    with pytest.raises(CoordinateChoiceAnswerKeyError, match="projection differs"):
        verify_coordinate_choice_content(
            _content_page(),
            pdf_sha256=PDF_SHA256,
            physical_page=54,
            bbox=(50.0, 290.0, 305.0, 355.0),
            question_number=6,
            question_text="Toplum nasıl değişir?",
            expected_content_unit="1.ÜNİTE",
            expected_test_variant="1. ÜNİTE",
            expected_projection_sha256=attested.projection_sha256,
        )


def test_public_apis_have_no_task_or_outcome_inputs() -> None:
    for function in (
        attest_coordinate_choice_answer_key,
        verify_coordinate_choice_answer_key,
        attest_coordinate_choice_content,
        verify_coordinate_choice_content,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert "task_id" not in parameters
        assert "outcome" not in parameters
        assert "gold" not in parameters
