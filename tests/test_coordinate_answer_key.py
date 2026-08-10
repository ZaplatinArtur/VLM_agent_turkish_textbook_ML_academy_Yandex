from __future__ import annotations

import copy

import pytest

from evidence_os.coordinate_answer_key import (
    CoordinateAnswerKeyError,
    attest_coordinate_answer_key,
    verify_coordinate_answer_key,
)


PDF_SHA256 = "a" * 64
BBOX = (0.0, 30.0, 200.0, 70.0)


def _word(text: str, x0: float, top: float, x1: float, bottom: float) -> dict[str, object]:
    return {"text": text, "x0": x0, "top": top, "x1": x1, "bottom": bottom}


class FakePage:
    width = 200.0
    height = 100.0

    def __init__(self, words: list[dict[str, object]], lines: list[dict[str, float]]) -> None:
        self._words = words
        self.lines = lines

    def extract_words(self, **_: object) -> list[dict[str, object]]:
        return copy.deepcopy(self._words)


def _fraction_line(x0: float, x1: float) -> dict[str, float]:
    return {"x0": x0, "x1": x1, "top": 50.0, "bottom": 50.0}


def _labelled_page() -> FakePage:
    return FakePage(
        words=[
            _word("1.", 5, 46, 12, 54),
            _word("a.", 18, 46, 25, 54),
            _word("2", 32, 38, 37, 47),
            _word("5", 32, 55, 37, 64),
            _word("b.", 55, 46, 62, 54),
            _word("-3", 68, 46, 77, 54),
            _word("7", 81, 38, 86, 47),
            _word("100", 78, 55, 90, 64),
            _word("c.", 110, 46, 117, 54),
            _word("1", 124, 46, 129, 54),
        ],
        lines=[_fraction_line(31, 38), _fraction_line(78, 90)],
    )


def _attestation():
    return attest_coordinate_answer_key(
        _labelled_page(),
        pdf_sha256=PDF_SHA256,
        physical_page=9,
        bbox=BBOX,
        question_number=1,
        expected_answer="a=2/5; b=-3 7/100; c=1",
    )


def test_coordinate_key_recovers_fractions_and_mixed_numbers() -> None:
    result = _attestation()

    assert result.derived_answer == {"a": "2/5", "b": "-3 7/100", "c": "1"}
    assert all(result.component_matches.values())
    assert len(result.projection_sha256) == 64


def test_frozen_projection_and_answer_are_both_required() -> None:
    attestation = _attestation()
    result = verify_coordinate_answer_key(
        _labelled_page(),
        pdf_sha256=PDF_SHA256,
        physical_page=9,
        bbox=BBOX,
        question_number=1,
        expected_answer="a=2/5; b=-3 7/100; c=1",
        expected_projection_sha256=attestation.projection_sha256,
    )

    assert result == attestation
    with pytest.raises(CoordinateAnswerKeyError, match="projection differs"):
        verify_coordinate_answer_key(
            _labelled_page(),
            pdf_sha256=PDF_SHA256,
            physical_page=9,
            bbox=BBOX,
            question_number=1,
            expected_answer="a=2/5; b=-3 7/100; c=1",
            expected_projection_sha256="f" * 64,
        )


def test_answer_mutation_fails_closed() -> None:
    with pytest.raises(CoordinateAnswerKeyError, match="does not expose"):
        attest_coordinate_answer_key(
            _labelled_page(),
            pdf_sha256=PDF_SHA256,
            physical_page=9,
            bbox=BBOX,
            question_number=1,
            expected_answer="a=2/6; b=-3 7/100; c=1",
        )


def test_question_marker_must_be_unique() -> None:
    page = _labelled_page()
    page._words.append(_word("1.", 5, 58, 12, 66))

    with pytest.raises(CoordinateAnswerKeyError, match="exactly one"):
        attest_coordinate_answer_key(
            page,
            pdf_sha256=PDF_SHA256,
            physical_page=9,
            bbox=BBOX,
            question_number=1,
            expected_answer="a=2/5; b=-3 7/100; c=1",
        )


def test_unlabelled_scalar_uses_only_the_target_marker_row() -> None:
    page = FakePage(
        words=[
            _word("1.", 5, 38, 12, 46),
            _word("wrong", 20, 38, 42, 46),
            _word("2.", 5, 50, 12, 58),
            _word("98,8", 20, 50, 38, 58),
            _word("kg", 42, 50, 51, 58),
        ],
        lines=[],
    )
    result = attest_coordinate_answer_key(
        page,
        pdf_sha256=PDF_SHA256,
        physical_page=10,
        bbox=BBOX,
        question_number=2,
        expected_answer="98,8 kg",
    )

    assert result.derived_answer == "98,8 kg"
    assert result.component_matches == {"scalar": True}
