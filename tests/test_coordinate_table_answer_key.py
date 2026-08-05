from __future__ import annotations

import copy

import pytest

from evidence_os.coordinate_table_answer_key import (
    CoordinateTableAnswerKeyError,
    attest_content_question_marker,
    attest_coordinate_table_answer_key,
    verify_content_question_marker,
    verify_coordinate_table_answer_key,
)


PDF_SHA256 = "a" * 64
BBOX = (10.0, 20.0, 90.0, 80.0)
SECTION = "ÜNİTE 1 - OLASILIK ÖRNEKLER"
TEST_VARIANT = "ÖRNEKLER"


def _word(text: str, x0: float, top: float, x1: float, bottom: float):
    return {"text": text, "x0": x0, "top": top, "x1": x1, "bottom": bottom}


def _horizontal(y: float):
    return {
        "orientation": "h",
        "x0": 10.0,
        "top": y,
        "x1": 90.0,
        "bottom": y,
    }


def _vertical(x: float):
    return {
        "orientation": "v",
        "x0": x,
        "top": 20.0,
        "x1": x,
        "bottom": 80.0,
    }


class FakeRow:
    def __init__(self, bbox):
        self.bbox = bbox


class FakeTable:
    def __init__(self, bbox, rows):
        self.bbox = bbox
        self._values = [values for _, values in rows]
        self.rows = [FakeRow(row_bbox) for row_bbox, _ in rows]
        self.cells = [row_bbox for row_bbox, _ in rows]

    def extract(self):
        return copy.deepcopy(self._values)


def _same_page_table(*, duplicate_heading: bool = False) -> FakeTable:
    rows = [
        ((10.0, 5.0, 90.0, 20.0), [SECTION]),
        ((10.0, 20.0, 90.0, 40.0), ["23", "24", "25"]),
        ((10.0, 40.0, 90.0, 80.0), ["x", "a) 24 b) 10", "y"]),
    ]
    if duplicate_heading:
        rows.insert(1, ((10.0, 18.0, 90.0, 20.0), [SECTION]))
    return FakeTable((10.0, 5.0, 90.0, 80.0), rows)


class FakeCrop:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self):
        return self._text


class FakePage:
    width = 200.0
    height = 150.0

    def __init__(self, words, *, text: str = SECTION, tables=None, crop_text=None):
        self._words = words
        self._text = text
        self._tables = list(tables) if tables is not None else [_same_page_table()]
        self._crop_text = crop_text if crop_text is not None else text
        self.edges = [
            _horizontal(20.0),
            _horizontal(40.0),
            _horizontal(80.0),
            _vertical(10.0),
            _vertical(90.0),
        ]

    def extract_words(self, **_):
        return copy.deepcopy(self._words)

    def extract_text(self):
        return self._text

    def find_tables(self):
        return self._tables

    def crop(self, _bbox):
        return FakeCrop(self._crop_text)


def _multipart_page() -> FakePage:
    return FakePage(
        [
            _word("24", 45, 26, 55, 34),
            _word("a)", 15, 49, 24, 57),
            _word("24", 28, 49, 38, 57),
            _word("b)", 15, 63, 24, 71),
            _word("10", 28, 63, 38, 71),
        ]
    )


def _attestation():
    return attest_coordinate_table_answer_key(
        _multipart_page(),
        pdf_sha256=PDF_SHA256,
        physical_page=12,
        bbox=BBOX,
        question_number=24,
        expected_answer="a=24; b=10",
        expected_section=SECTION,
        expected_test_variant=TEST_VARIANT,
    )


def test_table_cell_recovers_numbered_multipart_answer() -> None:
    result = _attestation()

    assert result.derived_answer == {"a": "24", "b": "10"}
    assert result.component_matches == {"a": True, "b": True}
    assert len(result.projection_sha256) == 64


def test_frozen_projection_and_answer_are_both_required() -> None:
    attestation = _attestation()
    result = verify_coordinate_table_answer_key(
        _multipart_page(),
        pdf_sha256=PDF_SHA256,
        physical_page=12,
        bbox=BBOX,
        question_number=24,
        expected_answer="a=24; b=10",
        expected_section=SECTION,
        expected_test_variant=TEST_VARIANT,
        expected_projection_sha256=attestation.projection_sha256,
    )

    assert result == attestation
    with pytest.raises(CoordinateTableAnswerKeyError, match="projection differs"):
        verify_coordinate_table_answer_key(
            _multipart_page(),
            pdf_sha256=PDF_SHA256,
            physical_page=12,
            bbox=BBOX,
            question_number=24,
            expected_answer="a=24; b=10",
            expected_section=SECTION,
            expected_test_variant=TEST_VARIANT,
            expected_projection_sha256="f" * 64,
        )


def test_answer_and_header_mutations_fail_closed() -> None:
    with pytest.raises(CoordinateTableAnswerKeyError, match="does not expose"):
        attest_coordinate_table_answer_key(
            _multipart_page(),
            pdf_sha256=PDF_SHA256,
            physical_page=12,
            bbox=BBOX,
            question_number=24,
            expected_answer="a=24; b=11",
            expected_section=SECTION,
            expected_test_variant=TEST_VARIANT,
        )
    with pytest.raises(CoordinateTableAnswerKeyError, match="header"):
        attest_coordinate_table_answer_key(
            _multipart_page(),
            pdf_sha256=PDF_SHA256,
            physical_page=12,
            bbox=BBOX,
            question_number=23,
            expected_answer="a=24; b=10",
            expected_section=SECTION,
            expected_test_variant=TEST_VARIANT,
        )


def test_section_may_be_proved_on_previous_continuation_page() -> None:
    cell_page = _multipart_page()
    cell_page._text = "continuation table only"
    cell_page._tables = [
        FakeTable(
            (10.0, 0.0, 90.0, 80.0),
            [
                ((10.0, 20.0, 90.0, 40.0), ["24", "25"]),
                ((10.0, 40.0, 90.0, 80.0), ["a) 24 b) 10", "y"]),
            ],
        )
    ]
    context_page = FakePage(
        [],
        text=f"preface {SECTION} appendix",
        tables=[
            FakeTable(
                (10.0, 70.0, 90.0, 145.0),
                [
                    ((10.0, 70.0, 90.0, 90.0), [SECTION]),
                    ((10.0, 90.0, 90.0, 110.0), ["21", "22", "23"]),
                    ((10.0, 110.0, 90.0, 145.0), ["x", "y", "z"]),
                ],
            )
        ],
    )
    result = attest_coordinate_table_answer_key(
        cell_page,
        pdf_sha256=PDF_SHA256,
        physical_page=13,
        bbox=BBOX,
        question_number=24,
        expected_answer="a=24; b=10",
        expected_section=SECTION,
        expected_test_variant=TEST_VARIANT,
        section_page=context_page,
        section_physical_page=12,
    )

    assert all(result.component_matches.values())


def test_ambiguous_section_and_broken_grid_fail_closed() -> None:
    page = _multipart_page()
    page._tables = [_same_page_table(duplicate_heading=True)]
    with pytest.raises(CoordinateTableAnswerKeyError, match="not unique"):
        attest_coordinate_table_answer_key(
            page,
            pdf_sha256=PDF_SHA256,
            physical_page=12,
            bbox=BBOX,
            question_number=24,
            expected_answer="a=24; b=10",
            expected_section=SECTION,
            expected_test_variant=TEST_VARIANT,
        )

    page = _multipart_page()
    page.edges = [edge for edge in page.edges if edge.get("orientation") != "v"]
    with pytest.raises(CoordinateTableAnswerKeyError, match="vertical boundary"):
        attest_coordinate_table_answer_key(
            page,
            pdf_sha256=PDF_SHA256,
            physical_page=12,
            bbox=BBOX,
            question_number=24,
            expected_answer="a=24; b=10",
            expected_section=SECTION,
            expected_test_variant=TEST_VARIANT,
        )


def test_prefix_reordered_labels_and_generic_section_fail_closed() -> None:
    for answer in (
        "UNVERIFIED PREFIX a=24; b=10",
        "b=10; a=24",
        "a=24; b=10\u200b",
        "a=24; b=10;;;;",
        "a=24; b=10////",
        "a=24; b=10,;/",
    ):
        with pytest.raises(CoordinateTableAnswerKeyError):
            attest_coordinate_table_answer_key(
                _multipart_page(),
                pdf_sha256=PDF_SHA256,
                physical_page=12,
                bbox=BBOX,
                question_number=24,
                expected_answer=answer,
                expected_section=SECTION,
                expected_test_variant=TEST_VARIANT,
            )
    with pytest.raises(CoordinateTableAnswerKeyError, match="complete unit heading"):
        attest_coordinate_table_answer_key(
            _multipart_page(),
            pdf_sha256=PDF_SHA256,
            physical_page=12,
            bbox=BBOX,
            question_number=24,
            expected_answer="a=24; b=10",
            expected_section="OLASILIK",
            expected_test_variant=TEST_VARIANT,
        )


def test_nonadjacent_context_page_fails_closed() -> None:
    with pytest.raises(CoordinateTableAnswerKeyError, match="cell page or its predecessor"):
        attest_coordinate_table_answer_key(
            _multipart_page(),
            pdf_sha256=PDF_SHA256,
            physical_page=13,
            bbox=BBOX,
            question_number=24,
            expected_answer="a=24; b=10",
            expected_section=SECTION,
            expected_test_variant=TEST_VARIANT,
            section_page=FakePage([], text=SECTION),
            section_physical_page=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("physical_page", True),
        ("question_number", True),
        ("bbox", (True, 20.0, 90.0, 80.0)),
        ("expected_answer", 24),
        ("expected_section", 1),
        ("expected_test_variant", False),
    ),
)
def test_nonliteral_addresses_geometry_and_answers_fail_closed(
    field: str, value: object
) -> None:
    kwargs = {
        "pdf_sha256": PDF_SHA256,
        "physical_page": 12,
        "bbox": BBOX,
        "question_number": 24,
        "expected_answer": "a=24; b=10",
        "expected_section": SECTION,
        "expected_test_variant": TEST_VARIANT,
    }
    kwargs[field] = value
    with pytest.raises(CoordinateTableAnswerKeyError):
        attest_coordinate_table_answer_key(_multipart_page(), **kwargs)


def test_content_marker_projection_is_exact_and_ambiguous_markers_fail() -> None:
    words = [
        _word("Örnek", 12, 25, 35, 33),
        _word("24", 38, 25, 48, 33),
        _word("indexed", 12, 45, 35, 53),
        _word("question", 38, 45, 70, 53),
    ]
    page = FakePage(words, crop_text="Örnek 24 indexed question")
    attested = attest_content_question_marker(
        page,
        pdf_sha256=PDF_SHA256,
        physical_page=2,
        bbox=BBOX,
        question_number=24,
        marker_kind="example_label",
        question_text="indexed question",
    )
    assert verify_content_question_marker(
        page,
        pdf_sha256=PDF_SHA256,
        physical_page=2,
        bbox=BBOX,
        question_number=24,
        marker_kind="example_label",
        question_text="indexed question",
        expected_projection_sha256=attested.projection_sha256,
    ) == attested
    duplicate = FakePage(
        words + [_word("24", 44, 25, 54, 33)],
        crop_text="Örnek 24 indexed question",
    )
    with pytest.raises(CoordinateTableAnswerKeyError, match="absent or ambiguous"):
        attest_content_question_marker(
            duplicate,
            pdf_sha256=PDF_SHA256,
            physical_page=2,
            bbox=BBOX,
            question_number=24,
            marker_kind="example_label",
            question_text="indexed question",
        )
