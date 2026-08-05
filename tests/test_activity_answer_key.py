from __future__ import annotations

import copy
from collections.abc import Sequence

import pytest

from evidence_os.activity_answer_key import (
    ActivityAnswerKeyError,
    activity_marker_inventory,
    activity_joint_projection_sha256,
    attest_activity_answer_key,
    count_activity_markers,
    parse_activity_canonical_answer,
    verify_activity_answer_key,
)


PDF_SHA256 = "a" * 64
CONTENT_BBOX = (0.0, 10.0, 195.0, 35.0)
KEY_BBOX = (0.0, 30.0, 195.0, 92.0)


def _line(text: str, top: float, *, x0: float = 10.0) -> list[dict[str, object]]:
    words: list[dict[str, object]] = []
    cursor = x0
    for token in text.split():
        width = max(5.0, len(token) * 3.7)
        words.append(
            {
                "text": token,
                "x0": cursor,
                "top": top,
                "x1": cursor + width,
                "bottom": top + 8.0,
            }
        )
        cursor += width + 3.0
    return words


class FakePage:
    width = 400.0
    height = 300.0

    def __init__(self, lines: Sequence[tuple[str, float]]) -> None:
        self._words = [
            word
            for text, top in lines
            for word in _line(text, top)
        ]

    def extract_words(self, **_: object) -> list[dict[str, object]]:
        return copy.deepcopy(self._words)


def _content(marker: str = "ETKİNLİK-2") -> FakePage:
    return FakePage(
        [
            (marker, 18.0),
            ("unrelated prompt text", 50.0),
        ]
    )


def _labelled_key(
    answer_lines: Sequence[str] = ("a) Alpha", "b) Beta", "c) Gamma"),
    *,
    unit_heading: str = "1. ÜNİTE: HÜCRE BÖLÜNMELERİ",
    header: str = "Etkinlik 2 (20. Sayfa)",
) -> FakePage:
    lines: list[tuple[str, float]] = [
        (unit_heading, 8.0),
        (header, 38.0),
    ]
    lines.extend((text, 54.0 + index * 13.0) for index, text in enumerate(answer_lines))
    lines.append(("Etkinlik 3 (21. Sayfa)", 108.0))
    lines.append(("right column decoy", 54.0))
    page = FakePage(lines[:-1])
    page._words.extend(_line(lines[-1][0], lines[-1][1], x0=220.0))
    return page


def _labelled_attestation():
    return attest_activity_answer_key(
        _content(),
        _labelled_key(),
        pdf_sha256=PDF_SHA256,
        content_physical_page=20,
        key_physical_page=179,
        unit_number=1,
        activity_number=2,
        activity_page_number=20,
        content_bbox=CONTENT_BBOX,
        key_bbox=KEY_BBOX,
        answer_format="labelled",
        expected_components=[("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")],
    )


def _verify_labelled(attestation, **changes: object):
    arguments: dict[str, object] = {
        "pdf_sha256": PDF_SHA256,
        "content_physical_page": 20,
        "key_physical_page": 179,
        "unit_number": 1,
        "activity_number": 2,
        "activity_page_number": 20,
        "content_bbox": CONTENT_BBOX,
        "key_bbox": KEY_BBOX,
        "answer_format": "labelled",
        "expected_components": [("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")],
        "expected_content_projection_sha256": attestation.content_projection_sha256,
        "expected_key_projection_sha256": attestation.key_projection_sha256,
        "expected_projection_sha256": attestation.projection_sha256,
    }
    arguments.update(changes)
    return verify_activity_answer_key(_content(), _labelled_key(), **arguments)


def test_labelled_activity_attests_complete_address_and_all_components() -> None:
    attestation = _labelled_attestation()

    assert attestation.derived_answer == {
        "a": "Alpha",
        "b": "Beta",
        "c": "Gamma",
    }
    assert attestation.component_matches == {"a": True, "b": True, "c": True}
    assert all(
        len(value) == 64
        for value in (
            attestation.content_projection_sha256,
            attestation.key_projection_sha256,
            attestation.projection_sha256,
        )
    )
    assert _verify_labelled(attestation) == attestation


def test_full_page_activity_marker_inventory_detects_duplicate_title() -> None:
    page = FakePage(
        [
            ("ETKİNLİK-2", 18.0),
            ("unrelated prompt text", 50.0),
            ("ETKİNLİK-2", 180.0),
        ]
    )

    assert count_activity_markers(_content(), 2) == 1
    assert count_activity_markers(page, 2) == 2


def test_full_page_activity_marker_inventory_preserves_distinct_titles() -> None:
    page = FakePage(
        [
            ("ETKİNLİK-2", 18.0),
            ("unrelated prompt text", 50.0),
            ("ETKİNLİK-4", 180.0),
        ]
    )

    assert activity_marker_inventory(page) == (2, 4)


def test_raw_short_text_record_maps_directly_and_derives_the_joint_pin() -> None:
    result = attest_activity_answer_key(
        _content(),
        _labelled_key(),
        pdf_sha256=PDF_SHA256,
        content_physical_page=20,
        key_physical_page=179,
        unit_number=1,
        activity_number=2,
        activity_page_number=20,
        content_bbox=CONTENT_BBOX,
        key_bbox=KEY_BBOX,
        answer_format="labelled_short_text",
        canonical_answer="a=Alpha; b=Beta; c=Gamma",
    )

    assert parse_activity_canonical_answer(
        "a=Alpha; b=Beta; c=Gamma",
        answer_format="labelled_short_text",
    ) == (("a", "Alpha"), ("b", "Beta"), ("c", "Gamma"))
    assert result.projection_sha256 == activity_joint_projection_sha256(
        pdf_sha256=PDF_SHA256,
        unit_number=1,
        activity_number=2,
        activity_page_number=20,
        answer_format="labelled_short_text",
        content_projection_sha256=result.content_projection_sha256,
        key_projection_sha256=result.key_projection_sha256,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"activity_page_number": 21}, "physical page"),
        ({"unit_number": 2}, "unit number differs"),
        ({"activity_number": 3}, "marker activity number differs"),
    ],
)
def test_wrong_page_unit_or_activity_fails_closed(
    change: dict[str, int], message: str
) -> None:
    attestation = _labelled_attestation()

    with pytest.raises(ActivityAnswerKeyError, match=message):
        _verify_labelled(attestation, **change)


def test_truncated_key_bbox_fails_complete_section_check() -> None:
    with pytest.raises(ActivityAnswerKeyError, match="complete activity section"):
        attest_activity_answer_key(
            _content(),
            _labelled_key(),
            pdf_sha256=PDF_SHA256,
            content_physical_page=20,
            key_physical_page=179,
            unit_number=1,
            activity_number=2,
            activity_page_number=20,
            content_bbox=CONTENT_BBOX,
            key_bbox=(0.0, 30.0, 195.0, 74.0),
            answer_format="labelled",
            expected_components=[("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")],
        )


@pytest.mark.parametrize(
    "answer_lines",
    [
        ("a) Alpha", "c) Gamma", "b) Beta"),
        ("a) Alpha", "b) Beta", "b) Gamma"),
        ("a) Alpha", "b) Beta"),
    ],
)
def test_reordered_duplicate_or_missing_labels_fail_closed(
    answer_lines: tuple[str, ...],
) -> None:
    with pytest.raises(ActivityAnswerKeyError):
        attest_activity_answer_key(
            _content(),
            _labelled_key(answer_lines),
            pdf_sha256=PDF_SHA256,
            content_physical_page=20,
            key_physical_page=179,
            unit_number=1,
            activity_number=2,
            activity_page_number=20,
            content_bbox=CONTENT_BBOX,
            key_bbox=KEY_BBOX,
            answer_format="labelled",
            expected_components=[("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")],
        )


def test_changed_bboxes_and_forged_joint_hash_are_rejected() -> None:
    attestation = _labelled_attestation()

    with pytest.raises(ActivityAnswerKeyError, match="content projection differs"):
        _verify_labelled(attestation, content_bbox=(0.0, 9.0, 195.0, 35.0))
    with pytest.raises(ActivityAnswerKeyError, match="key projection differs"):
        _verify_labelled(attestation, key_bbox=(0.0, 29.0, 195.0, 92.0))
    with pytest.raises(ActivityAnswerKeyError, match="joint projection differs"):
        _verify_labelled(attestation, expected_projection_sha256="f" * 64)


def test_numbered_crossword_preserves_printed_category_order() -> None:
    key = FakePage(
        [
            ("1. ÜNİTE: HÜCRE BÖLÜNMELERİ", 8.0),
            ("Etkinlik 5 (27. Sayfa)", 38.0),
            ("Soldan sağa", 51.0),
            ("1. Bir", 64.0),
            ("3. Üç", 77.0),
            ("5. Beş", 90.0),
            ("7. Yedi", 103.0),
            ("Yukarıdan aşağıya", 116.0),
            ("2. İki", 129.0),
            ("4. Dört", 142.0),
            ("6. Altı", 155.0),
            ("8. Sekiz", 168.0),
            ("9. Dokuz", 181.0),
            ("Etkinlik 6 (27. Sayfa)", 207.0),
        ]
    )
    expected = [
        ("1", "Bir"),
        ("3", "Üç"),
        ("5", "Beş"),
        ("7", "Yedi"),
        ("2", "İki"),
        ("4", "Dört"),
        ("6", "Altı"),
        ("8", "Sekiz"),
        ("9", "Dokuz"),
    ]

    result = attest_activity_answer_key(
        _content("ETKİNLİK-S"),
        key,
        pdf_sha256=PDF_SHA256,
        content_physical_page=27,
        key_physical_page=179,
        unit_number=1,
        activity_number=5,
        activity_page_number=27,
        content_bbox=CONTENT_BBOX,
        key_bbox=(0.0, 30.0, 195.0, 196.0),
        answer_format="numbered",
        expected_components=expected,
    )

    assert list(result.derived_answer) == [label for label, _ in expected]
    assert all(result.component_matches.values())


def test_scalar_exit_binds_the_exit_while_pinning_the_full_section() -> None:
    key = FakePage(
        [
            ("2. ÜNİTE: KALITIM", 8.0),
            ("Etkinlik 3 (74. Sayfa)", 38.0),
            ("Doğru çıkış: 7", 54.0),
            ("1. D yolundan ilerler.", 67.0),
            ("2. Y yolundan ilerler.", 80.0),
            ("Etkinlik 4 (75. Sayfa)", 108.0),
        ]
    )

    result = attest_activity_answer_key(
        _content("ETKİNLİK-3"),
        key,
        pdf_sha256=PDF_SHA256,
        content_physical_page=74,
        key_physical_page=180,
        unit_number=2,
        activity_number=3,
        activity_page_number=74,
        content_bbox=CONTENT_BBOX,
        key_bbox=(0.0, 30.0, 195.0, 92.0),
        answer_format="scalar_exit",
        canonical_answer="7",
    )

    assert result.derived_answer == {"scalar": "7"}


@pytest.mark.parametrize("marker", ["ETKİNLİK-S", "## ETKINLIK-S"])
def test_narrow_terminal_s_is_accepted_only_as_activity_five(marker: str) -> None:
    key = FakePage(
        [
            ("1. ÜNİTE: HÜCRE BÖLÜNMELERİ", 8.0),
            ("Etkinlik 5 (27. Sayfa)", 38.0),
            ("1. Bir", 54.0),
            ("Etkinlik 6 (27. Sayfa)", 82.0),
        ]
    )

    result = attest_activity_answer_key(
        _content(marker),
        key,
        pdf_sha256=PDF_SHA256,
        content_physical_page=27,
        key_physical_page=179,
        unit_number=1,
        activity_number=5,
        activity_page_number=27,
        content_bbox=CONTENT_BBOX,
        key_bbox=(0.0, 30.0, 195.0, 70.0),
        answer_format="numbered",
        expected_components={"1": "Bir"},
    )

    assert result.derived_answer == {"1": "Bir"}


@pytest.mark.parametrize(
    "marker",
    [
        "ETKINLIK-X",
        "ETKINLIK-S2",
        "S-ETKINLIK-5",
        "ETKINLIKS",
        "ETKINLIK-AS",
        "ETKINLIK—S",
    ],
)
def test_other_letters_and_positions_never_canonicalize(marker: str) -> None:
    with pytest.raises(ActivityAnswerKeyError):
        attest_activity_answer_key(
            _content(marker),
            _labelled_key(),
            pdf_sha256=PDF_SHA256,
            content_physical_page=20,
            key_physical_page=179,
            unit_number=1,
            activity_number=2,
            activity_page_number=20,
            content_bbox=CONTENT_BBOX,
            key_bbox=KEY_BBOX,
            answer_format="labelled",
            expected_components=[("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")],
        )


def test_terminal_s_cannot_claim_an_activity_other_than_five() -> None:
    with pytest.raises(ActivityAnswerKeyError, match="activity number differs"):
        attest_activity_answer_key(
            _content("ETKINLIK-S"),
            _labelled_key(),
            pdf_sha256=PDF_SHA256,
            content_physical_page=20,
            key_physical_page=179,
            unit_number=1,
            activity_number=2,
            activity_page_number=20,
            content_bbox=CONTENT_BBOX,
            key_bbox=KEY_BBOX,
            answer_format="labelled",
            expected_components=[("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")],
        )


def test_key_header_remains_strictly_numeric() -> None:
    with pytest.raises(ActivityAnswerKeyError, match="numeric header"):
        attest_activity_answer_key(
            _content("ETKINLIK-S"),
            _labelled_key(
                ("a) Alpha", "b) Beta", "c) Gamma"),
                header="Etkinlik S (20. Sayfa)",
            ),
            pdf_sha256=PDF_SHA256,
            content_physical_page=20,
            key_physical_page=179,
            unit_number=1,
            activity_number=5,
            activity_page_number=20,
            content_bbox=CONTENT_BBOX,
            key_bbox=KEY_BBOX,
            answer_format="labelled",
            expected_components=[("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")],
        )


def test_duplicate_expected_labels_are_rejected_before_attestation() -> None:
    with pytest.raises(ActivityAnswerKeyError, match="repeat a label"):
        attest_activity_answer_key(
            _content(),
            _labelled_key(),
            pdf_sha256=PDF_SHA256,
            content_physical_page=20,
            key_physical_page=179,
            unit_number=1,
            activity_number=2,
            activity_page_number=20,
            content_bbox=CONTENT_BBOX,
            key_bbox=KEY_BBOX,
            answer_format="labelled",
            expected_components=[("a", "Alpha"), ("a", "Beta")],
        )
