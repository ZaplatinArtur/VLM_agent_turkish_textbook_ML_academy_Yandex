"""Fail-closed attestation for coordinate-bound multiple-choice answer keys.

The sociology workbook uses compact answer-key blocks without drawn table
edges.  Each entry is a printed ``<number>. <A-E>`` pair beneath a full
``Cevap Anahtarı`` / ``<number>. ÜNİTE`` descriptor.  This module proves
that geometry directly from positioned PDF words and freezes a canonical
projection.  It intentionally accepts no benchmark task identifiers,
candidates, scores, or outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence
import unicodedata


KEY_PROJECTION_SCHEMA = "pdf-coordinate-choice-answer-key-projection-v1"
CONTENT_PROJECTION_SCHEMA = "pdf-coordinate-choice-content-projection-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"[a-z0-9]+")
_QUESTION_MARKER = re.compile(r"^([1-9]\d*)\.$")
_CHOICE = re.compile(r"^[A-E]$")


class CoordinateChoiceAnswerKeyError(ValueError):
    """The supplied PDF geometry does not prove the indexed source claim."""


@dataclass(frozen=True, slots=True)
class CoordinateChoiceAnswerKeyVerification:
    projection_sha256: str
    derived_answer: str
    unit_number: int
    table_entry_count: int


@dataclass(frozen=True, slots=True)
class CoordinateChoiceContentVerification:
    projection_sha256: str
    marker_count: int
    unit_number: int


@dataclass(frozen=True, slots=True)
class _Descriptor:
    section_words: tuple[Mapping[str, Any], ...]
    variant_words: tuple[Mapping[str, Any], ...]
    unit_number: int

    @property
    def top(self) -> float:
        return min(
            float(word["top"])
            for word in self.section_words + self.variant_words
        )

    @property
    def bottom(self) -> float:
        return max(
            float(word["bottom"])
            for word in self.section_words + self.variant_words
        )


@dataclass(frozen=True, slots=True)
class _KeyEntry:
    question_number: int
    answer: str
    marker: Mapping[str, Any]
    answer_word: Mapping[str, Any]


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plain_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CoordinateChoiceAnswerKeyError(f"{label} must be text")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized or any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in normalized
    ):
        raise CoordinateChoiceAnswerKeyError(
            f"{label} must be non-empty plain Unicode text"
        )
    return normalized


def _tokens(value: Any, label: str) -> tuple[str, ...]:
    normalized = _plain_text(value, label).casefold().translate(
        str.maketrans(
            {
                "ı": "i",
                "ş": "s",
                "ğ": "g",
                "ç": "c",
                "ö": "o",
                "ü": "u",
            }
        )
    )
    decomposed = unicodedata.normalize("NFKD", normalized)
    ascii_like = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return tuple(_TOKEN.findall(ascii_like))


def _full_section(value: Any) -> tuple[str, str]:
    tokens = _tokens(value, "expected section")
    if tokens != ("cevap", "anahtari"):
        raise CoordinateChoiceAnswerKeyError(
            "expected section is not the full 'Cevap Anahtarı' descriptor"
        )
    return tokens


def _full_unit(value: Any, label: str) -> tuple[int, tuple[str, str]]:
    tokens = _tokens(value, label)
    if (
        len(tokens) != 2
        or not tokens[0].isdigit()
        or int(tokens[0]) < 1
        or tokens[1] != "unite"
    ):
        raise CoordinateChoiceAnswerKeyError(
            f"{label} is not a full '<number>. ÜNİTE' descriptor"
        )
    return int(tokens[0]), (tokens[0], tokens[1])


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CoordinateChoiceAnswerKeyError(f"{label} must be a positive integer")
    return value


def _page_size(page: Any) -> tuple[float, float]:
    try:
        width = float(page.width)
        height = float(page.height)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CoordinateChoiceAnswerKeyError("PDF page size is unavailable") from exc
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise CoordinateChoiceAnswerKeyError("PDF page size is malformed")
    return width, height


def _strict_bbox(
    value: Any,
    label: str,
    *,
    page_size: tuple[float, float],
) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 4
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise CoordinateChoiceAnswerKeyError(f"{label} is malformed")
    bbox = tuple(float(item) for item in value)
    width, height = page_size
    if not (
        0.0 <= bbox[0] < bbox[2] <= width
        and 0.0 <= bbox[1] < bbox[3] <= height
    ):
        raise CoordinateChoiceAnswerKeyError(f"{label} is outside the PDF page")
    return bbox  # type: ignore[return-value]


def _word_projection(word: Mapping[str, Any]) -> dict[str, Any]:
    try:
        text = _plain_text(word.get("text"), "PDF word")
        coordinates = tuple(
            float(word[key]) for key in ("x0", "top", "x1", "bottom")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CoordinateChoiceAnswerKeyError("positioned PDF word is malformed") from exc
    if any(not math.isfinite(value) for value in coordinates) or not (
        coordinates[0] < coordinates[2] and coordinates[1] < coordinates[3]
    ):
        raise CoordinateChoiceAnswerKeyError("positioned PDF word is malformed")
    return {
        "text": text,
        "x0": round(coordinates[0], 3),
        "top": round(coordinates[1], 3),
        "x1": round(coordinates[2], 3),
        "bottom": round(coordinates[3], 3),
    }


def _all_words(page: Any) -> list[dict[str, Any]]:
    try:
        extracted = page.extract_words(
            x_tolerance=2,
            y_tolerance=2,
            keep_blank_chars=False,
            use_text_flow=False,
        ) or []
    except (AttributeError, TypeError, ValueError) as exc:
        raise CoordinateChoiceAnswerKeyError(
            "positioned PDF words cannot be extracted"
        ) from exc
    words = [_word_projection(word) for word in extracted]
    return sorted(
        words,
        key=lambda word: (
            word["top"],
            word["x0"],
            word["bottom"],
            word["x1"],
            word["text"],
        ),
    )


def _center(word: Mapping[str, Any]) -> tuple[float, float]:
    return (
        (float(word["x0"]) + float(word["x1"])) / 2.0,
        (float(word["top"]) + float(word["bottom"])) / 2.0,
    )


def _inside_bbox(
    word: Mapping[str, Any], bbox: tuple[float, float, float, float]
) -> bool:
    x, y = _center(word)
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _same_line(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return abs(_center(left)[1] - _center(right)[1]) <= 1.5


def _right_neighbor(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    max_gap: float,
) -> bool:
    gap = float(right["x0"]) - float(left["x1"])
    return _same_line(left, right) and 0.0 <= gap <= max_gap


def _section_candidates(words: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], ...]]:
    candidates: list[tuple[Mapping[str, Any], ...]] = []
    for first in words:
        if _tokens(first["text"], "PDF section word") != ("cevap",):
            continue
        seconds = [
            second
            for second in words
            if _tokens(second["text"], "PDF section word") == ("anahtari",)
            and _right_neighbor(first, second, max_gap=16.0)
        ]
        if len(seconds) == 1:
            candidates.append((first, seconds[0]))
    return candidates


def _unit_candidates(
    words: Sequence[Mapping[str, Any]],
) -> list[tuple[int, tuple[Mapping[str, Any], ...]]]:
    candidates: list[tuple[int, tuple[Mapping[str, Any], ...]]] = []
    seen: set[tuple[tuple[object, ...], ...]] = set()
    for word in words:
        tokens = _tokens(word["text"], "PDF unit word")
        if len(tokens) == 2 and tokens[0].isdigit() and tokens[1] == "unite":
            number = int(tokens[0])
            if number >= 1:
                key = ((word["text"], word["x0"], word["top"], word["x1"], word["bottom"]),)
                if key not in seen:
                    seen.add(key)
                    candidates.append((number, (word,)))
    for number_word in words:
        number_tokens = _tokens(number_word["text"], "PDF unit number")
        if len(number_tokens) != 1 or not number_tokens[0].isdigit():
            continue
        for unit_word in words:
            if (
                _tokens(unit_word["text"], "PDF unit word") == ("unite",)
                and _right_neighbor(number_word, unit_word, max_gap=12.0)
            ):
                number = int(number_tokens[0])
                key = tuple(
                    (word["text"], word["x0"], word["top"], word["x1"], word["bottom"])
                    for word in (number_word, unit_word)
                )
                if number >= 1 and key not in seen:
                    seen.add(key)
                    candidates.append((number, (number_word, unit_word)))
    return candidates


def _descriptors(words: Sequence[Mapping[str, Any]]) -> list[_Descriptor]:
    result: list[_Descriptor] = []
    for section_words in _section_candidates(words):
        section_y = sum(_center(word)[1] for word in section_words) / len(section_words)
        matching_units = []
        for unit_number, variant_words in _unit_candidates(words):
            variant_y = sum(_center(word)[1] for word in variant_words) / len(variant_words)
            if abs(section_y - variant_y) <= 12.0:
                matching_units.append((unit_number, variant_words))
        if len(matching_units) == 1:
            unit_number, variant_words = matching_units[0]
            result.append(
                _Descriptor(
                    section_words=section_words,
                    variant_words=variant_words,
                    unit_number=unit_number,
                )
            )
    return sorted(result, key=lambda descriptor: (descriptor.top, descriptor.unit_number))


def _nearest_preceding_descriptor(
    descriptors: Sequence[_Descriptor], *, before_y: float
) -> _Descriptor:
    preceding = [descriptor for descriptor in descriptors if descriptor.bottom < before_y]
    if not preceding:
        raise CoordinateChoiceAnswerKeyError(
            "answer-key cell has no preceding full section/unit descriptor"
        )
    nearest_bottom = max(descriptor.bottom for descriptor in preceding)
    nearest = [
        descriptor
        for descriptor in preceding
        if abs(descriptor.bottom - nearest_bottom) <= 0.5
    ]
    if len(nearest) != 1:
        raise CoordinateChoiceAnswerKeyError(
            "answer-key cell has an ambiguous preceding descriptor"
        )
    return nearest[0]


def _key_entries(words: Sequence[Mapping[str, Any]]) -> list[_KeyEntry]:
    entries: list[_KeyEntry] = []
    for marker in words:
        match = _QUESTION_MARKER.fullmatch(marker["text"])
        if match is None:
            continue
        answers = [
            word
            for word in words
            if _CHOICE.fullmatch(word["text"])
            and _right_neighbor(marker, word, max_gap=12.0)
        ]
        if len(answers) == 1:
            entries.append(
                _KeyEntry(
                    question_number=int(match.group(1)),
                    answer=answers[0]["text"],
                    marker=marker,
                    answer_word=answers[0],
                )
            )
    return entries


def _tight_cell(
    words: Sequence[Mapping[str, Any]],
    bbox: tuple[float, float, float, float],
    *,
    question_number: int,
    expected_answer: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    cell_words = sorted(
        (word for word in words if _inside_bbox(word, bbox)),
        key=lambda word: (word["top"], word["x0"], word["text"]),
    )
    if len(cell_words) != 2:
        raise CoordinateChoiceAnswerKeyError(
            "coordinate choice bbox is not one exact two-word key cell"
        )
    marker_matches = [
        word for word in cell_words if word["text"] == f"{question_number}."
    ]
    answer_matches = [word for word in cell_words if word["text"] == expected_answer]
    if len(marker_matches) != 1 or len(answer_matches) != 1:
        raise CoordinateChoiceAnswerKeyError(
            "coordinate choice cell does not expose the indexed number and answer"
        )
    marker, answer_word = marker_matches[0], answer_matches[0]
    if not _right_neighbor(marker, answer_word, max_gap=12.0):
        raise CoordinateChoiceAnswerKeyError(
            "coordinate choice number and answer are not one printed pair"
        )
    envelope = (
        min(float(word["x0"]) for word in cell_words),
        min(float(word["top"]) for word in cell_words),
        max(float(word["x1"]) for word in cell_words),
        max(float(word["bottom"]) for word in cell_words),
    )
    top_padding = envelope[1] - bbox[1]
    if (
        abs(bbox[0] - envelope[0]) > 0.75
        or abs(bbox[2] - envelope[2]) > 0.75
        or abs(bbox[3] - envelope[3]) > 0.75
        or not -0.1 <= top_padding <= 3.5
    ):
        raise CoordinateChoiceAnswerKeyError(
            "coordinate choice bbox is not tight around the printed key cell"
        )
    return marker, answer_word


def _page_words_hash(words: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_json_sha256({"positioned_words": list(words)})


def attest_coordinate_choice_answer_key(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    bbox: tuple[float, float, float, float],
    question_number: int,
    expected_answer: str,
    expected_section: str,
    expected_test_variant: str,
) -> CoordinateChoiceAnswerKeyVerification:
    """Prove one A-E answer-key cell and its nearest full descriptor."""

    if not isinstance(pdf_sha256, str) or _HEX64.fullmatch(pdf_sha256) is None:
        raise CoordinateChoiceAnswerKeyError("coordinate choice PDF hash is malformed")
    physical_page = _positive_integer(physical_page, "coordinate physical page")
    question_number = _positive_integer(question_number, "coordinate question number")
    answer = _plain_text(expected_answer, "expected answer")
    if _CHOICE.fullmatch(answer) is None:
        raise CoordinateChoiceAnswerKeyError("expected answer must be one uppercase A-E")
    section_text = _plain_text(expected_section, "expected section")
    section_tokens = _full_section(section_text)
    variant_text = _plain_text(expected_test_variant, "expected test variant")
    expected_unit, variant_tokens = _full_unit(
        variant_text, "expected test variant"
    )
    page_size = _page_size(page)
    strict_bbox = _strict_bbox(bbox, "coordinate choice bbox", page_size=page_size)
    words = _all_words(page)
    marker, answer_word = _tight_cell(
        words,
        strict_bbox,
        question_number=question_number,
        expected_answer=answer,
    )

    descriptors = _descriptors(words)
    descriptor = _nearest_preceding_descriptor(
        descriptors, before_y=strict_bbox[1]
    )
    if descriptor.unit_number != expected_unit:
        raise CoordinateChoiceAnswerKeyError(
            "nearest answer-key descriptor belongs to a different full unit"
        )
    next_descriptor_top = min(
        (
            candidate.top
            for candidate in descriptors
            if candidate.top > descriptor.bottom + 0.5
        ),
        default=page_size[1] + 1.0,
    )
    block_entries = [
        entry
        for entry in _key_entries(words)
        if descriptor.bottom < _center(entry.marker)[1] < next_descriptor_top
    ]
    counts: dict[int, int] = {}
    for entry in block_entries:
        counts[entry.question_number] = counts.get(entry.question_number, 0) + 1
    if len(block_entries) < 3 or any(count != 1 for count in counts.values()):
        raise CoordinateChoiceAnswerKeyError(
            "full descriptor is not followed by one unambiguous answer-key table"
        )
    target_entries = [
        entry
        for entry in block_entries
        if entry.question_number == question_number
        and entry.answer == answer
        and entry.marker == marker
        and entry.answer_word == answer_word
    ]
    if len(target_entries) != 1 or not any(
        abs(entry.question_number - question_number) == 1
        for entry in block_entries
    ):
        raise CoordinateChoiceAnswerKeyError(
            "coordinate cell is not the indexed member of its answer-key table"
        )

    projection = {
        "projection_schema": KEY_PROJECTION_SCHEMA,
        "pdf_sha256": pdf_sha256,
        "physical_page": physical_page,
        "page_size": [round(value, 3) for value in page_size],
        "bbox": [round(value, 3) for value in strict_bbox],
        "question_number": question_number,
        "expected_answer": answer,
        "expected_section": section_text,
        "expected_section_tokens": list(section_tokens),
        "expected_test_variant": variant_text,
        "expected_test_variant_tokens": list(variant_tokens),
        "descriptor": {
            "section_words": [
                dict(word) for word in descriptor.section_words
            ],
            "variant_words": [
                dict(word) for word in descriptor.variant_words
            ],
            "unit_number": descriptor.unit_number,
        },
        "cell_words": [dict(marker), dict(answer_word)],
        "table_entries": [
            {
                "question_number": entry.question_number,
                "answer": entry.answer,
                "marker": dict(entry.marker),
                "answer_word": dict(entry.answer_word),
            }
            for entry in sorted(
                block_entries,
                key=lambda entry: (
                    entry.question_number,
                    entry.marker["top"],
                    entry.marker["x0"],
                ),
            )
        ],
        "next_descriptor_top": round(next_descriptor_top, 3),
        "page_positioned_words_sha256": _page_words_hash(words),
    }
    return CoordinateChoiceAnswerKeyVerification(
        projection_sha256=_canonical_json_sha256(projection),
        derived_answer=answer_word["text"],
        unit_number=descriptor.unit_number,
        table_entry_count=len(block_entries),
    )


def verify_coordinate_choice_answer_key(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    bbox: tuple[float, float, float, float],
    question_number: int,
    expected_answer: str,
    expected_section: str,
    expected_test_variant: str,
    expected_projection_sha256: str,
) -> CoordinateChoiceAnswerKeyVerification:
    """Recompute a choice-key attestation and require its frozen projection."""

    if (
        not isinstance(expected_projection_sha256, str)
        or _HEX64.fullmatch(expected_projection_sha256) is None
    ):
        raise CoordinateChoiceAnswerKeyError(
            "coordinate choice projection hash is malformed"
        )
    verification = attest_coordinate_choice_answer_key(
        page,
        pdf_sha256=pdf_sha256,
        physical_page=physical_page,
        bbox=bbox,
        question_number=question_number,
        expected_answer=expected_answer,
        expected_section=expected_section,
        expected_test_variant=expected_test_variant,
    )
    if verification.projection_sha256 != expected_projection_sha256:
        raise CoordinateChoiceAnswerKeyError(
            "coordinate choice projection differs from source index"
        )
    return verification


def _contains_once(
    haystack: Sequence[str], needle: Sequence[str]
) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return (
        sum(
            tuple(haystack[index : index + len(needle)]) == tuple(needle)
            for index in range(len(haystack) - len(needle) + 1)
        )
        == 1
    )


def attest_coordinate_choice_content(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    bbox: tuple[float, float, float, float],
    question_number: int,
    question_text: str,
    expected_content_unit: str,
    expected_test_variant: str,
) -> CoordinateChoiceContentVerification:
    """Pin one numbered question crop and its page-level unit heading.

    ``expected_content_unit`` is checked against the source page while
    ``expected_test_variant`` is the key-side unit descriptor.  Requiring the
    same parsed unit number mechanically joins the two independently supplied
    source addresses.
    """

    if not isinstance(pdf_sha256, str) or _HEX64.fullmatch(pdf_sha256) is None:
        raise CoordinateChoiceAnswerKeyError("content choice PDF hash is malformed")
    physical_page = _positive_integer(physical_page, "content physical page")
    question_number = _positive_integer(question_number, "content question number")
    source_question = _plain_text(question_text, "indexed question text")
    question_tokens = _tokens(source_question, "indexed question text")
    if len(question_tokens) < 3:
        raise CoordinateChoiceAnswerKeyError(
            "indexed question text is too short for exact source binding"
        )
    content_unit_text = _plain_text(expected_content_unit, "expected content unit")
    content_unit, content_unit_tokens = _full_unit(
        content_unit_text, "expected content unit"
    )
    variant_text = _plain_text(expected_test_variant, "expected test variant")
    key_unit, variant_tokens = _full_unit(
        variant_text, "expected test variant"
    )
    if content_unit != key_unit:
        raise CoordinateChoiceAnswerKeyError(
            "content unit and key test variant do not identify the same unit"
        )
    page_size = _page_size(page)
    strict_bbox = _strict_bbox(bbox, "content choice bbox", page_size=page_size)
    words = _all_words(page)
    crop_words = [word for word in words if _inside_bbox(word, strict_bbox)]
    numbered_markers = [
        word for word in crop_words if _QUESTION_MARKER.fullmatch(word["text"])
    ]
    target_markers = [
        word for word in numbered_markers if word["text"] == f"{question_number}."
    ]
    if len(numbered_markers) != 1 or len(target_markers) != 1:
        raise CoordinateChoiceAnswerKeyError(
            "content crop does not expose one exact printed question marker"
        )
    marker = target_markers[0]
    # The marker is independently proven above.  In some two-column PDFs its
    # baseline is a fraction lower than the first prompt line, so a geometric
    # top/x sort legitimately places ``6.`` in the middle of that line.  Drop
    # that exact word before testing the contiguous indexed prompt.
    crop_tokens = tuple(
        token
        for word in crop_words
        if word is not marker
        for token in _tokens(word["text"], "content PDF word")
    )
    if not _contains_once(crop_tokens, question_tokens):
        raise CoordinateChoiceAnswerKeyError(
            "content crop does not expose one exact indexed question text"
        )

    preceding_units = [
        (number, unit_words)
        for number, unit_words in _unit_candidates(words)
        if max(float(word["bottom"]) for word in unit_words) < float(marker["top"])
        and min(float(word["top"]) for word in unit_words) <= page_size[1] * 0.20
    ]
    if len(preceding_units) != 1:
        raise CoordinateChoiceAnswerKeyError(
            "content question has no unique preceding page-level unit heading"
        )
    observed_unit, heading_words = preceding_units[0]
    if observed_unit != content_unit:
        raise CoordinateChoiceAnswerKeyError(
            "content question belongs to a different full unit heading"
        )

    projection = {
        "projection_schema": CONTENT_PROJECTION_SCHEMA,
        "pdf_sha256": pdf_sha256,
        "physical_page": physical_page,
        "page_size": [round(value, 3) for value in page_size],
        "bbox": [round(value, 3) for value in strict_bbox],
        "question_number": question_number,
        "question_text": source_question,
        "question_tokens": list(question_tokens),
        "expected_content_unit": content_unit_text,
        "expected_content_unit_tokens": list(content_unit_tokens),
        "expected_test_variant": variant_text,
        "expected_test_variant_tokens": list(variant_tokens),
        "marker": dict(marker),
        "unit_heading_words": [dict(word) for word in heading_words],
        "crop_words": [dict(word) for word in crop_words],
        "crop_tokens_sha256": _canonical_json_sha256(
            {"tokens": list(crop_tokens)}
        ),
        "page_positioned_words_sha256": _page_words_hash(words),
    }
    return CoordinateChoiceContentVerification(
        projection_sha256=_canonical_json_sha256(projection),
        marker_count=1,
        unit_number=observed_unit,
    )


def verify_coordinate_choice_content(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    bbox: tuple[float, float, float, float],
    question_number: int,
    question_text: str,
    expected_content_unit: str,
    expected_test_variant: str,
    expected_projection_sha256: str,
) -> CoordinateChoiceContentVerification:
    """Recompute a content attestation and require its frozen projection."""

    if (
        not isinstance(expected_projection_sha256, str)
        or _HEX64.fullmatch(expected_projection_sha256) is None
    ):
        raise CoordinateChoiceAnswerKeyError(
            "content choice projection hash is malformed"
        )
    verification = attest_coordinate_choice_content(
        page,
        pdf_sha256=pdf_sha256,
        physical_page=physical_page,
        bbox=bbox,
        question_number=question_number,
        question_text=question_text,
        expected_content_unit=expected_content_unit,
        expected_test_variant=expected_test_variant,
    )
    if verification.projection_sha256 != expected_projection_sha256:
        raise CoordinateChoiceAnswerKeyError(
            "content choice projection differs from source index"
        )
    return verification
