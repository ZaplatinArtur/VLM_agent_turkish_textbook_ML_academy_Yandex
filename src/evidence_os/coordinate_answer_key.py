"""Deterministic verification for two-dimensional PDF answer-key rows.

Plain PDF text extraction linearizes stacked numerators and denominators, so an
exact string comparison cannot prove answers containing fractions.  This module
uses positioned words and PDF fraction-bar lines to recover a small, strict
answer-key grammar.  It has no benchmark or task-ID inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence

from .official_ogm import canonical_json_sha256


PROJECTION_SCHEMA = "pdf-coordinate-projection-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NUMBER = re.compile(r"^-?\d+(?:,\d+)?$")
_INTEGER = re.compile(r"^-?\d+$")
_MAIN_MARKER = re.compile(r"^(\d+)\.$")
_LABELLED_PART = re.compile(
    r"^\s*([A-Za-zÇçĞğİıÖöŞşÜü])\s*(?:=|:|→)\s*(.*?)\s*$"
)


class CoordinateAnswerKeyError(ValueError):
    """A PDF row could not prove the indexed short-text answer."""


@dataclass(frozen=True, slots=True)
class CoordinateAnswerKeyVerification:
    projection_sha256: str
    derived_answer: Mapping[str, str] | str
    component_matches: Mapping[str, bool]


@dataclass(frozen=True, slots=True)
class _Item:
    text: str
    x0: float
    x1: float
    y: float
    kind: str
    source_indices: tuple[int, ...] = ()


def _normalized_text(value: str) -> str:
    return (
        unicodedata.normalize("NFKC", value)
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .strip()
    )


def _canonical_expression(value: str) -> str:
    return re.sub(r"\s+", "", _normalized_text(value).casefold())


def _normalized_label(value: str) -> str:
    return _normalized_text(value).rstrip(".").casefold()


def _expected_parts(answer: str) -> tuple[dict[str, str] | None, str | None]:
    parts = [part.strip() for part in answer.split(";") if part.strip()]
    labelled: dict[str, str] = {}
    for part in parts:
        match = _LABELLED_PART.fullmatch(part)
        if match is None:
            if len(parts) != 1:
                raise CoordinateAnswerKeyError("short-text answer mixes labelled and scalar parts")
            return None, answer.strip()
        label = _normalized_label(match.group(1))
        if label in labelled:
            raise CoordinateAnswerKeyError("short-text answer repeats a component label")
        component = match.group(2).strip()
        if not component:
            raise CoordinateAnswerKeyError("short-text answer has an empty component")
        labelled[label] = component
    if not labelled:
        raise CoordinateAnswerKeyError("short-text answer is empty")
    return labelled, None


def _word_center(word: Mapping[str, Any]) -> tuple[float, float]:
    return (
        (float(word["x0"]) + float(word["x1"])) / 2.0,
        (float(word["top"]) + float(word["bottom"])) / 2.0,
    )


def _inside_bbox(
    word: Mapping[str, Any], bbox: tuple[float, float, float, float]
) -> bool:
    x, y = _word_center(word)
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _fraction_lines(
    page: Any, bbox: tuple[float, float, float, float]
) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for raw_line in page.lines:
        x0 = float(raw_line["x0"])
        x1 = float(raw_line["x1"])
        top = float(raw_line["top"])
        bottom = float(raw_line["bottom"])
        width = x1 - x0
        if abs(top - bottom) > 0.05 or not 4.0 <= width <= 24.0:
            continue
        if not (
            bbox[0] <= (x0 + x1) / 2.0 <= bbox[2]
            and bbox[1] <= top <= bbox[3]
        ):
            continue
        result.append({"x0": x0, "x1": x1, "y": top})
    return result


def _numeric_candidates(
    words: Sequence[Mapping[str, Any]], line: Mapping[str, float], *, above: bool
) -> list[tuple[float, int, Mapping[str, Any]]]:
    result: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, word in enumerate(words):
        text = _normalized_text(str(word.get("text") or ""))
        if _NUMBER.fullmatch(text) is None:
            continue
        center_x, _ = _word_center(word)
        if not line["x0"] - 2.0 <= center_x <= line["x1"] + 2.0:
            continue
        distance = (
            line["y"] - float(word["bottom"])
            if above
            else float(word["top"]) - line["y"]
        )
        if -0.2 <= distance <= 12.0:
            result.append((distance, index, word))
    return sorted(result, key=lambda item: (item[0], item[1]))


def _positioned_items(
    page: Any, bbox: tuple[float, float, float, float]
) -> tuple[list[Mapping[str, Any]], list[_Item]]:
    words = [
        word
        for word in (
            page.extract_words(
                x_tolerance=2,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=False,
            )
            or []
        )
        if _inside_bbox(word, bbox)
    ]
    consumed: set[int] = set()
    fractions: list[_Item] = []
    for line in _fraction_lines(page, bbox):
        numerators = _numeric_candidates(words, line, above=True)
        denominators = _numeric_candidates(words, line, above=False)
        if not numerators or not denominators:
            continue
        _, numerator_index, numerator = numerators[0]
        _, denominator_index, denominator = denominators[0]
        if numerator_index in consumed or denominator_index in consumed:
            continue
        numerator_text = _normalized_text(str(numerator["text"]))
        denominator_text = _normalized_text(str(denominator["text"]))
        if denominator_text.startswith("-"):
            continue
        consumed.update((numerator_index, denominator_index))
        fractions.append(
            _Item(
                text=f"{numerator_text}/{denominator_text}",
                x0=line["x0"],
                x1=line["x1"],
                y=line["y"],
                kind="fraction",
                source_indices=(numerator_index, denominator_index),
            )
        )

    mixed_consumed: set[int] = set()
    fraction_items: list[_Item] = []
    for fraction in fractions:
        left: list[tuple[float, int, Mapping[str, Any]]] = []
        for index, word in enumerate(words):
            if index in consumed or index in mixed_consumed:
                continue
            text = _normalized_text(str(word.get("text") or ""))
            if _INTEGER.fullmatch(text) is None:
                continue
            _, word_y = _word_center(word)
            gap = fraction.x0 - float(word["x1"])
            if -0.5 <= gap <= 5.0 and abs(word_y - fraction.y) <= 5.0:
                left.append((gap, index, word))
        if not left:
            fraction_items.append(fraction)
            continue
        _, index, whole = min(left, key=lambda item: (item[0], item[1]))
        mixed_consumed.add(index)
        fraction_items.append(
            _Item(
                text=f"{_normalized_text(str(whole['text']))} {fraction.text}",
                x0=float(whole["x0"]),
                x1=fraction.x1,
                y=fraction.y,
                kind="mixed_fraction",
                source_indices=(index,) + fraction.source_indices,
            )
        )

    items = list(fraction_items)
    for index, word in enumerate(words):
        if index in consumed or index in mixed_consumed:
            continue
        _, word_y = _word_center(word)
        items.append(
            _Item(
                text=_normalized_text(str(word.get("text") or "")),
                x0=float(word["x0"]),
                x1=float(word["x1"]),
                y=word_y,
                kind="word",
                source_indices=(index,),
            )
        )
    return words, items


def _question_bounds(
    items: Sequence[_Item],
    question_number: int,
    bbox: tuple[float, float, float, float],
) -> tuple[_Item, float, float]:
    marker_text = f"{question_number}."
    markers = [item for item in items if item.kind == "word" and item.text == marker_text]
    if len(markers) != 1:
        raise CoordinateAnswerKeyError(
            f"coordinate key requires exactly one {marker_text!r} marker"
        )
    marker = markers[0]
    later_markers = [
        item
        for item in items
        if item.kind == "word"
        and _MAIN_MARKER.fullmatch(item.text)
        and item.y > marker.y + 4.0
        and abs(item.x0 - marker.x0) <= 3.0
    ]
    end_y = min((item.y for item in later_markers), default=bbox[3] + 1.0) - 3.0
    return marker, marker.y - 4.0, end_y


def _derive_answer(
    items: Sequence[_Item],
    question_number: int,
    bbox: tuple[float, float, float, float],
    expected_answer: str,
) -> tuple[Mapping[str, str] | str, Mapping[str, bool]]:
    expected_map, expected_scalar = _expected_parts(expected_answer)
    marker, start_y, end_y = _question_bounds(items, question_number, bbox)
    area_items = [item for item in items if start_y <= item.y <= end_y and item is not marker]

    if expected_map is None:
        scalar_items = sorted(
            (
                item
                for item in area_items
                if abs(item.y - marker.y) <= 4.0 and item.x0 >= marker.x1 - 0.2
            ),
            key=lambda item: (item.x0, item.y, item.text),
        )
        if not scalar_items:
            raise CoordinateAnswerKeyError("coordinate key has no scalar value")
        actual_scalar = " ".join(item.text for item in scalar_items)
        matches = {
            "scalar": _canonical_expression(actual_scalar)
            == _canonical_expression(expected_scalar or "")
        }
        return actual_scalar, matches

    expected_labels = set(expected_map)
    labels: dict[str, _Item] = {}
    for item in area_items:
        if item.kind != "word":
            continue
        label = _normalized_label(item.text)
        if label not in expected_labels:
            continue
        if label in labels:
            raise CoordinateAnswerKeyError("coordinate key repeats a source label")
        labels[label] = item
    if set(labels) != expected_labels:
        raise CoordinateAnswerKeyError("coordinate key labels do not match the indexed answer")

    label_item_ids = {id(item) for item in labels.values()}
    actual: dict[str, str] = {}
    for label, label_item in labels.items():
        following_labels = [
            other
            for other in labels.values()
            if abs(other.y - label_item.y) <= 5.0 and other.x0 > label_item.x0
        ]
        end_x = min((other.x0 for other in following_labels), default=bbox[2] + 1.0)
        values = sorted(
            (
                item
                for item in area_items
                if id(item) not in label_item_ids
                and abs(item.y - label_item.y) <= 5.0
                and item.x0 >= label_item.x1 - 0.2
                and item.x0 < end_x - 0.2
                and _MAIN_MARKER.fullmatch(item.text) is None
            ),
            key=lambda item: (item.x0, item.y, item.text),
        )
        if not values:
            raise CoordinateAnswerKeyError("coordinate key has a label without a value")
        actual[label] = " ".join(item.text for item in values)
    matches = {
        label: _canonical_expression(actual[label]) == _canonical_expression(expected)
        for label, expected in expected_map.items()
    }
    return actual, matches


def _projection(
    *,
    pdf_sha256: str,
    physical_page: int,
    page: Any,
    bbox: tuple[float, float, float, float],
    words: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "projection_schema": PROJECTION_SCHEMA,
        "pdf_sha256": pdf_sha256,
        "physical_page": physical_page,
        "page_size": [round(float(page.width), 3), round(float(page.height), 3)],
        "bbox": [round(value, 3) for value in bbox],
        "words": [
            {
                "text": _normalized_text(str(word.get("text") or "")),
                "x0": round(float(word["x0"]), 3),
                "top": round(float(word["top"]), 3),
                "x1": round(float(word["x1"]), 3),
                "bottom": round(float(word["bottom"]), 3),
            }
            for word in words
        ],
        "candidate_horizontal_lines": [
            {key: round(value, 3) for key, value in line.items()}
            for line in _fraction_lines(page, bbox)
        ],
    }


def attest_coordinate_answer_key(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    bbox: tuple[float, float, float, float],
    question_number: int,
    expected_answer: str,
) -> CoordinateAnswerKeyVerification:
    """Derive and attest one source answer without benchmark-side inputs."""

    if _HEX64.fullmatch(pdf_sha256) is None:
        raise CoordinateAnswerKeyError("coordinate key PDF hash is malformed")
    if physical_page < 1 or question_number < 1:
        raise CoordinateAnswerKeyError("coordinate key source address is malformed")

    words, items = _positioned_items(page, bbox)
    projection_sha256 = canonical_json_sha256(
        _projection(
            pdf_sha256=pdf_sha256,
            physical_page=physical_page,
            page=page,
            bbox=bbox,
            words=words,
        )
    )
    derived_answer, matches = _derive_answer(items, question_number, bbox, expected_answer)
    if not matches or not all(matches.values()):
        raise CoordinateAnswerKeyError("coordinate key does not expose the indexed answer")
    return CoordinateAnswerKeyVerification(
        projection_sha256=projection_sha256,
        derived_answer=derived_answer,
        component_matches=matches,
    )


def verify_coordinate_answer_key(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    bbox: tuple[float, float, float, float],
    question_number: int,
    expected_answer: str,
    expected_projection_sha256: str,
) -> CoordinateAnswerKeyVerification:
    """Prove one indexed answer and its frozen coordinate projection."""

    if _HEX64.fullmatch(expected_projection_sha256) is None:
        raise CoordinateAnswerKeyError("coordinate key projection hash is malformed")
    verification = attest_coordinate_answer_key(
        page,
        pdf_sha256=pdf_sha256,
        physical_page=physical_page,
        bbox=bbox,
        question_number=question_number,
        expected_answer=expected_answer,
    )
    if verification.projection_sha256 != expected_projection_sha256:
        raise CoordinateAnswerKeyError("coordinate key projection differs from source index")
    return verification
