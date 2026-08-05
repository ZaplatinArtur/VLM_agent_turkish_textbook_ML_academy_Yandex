"""Deterministic PDF attestation for a numbered fill-in activity.

The activity page and its answer-key page are verified independently.  The
content projection proves the activity title, instruction, complete numbered
item inventory, and word-bank multiset.  The key projection then derives every
numbered answer from PDF coordinates.  A joint projection hash binds both
pages to one immutable PDF without using task IDs, benchmark answers, model
candidates, scores, judges, or outcomes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Any

from .official_ogm import canonical_json_sha256, normalize_tokens


PROJECTION_SCHEMA = "pdf-fill-blank-answer-binding-projection-v1"
CONTENT_PROJECTION_SCHEMA = "pdf-fill-blank-content-projection-v1"
KEY_PROJECTION_SCHEMA = "pdf-fill-blank-key-projection-v1"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NUMBERED_MARKER = re.compile(r"^([1-9]\d*)\.$")


class FillBlankAnswerKeyError(ValueError):
    """The two PDF pages could not prove a numbered fill-in answer."""


@dataclass(frozen=True, slots=True)
class FillBlankAnswerKeyVerification:
    projection_sha256: str
    content_projection_sha256: str
    key_projection_sha256: str
    derived_answer: Mapping[int, str]
    component_matches: Mapping[int, bool]
    content_text: str
    key_text: str


def _plain_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in value
    ):
        raise FillBlankAnswerKeyError(f"{label} must be plain Unicode text")
    return unicodedata.normalize("NFKC", value).strip()


def _fold(value: Any, label: str = "PDF text") -> str:
    text = (
        _plain_text(value, label)
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .casefold()
        .replace("i\u0307", "i")
    )
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"\s+", " ", without_marks.replace("ı", "i")).strip()


def _canonical_component(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _fold(value, "answer component"))


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise FillBlankAnswerKeyError(f"{label} must be a positive integer")
    return value


def _strict_bbox(
    value: Sequence[float], label: str
) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise FillBlankAnswerKeyError(f"{label} is malformed")
    bbox = tuple(float(item) for item in value)
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise FillBlankAnswerKeyError(f"{label} is empty")
    return bbox  # type: ignore[return-value]


def _page_size(page: Any, label: str) -> tuple[float, float]:
    try:
        width = float(page.width)
        height = float(page.height)
    except (AttributeError, TypeError, ValueError) as exc:
        raise FillBlankAnswerKeyError(f"{label} has no finite geometry") from exc
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise FillBlankAnswerKeyError(f"{label} has no finite geometry")
    return width, height


def _validate_bbox_on_page(
    page: Any, bbox: tuple[float, float, float, float], label: str
) -> None:
    width, height = _page_size(page, label)
    if not (
        0.0 <= bbox[0] < bbox[2] <= width
        and 0.0 <= bbox[1] < bbox[3] <= height
    ):
        raise FillBlankAnswerKeyError(f"{label} bbox is outside the PDF page")


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


def _extract_words(page: Any, label: str) -> list[Mapping[str, Any]]:
    try:
        raw_words = page.extract_words(
            x_tolerance=2,
            y_tolerance=2,
            keep_blank_chars=False,
            use_text_flow=False,
        ) or []
    except (AttributeError, TypeError, ValueError) as exc:
        raise FillBlankAnswerKeyError(f"{label} words cannot be extracted") from exc
    words: list[Mapping[str, Any]] = []
    for word in raw_words:
        if not isinstance(word, Mapping):
            raise FillBlankAnswerKeyError(f"{label} contains a malformed word")
        try:
            x0, top, x1, bottom = (
                float(word[field]) for field in ("x0", "top", "x1", "bottom")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FillBlankAnswerKeyError(
                f"{label} contains a malformed word"
            ) from exc
        if (
            not all(math.isfinite(value) for value in (x0, top, x1, bottom))
            or not x0 < x1
            or not top < bottom
        ):
            raise FillBlankAnswerKeyError(f"{label} contains a malformed word")
        if _plain_text(word.get("text"), f"{label} word"):
            words.append(word)
    return words


def _ordered_words(
    words: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return sorted(
        words,
        key=lambda word: (
            round(float(word["top"]), 1),
            float(word["x0"]),
            str(word.get("text") or ""),
        ),
    )


def _word_projection(word: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": _plain_text(word.get("text"), "PDF word"),
        "x0": round(float(word["x0"]), 3),
        "top": round(float(word["top"]), 3),
        "x1": round(float(word["x1"]), 3),
        "bottom": round(float(word["bottom"]), 3),
    }


def _bbox_projection(
    bbox: tuple[float, float, float, float]
) -> list[float]:
    return [round(value, 3) for value in bbox]


def _crop_text(page: Any, bbox: tuple[float, float, float, float]) -> str:
    try:
        text = page.crop(tuple(bbox)).extract_text() or ""
    except (AttributeError, TypeError, ValueError) as exc:
        raise FillBlankAnswerKeyError("PDF crop text cannot be extracted") from exc
    # ``pdfplumber`` legitimately separates lines with newline controls.  Fold
    # those layout separators before applying the plain-text safety check.
    return _plain_text(" ".join(str(text).split()), "PDF crop text")


def _phrase_count(tokens: Sequence[str], phrase: Sequence[str]) -> int:
    if not phrase or len(phrase) > len(tokens):
        return 0
    return sum(
        tuple(tokens[index : index + len(phrase)]) == tuple(phrase)
        for index in range(len(tokens) - len(phrase) + 1)
    )


def parse_fill_blank_canonical_answer(
    canonical_answer: str, *, expected_item_count: int
) -> tuple[tuple[int, str], ...]:
    """Parse the source-native ``1=value; ...`` representation."""

    item_count = _positive_integer(expected_item_count, "expected_item_count")
    text = _plain_text(canonical_answer, "canonical fill-blank answer")
    if not text:
        raise FillBlankAnswerKeyError("canonical fill-blank answer is empty")
    parts = text.split(";")
    result: list[tuple[int, str]] = []
    for expected_number, raw_part in enumerate(parts, start=1):
        part = raw_part.strip()
        if part.count("=") != 1:
            raise FillBlankAnswerKeyError(
                "canonical fill-blank answer must use 'number=value; ...'"
            )
        raw_number, raw_value = (piece.strip() for piece in part.split("=", 1))
        if not raw_number.isdigit() or int(raw_number) != expected_number:
            raise FillBlankAnswerKeyError(
                "canonical fill-blank labels must be consecutive from one"
            )
        if not raw_value:
            raise FillBlankAnswerKeyError(
                "canonical fill-blank answer contains an empty value"
            )
        result.append((expected_number, raw_value))
    if len(result) != item_count:
        raise FillBlankAnswerKeyError(
            "canonical fill-blank answer has the wrong item count"
        )
    return tuple(result)


def _answer_bank_counter(answer_items: Sequence[tuple[int, str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for _number, value in answer_items:
        for component in value.split("/"):
            tokens = normalize_tokens(_fold(component, "answer component"))
            if not tokens:
                raise FillBlankAnswerKeyError(
                    "canonical fill-blank answer contains an empty component"
                )
            counter.update(tokens)
    return counter


def _content_projection(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    content_bbox: tuple[float, float, float, float],
    word_bank_bbox: tuple[float, float, float, float],
    activity_title: str,
    instruction_text: str,
    expected_item_count: int,
    answer_items: Sequence[tuple[int, str]],
) -> tuple[dict[str, Any], str]:
    _validate_bbox_on_page(page, content_bbox, "content page")
    _validate_bbox_on_page(page, word_bank_bbox, "word-bank page")
    if not (
        content_bbox[0] <= word_bank_bbox[0] < word_bank_bbox[2] <= content_bbox[2]
        and content_bbox[1] <= word_bank_bbox[1] < word_bank_bbox[3] <= content_bbox[3]
    ):
        raise FillBlankAnswerKeyError("word-bank bbox is outside the content bbox")
    all_words = _extract_words(page, "content page")
    content_words = _ordered_words(
        [word for word in all_words if _inside_bbox(word, content_bbox)]
    )
    bank_words = _ordered_words(
        [word for word in all_words if _inside_bbox(word, word_bank_bbox)]
    )
    if not content_words or not bank_words:
        raise FillBlankAnswerKeyError("content or word-bank crop is empty")
    content_text = _crop_text(page, content_bbox)
    content_tokens = normalize_tokens(_fold(content_text))
    title_tokens = normalize_tokens(_fold(activity_title, "activity_title"))
    instruction_tokens = normalize_tokens(
        _fold(instruction_text, "instruction_text")
    )
    title_count = _phrase_count(content_tokens, title_tokens)
    instruction_count = _phrase_count(content_tokens, instruction_tokens)
    if title_count != 1 or instruction_count != 1:
        raise FillBlankAnswerKeyError(
            "content page does not expose the exact title and instruction once"
        )
    raw_markers: list[tuple[int, Mapping[str, Any]]] = []
    for word in content_words:
        match = _NUMBERED_MARKER.fullmatch(
            _plain_text(word.get("text"), "content marker")
        )
        if match is not None:
            raw_markers.append((int(match.group(1)), word))
    expected_numbers = set(range(1, expected_item_count + 1))
    complete_columns: list[dict[int, Mapping[str, Any]]] = []
    for column in _marker_columns(raw_markers):
        grouped: dict[int, list[Mapping[str, Any]]] = {}
        for number, word in column:
            grouped.setdefault(number, []).append(word)
        if set(grouped) == expected_numbers and all(
            len(values) == 1 for values in grouped.values()
        ):
            complete_columns.append(
                {number: values[0] for number, values in grouped.items()}
            )
    if len(complete_columns) != 1:
        raise FillBlankAnswerKeyError(
            "content page does not expose one unique complete marker column"
        )
    marker_words = complete_columns[0]
    bank_counter: Counter[str] = Counter()
    for word in bank_words:
        bank_counter.update(normalize_tokens(_fold(word.get("text"))))
    expected_bank_counter = _answer_bank_counter(answer_items)
    if bank_counter != expected_bank_counter:
        raise FillBlankAnswerKeyError(
            "content word bank does not equal the canonical answer multiset"
        )
    width, height = _page_size(page, "content page")
    return (
        {
            "projection_schema": CONTENT_PROJECTION_SCHEMA,
            "pdf_sha256": pdf_sha256,
            "physical_page": physical_page,
            "page_size": [round(width, 3), round(height, 3)],
            "content_bbox": _bbox_projection(content_bbox),
            "word_bank_bbox": _bbox_projection(word_bank_bbox),
            "activity_title_tokens": list(title_tokens),
            "instruction_tokens": list(instruction_tokens),
            "activity_title_count": title_count,
            "instruction_count": instruction_count,
            "expected_item_count": expected_item_count,
            "item_markers": {
                str(number): _word_projection(marker_words[number])
                for number in sorted(marker_words)
            },
            "word_bank_token_counts": dict(sorted(bank_counter.items())),
            "words": [_word_projection(word) for word in content_words],
        },
        content_text,
    )


def _marker_columns(
    markers: Sequence[tuple[int, Mapping[str, Any]]],
) -> list[list[tuple[int, Mapping[str, Any]]]]:
    columns: list[list[tuple[int, Mapping[str, Any]]]] = []
    centers: list[float] = []
    for item in sorted(markers, key=lambda value: float(value[1]["x0"])):
        x0 = float(item[1]["x0"])
        destination = next(
            (index for index, center in enumerate(centers) if abs(x0 - center) <= 8.0),
            None,
        )
        if destination is None:
            columns.append([item])
            centers.append(x0)
        else:
            columns[destination].append(item)
            centers[destination] = sum(
                float(marker["x0"]) for _, marker in columns[destination]
            ) / len(columns[destination])
    order = sorted(range(len(columns)), key=lambda index: centers[index])
    return [columns[index] for index in order]


def _derive_key_answers(
    words: Sequence[Mapping[str, Any]],
    *,
    key_bbox: tuple[float, float, float, float],
    expected_item_count: int,
    expected_column_count: int,
) -> tuple[dict[int, str], dict[int, Mapping[str, Any]], list[float]]:
    markers: list[tuple[int, Mapping[str, Any]]] = []
    for word in words:
        match = _NUMBERED_MARKER.fullmatch(
            _plain_text(word.get("text"), "answer-key marker")
        )
        if match is not None:
            markers.append((int(match.group(1)), word))
    by_number: dict[int, list[Mapping[str, Any]]] = {}
    for number, word in markers:
        by_number.setdefault(number, []).append(word)
    expected_numbers = set(range(1, expected_item_count + 1))
    if set(by_number) != expected_numbers or any(
        len(values) != 1 for values in by_number.values()
    ):
        raise FillBlankAnswerKeyError(
            "answer-key crop does not expose one marker for every item"
        )
    unique_markers = [(number, values[0]) for number, values in by_number.items()]
    columns = _marker_columns(unique_markers)
    if len(columns) != expected_column_count:
        raise FillBlankAnswerKeyError(
            "answer-key crop has an unexpected marker-column count"
        )
    column_x = [
        sum(float(marker["x0"]) for _, marker in column) / len(column)
        for column in columns
    ]
    derived: dict[int, str] = {}
    for column_index, raw_column in enumerate(columns):
        column = sorted(raw_column, key=lambda item: float(item[1]["top"]))
        numbers = [number for number, _ in column]
        if numbers != sorted(numbers):
            raise FillBlankAnswerKeyError(
                "answer-key numbers are not increasing within a column"
            )
        left = key_bbox[0] if column_index == 0 else column_x[column_index] - 2.0
        right = (
            column_x[column_index + 1] - 2.0
            if column_index + 1 < len(columns)
            else key_bbox[2]
        )
        for position, (number, marker) in enumerate(column):
            top = float(marker["top"]) - 1.0
            bottom = (
                float(column[position + 1][1]["top"]) - 1.0
                if position + 1 < len(column)
                else key_bbox[3]
            )
            answer_words = _ordered_words(
                [
                    word
                    for word in words
                    if not _NUMBERED_MARKER.fullmatch(
                        _plain_text(word.get("text"), "answer-key word")
                    )
                    and left <= _word_center(word)[0] < right
                    and top <= _word_center(word)[1] < bottom
                    and (
                        _word_center(word)[1] > _word_center(marker)[1] + 2.5
                        or float(word["x0"]) >= float(marker["x1"]) - 0.2
                    )
                ]
            )
            value = " ".join(
                _plain_text(word.get("text"), "answer-key word")
                for word in answer_words
            ).strip()
            if not value:
                raise FillBlankAnswerKeyError(
                    f"answer-key item {number} has no value"
                )
            derived[number] = value
    return derived, {number: values[0] for number, values in by_number.items()}, column_x


def _key_projection(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    key_bbox: tuple[float, float, float, float],
    activity_title: str,
    expected_item_count: int,
    expected_column_count: int,
) -> tuple[dict[str, Any], dict[int, str], str]:
    _validate_bbox_on_page(page, key_bbox, "answer-key page")
    all_words = _extract_words(page, "answer-key page")
    words = _ordered_words(
        [word for word in all_words if _inside_bbox(word, key_bbox)]
    )
    if not words:
        raise FillBlankAnswerKeyError("answer-key crop is empty")
    key_text = _crop_text(page, key_bbox)
    title_tokens = normalize_tokens(_fold(activity_title, "activity_title"))
    key_tokens = normalize_tokens(_fold(key_text))
    title_count = _phrase_count(key_tokens, title_tokens)
    if title_count != 1:
        raise FillBlankAnswerKeyError(
            "answer-key crop does not expose the exact activity heading once"
        )
    derived, marker_words, column_x = _derive_key_answers(
        words,
        key_bbox=key_bbox,
        expected_item_count=expected_item_count,
        expected_column_count=expected_column_count,
    )
    width, height = _page_size(page, "answer-key page")
    return (
        {
            "projection_schema": KEY_PROJECTION_SCHEMA,
            "pdf_sha256": pdf_sha256,
            "physical_page": physical_page,
            "page_size": [round(width, 3), round(height, 3)],
            "key_bbox": _bbox_projection(key_bbox),
            "activity_title_tokens": list(title_tokens),
            "activity_title_count": title_count,
            "expected_item_count": expected_item_count,
            "expected_column_count": expected_column_count,
            "marker_column_x": [round(value, 3) for value in column_x],
            "item_markers": {
                str(number): _word_projection(marker_words[number])
                for number in sorted(marker_words)
            },
            "derived_answers": {
                str(number): derived[number] for number in sorted(derived)
            },
            "words": [_word_projection(word) for word in words],
        },
        derived,
        key_text,
    )


def attest_fill_blank_answer_key(
    content_page: Any,
    key_page: Any,
    *,
    pdf_sha256: str,
    content_page_number: int,
    key_page_number: int,
    content_bbox: Sequence[float],
    word_bank_bbox: Sequence[float],
    key_bbox: Sequence[float],
    activity_title: str,
    instruction_text: str,
    expected_item_count: int,
    expected_column_count: int,
    expected_answer: str,
) -> FillBlankAnswerKeyVerification:
    """Derive and attest one complete fill-in activity from PDF geometry."""

    if _HEX64.fullmatch(pdf_sha256) is None:
        raise FillBlankAnswerKeyError("fill-blank PDF hash is malformed")
    content_number = _positive_integer(content_page_number, "content_page_number")
    key_number = _positive_integer(key_page_number, "key_page_number")
    if content_number == key_number:
        raise FillBlankAnswerKeyError(
            "fill-blank content and answer-key pages must be independent"
        )
    item_count = _positive_integer(expected_item_count, "expected_item_count")
    column_count = _positive_integer(expected_column_count, "expected_column_count")
    answer_items = parse_fill_blank_canonical_answer(
        expected_answer, expected_item_count=item_count
    )
    content_box = _strict_bbox(content_bbox, "content_bbox")
    bank_box = _strict_bbox(word_bank_bbox, "word_bank_bbox")
    key_box = _strict_bbox(key_bbox, "key_bbox")
    content_projection, content_text = _content_projection(
        content_page,
        pdf_sha256=pdf_sha256,
        physical_page=content_number,
        content_bbox=content_box,
        word_bank_bbox=bank_box,
        activity_title=activity_title,
        instruction_text=instruction_text,
        expected_item_count=item_count,
        answer_items=answer_items,
    )
    key_projection, derived, key_text = _key_projection(
        key_page,
        pdf_sha256=pdf_sha256,
        physical_page=key_number,
        key_bbox=key_box,
        activity_title=activity_title,
        expected_item_count=item_count,
        expected_column_count=column_count,
    )
    expected = dict(answer_items)
    matches = {
        number: _canonical_component(derived[number])
        == _canonical_component(expected[number])
        for number in range(1, item_count + 1)
    }
    if not all(matches.values()):
        mismatches = [number for number, passed in matches.items() if not passed]
        raise FillBlankAnswerKeyError(
            f"answer-key values disagree with canonical source entries: {mismatches}"
        )
    content_sha256 = canonical_json_sha256(content_projection)
    key_sha256 = canonical_json_sha256(key_projection)
    binding_projection = {
        "projection_schema": PROJECTION_SCHEMA,
        "pdf_sha256": pdf_sha256,
        "content_page_number": content_number,
        "key_page_number": key_number,
        "content_projection_sha256": content_sha256,
        "key_projection_sha256": key_sha256,
        "activity_title_tokens": list(
            normalize_tokens(_fold(activity_title, "activity_title"))
        ),
        "instruction_tokens": list(
            normalize_tokens(_fold(instruction_text, "instruction_text"))
        ),
        "expected_item_count": item_count,
        "expected_column_count": column_count,
        "canonical_answer_sha256": canonical_json_sha256(
            {str(number): expected[number] for number in sorted(expected)}
        ),
    }
    return FillBlankAnswerKeyVerification(
        projection_sha256=canonical_json_sha256(binding_projection),
        content_projection_sha256=content_sha256,
        key_projection_sha256=key_sha256,
        derived_answer=derived,
        component_matches=matches,
        content_text=content_text,
        key_text=key_text,
    )


def verify_fill_blank_answer_key(
    content_page: Any,
    key_page: Any,
    *,
    expected_projection_sha256: str,
    expected_content_projection_sha256: str,
    expected_key_projection_sha256: str,
    **kwargs: Any,
) -> FillBlankAnswerKeyVerification:
    """Replay an attestation and require all three frozen projection pins."""

    expected_hashes = {
        "binding": expected_projection_sha256,
        "content": expected_content_projection_sha256,
        "key": expected_key_projection_sha256,
    }
    if any(_HEX64.fullmatch(value) is None for value in expected_hashes.values()):
        raise FillBlankAnswerKeyError("fill-blank projection hash is malformed")
    verification = attest_fill_blank_answer_key(content_page, key_page, **kwargs)
    actual = {
        "binding": verification.projection_sha256,
        "content": verification.content_projection_sha256,
        "key": verification.key_projection_sha256,
    }
    mismatches = [name for name in expected_hashes if expected_hashes[name] != actual[name]]
    if mismatches:
        raise FillBlankAnswerKeyError(
            "fill-blank projection differs from source index: " + ", ".join(mismatches)
        )
    return verification
