"""Fail-closed binding for activity answers in an official MEB PDF.

The content page proves an exact ``ETKİNLİK-N`` marker.  Independently,
the answer-key page proves the unit heading, ``Etkinlik N (P. Sayfa)`` header,
the complete bounded section, and every canonical answer component.  Both PDF
projections and their joint address are hash-pinned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Any

from .official_ogm import canonical_json_sha256, normalize_tokens


PROJECTION_SCHEMA = "pdf-activity-answer-key-projection-v1"
CONTENT_PROJECTION_SCHEMA = "pdf-activity-content-projection-v1"
KEY_PROJECTION_SCHEMA = "pdf-activity-key-projection-v1"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_MARKER_PREFIX = re.compile(r"^(?:##\s*)?etkinlik-")
_CONTENT_MARKER = re.compile(r"^(?:##\s*)?etkinlik-([1-9]\d*|s)$")
_KEY_HEADER = re.compile(
    r"^etkinlik\s+([1-9]\d*)\s*\(\s*([1-9]\d*)\.\s*sayfa\s*\)$"
)
_UNIT_HEADING = re.compile(r"^([1-9]\d*)\.\s*unite\s*:\s*\S.*$")
_UNIT_BOUNDARY = re.compile(r"^[1-9]\d*\.\s*unite(?:\s|:|$)")
_LABEL_TOKEN = re.compile(r"(?<![\w])([a-zçğıöşü])\)\s*")
_NUMBER_TOKEN = re.compile(r"(?:^|\s)([1-9]\d*)\.\s+")
_SCALAR_EXIT = re.compile(r"^dogru\s+cikis\s*:\s*([1-9]\d*)$")
_NUMBERED_HEADINGS = frozenset({"soldan saga", "yukaridan asagiya"})
_TURKISH_LABELS = tuple("abcçdefgğhıijklmnoöprsştuüvyz")
_ANSWER_FORMAT_ALIASES = {
    "labelled": "labelled_short_text",
    "labelled_short_text": "labelled_short_text",
    "numbered": "numbered_short_text",
    "numbered_short_text": "numbered_short_text",
    "scalar_exit": "scalar_exit",
}


class ActivityAnswerKeyError(ValueError):
    """The two PDF pages could not prove one activity answer binding."""


@dataclass(frozen=True, slots=True)
class ActivityAnswerKeyVerification:
    projection_sha256: str
    content_projection_sha256: str
    key_projection_sha256: str
    derived_answer: Mapping[str, str]
    component_matches: Mapping[str, bool]


@dataclass(frozen=True, slots=True)
class _Line:
    words: tuple[Mapping[str, Any], ...]
    text: str
    top: float
    bottom: float


def _plain_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in value
    ):
        raise ActivityAnswerKeyError(f"{label} must be plain Unicode text")
    return unicodedata.normalize("NFKC", value).strip()


def _fold(value: Any, label: str = "PDF text") -> str:
    text = _plain_text(value, label).casefold().replace("i\u0307", "i")
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    without_marks = without_marks.replace("ı", "i")
    return re.sub(r"\s+", " ", without_marks).strip()


def _canonical_component(value: Any) -> str:
    text = (
        _plain_text(value, "answer component")
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .casefold()
        .replace("i\u0307", "i")
    )
    return re.sub(r"\s+", "", text)


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ActivityAnswerKeyError(f"{label} must be a positive integer")
    return value


def _strict_bbox(value: Any, label: str) -> tuple[float, float, float, float]:
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
        raise ActivityAnswerKeyError(f"{label} is malformed")
    bbox = tuple(float(item) for item in value)
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ActivityAnswerKeyError(f"{label} is empty")
    return bbox  # type: ignore[return-value]


def _page_size(page: Any, label: str) -> tuple[float, float]:
    try:
        width = float(page.width)
        height = float(page.height)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ActivityAnswerKeyError(f"{label} has no finite page geometry") from exc
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ActivityAnswerKeyError(f"{label} has no finite page geometry")
    return width, height


def _validate_bbox_on_page(
    page: Any,
    bbox: tuple[float, float, float, float],
    label: str,
) -> None:
    width, height = _page_size(page, label)
    if not (
        0.0 <= bbox[0] < bbox[2] <= width
        and 0.0 <= bbox[1] < bbox[3] <= height
    ):
        raise ActivityAnswerKeyError(f"{label} bbox is outside the PDF page")


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


def _extract_words(page: Any, page_label: str) -> list[Mapping[str, Any]]:
    try:
        raw_words = page.extract_words(
            x_tolerance=2,
            y_tolerance=2,
            keep_blank_chars=False,
            use_text_flow=False,
        ) or []
    except (AttributeError, TypeError, ValueError) as exc:
        raise ActivityAnswerKeyError(f"{page_label} words cannot be extracted") from exc

    result: list[Mapping[str, Any]] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, Mapping):
            raise ActivityAnswerKeyError(f"{page_label} contains a malformed word")
        try:
            coordinates = tuple(
                float(raw_word[key]) for key in ("x0", "top", "x1", "bottom")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ActivityAnswerKeyError(f"{page_label} contains a malformed word") from exc
        if any(not math.isfinite(value) for value in coordinates) or not (
            coordinates[0] < coordinates[2] and coordinates[1] < coordinates[3]
        ):
            raise ActivityAnswerKeyError(f"{page_label} contains a malformed word")
        text = _plain_text(raw_word.get("text"), f"{page_label} word")
        if not text:
            continue
        result.append(raw_word)
    return result


def _lines(words: Sequence[Mapping[str, Any]]) -> list[_Line]:
    ordered = sorted(
        words,
        key=lambda word: (
            _word_center(word)[1],
            float(word["x0"]),
            _plain_text(word.get("text"), "PDF word"),
        ),
    )
    groups: list[list[Mapping[str, Any]]] = []
    centers: list[float] = []
    for word in ordered:
        center_y = _word_center(word)[1]
        if groups and abs(center_y - centers[-1]) <= 2.5:
            groups[-1].append(word)
            centers[-1] = sum(_word_center(item)[1] for item in groups[-1]) / len(
                groups[-1]
            )
        else:
            groups.append([word])
            centers.append(center_y)

    result: list[_Line] = []
    for group in groups:
        line_words = tuple(
            sorted(group, key=lambda word: (float(word["x0"]), float(word["x1"])))
        )
        text = " ".join(
            _plain_text(word.get("text"), "PDF word") for word in line_words
        )
        result.append(
            _Line(
                words=line_words,
                text=re.sub(r"\s+", " ", text).strip(),
                top=min(float(word["top"]) for word in line_words),
                bottom=max(float(word["bottom"]) for word in line_words),
            )
        )
    return result


def _word_projection(word: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": str(word.get("text") or "").strip(),
        "x0": round(float(word["x0"]), 3),
        "top": round(float(word["top"]), 3),
        "x1": round(float(word["x1"]), 3),
        "bottom": round(float(word["bottom"]), 3),
    }


def _bbox_projection(bbox: tuple[float, float, float, float]) -> list[float]:
    return [round(value, 3) for value in bbox]


def _projection_words(
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


def _union_bbox(words: Sequence[Mapping[str, Any]]) -> list[float]:
    if not words:
        raise ActivityAnswerKeyError("cannot derive a bbox from no words")
    return [
        round(min(float(word["x0"]) for word in words), 3),
        round(min(float(word["top"]) for word in words), 3),
        round(max(float(word["x1"]) for word in words), 3),
        round(max(float(word["bottom"]) for word in words), 3),
    ]


def _crop_text(
    page: Any,
    bbox: tuple[float, float, float, float],
    words: Sequence[Mapping[str, Any]],
) -> str:
    try:
        extracted = page.crop(tuple(bbox)).extract_text() or ""
    except (AttributeError, TypeError, ValueError):
        extracted = " ".join(line.text for line in _lines(words))
    return " ".join(str(extracted).split())


def _crop_tokens_sha256(
    page: Any,
    bbox: tuple[float, float, float, float],
    words: Sequence[Mapping[str, Any]],
) -> str:
    return canonical_json_sha256(
        {"tokens": list(normalize_tokens(_crop_text(page, bbox, words)))}
    )


def _canonical_answer_format(value: Any) -> str:
    if not isinstance(value, str) or value not in _ANSWER_FORMAT_ALIASES:
        raise ActivityAnswerKeyError("activity answer format is unsupported")
    return _ANSWER_FORMAT_ALIASES[value]


def _answer_kind(answer_format: str) -> str:
    if answer_format == "labelled_short_text":
        return "labelled"
    if answer_format == "numbered_short_text":
        return "numbered"
    return "scalar_exit"


def parse_activity_canonical_answer(
    canonical_answer: str,
    *,
    answer_format: str,
) -> tuple[tuple[str, str], ...]:
    """Parse the literal answer form used by canonical short-text records."""

    canonical_format = _canonical_answer_format(answer_format)
    text = _plain_text(canonical_answer, "canonical activity answer")
    if not text:
        raise ActivityAnswerKeyError("canonical activity answer is empty")
    if canonical_format == "scalar_exit":
        return (("scalar", text),)

    parts = text.split(";")
    result: list[tuple[str, str]] = []
    for part in parts:
        stripped = part.strip()
        if not stripped or stripped.count("=") != 1:
            raise ActivityAnswerKeyError(
                "canonical activity answer must use 'label=value; ...'"
            )
        label, value = (item.strip() for item in stripped.split("=", 1))
        if not label or not value:
            raise ActivityAnswerKeyError(
                "canonical activity answer contains an empty component"
            )
        result.append((label, value))
    return tuple(result)


def _expected_components(
    expected: Mapping[str, str] | Sequence[tuple[str, str]],
    answer_format: str,
) -> list[tuple[str, str]]:
    if isinstance(expected, Mapping):
        raw_items = list(expected.items())
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        raw_items = list(expected)
    else:
        raise ActivityAnswerKeyError("expected components are malformed")
    if not raw_items:
        raise ActivityAnswerKeyError("expected components are empty")

    items: list[tuple[str, str]] = []
    labels: set[str] = set()
    for raw_item in raw_items:
        if (
            not isinstance(raw_item, Sequence)
            or isinstance(raw_item, (str, bytes))
            or len(raw_item) != 2
        ):
            raise ActivityAnswerKeyError("expected components are malformed")
        label = _plain_text(raw_item[0], "component label").casefold().replace(
            "i\u0307", "i"
        )
        value = _plain_text(raw_item[1], "answer component")
        if not label or not value:
            raise ActivityAnswerKeyError("expected components contain an empty value")
        if label in labels:
            raise ActivityAnswerKeyError("expected components repeat a label")
        labels.add(label)
        items.append((label, value))

    item_labels = tuple(label for label, _ in items)
    answer_kind = _answer_kind(answer_format)
    if answer_kind == "labelled":
        if item_labels != _TURKISH_LABELS[: len(item_labels)]:
            raise ActivityAnswerKeyError(
                "labelled components must be one canonical Turkish alphabet prefix"
            )
    elif answer_kind == "numbered":
        if any(re.fullmatch(r"[1-9]\d*", label) is None for label in item_labels):
            raise ActivityAnswerKeyError("numbered components require decimal labels")
        numeric = {int(label) for label in item_labels}
        if numeric != set(range(1, max(numeric) + 1)):
            raise ActivityAnswerKeyError(
                "numbered components must cover one contiguous range from 1"
            )
    elif item_labels not in {("scalar",), ("exit",)}:
        raise ActivityAnswerKeyError(
            "scalar_exit requires one component labelled 'scalar' or 'exit'"
        )
    return items


def _answer_binding(
    *,
    answer_format: str,
    expected_components: Mapping[str, str] | Sequence[tuple[str, str]] | None,
    canonical_answer: str | None,
) -> tuple[list[tuple[str, str]], str]:
    if (expected_components is None) == (canonical_answer is None):
        raise ActivityAnswerKeyError(
            "provide exactly one of expected_components or canonical_answer"
        )
    if canonical_answer is not None:
        literal = _plain_text(canonical_answer, "canonical activity answer")
        expected = _expected_components(
            parse_activity_canonical_answer(
                literal,
                answer_format=answer_format,
            ),
            answer_format,
        )
        return expected, literal

    expected = _expected_components(expected_components, answer_format)  # type: ignore[arg-type]
    if answer_format == "scalar_exit":
        literal = expected[0][1]
    else:
        literal = "; ".join(f"{label}={value}" for label, value in expected)
    return expected, unicodedata.normalize("NFKC", literal)


def _content_marker_number(line: str) -> int | None:
    folded = _fold(line)
    if _CONTENT_MARKER_PREFIX.match(folded) is None:
        return None
    match = _CONTENT_MARKER.fullmatch(folded)
    if match is None:
        raise ActivityAnswerKeyError("content marker has a non-canonical shape")
    terminal = match.group(1)
    # The official activity-5 glyph is visually S-like.  This exception is
    # deliberately confined to the one terminal immediately after the exact
    # marker prefix; no other letter or position is canonicalized.
    return 5 if terminal == "s" else int(terminal)


def _prove_content_marker(
    page: Any,
    words: Sequence[Mapping[str, Any]],
    bbox: tuple[float, float, float, float],
    activity_number: int,
) -> tuple[str, Mapping[str, Any], list[Mapping[str, Any]]]:
    bbox_words = [word for word in words if _inside_bbox(word, bbox)]
    markers: list[tuple[str, int, Mapping[str, Any]]] = []
    for line in _lines(bbox_words):
        marker_number = _content_marker_number(line.text)
        if marker_number is not None:
            marker_words = [
                word
                for word in line.words
                if _CONTENT_MARKER.fullmatch(_fold(word.get("text"))) is not None
            ]
            if len(marker_words) != 1:
                raise ActivityAnswerKeyError(
                    "content marker must expose one exact positioned marker word"
                )
            markers.append((line.text, marker_number, marker_words[0]))
    if len(markers) != 1:
        raise ActivityAnswerKeyError(
            "content bbox must contain exactly one canonical activity marker"
        )
    marker_line, marker_number, marker_word = markers[0]
    if marker_number != activity_number:
        raise ActivityAnswerKeyError("content marker activity number differs")
    return marker_line, marker_word, bbox_words


def count_activity_markers(page: Any, activity_number: int) -> int:
    """Count canonical activity titles across the complete physical page."""

    activity_number = _positive_integer(activity_number, "activity number")
    words = _extract_words(page, "content page")
    return sum(
        _content_marker_number(line.text) == activity_number
        for line in _lines(words)
    )


def _key_section(
    page: Any,
    words: Sequence[Mapping[str, Any]],
    bbox: tuple[float, float, float, float],
    *,
    unit_number: int,
    activity_number: int,
    activity_page_number: int,
) -> tuple[str, str, str | None, list[_Line], list[Mapping[str, Any]]]:
    column_words = [
        word
        for word in words
        if bbox[0] <= _word_center(word)[0] <= bbox[2]
    ]
    column_lines = _lines(column_words)
    headers: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(column_lines):
        match = _KEY_HEADER.fullmatch(_fold(line.text))
        if match is not None and int(match.group(1)) == activity_number:
            headers.append((index, match))
    if len(headers) != 1:
        raise ActivityAnswerKeyError(
            "key column must contain exactly one numeric header for the activity"
        )
    header_index, header_match = headers[0]
    if int(header_match.group(2)) != activity_page_number:
        raise ActivityAnswerKeyError("key header printed page differs")

    unit_candidates: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(column_lines[:header_index]):
        match = _UNIT_HEADING.fullmatch(_fold(line.text))
        if match is not None:
            unit_candidates.append((index, match))
    if not unit_candidates:
        raise ActivityAnswerKeyError("key header has no preceding unit heading")
    unit_index, unit_match = unit_candidates[-1]
    if int(unit_match.group(1)) != unit_number:
        raise ActivityAnswerKeyError("key unit number differs")

    end_index = len(column_lines)
    boundary_text: str | None = None
    for index in range(header_index + 1, len(column_lines)):
        folded = _fold(column_lines[index].text)
        if _KEY_HEADER.fullmatch(folded) is not None or _UNIT_BOUNDARY.match(folded):
            end_index = index
            boundary_text = column_lines[index].text
            break
    section_lines = column_lines[header_index:end_index]
    if not section_lines:
        raise ActivityAnswerKeyError("key activity section is empty")
    section_words = [word for line in section_lines for word in line.words]
    bbox_words = [word for word in words if _inside_bbox(word, bbox)]
    if {id(word) for word in bbox_words} != {id(word) for word in section_words}:
        raise ActivityAnswerKeyError(
            "key bbox must contain the complete activity section and nothing else"
        )
    if unit_index >= header_index:
        raise ActivityAnswerKeyError("key unit heading order is invalid")
    return (
        column_lines[unit_index].text,
        column_lines[header_index].text,
        boundary_text,
        section_lines,
        section_words,
    )


def _trim_component(value: str) -> str:
    result = value.strip()
    if result.endswith(";"):
        result = result[:-1].rstrip()
    if not result:
        raise ActivityAnswerKeyError("key section has an empty component")
    return result


def _parse_labelled(lines: Sequence[_Line]) -> list[tuple[str, str]]:
    text = " ".join(line.text for line in lines[1:]).strip()
    matches = list(_LABEL_TOKEN.finditer(text))
    if not matches or text[: matches[0].start()].strip():
        raise ActivityAnswerKeyError("labelled key has text before its first label")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        label = match.group(1).casefold()
        if label in seen:
            raise ActivityAnswerKeyError("labelled key repeats a component label")
        seen.add(label)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.append((label, _trim_component(text[match.end() : end])))
    return result


def _parse_numbered(lines: Sequence[_Line]) -> list[tuple[str, str]]:
    answer_lines = [
        line.text
        for line in lines[1:]
        if _fold(line.text) not in _NUMBERED_HEADINGS
    ]
    text = " ".join(answer_lines).strip()
    matches = list(_NUMBER_TOKEN.finditer(text))
    if not matches or text[: matches[0].start()].strip():
        raise ActivityAnswerKeyError("numbered key has text before its first number")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        label = match.group(1)
        if label in seen:
            raise ActivityAnswerKeyError("numbered key repeats a component label")
        seen.add(label)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.append((label, _trim_component(text[match.end() : end])))
    return result


def _parse_scalar_exit(lines: Sequence[_Line]) -> list[tuple[str, str]]:
    exits: list[str] = []
    for line in lines[1:]:
        match = _SCALAR_EXIT.fullmatch(_fold(line.text))
        if match is not None:
            exits.append(match.group(1))
    if len(exits) != 1:
        raise ActivityAnswerKeyError(
            "scalar_exit key must contain exactly one canonical exit line"
        )
    return [("scalar", exits[0])]


def _derive_answer(
    section_lines: Sequence[_Line],
    expected: Sequence[tuple[str, str]],
    answer_format: str,
) -> tuple[dict[str, str], dict[str, bool]]:
    answer_kind = _answer_kind(answer_format)
    if answer_kind == "labelled":
        actual = _parse_labelled(section_lines)
    elif answer_kind == "numbered":
        actual = _parse_numbered(section_lines)
    else:
        actual = _parse_scalar_exit(section_lines)

    if answer_kind == "scalar_exit" and expected[0][0] == "exit":
        actual = [("exit", actual[0][1])]

    actual_labels = tuple(label for label, _ in actual)
    expected_labels = tuple(label for label, _ in expected)
    if len(actual_labels) != len(set(actual_labels)):
        raise ActivityAnswerKeyError("key section repeats a component label")
    labels_match = (
        set(actual_labels) == set(expected_labels)
        if answer_kind == "numbered"
        else actual_labels == expected_labels
    )
    if not labels_match:
        raise ActivityAnswerKeyError(
            "key component labels or their order differ from the canonical binding"
        )
    derived = dict(actual)
    matches = {
        label: _canonical_component(derived[label]) == _canonical_component(value)
        for label, value in expected
    }
    if not matches or not all(matches.values()):
        raise ActivityAnswerKeyError(
            "key section does not expose every canonical answer component"
        )
    return derived, matches


def _content_projection(
    *,
    pdf_sha256: str,
    physical_page: int,
    printed_page_number: int,
    unit_number: int,
    activity_number: int,
    page: Any,
    bbox: tuple[float, float, float, float],
    marker_word: Mapping[str, Any],
    words: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    width, height = _page_size(page, "content page")
    ordered_words = _projection_words(words)
    return {
        "projection_schema": CONTENT_PROJECTION_SCHEMA,
        "pdf_sha256": pdf_sha256,
        "physical_page": physical_page,
        "printed_page_number": printed_page_number,
        "page_size": [round(width, 3), round(height, 3)],
        "unit_number": unit_number,
        "activity_number": activity_number,
        "marker_text": str(marker_word.get("text") or "").strip(),
        "marker_bbox": _union_bbox((marker_word,)),
        "content_bbox": _bbox_projection(bbox),
        "marker": _word_projection(marker_word),
        "content_tokens_sha256": _crop_tokens_sha256(page, bbox, words),
        "words": [_word_projection(word) for word in ordered_words],
    }


def _key_projection(
    *,
    pdf_sha256: str,
    physical_page: int,
    unit_number: int,
    printed_source_page_number: int,
    activity_number: int,
    answer_format: str,
    page: Any,
    bbox: tuple[float, float, float, float],
    header_line: str,
    header_words: Sequence[Mapping[str, Any]],
    words: Sequence[Mapping[str, Any]],
    canonical_answer: str,
) -> dict[str, Any]:
    width, height = _page_size(page, "key page")
    ordered_words = _projection_words(words)
    return {
        "projection_schema": KEY_PROJECTION_SCHEMA,
        "pdf_sha256": pdf_sha256,
        "physical_page": physical_page,
        "unit_number": unit_number,
        "activity_number": activity_number,
        "printed_source_page_number": printed_source_page_number,
        "page_size": [round(width, 3), round(height, 3)],
        "key_header_text": header_line,
        "key_header_bbox": _union_bbox(header_words),
        "key_bbox": _bbox_projection(bbox),
        "answer_format": answer_format,
        "canonical_answer": unicodedata.normalize("NFKC", canonical_answer),
        "key_tokens_sha256": _crop_tokens_sha256(page, bbox, words),
        "words": [_word_projection(word) for word in ordered_words],
    }


def activity_joint_projection_sha256(
    *,
    pdf_sha256: str,
    unit_number: int,
    activity_number: int,
    activity_page_number: int,
    answer_format: str,
    content_projection_sha256: str,
    key_projection_sha256: str,
) -> str:
    """Derive the third pin from the two raw projections and full address."""

    if not isinstance(pdf_sha256, str) or _HEX64.fullmatch(pdf_sha256) is None:
        raise ActivityAnswerKeyError("activity PDF hash is malformed")
    unit_number = _positive_integer(unit_number, "unit number")
    activity_number = _positive_integer(activity_number, "activity number")
    activity_page_number = _positive_integer(
        activity_page_number, "activity printed page"
    )
    canonical_format = _canonical_answer_format(answer_format)
    if (
        not isinstance(content_projection_sha256, str)
        or _HEX64.fullmatch(content_projection_sha256) is None
        or not isinstance(key_projection_sha256, str)
        or _HEX64.fullmatch(key_projection_sha256) is None
    ):
        raise ActivityAnswerKeyError("activity child projection hash is malformed")
    return canonical_json_sha256(
        {
            "projection_schema": PROJECTION_SCHEMA,
            "pdf_sha256": pdf_sha256,
            "unit_number": unit_number,
            "activity_page_number": activity_page_number,
            "activity_number": activity_number,
            "answer_format": canonical_format,
            "content_projection_sha256": content_projection_sha256,
            "key_projection_sha256": key_projection_sha256,
        }
    )


def attest_activity_answer_key(
    content_page: Any,
    key_page: Any,
    *,
    pdf_sha256: str,
    content_physical_page: int,
    key_physical_page: int,
    unit_number: int,
    activity_number: int,
    activity_page_number: int,
    content_bbox: Sequence[float],
    key_bbox: Sequence[float],
    answer_format: str,
    expected_components: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    canonical_answer: str | None = None,
) -> ActivityAnswerKeyVerification:
    """Derive and attest one complete activity answer from two PDF pages."""

    if not isinstance(pdf_sha256, str) or _HEX64.fullmatch(pdf_sha256) is None:
        raise ActivityAnswerKeyError("activity PDF hash is malformed")
    content_physical_page = _positive_integer(
        content_physical_page, "content physical page"
    )
    key_physical_page = _positive_integer(key_physical_page, "key physical page")
    unit_number = _positive_integer(unit_number, "unit number")
    activity_number = _positive_integer(activity_number, "activity number")
    activity_page_number = _positive_integer(
        activity_page_number, "activity printed page"
    )
    if content_physical_page != activity_page_number:
        raise ActivityAnswerKeyError(
            "content physical page must equal the key header printed page"
        )
    if key_physical_page <= content_physical_page:
        raise ActivityAnswerKeyError("key page must follow the content page")
    canonical_format = _canonical_answer_format(answer_format)

    strict_content_bbox = _strict_bbox(content_bbox, "content bbox")
    strict_key_bbox = _strict_bbox(key_bbox, "key bbox")
    _validate_bbox_on_page(content_page, strict_content_bbox, "content page")
    _validate_bbox_on_page(key_page, strict_key_bbox, "key page")
    expected, canonical_answer_literal = _answer_binding(
        answer_format=canonical_format,
        expected_components=expected_components,
        canonical_answer=canonical_answer,
    )

    content_words = _extract_words(content_page, "content page")
    key_words = _extract_words(key_page, "key page")
    _marker_line, marker_word, marker_words = _prove_content_marker(
        content_page,
        content_words,
        strict_content_bbox,
        activity_number,
    )
    (
        _unit_heading,
        header_line,
        _boundary_line,
        section_lines,
        section_words,
    ) = _key_section(
        key_page,
        key_words,
        strict_key_bbox,
        unit_number=unit_number,
        activity_number=activity_number,
        activity_page_number=activity_page_number,
    )
    derived_answer, component_matches = _derive_answer(
        section_lines,
        expected,
        canonical_format,
    )

    content_projection_sha256 = canonical_json_sha256(
        _content_projection(
            pdf_sha256=pdf_sha256,
            physical_page=content_physical_page,
            printed_page_number=activity_page_number,
            unit_number=unit_number,
            activity_number=activity_number,
            page=content_page,
            bbox=strict_content_bbox,
            marker_word=marker_word,
            words=marker_words,
        )
    )
    key_projection_sha256 = canonical_json_sha256(
        _key_projection(
            pdf_sha256=pdf_sha256,
            physical_page=key_physical_page,
            unit_number=unit_number,
            printed_source_page_number=activity_page_number,
            activity_number=activity_number,
            answer_format=canonical_format,
            page=key_page,
            bbox=strict_key_bbox,
            header_line=header_line,
            header_words=section_lines[0].words,
            words=section_words,
            canonical_answer=canonical_answer_literal,
        )
    )
    projection_sha256 = activity_joint_projection_sha256(
        pdf_sha256=pdf_sha256,
        unit_number=unit_number,
        activity_number=activity_number,
        activity_page_number=activity_page_number,
        answer_format=canonical_format,
        content_projection_sha256=content_projection_sha256,
        key_projection_sha256=key_projection_sha256,
    )
    return ActivityAnswerKeyVerification(
        projection_sha256=projection_sha256,
        content_projection_sha256=content_projection_sha256,
        key_projection_sha256=key_projection_sha256,
        derived_answer=derived_answer,
        component_matches=component_matches,
    )


def verify_activity_answer_key(
    content_page: Any,
    key_page: Any,
    *,
    pdf_sha256: str,
    content_physical_page: int,
    key_physical_page: int,
    unit_number: int,
    activity_number: int,
    activity_page_number: int,
    content_bbox: Sequence[float],
    key_bbox: Sequence[float],
    answer_format: str,
    expected_components: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    canonical_answer: str | None = None,
    expected_content_projection_sha256: str,
    expected_key_projection_sha256: str,
    expected_projection_sha256: str,
) -> ActivityAnswerKeyVerification:
    """Recompute the proof and require all three frozen projection hashes."""

    frozen = {
        "content": expected_content_projection_sha256,
        "key": expected_key_projection_sha256,
        "joint": expected_projection_sha256,
    }
    if any(
        not isinstance(value, str) or _HEX64.fullmatch(value) is None
        for value in frozen.values()
    ):
        raise ActivityAnswerKeyError("activity projection hash is malformed")

    verification = attest_activity_answer_key(
        content_page,
        key_page,
        pdf_sha256=pdf_sha256,
        content_physical_page=content_physical_page,
        key_physical_page=key_physical_page,
        unit_number=unit_number,
        activity_number=activity_number,
        activity_page_number=activity_page_number,
        content_bbox=content_bbox,
        key_bbox=key_bbox,
        answer_format=answer_format,
        expected_components=expected_components,
        canonical_answer=canonical_answer,
    )
    if verification.content_projection_sha256 != frozen["content"]:
        raise ActivityAnswerKeyError("content projection differs from its frozen hash")
    if verification.key_projection_sha256 != frozen["key"]:
        raise ActivityAnswerKeyError("key projection differs from its frozen hash")
    if verification.projection_sha256 != frozen["joint"]:
        raise ActivityAnswerKeyError("joint projection differs from its frozen hash")
    return verification
