"""Deterministic attestation for header-over-answer PDF table cells.

Some official workbooks publish short-text answers in a grid: a numbered
header cell sits directly above its answer cell.  Plain text extraction loses
that relationship.  This module proves the relationship from PDF geometry,
checks the surrounding source section, and pins the complete projection.  It
has no task IDs, benchmark inputs, candidates, or outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

from .official_ogm import canonical_json_sha256, normalize_tokens


PROJECTION_SCHEMA = "pdf-coordinate-table-cell-projection-v2"
CONTENT_PROJECTION_SCHEMA = "pdf-content-question-marker-projection-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LABEL = re.compile(
    r"(?<![\w(])([A-Za-zÇĞİÖŞÜçğıöşü])\s*(?:=|\))\s*"
)


class CoordinateTableAnswerKeyError(ValueError):
    """A table cell could not prove the indexed source answer."""


@dataclass(frozen=True, slots=True)
class CoordinateTableAnswerKeyVerification:
    projection_sha256: str
    derived_answer: Mapping[str, str] | str
    component_matches: Mapping[str, bool]


@dataclass(frozen=True, slots=True)
class ContentQuestionMarkerVerification:
    projection_sha256: str
    marker_count: int


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CoordinateTableAnswerKeyError(f"{label} must be a positive integer")
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
        raise CoordinateTableAnswerKeyError(f"{label} is malformed")
    result = tuple(float(item) for item in value)
    if not (result[0] < result[2] and result[1] < result[3]):
        raise CoordinateTableAnswerKeyError(f"{label} is empty")
    return result  # type: ignore[return-value]


def _normalized_text(value: str) -> str:
    if not isinstance(value, str) or any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in value
    ):
        raise CoordinateTableAnswerKeyError(
            "table answer must be plain Unicode text without controls"
        )
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
    return _normalized_text(value).casefold()


def _labelled_parts(value: str) -> dict[str, str] | None:
    normalized = _normalized_text(value)
    matches = list(_LABEL.finditer(normalized))
    if not matches:
        return None
    if matches[0].start() != 0:
        raise CoordinateTableAnswerKeyError(
            "table answer has text before its first component label"
        )
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        label = _normalized_label(match.group(1))
        if label in result:
            raise CoordinateTableAnswerKeyError("table answer repeats a component label")
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(normalized)
        )
        raw_component = normalized[match.end() : end]
        if index + 1 < len(matches):
            separated = re.fullmatch(
                r"(?s)(.+?)(?:\s*;\s*|\s*,\s*|\s+)", raw_component
            )
            if separated is None:
                raise CoordinateTableAnswerKeyError(
                    "multipart table answer lacks one explicit component delimiter"
                )
            component = separated.group(1).strip()
        else:
            component = raw_component.strip()
            if re.search(r"[;,/]\Z", component):
                raise CoordinateTableAnswerKeyError(
                    "multipart table answer has a trailing delimiter"
                )
        if not component:
            raise CoordinateTableAnswerKeyError("table answer has an empty component")
        result[label] = component
    if len(result) < 2:
        raise CoordinateTableAnswerKeyError(
            "multipart table answer requires at least two labelled components"
        )
    return result


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


def _ordered_words(
    page: Any, bbox: tuple[float, float, float, float]
) -> list[Mapping[str, Any]]:
    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=False,
    ) or []
    return sorted(
        (word for word in words if _inside_bbox(word, bbox)),
        key=lambda word: (
            round(float(word["top"]), 1),
            float(word["x0"]),
            str(word.get("text") or ""),
        ),
    )


def _words_text(words: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(
        _normalized_text(str(word.get("text") or ""))
        for word in words
        if _normalized_text(str(word.get("text") or ""))
    )


def _edge_coordinate(edge: Mapping[str, Any], orientation: str) -> float:
    if orientation == "h":
        return (float(edge["top"]) + float(edge["bottom"])) / 2.0
    return (float(edge["x0"]) + float(edge["x1"])) / 2.0


def _edge_interval(edge: Mapping[str, Any], orientation: str) -> tuple[float, float]:
    if orientation == "h":
        return float(edge["x0"]), float(edge["x1"])
    return float(edge["top"]), float(edge["bottom"])


def _coverage_at(
    edges: Sequence[Mapping[str, Any]],
    *,
    orientation: str,
    coordinate: float,
    start: float,
    end: float,
    tolerance: float = 0.8,
) -> bool:
    intervals = sorted(
        _edge_interval(edge, orientation)
        for edge in edges
        if str(edge.get("orientation") or "") == orientation
        and abs(_edge_coordinate(edge, orientation) - coordinate) <= tolerance
    )
    if not intervals:
        return False
    cursor = start
    for left, right in intervals:
        if right < cursor - tolerance:
            continue
        if left > cursor + tolerance:
            return False
        cursor = max(cursor, right)
        if cursor >= end - tolerance:
            return True
    return cursor >= end - tolerance


def _horizontal_positions(
    edges: Sequence[Mapping[str, Any]],
    bbox: tuple[float, float, float, float],
) -> list[float]:
    raw = sorted(
        _edge_coordinate(edge, "h")
        for edge in edges
        if str(edge.get("orientation") or "") == "h"
        and bbox[1] - 0.8 <= _edge_coordinate(edge, "h") <= bbox[3] + 0.8
    )
    clustered: list[float] = []
    for value in raw:
        if clustered and abs(clustered[-1] - value) <= 0.8:
            clustered[-1] = (clustered[-1] + value) / 2.0
        else:
            clustered.append(value)
    return [
        value
        for value in clustered
        if _coverage_at(
            edges,
            orientation="h",
            coordinate=value,
            start=bbox[0],
            end=bbox[2],
        )
    ]


def _table_split(
    page: Any, bbox: tuple[float, float, float, float]
) -> tuple[float, list[Mapping[str, Any]]]:
    if not (
        0.0 <= bbox[0] < bbox[2] <= float(page.width)
        and 0.0 <= bbox[1] < bbox[3] <= float(page.height)
    ):
        raise CoordinateTableAnswerKeyError("table cell bbox is outside the PDF page")
    edges = list(page.edges or [])
    if not _coverage_at(
        edges, orientation="v", coordinate=bbox[0], start=bbox[1], end=bbox[3]
    ) or not _coverage_at(
        edges, orientation="v", coordinate=bbox[2], start=bbox[1], end=bbox[3]
    ):
        raise CoordinateTableAnswerKeyError("table cell lacks vertical boundary proof")
    positions = _horizontal_positions(edges, bbox)
    if not any(abs(value - bbox[1]) <= 0.8 for value in positions) or not any(
        abs(value - bbox[3]) <= 0.8 for value in positions
    ):
        raise CoordinateTableAnswerKeyError("table cell lacks outer horizontal boundary proof")
    internal = [
        value for value in positions if bbox[1] + 1.0 < value < bbox[3] - 1.0
    ]
    if len(internal) != 1:
        raise CoordinateTableAnswerKeyError(
            "table cell requires exactly one header-answer divider"
        )
    return internal[0], edges


def _contains_once(values: Sequence[str], target: Sequence[str]) -> bool:
    if not target or len(target) > len(values):
        return False
    count = sum(
        tuple(values[index : index + len(target)]) == tuple(target)
        for index in range(len(values) - len(target) + 1)
    )
    return count == 1


def _tokenized_page_words(page: Any) -> list[tuple[str, float, float]]:
    projected: list[tuple[str, float, float]] = []
    words = sorted(
        page.extract_words(
            x_tolerance=2,
            y_tolerance=2,
            keep_blank_chars=False,
            use_text_flow=False,
        )
        or [],
        key=lambda word: (
            round(float(word["top"]), 1),
            float(word["x0"]),
            str(word.get("text") or ""),
        ),
    )
    for word in words:
        for token in normalize_tokens(str(word.get("text") or "")):
            projected.append((token, float(word["top"]), float(word["x0"])))
    return projected


def _table_bbox(table: Any) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in table.bbox)  # type: ignore[return-value]


def _table_rows(table: Any) -> list[dict[str, Any]]:
    extracted = table.extract()
    rows = list(table.rows)
    if len(extracted) != len(rows):
        raise CoordinateTableAnswerKeyError("PDF table row extraction is inconsistent")
    result: list[dict[str, Any]] = []
    for row, values in zip(rows, extracted, strict=True):
        bbox = tuple(float(value) for value in row.bbox)
        result.append(
            {
                "bbox": bbox,
                "values": [None if value is None else str(value).strip() for value in values],
            }
        )
    return result


def _row_tokens(row: Mapping[str, Any]) -> tuple[str, ...]:
    return normalize_tokens(
        " ".join(value for value in row["values"] if isinstance(value, str) and value)
    )


def _numeric_header(row: Mapping[str, Any]) -> tuple[int, ...] | None:
    raw_values = [
        value for value in row["values"] if isinstance(value, str) and value.strip()
    ]
    if len(raw_values) < 2 or any(
        re.fullmatch(r"[1-9]\d{0,2}", value) is None for value in raw_values
    ):
        return None
    values = tuple(int(value) for value in raw_values)
    if values != tuple(range(values[0], values[-1] + 1)):
        return None
    return values


def _table_contains_bbox(
    table: Any,
    bbox: tuple[float, float, float, float],
    *,
    tolerance: float = 0.8,
) -> bool:
    left, top, right, bottom = _table_bbox(table)
    return (
        left - tolerance <= bbox[0]
        and top - tolerance <= bbox[1]
        and right + tolerance >= bbox[2]
        and bottom + tolerance >= bbox[3]
    )


def _table_projection(table: Any) -> dict[str, Any]:
    cells = sorted(
        {
            tuple(round(float(value), 3) for value in cell)
            for cell in table.cells
            if cell is not None
        }
    )
    return {
        "bbox": [round(value, 3) for value in _table_bbox(table)],
        "cells": [list(cell) for cell in cells],
        "rows": [
            {
                "bbox": [round(value, 3) for value in row["bbox"]],
                "values": row["values"],
            }
            for row in _table_rows(table)
        ],
    }


def _page_tables(page: Any) -> list[Any]:
    try:
        tables = list(page.find_tables())
    except (AttributeError, TypeError, ValueError) as exc:
        raise CoordinateTableAnswerKeyError(
            "PDF table components cannot be derived"
        ) from exc
    if not tables:
        raise CoordinateTableAnswerKeyError("PDF page exposes no table components")
    return tables


def _unit_heading_rows(tables: Sequence[Any]) -> list[tuple[Any, dict[str, Any]]]:
    result: list[tuple[Any, dict[str, Any]]] = []
    for table in tables:
        for row in _table_rows(table):
            tokens = _row_tokens(row)
            if "unite" in tokens[:2]:
                result.append((table, row))
    return result


def _strict_section_context(
    page: Any,
    context_page: Any,
    *,
    physical_page: int,
    context_number: int,
    bbox: tuple[float, float, float, float],
    question_number: int,
    expected_section: str,
    expected_test_variant: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str, Mapping[str, Any]]:
    section_tokens = normalize_tokens(expected_section)
    variant_tokens = normalize_tokens(expected_test_variant)
    if (
        len(section_tokens) < 4
        or len(variant_tokens) < 1
        or section_tokens[0] != "unite"
        or not section_tokens[1].isdigit()
        or tuple(section_tokens[-len(variant_tokens) :]) != tuple(variant_tokens)
    ):
        raise CoordinateTableAnswerKeyError(
            "coordinate table section descriptor is not a complete unit heading"
        )
    if context_number not in {physical_page, physical_page - 1}:
        raise CoordinateTableAnswerKeyError(
            "coordinate table section page is not the cell page or its predecessor"
        )

    key_tables = _page_tables(page)
    target_tables = [
        table for table in key_tables if _table_contains_bbox(table, bbox)
    ]
    if len(target_tables) != 1:
        raise CoordinateTableAnswerKeyError(
            "coordinate table answer cell has no unique connected grid component"
        )
    target_table = target_tables[0]
    if context_number == physical_page:
        heading_rows = [
            (table, row)
            for table, row in _unit_heading_rows(key_tables)
            if float(row["bbox"][3]) <= bbox[1] + 0.8
        ]
        exact_rows = [
            (table, row)
            for table, row in heading_rows
            if _row_tokens(row) == section_tokens
        ]
        if len(exact_rows) != 1 or exact_rows[0][0] is not target_table:
            raise CoordinateTableAnswerKeyError(
                "coordinate table section heading is not unique in the target grid"
            )
        nearest = max(heading_rows, key=lambda item: float(item[1]["bbox"][1]))
        if nearest != exact_rows[0]:
            raise CoordinateTableAnswerKeyError(
                "coordinate table section is not the nearest heading in its grid"
            )
        target_headers = [
            header
            for row in _table_rows(target_table)
            if float(row["bbox"][1]) >= float(exact_rows[0][1]["bbox"][3]) - 0.8
            and float(row["bbox"][3]) <= bbox[3] + 0.8
            and (header := _numeric_header(row)) is not None
        ]
        if sum(question_number in header for header in target_headers) != 1:
            raise CoordinateTableAnswerKeyError(
                "coordinate table target header is absent or ambiguous in its section grid"
            )
        relation = "same_page_nearest_preceding_unit_heading"
        table_context = {
            "target_table": _table_projection(target_table),
            "section_heading_bbox": [
                round(value, 3) for value in exact_rows[0][1]["bbox"]
            ],
        }
    else:
        if bbox[1] > float(page.height) * 0.20:
            raise CoordinateTableAnswerKeyError(
                "continued coordinate table cell is not near the next-page top"
            )
        context_tables = _page_tables(context_page)
        context_headings = _unit_heading_rows(context_tables)
        exact_rows = [
            (table, row)
            for table, row in context_headings
            if _row_tokens(row) == section_tokens
        ]
        if len(exact_rows) != 1:
            raise CoordinateTableAnswerKeyError(
                "continued coordinate table section heading is absent or ambiguous"
            )
        context_table, heading_row = exact_rows[0]
        if max(
            context_headings,
            key=lambda item: float(item[1]["bbox"][1]),
        ) != exact_rows[0]:
            raise CoordinateTableAnswerKeyError(
                "continued coordinate table section is not the final prior-page heading"
            )
        current_headings = _unit_heading_rows(key_tables)
        if any(float(row["bbox"][1]) < bbox[3] for _, row in current_headings):
            raise CoordinateTableAnswerKeyError(
                "continued coordinate table has a nearer unit heading on the cell page"
            )
        previous_bbox = _table_bbox(context_table)
        current_bbox = _table_bbox(target_table)
        if (
            previous_bbox[3] < float(context_page.height) * 0.88
            or current_bbox[1] > float(page.height) * 0.10
            or abs(previous_bbox[0] - current_bbox[0]) > 0.75
            or abs(previous_bbox[2] - current_bbox[2]) > 0.75
        ):
            raise CoordinateTableAnswerKeyError(
                "continued coordinate table grids are not page-edge adjacent and aligned"
            )
        previous_headers = [
            header
            for row in _table_rows(context_table)
            if float(row["bbox"][1]) >= float(heading_row["bbox"][3]) - 0.8
            and (header := _numeric_header(row)) is not None
        ]
        next_heading_top = min(
            (
                float(row["bbox"][1])
                for _, row in current_headings
                if float(row["bbox"][1]) > bbox[3]
            ),
            default=current_bbox[3] + 0.8,
        )
        current_headers = [
            header
            for row in _table_rows(target_table)
            if float(row["bbox"][1]) < next_heading_top
            and (header := _numeric_header(row)) is not None
        ]
        if (
            not previous_headers
            or not current_headers
            or current_headers[0][0] != previous_headers[-1][-1] + 1
            or sum(question_number in header for header in current_headers) != 1
        ):
            raise CoordinateTableAnswerKeyError(
                "continued coordinate table does not prove a sequential header run"
            )
        relation = "previous_page_nearest_unit_heading_top_continuation"
        table_context = {
            "previous_table": _table_projection(context_table),
            "continuation_table": _table_projection(target_table),
            "section_heading_bbox": [round(value, 3) for value in heading_row["bbox"]],
            "previous_terminal_header": previous_headers[-1][-1],
            "continuation_initial_header": current_headers[0][0],
            "continuation_final_header": current_headers[-1][-1],
            "next_section_heading_top": round(next_heading_top, 3),
        }
    return section_tokens, variant_tokens, relation, table_context


def _derive_answer(
    actual_text: str, expected_answer: str
) -> tuple[Mapping[str, str] | str, Mapping[str, bool]]:
    actual_parts = _labelled_parts(actual_text)
    expected_parts = _labelled_parts(expected_answer)
    if (actual_parts is None) != (expected_parts is None):
        raise CoordinateTableAnswerKeyError(
            "source and indexed answer structures differ"
        )
    if actual_parts is None:
        return actual_text, {
            "scalar": _canonical_expression(actual_text)
            == _canonical_expression(expected_answer)
        }
    assert expected_parts is not None
    if tuple(actual_parts) != tuple(expected_parts):
        raise CoordinateTableAnswerKeyError(
            "source and indexed answer labels differ or are reordered"
        )
    return actual_parts, {
        label: _canonical_expression(actual_parts[label])
        == _canonical_expression(expected_parts[label])
        for label in expected_parts
    }


def _projection_word(word: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": _normalized_text(str(word.get("text") or "")),
        "x0": round(float(word["x0"]), 3),
        "top": round(float(word["top"]), 3),
        "x1": round(float(word["x1"]), 3),
        "bottom": round(float(word["bottom"]), 3),
    }


def _content_projection_word(word: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": str(word.get("text") or "").strip(),
        "x0": round(float(word["x0"]), 3),
        "top": round(float(word["top"]), 3),
        "x1": round(float(word["x1"]), 3),
        "bottom": round(float(word["bottom"]), 3),
    }


def attest_content_question_marker(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    bbox: tuple[float, float, float, float],
    question_number: int,
    marker_kind: str,
    question_text: str,
) -> ContentQuestionMarkerVerification:
    """Pin one printed question marker and its source-question crop."""

    if not isinstance(pdf_sha256, str) or _HEX64.fullmatch(pdf_sha256) is None:
        raise CoordinateTableAnswerKeyError("content-marker PDF hash is malformed")
    physical_page = _positive_integer(physical_page, "content physical page")
    question_number = _positive_integer(question_number, "content question number")
    bbox = _strict_bbox(bbox, "content bbox")
    if not isinstance(question_text, str) or not _normalized_text(question_text):
        raise CoordinateTableAnswerKeyError("content-marker source address is malformed")
    if not (
        0.0 <= bbox[0] < bbox[2] <= float(page.width)
        and 0.0 <= bbox[1] < bbox[3] <= float(page.height)
    ):
        raise CoordinateTableAnswerKeyError("content bbox is outside the PDF page")
    words = sorted(
        (
            word
            for word in (page.extract_words(use_text_flow=False) or [])
            if _inside_bbox(word, bbox)
        ),
        key=lambda word: (
            round(float(word["top"]), 1),
            float(word["x0"]),
            str(word.get("text") or ""),
        ),
    )
    markers: list[dict[str, Any]] = []
    if marker_kind == "numbered_item":
        markers = [
            _content_projection_word(word)
            for word in words
            if str(word.get("text") or "").strip() == f"{question_number}."
        ]
    elif marker_kind == "example_label":
        for label in words:
            if str(label.get("text") or "").strip().casefold() != "\u00f6rnek":
                continue
            for number in words:
                if str(number.get("text") or "").strip() != str(question_number):
                    continue
                same_line = abs(float(label["top"]) - float(number["top"])) <= 2.5
                gap = float(number["x0"]) - float(label["x1"])
                if same_line and 0.0 <= gap <= 10.0:
                    markers.append(
                        {
                            "label": _content_projection_word(label),
                            "number": _content_projection_word(number),
                        }
                    )
    else:
        raise CoordinateTableAnswerKeyError("content marker kind is unsupported")
    if len(markers) != 1:
        raise CoordinateTableAnswerKeyError(
            "printed source question marker is absent or ambiguous"
        )
    crop_text = page.crop(bbox).extract_text() or ""
    crop_tokens = normalize_tokens(crop_text)
    question_tokens = normalize_tokens(question_text)
    if not _contains_once(crop_tokens, question_tokens):
        raise CoordinateTableAnswerKeyError(
            "content crop does not expose one exact indexed source question"
        )
    projection = {
        "projection_schema": CONTENT_PROJECTION_SCHEMA,
        "pdf_sha256": pdf_sha256,
        "physical_page": physical_page,
        "page_size": [round(float(page.width), 3), round(float(page.height), 3)],
        "bbox": [round(item, 3) for item in bbox],
        "question_number": question_number,
        "marker_kind": marker_kind,
        "marker": markers[0],
        "question_tokens": list(question_tokens),
        "crop_tokens_sha256": canonical_json_sha256({"tokens": list(crop_tokens)}),
        "words": [_content_projection_word(word) for word in words],
    }
    return ContentQuestionMarkerVerification(
        projection_sha256=canonical_json_sha256(projection),
        marker_count=1,
    )


def verify_content_question_marker(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    bbox: tuple[float, float, float, float],
    question_number: int,
    marker_kind: str,
    question_text: str,
    expected_projection_sha256: str,
) -> ContentQuestionMarkerVerification:
    if (
        not isinstance(expected_projection_sha256, str)
        or _HEX64.fullmatch(expected_projection_sha256) is None
    ):
        raise CoordinateTableAnswerKeyError(
            "content-marker projection hash is malformed"
        )
    verification = attest_content_question_marker(
        page,
        pdf_sha256=pdf_sha256,
        physical_page=physical_page,
        bbox=bbox,
        question_number=question_number,
        marker_kind=marker_kind,
        question_text=question_text,
    )
    if verification.projection_sha256 != expected_projection_sha256:
        raise CoordinateTableAnswerKeyError(
            "content-marker projection differs from source index"
        )
    return verification


def attest_coordinate_table_answer_key(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    bbox: tuple[float, float, float, float],
    question_number: int,
    expected_answer: str,
    expected_section: str,
    expected_test_variant: str,
    section_page: Any | None = None,
    section_physical_page: int | None = None,
) -> CoordinateTableAnswerKeyVerification:
    """Derive one source answer from a numbered header-over-answer cell."""

    if not isinstance(pdf_sha256, str) or _HEX64.fullmatch(pdf_sha256) is None:
        raise CoordinateTableAnswerKeyError("coordinate table PDF hash is malformed")
    physical_page = _positive_integer(physical_page, "coordinate physical page")
    question_number = _positive_integer(question_number, "coordinate question number")
    bbox = _strict_bbox(bbox, "coordinate table bbox")
    if (
        not isinstance(expected_answer, str)
        or not isinstance(expected_section, str)
        or not isinstance(expected_test_variant, str)
        or not _normalized_text(expected_answer)
        or not _normalized_text(expected_section)
        or not _normalized_text(expected_test_variant)
    ):
        raise CoordinateTableAnswerKeyError("coordinate table expectation is empty")
    context_page = section_page if section_page is not None else page
    context_number = (
        section_physical_page
        if section_physical_page is not None
        else physical_page
    )
    context_number = _positive_integer(context_number, "section physical page")
    context_tokens = normalize_tokens(context_page.extract_text() or "")
    section_tokens, variant_tokens, context_relation, table_context = (
        _strict_section_context(
        page,
        context_page,
        physical_page=physical_page,
        context_number=context_number,
        bbox=bbox,
        question_number=question_number,
        expected_section=expected_section,
        expected_test_variant=expected_test_variant,
        )
    )

    split_y, edges = _table_split(page, bbox)
    words = _ordered_words(page, bbox)
    header_words = [word for word in words if _word_center(word)[1] < split_y]
    answer_words = [word for word in words if _word_center(word)[1] > split_y]
    if _canonical_expression(_words_text(header_words)) != str(question_number):
        raise CoordinateTableAnswerKeyError(
            "coordinate table header does not expose the indexed question number"
        )
    actual_text = _words_text(answer_words)
    if not actual_text:
        raise CoordinateTableAnswerKeyError("coordinate table answer cell is empty")
    derived_answer, matches = _derive_answer(actual_text, expected_answer)
    if not matches or not all(matches.values()):
        raise CoordinateTableAnswerKeyError(
            "coordinate table cell does not expose the indexed answer"
        )

    relevant_edges = sorted(
        (
            {
                "orientation": str(edge.get("orientation") or ""),
                "x0": round(float(edge["x0"]), 3),
                "top": round(float(edge["top"]), 3),
                "x1": round(float(edge["x1"]), 3),
                "bottom": round(float(edge["bottom"]), 3),
            }
            for edge in edges
            if (
                bbox[0] - 0.8 <= float(edge["x0"]) <= bbox[2] + 0.8
                or bbox[0] - 0.8 <= float(edge["x1"]) <= bbox[2] + 0.8
            )
            and (
                bbox[1] - 0.8 <= float(edge["top"]) <= bbox[3] + 0.8
                or bbox[1] - 0.8 <= float(edge["bottom"]) <= bbox[3] + 0.8
            )
        ),
        key=lambda edge: (
            edge["orientation"],
            edge["top"],
            edge["x0"],
            edge["bottom"],
            edge["x1"],
        ),
    )
    projection = {
        "projection_schema": PROJECTION_SCHEMA,
        "pdf_sha256": pdf_sha256,
        "physical_page": physical_page,
        "section_physical_page": context_number,
        "page_size": [round(float(page.width), 3), round(float(page.height), 3)],
        "bbox": [round(value, 3) for value in bbox],
        "split_y": round(split_y, 3),
        "question_number": question_number,
        "header_words": [_projection_word(word) for word in header_words],
        "answer_words": [_projection_word(word) for word in answer_words],
        "answer_binding": (
            {
                "shape": "multipart",
                "ordered_labels": list(derived_answer),
                "canonical_components": {
                    label: _canonical_expression(value)
                    for label, value in derived_answer.items()
                },
            }
            if isinstance(derived_answer, Mapping)
            else {
                "shape": "scalar",
                "canonical_value": _canonical_expression(derived_answer),
            }
        ),
        "table_edges": relevant_edges,
        "section_tokens": list(section_tokens),
        "test_variant_tokens": list(variant_tokens),
        "section_context_relation": context_relation,
        "table_context": table_context,
        "section_page_tokens_sha256": canonical_json_sha256(
            {"tokens": list(context_tokens)}
        ),
    }
    return CoordinateTableAnswerKeyVerification(
        projection_sha256=canonical_json_sha256(projection),
        derived_answer=derived_answer,
        component_matches=matches,
    )


def verify_coordinate_table_answer_key(
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
    section_page: Any | None = None,
    section_physical_page: int | None = None,
) -> CoordinateTableAnswerKeyVerification:
    """Prove one source cell and its frozen coordinate projection."""

    if (
        not isinstance(expected_projection_sha256, str)
        or _HEX64.fullmatch(expected_projection_sha256) is None
    ):
        raise CoordinateTableAnswerKeyError(
            "coordinate table projection hash is malformed"
        )
    verification = attest_coordinate_table_answer_key(
        page,
        pdf_sha256=pdf_sha256,
        physical_page=physical_page,
        bbox=bbox,
        question_number=question_number,
        expected_answer=expected_answer,
        expected_section=expected_section,
        expected_test_variant=expected_test_variant,
        section_page=section_page,
        section_physical_page=section_physical_page,
    )
    if verification.projection_sha256 != expected_projection_sha256:
        raise CoordinateTableAnswerKeyError(
            "coordinate table projection differs from source index"
        )
    return verification
