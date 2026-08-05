"""Exact PDF projection for a numbered inline ``Cevap: A-E`` solution.

The projection deliberately covers the complete reviewed question/solution
column crop, not only the final answer glyph.  This prevents an index from
pairing a genuine inline key with fabricated routing text while preserving the
source-native page/question address.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any
import unicodedata

from .official_ogm import canonical_json_sha256


PROJECTION_SCHEMA = "pdf-inline-solution-content-projection-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_QUESTION_MARKER = re.compile(r"^([1-9]\d*)[.)]$")


class InlineSolutionError(ValueError):
    """An inline solution crop is not exactly bound to its pinned PDF."""


@dataclass(frozen=True, slots=True)
class InlineSolutionVerification:
    projection_sha256: str
    content_text: str
    marker_count: int


def _plain(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise InlineSolutionError(f"{label} must be literal Unicode text")
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in normalized):
        raise InlineSolutionError(f"{label} contains unsafe Unicode controls")
    return normalized.strip()


def normalized_crop_text(value: Any) -> str:
    """Canonicalize only PDF layout whitespace, preserving all source text."""

    if not isinstance(value, str):
        raise InlineSolutionError("inline content text must be literal Unicode text")
    return _plain(" ".join(value.split()), "inline content text")


def _bbox(value: Sequence[float]) -> tuple[float, float, float, float]:
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
        raise InlineSolutionError("inline content bbox is malformed")
    result = tuple(float(item) for item in value)
    if not (result[0] < result[2] and result[1] < result[3]):
        raise InlineSolutionError("inline content bbox is empty")
    return result  # type: ignore[return-value]


def _word_projection(word: Mapping[str, Any]) -> dict[str, Any]:
    try:
        text = _plain(word.get("text"), "inline PDF word")
        coordinates = tuple(float(word[key]) for key in ("x0", "top", "x1", "bottom"))
    except (KeyError, TypeError, ValueError) as exc:
        raise InlineSolutionError("inline PDF word is malformed") from exc
    if (
        not text
        or not all(math.isfinite(value) for value in coordinates)
        or not coordinates[0] < coordinates[2]
        or not coordinates[1] < coordinates[3]
    ):
        raise InlineSolutionError("inline PDF word is malformed")
    return {
        "text": text,
        "x0": round(coordinates[0], 3),
        "top": round(coordinates[1], 3),
        "x1": round(coordinates[2], 3),
        "bottom": round(coordinates[3], 3),
    }


def _inside(word: Mapping[str, Any], bbox: tuple[float, float, float, float]) -> bool:
    try:
        center_x = (float(word["x0"]) + float(word["x1"])) / 2.0
        center_y = (float(word["top"]) + float(word["bottom"])) / 2.0
    except (KeyError, TypeError, ValueError) as exc:
        raise InlineSolutionError("inline PDF word is malformed") from exc
    return bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]


def attest_inline_solution_content(
    page: Any,
    *,
    pdf_sha256: str,
    physical_page: int,
    bbox: Sequence[float],
    question_number: int,
    question_text: str,
) -> InlineSolutionVerification:
    """Attest one complete inline question/solution crop from PDF geometry."""

    if _HEX64.fullmatch(pdf_sha256) is None:
        raise InlineSolutionError("inline PDF hash is malformed")
    if (
        not isinstance(physical_page, int)
        or isinstance(physical_page, bool)
        or physical_page < 1
        or not isinstance(question_number, int)
        or isinstance(question_number, bool)
        or question_number < 1
    ):
        raise InlineSolutionError("inline source address is malformed")
    content_bbox = _bbox(bbox)
    try:
        width = float(page.width)
        height = float(page.height)
    except (AttributeError, TypeError, ValueError) as exc:
        raise InlineSolutionError("inline PDF page has no finite geometry") from exc
    if (
        not all(math.isfinite(value) and value > 0 for value in (width, height))
        or not 0.0 <= content_bbox[0] < content_bbox[2] <= width
        or not 0.0 <= content_bbox[1] < content_bbox[3] <= height
    ):
        raise InlineSolutionError("inline content bbox is outside the PDF page")
    try:
        raw_words = page.extract_words() or []
        raw_crop_text = page.crop(content_bbox).extract_text() or ""
    except (AttributeError, TypeError, ValueError) as exc:
        raise InlineSolutionError("inline PDF content cannot be extracted") from exc
    if not isinstance(raw_words, list) or any(
        not isinstance(word, Mapping) for word in raw_words
    ):
        raise InlineSolutionError("inline PDF word inventory is malformed")
    words = sorted(
        (_word_projection(word) for word in raw_words if _inside(word, content_bbox)),
        key=lambda word: (word["top"], word["x0"], word["text"]),
    )
    content_text = normalized_crop_text(raw_crop_text)
    expected_text = normalized_crop_text(question_text)
    if not content_text or content_text.casefold() != expected_text.casefold():
        raise InlineSolutionError("inline content crop differs from its indexed source text")
    marker_count = sum(
        1
        for word in words
        if (
            (match := _QUESTION_MARKER.fullmatch(str(word["text"]))) is not None
            and int(match.group(1)) == question_number
        )
    )
    if marker_count != 1:
        raise InlineSolutionError("inline content crop lacks one unique question marker")
    projection = {
        "projection_schema": PROJECTION_SCHEMA,
        "pdf_sha256": pdf_sha256,
        "physical_page": physical_page,
        "page_size": [round(width, 3), round(height, 3)],
        "content_bbox": [round(value, 3) for value in content_bbox],
        "question_number": question_number,
        "question_text": content_text,
        "marker_count": marker_count,
        "words": words,
    }
    return InlineSolutionVerification(
        projection_sha256=canonical_json_sha256(projection),
        content_text=content_text,
        marker_count=marker_count,
    )


def verify_inline_solution_content(
    page: Any,
    *,
    expected_projection_sha256: str,
    **kwargs: Any,
) -> InlineSolutionVerification:
    """Replay an inline content attestation and require its frozen pin."""

    if _HEX64.fullmatch(expected_projection_sha256) is None:
        raise InlineSolutionError("inline content projection hash is malformed")
    verification = attest_inline_solution_content(page, **kwargs)
    if verification.projection_sha256 != expected_projection_sha256:
        raise InlineSolutionError("inline content projection differs from source index")
    return verification
