"""Source-only Math12 activity inventory and fail-closed visual resolver.

The adapter is deliberately independent from benchmark rows and answers.  It
parses one pinned official workbook, binds an arbitrary image to *all* content
pages with SIFT/RANSAC, and returns an activity/key address only after the
existing visual safety floor passes.  It never derives a correctness label or
an answer value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .official_ogm import canonical_json_bytes, canonical_json_sha256, sha256_file
from .visual_coordinate_binding import (
    SiftRuntimeProfile,
    VisualBindingThresholds,
    VisualCoordinateBindingError,
    VisualPageEvidence,
    compute_sift_page_evidence,
    geometry_checks,
    visual_page_evidence_from_mapping,
)


INVENTORY_SCHEMA = "math12-official-activity-inventory-v1"
CERTIFICATE_SCHEMA = "math12-source-binding-certificate-v1"
RENDER_MANIFEST_SCHEMA = "math12-poppler-content-render-manifest-v1"
DOCUMENT_PREFIX = "meb_math12"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ACTIVITY_COUNT = 95

# This adapter is intentionally narrower than the generic visual-binding
# helpers.  A Math12 certificate is meaningful only under this exact profile;
# accepting a "stronger" caller override would still create a different,
# post-freeze policy.
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_PDFPLUMBER_VERSION = "0.11.9"
EXPECTED_NUMPY_VERSION = "2.5.1"
EXPECTED_OPENCV_VERSION = "5.0.0"
EXPECTED_POPPLER_VERSION = "26.05.0"
FROZEN_VISUAL_THRESHOLDS = VisualBindingThresholds()
FROZEN_SIFT_RUNTIME_PROFILE = SiftRuntimeProfile(
    render_dpi=144,
    nfeatures=12_000,
    contrast_threshold=0.02,
    edge_threshold=12.0,
    ratio_test=0.72,
    ransac_reprojection_px=4.0,
    ransac_max_iters=5_000,
    ransac_confidence=0.999,
    rng_seed=19_870_511,
    expected_opencv_version=EXPECTED_OPENCV_VERSION,
)


class Math12SourceError(ValueError):
    """The official source or its visual evidence is not certifiable."""


def assert_math12_runtime(
    *, require_pdfplumber: bool = False, require_visual: bool = False
) -> dict[str, str]:
    """Fail closed unless the process matches the frozen Math12 runtime.

    Parsing already-written JSON does not need this preflight.  Building or
    extracting PDF text pins pdfplumber, while computing visual evidence pins
    NumPy and OpenCV.  Python is pinned for every executable Math12 operation.
    """

    observed: dict[str, str] = {
        "python": ".".join(str(value) for value in sys.version_info[:3])
    }
    if observed["python"] != EXPECTED_PYTHON_VERSION:
        raise Math12SourceError(
            f"Python {observed['python']} differs from pinned "
            f"{EXPECTED_PYTHON_VERSION}"
        )
    if require_pdfplumber:
        try:
            import pdfplumber  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise Math12SourceError("pdfplumber is required by the Math12 runtime") from exc
        observed["pdfplumber"] = str(getattr(pdfplumber, "__version__", ""))
        if observed["pdfplumber"] != EXPECTED_PDFPLUMBER_VERSION:
            raise Math12SourceError(
                f"pdfplumber {observed['pdfplumber']} differs from pinned "
                f"{EXPECTED_PDFPLUMBER_VERSION}"
            )
    if require_visual:
        try:
            import cv2  # type: ignore
            import numpy  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependencies
            raise Math12SourceError("NumPy and OpenCV are required by the Math12 runtime") from exc
        observed["numpy"] = str(getattr(numpy, "__version__", ""))
        observed["opencv"] = str(getattr(cv2, "__version__", ""))
        if observed["numpy"] != EXPECTED_NUMPY_VERSION:
            raise Math12SourceError(
                f"NumPy {observed['numpy']} differs from pinned {EXPECTED_NUMPY_VERSION}"
            )
        if observed["opencv"] != EXPECTED_OPENCV_VERSION:
            raise Math12SourceError(
                f"OpenCV {observed['opencv']} differs from pinned {EXPECTED_OPENCV_VERSION}"
            )
    return observed


def _require_frozen_math12_profile(
    thresholds: VisualBindingThresholds,
    runtime_profile: SiftRuntimeProfile | None = None,
) -> None:
    if thresholds != FROZEN_VISUAL_THRESHOLDS:
        raise VisualCoordinateBindingError(
            "Math12 visual thresholds differ from the exact frozen profile"
        )
    if runtime_profile is not None and runtime_profile != FROZEN_SIFT_RUNTIME_PROFILE:
        raise VisualCoordinateBindingError(
            "Math12 SIFT runtime differs from the exact frozen profile"
        )


@dataclass(frozen=True, slots=True)
class KeyLocator:
    page_number: int
    column: str
    top: float

    def __post_init__(self) -> None:
        if self.page_number < 1 or self.column not in {"left", "right"}:
            raise Math12SourceError("malformed answer-key locator")
        if not math.isfinite(self.top) or self.top < 0.0:
            raise Math12SourceError("malformed answer-key top coordinate")


@dataclass(frozen=True, slots=True)
class Math12ActivityRecord:
    activity_number: int
    index_page_number: int
    index_column: str
    index_top: float
    content_page_start: int
    content_page_end: int
    key_page_start: int
    key_page_end: int
    key_start: KeyLocator
    key_end_exclusive: KeyLocator
    index_projection_sha256: str
    content_projection_sha256: str
    key_projection_sha256: str
    binding_projection_sha256: str

    def __post_init__(self) -> None:
        if not 1 <= self.activity_number <= _ACTIVITY_COUNT:
            raise Math12SourceError("activity number is outside 1..95")
        if self.index_page_number not in {2, 3} or self.index_column not in {
            "left",
            "right",
        }:
            raise Math12SourceError("malformed contents-table address")
        if not self.content_page_start <= self.content_page_end:
            raise Math12SourceError("empty activity content range")
        if not self.key_page_start <= self.key_page_end:
            raise Math12SourceError("empty answer-key page range")
        if self.key_start.page_number != self.key_page_start:
            raise Math12SourceError("answer-key start page disagrees with locator")
        for value in (
            self.index_projection_sha256,
            self.content_projection_sha256,
            self.key_projection_sha256,
            self.binding_projection_sha256,
        ):
            if _HEX64.fullmatch(value) is None:
                raise Math12SourceError("activity record lacks a SHA-256 pin")


@dataclass(frozen=True, slots=True)
class Math12Inventory:
    schema_version: str
    document_id: str
    pdf_basename: str
    pdf_sha256: str
    pdf_size_bytes: int
    page_count: int
    index_page_start: int
    index_page_end: int
    content_page_start: int
    content_page_end: int
    key_page_start: int
    key_page_end: int
    source_page_projection_sha256: tuple[tuple[int, str], ...]
    activities: tuple[Math12ActivityRecord, ...]
    inventory_projection_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != INVENTORY_SCHEMA:
            raise Math12SourceError("unknown Math12 inventory schema")
        if not self.document_id.startswith(f"{DOCUMENT_PREFIX}_"):
            raise Math12SourceError("Math12 document identity is malformed")
        if _HEX64.fullmatch(self.pdf_sha256) is None or self.pdf_size_bytes < 1:
            raise Math12SourceError("Math12 PDF pin is malformed")
        if _HEX64.fullmatch(self.inventory_projection_sha256) is None:
            raise Math12SourceError("inventory projection pin is malformed")
        if tuple(item.activity_number for item in self.activities) != tuple(
            range(1, _ACTIVITY_COUNT + 1)
        ):
            raise Math12SourceError("inventory must contain activities 1..95 exactly once")
        expected_pages = tuple(range(self.content_page_start, self.content_page_end + 1))
        covered: list[int] = []
        for item in self.activities:
            covered.extend(range(item.content_page_start, item.content_page_end + 1))
        if tuple(covered) != expected_pages:
            raise Math12SourceError("activity ranges do not partition all content pages")
        pins = dict(self.source_page_projection_sha256)
        if tuple(sorted(pins)) != tuple(
            range(self.content_page_start, self.key_page_end + 1)
        ) or any(_HEX64.fullmatch(value) is None for value in pins.values()):
            raise Math12SourceError("source page projection pins are incomplete")

    def activity_for_page(self, page_number: int) -> Math12ActivityRecord | None:
        matches = [
            item
            for item in self.activities
            if item.content_page_start <= page_number <= item.content_page_end
        ]
        return matches[0] if len(matches) == 1 else None

    def projection(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "pdf_basename": self.pdf_basename,
            "pdf_sha256": self.pdf_sha256,
            "pdf_size_bytes": self.pdf_size_bytes,
            "page_count": self.page_count,
            "index_page_start": self.index_page_start,
            "index_page_end": self.index_page_end,
            "content_page_start": self.content_page_start,
            "content_page_end": self.content_page_end,
            "key_page_start": self.key_page_start,
            "key_page_end": self.key_page_end,
            "source_page_projection_sha256": {
                str(page): pin for page, pin in self.source_page_projection_sha256
            },
            "activities": [_record_to_mapping(item) for item in self.activities],
        }

    def to_mapping(self) -> dict[str, Any]:
        result = self.projection()
        result["inventory_projection_sha256"] = self.inventory_projection_sha256
        return result


@dataclass(frozen=True, slots=True)
class Math12RenderedPage:
    page_number: int
    manifest_path: str
    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.page_number < 1 or _HEX64.fullmatch(self.sha256) is None:
            raise Math12SourceError("rendered page pin is malformed")
        expected_name = f"page-{self.page_number:03d}.png"
        if self.manifest_path != expected_name:
            raise Math12SourceError("rendered page manifest path is not portable")
        if self.size_bytes < 1:
            raise Math12SourceError("rendered page size pin is malformed")


@dataclass(frozen=True, slots=True)
class Math12RenderManifest:
    schema_version: str
    document_id: str
    pdf_sha256: str
    inventory_projection_sha256: str
    render_dpi: int
    color_mode: str
    poppler_version: str
    pages: tuple[Math12RenderedPage, ...]
    render_manifest_projection_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != RENDER_MANIFEST_SCHEMA:
            raise Math12SourceError("unknown render manifest schema")
        if self.render_dpi < 1 or self.color_mode != "gray_png":
            raise Math12SourceError("render profile is malformed")
        if _HEX64.fullmatch(self.pdf_sha256) is None or _HEX64.fullmatch(
            self.inventory_projection_sha256
        ) is None:
            raise Math12SourceError("render source pins are malformed")
        if _HEX64.fullmatch(self.render_manifest_projection_sha256) is None:
            raise Math12SourceError("render manifest projection pin is malformed")

    def page_map(self) -> dict[int, Math12RenderedPage]:
        return {item.page_number: item for item in self.pages}

    def projection(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "pdf_sha256": self.pdf_sha256,
            "inventory_projection_sha256": self.inventory_projection_sha256,
            "render_dpi": self.render_dpi,
            "color_mode": self.color_mode,
            "poppler_version": self.poppler_version,
            "pages": {
                str(item.page_number): {
                    "path": item.manifest_path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in sorted(self.pages, key=lambda value: value.page_number)
            },
        }


@dataclass(frozen=True, slots=True)
class Math12BindingDecision:
    accepted: bool
    reason: str
    checks: tuple[tuple[str, bool], ...]
    selected_content_page: int | None = None
    selected_activity_number: int | None = None
    key_page_start: int | None = None
    key_page_end: int | None = None
    key_start: KeyLocator | None = None
    key_end_exclusive: KeyLocator | None = None
    binding_projection_sha256: str | None = None
    best_rank_score: float = 0.0
    runner_rank_score: float = 0.0


@dataclass(frozen=True, slots=True)
class Math12SourceCertificate:
    schema_version: str
    document_id: str
    pdf_sha256: str
    inventory_projection_sha256: str
    render_manifest_projection_sha256: str
    task_image_sha256: str
    thresholds: VisualBindingThresholds
    runtime_profile: SiftRuntimeProfile
    evidences: tuple[VisualPageEvidence, ...]
    decision: Math12BindingDecision
    evidence_projection_sha256: str
    certificate_projection_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "component_scope": "source_binding_only_no_answer_no_correctness",
            "document_id": self.document_id,
            "pdf_sha256": self.pdf_sha256,
            "inventory_projection_sha256": self.inventory_projection_sha256,
            "render_manifest_projection_sha256": self.render_manifest_projection_sha256,
            "task_image_sha256": self.task_image_sha256,
            "thresholds": asdict(self.thresholds),
            "runtime_profile": asdict(self.runtime_profile),
            "evidences": [_evidence_to_mapping(item) for item in self.evidences],
            "decision": _decision_to_mapping(self.decision),
            "evidence_projection_sha256": self.evidence_projection_sha256,
        }
        result["certificate_projection_sha256"] = self.certificate_projection_sha256
        return result


@dataclass(frozen=True, slots=True)
class Math12OfficialSolutionRecord:
    """PDF-native solution text bound to one accepted visual certificate."""

    schema_version: str
    component_scope: str
    document_id: str
    pdf_sha256: str
    inventory_projection_sha256: str
    source_certificate_projection_sha256: str
    task_image_sha256: str
    selected_content_page: int
    activity_number: int
    key_page_start: int
    key_page_end: int
    key_start: KeyLocator
    key_end_exclusive: KeyLocator
    binding_projection_sha256: str
    key_projection_sha256: str
    official_solution_text: str
    official_solution_text_sha256: str
    answer_bound_certificate_projection_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def _rounded(value: Any) -> float:
    return round(float(value), 4)


def _floor4(value: Any) -> float:
    """Four-decimal lower bound used for half-open key-span markers."""

    return math.floor(float(value) * 10_000.0) / 10_000.0


def _word_projection(word: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": str(word.get("text") or ""),
        "x0": _rounded(word["x0"]),
        "x1": _rounded(word["x1"]),
        "top": _rounded(word["top"]),
        "bottom": _rounded(word["bottom"]),
    }


def _page_projection(page_number: int, page: Any) -> dict[str, Any]:
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
    return {
        "page_number": page_number,
        "width": _rounded(page.width),
        "height": _rounded(page.height),
        "words": [_word_projection(item) for item in words],
    }


@dataclass(frozen=True, slots=True)
class _IndexAnchor:
    activity_number: int
    page_number: int
    column: str
    top: float
    source_page: int
    projection_sha256: str


def _parse_index_anchors(document: Any) -> tuple[_IndexAnchor, ...]:
    anchors: list[_IndexAnchor] = []
    for physical_page in (2, 3):
        page = document.pages[physical_page - 1]
        width = float(page.width)
        words = page.extract_words(use_text_flow=False, keep_blank_chars=False) or []
        for column, number_band, page_band in (
            ("left", (0.07 * width, 0.20 * width), (0.40 * width, 0.50 * width)),
            ("right", (0.50 * width, 0.63 * width), (0.80 * width, 0.91 * width)),
        ):
            candidates = [
                item
                for item in words
                if str(item.get("text") or "").isdigit()
                and number_band[0] <= float(item["x0"]) <= number_band[1]
                and 1 <= int(item["text"]) <= _ACTIVITY_COUNT
            ]
            for number_word in candidates:
                row_pages = [
                    item
                    for item in words
                    if str(item.get("text") or "").isdigit()
                    and page_band[0] <= float(item["x0"]) <= page_band[1]
                    and abs(float(item["top"]) - float(number_word["top"])) <= 1.25
                ]
                if len(row_pages) != 1:
                    continue
                activity = int(number_word["text"])
                source_page = int(row_pages[0]["text"])
                anchors.append(
                    _IndexAnchor(
                        activity_number=activity,
                        page_number=physical_page,
                        column=column,
                        top=_rounded(number_word["top"]),
                        source_page=source_page,
                        projection_sha256=canonical_json_sha256(
                            {
                                "physical_page": physical_page,
                                "column": column,
                                "activity_word": _word_projection(number_word),
                                "content_start_word": _word_projection(row_pages[0]),
                            }
                        ),
                    )
                )
    by_number: dict[int, _IndexAnchor] = {}
    for item in anchors:
        if item.activity_number in by_number:
            raise Math12SourceError("duplicate activity row in contents table")
        by_number[item.activity_number] = item
    if tuple(sorted(by_number)) != tuple(range(1, _ACTIVITY_COUNT + 1)):
        raise Math12SourceError("contents table does not attest activities 1..95")
    ordered = tuple(by_number[number] for number in range(1, _ACTIVITY_COUNT + 1))
    starts = tuple(item.source_page for item in ordered)
    if starts[0] != 4 or any(right <= left for left, right in zip(starts, starts[1:])):
        raise Math12SourceError("contents-table content starts are not strictly increasing")
    return ordered


@dataclass(frozen=True, slots=True)
class _KeyMarker:
    activity_number: int
    locator: KeyLocator
    marker_projection_sha256: str


def _key_marker_locator(
    physical_page: int, page_width: float, marker_words: Sequence[Mapping[str, Any]]
) -> KeyLocator:
    if len(marker_words) != 3:
        raise Math12SourceError("answer-key marker must contain exactly three words")
    tops = [float(item["top"]) for item in marker_words]
    column = "left" if float(marker_words[0]["x0"]) < page_width / 2.0 else "right"
    return KeyLocator(physical_page, column, _floor4(min(tops)))


def _parse_key_markers(document: Any, start: int, end: int) -> tuple[_KeyMarker, ...]:
    markers: list[_KeyMarker] = []
    for physical_page in range(start, end + 1):
        page = document.pages[physical_page - 1]
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
        width = float(page.width)
        for index, first in enumerate(words[:-2]):
            second, third = words[index + 1], words[index + 2]
            if (
                str(first.get("text") or "") != "Etkinlik"
                or str(second.get("text") or "") != "No.:"
                or not str(third.get("text") or "").isdigit()
            ):
                continue
            number = int(third["text"])
            if not 1 <= number <= _ACTIVITY_COUNT:
                continue
            marker_words = (first, second, third)
            tops = [float(item["top"]) for item in marker_words]
            if max(tops) - min(tops) > 2.0:
                continue
            locator = _key_marker_locator(physical_page, width, marker_words)
            markers.append(
                _KeyMarker(
                    number,
                    # A start locator must precede every word in its own
                    # three-token header.  An end-exclusive locator must also
                    # precede every word in the next header.  Flooring the
                    # minimum (rather than rounding the first word) gives both
                    # invariants even when PDF glyph tops differ by fractions
                    # of a point.
                    locator,
                    canonical_json_sha256(
                        {
                            "physical_page": physical_page,
                            "column": locator.column,
                            "words": [
                                _word_projection(item) for item in marker_words
                            ],
                        }
                    ),
                )
            )
    unique: dict[int, _KeyMarker] = {}
    for item in markers:
        if item.activity_number in unique:
            raise Math12SourceError("duplicate official answer-key marker")
        unique[item.activity_number] = item
    if tuple(sorted(unique)) != tuple(range(1, _ACTIVITY_COUNT + 1)):
        raise Math12SourceError("official key does not attest activities 1..95")
    ordered = tuple(unique[number] for number in range(1, _ACTIVITY_COUNT + 1))
    logical = tuple(
        sorted(
            ordered,
            key=lambda item: (
                item.locator.page_number,
                0 if item.locator.column == "left" else 1,
                item.locator.top,
            ),
        )
    )
    if tuple(item.activity_number for item in logical) != tuple(
        range(1, _ACTIVITY_COUNT + 1)
    ):
        raise Math12SourceError("official key markers are not in logical 1..95 order")
    return ordered


def _locator_key(locator: KeyLocator) -> tuple[int, int, float]:
    return (
        locator.page_number,
        0 if locator.column == "left" else 1,
        locator.top,
    )


def _key_section_projection(
    document: Any,
    activity_number: int,
    start: KeyLocator,
    end: KeyLocator,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for physical_page in range(start.page_number, end.page_number + 1):
        page = document.pages[physical_page - 1]
        width = float(page.width)
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
        for word in words:
            center = (float(word["x0"]) + float(word["x1"])) / 2.0
            column = "left" if center < width / 2.0 else "right"
            logical = (physical_page, 0 if column == "left" else 1, float(word["top"]))
            if _locator_key(start) <= logical < _locator_key(end):
                selected.append(
                    {
                        "physical_page": physical_page,
                        "column": column,
                        **_word_projection(word),
                    }
                )
    if not selected:
        raise Math12SourceError("empty official answer-key section")
    return {
        "activity_number": activity_number,
        "start": asdict(start),
        "end_exclusive": asdict(end),
        "words": selected,
    }


def build_math12_inventory(pdf_path: Path) -> Math12Inventory:
    """Parse and hash the complete 95-activity family from one source PDF."""

    assert_math12_runtime(require_pdfplumber=True)
    import pdfplumber  # type: ignore
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise Math12SourceError("Math12 source PDF is missing")
    pdf_sha = sha256_file(pdf_path)
    with pdfplumber.open(pdf_path) as document:
        if len(document.pages) != 182:
            raise Math12SourceError("unexpected Math12 PDF page count")
        anchors = _parse_index_anchors(document)
        key_start, key_end = 131, 179
        markers = _parse_key_markers(document, key_start, key_end)
        page_pins = tuple(
            (
                page_number,
                canonical_json_sha256(
                    _page_projection(page_number, document.pages[page_number - 1])
                ),
            )
            for page_number in range(4, key_end + 1)
        )
        page_pin_map = dict(page_pins)
        activities: list[Math12ActivityRecord] = []
        sentinel = KeyLocator(key_end + 1, "left", 0.0)
        for offset, (anchor, marker) in enumerate(zip(anchors, markers)):
            content_end = (
                anchors[offset + 1].source_page - 1
                if offset + 1 < len(anchors)
                else key_start - 1
            )
            end_locator = markers[offset + 1].locator if offset + 1 < len(markers) else sentinel
            key_projection = _key_section_projection(
                document, anchor.activity_number, marker.locator, end_locator
            )
            content_projection_sha = canonical_json_sha256(
                {
                    "pdf_sha256": pdf_sha,
                    "activity_number": anchor.activity_number,
                    "content_page_start": anchor.source_page,
                    "content_page_end": content_end,
                    "page_projection_sha256": [
                        page_pin_map[page]
                        for page in range(anchor.source_page, content_end + 1)
                    ],
                }
            )
            key_projection_sha = canonical_json_sha256(key_projection)
            key_page_end = min(key_end, end_locator.page_number)
            binding_projection_sha = canonical_json_sha256(
                {
                    "pdf_sha256": pdf_sha,
                    "activity_number": anchor.activity_number,
                    "index_projection_sha256": anchor.projection_sha256,
                    "content_projection_sha256": content_projection_sha,
                    "key_projection_sha256": key_projection_sha,
                    "content_page_range": [anchor.source_page, content_end],
                    "key_start": asdict(marker.locator),
                    "key_end_exclusive": asdict(end_locator),
                }
            )
            activities.append(
                Math12ActivityRecord(
                    activity_number=anchor.activity_number,
                    index_page_number=anchor.page_number,
                    index_column=anchor.column,
                    index_top=anchor.top,
                    content_page_start=anchor.source_page,
                    content_page_end=content_end,
                    key_page_start=marker.locator.page_number,
                    key_page_end=key_page_end,
                    key_start=marker.locator,
                    key_end_exclusive=end_locator,
                    index_projection_sha256=anchor.projection_sha256,
                    content_projection_sha256=content_projection_sha,
                    key_projection_sha256=key_projection_sha,
                    binding_projection_sha256=binding_projection_sha,
                )
            )
    projection = {
        "schema_version": INVENTORY_SCHEMA,
        "document_id": f"{DOCUMENT_PREFIX}_{pdf_sha[:12]}",
        "pdf_basename": pdf_path.name,
        "pdf_sha256": pdf_sha,
        "pdf_size_bytes": pdf_path.stat().st_size,
        "page_count": 182,
        "index_page_start": 2,
        "index_page_end": 3,
        "content_page_start": 4,
        "content_page_end": 130,
        "key_page_start": key_start,
        "key_page_end": key_end,
        "source_page_projection_sha256": {str(page): pin for page, pin in page_pins},
        "activities": [_record_to_mapping(item) for item in activities],
    }
    return Math12Inventory(
        schema_version=INVENTORY_SCHEMA,
        document_id=projection["document_id"],
        pdf_basename=pdf_path.name,
        pdf_sha256=pdf_sha,
        pdf_size_bytes=pdf_path.stat().st_size,
        page_count=182,
        index_page_start=2,
        index_page_end=3,
        content_page_start=4,
        content_page_end=130,
        key_page_start=key_start,
        key_page_end=key_end,
        source_page_projection_sha256=page_pins,
        activities=tuple(activities),
        inventory_projection_sha256=canonical_json_sha256(projection),
    )


def _record_to_mapping(item: Math12ActivityRecord) -> dict[str, Any]:
    value = asdict(item)
    return value


def load_math12_inventory(path: Path) -> Math12Inventory:
    """Load canonical inventory JSON and recompute its projection pin."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Math12SourceError("Math12 inventory cannot be loaded") from exc
    if not isinstance(raw, dict):
        raise Math12SourceError("Math12 inventory root is not an object")
    activities = tuple(
        Math12ActivityRecord(
            **{
                **item,
                "key_start": KeyLocator(**item["key_start"]),
                "key_end_exclusive": KeyLocator(**item["key_end_exclusive"]),
            }
        )
        for item in raw["activities"]
    )
    pins_raw = raw["source_page_projection_sha256"]
    inventory = Math12Inventory(
        schema_version=raw["schema_version"],
        document_id=raw["document_id"],
        pdf_basename=raw["pdf_basename"],
        pdf_sha256=raw["pdf_sha256"],
        pdf_size_bytes=raw["pdf_size_bytes"],
        page_count=raw["page_count"],
        index_page_start=raw["index_page_start"],
        index_page_end=raw["index_page_end"],
        content_page_start=raw["content_page_start"],
        content_page_end=raw["content_page_end"],
        key_page_start=raw["key_page_start"],
        key_page_end=raw["key_page_end"],
        source_page_projection_sha256=tuple(
            (int(page), pin)
            for page, pin in sorted(pins_raw.items(), key=lambda pair: int(pair[0]))
        ),
        activities=activities,
        inventory_projection_sha256=raw["inventory_projection_sha256"],
    )
    projection = inventory.projection()
    if canonical_json_sha256(projection) != inventory.inventory_projection_sha256:
        raise Math12SourceError("Math12 inventory projection pin mismatch")
    return inventory


def load_math12_render_manifest(
    path: Path,
    inventory: Math12Inventory,
    *,
    page_root: Path | None = None,
) -> Math12RenderManifest:
    """Parse a portable render manifest and verify every external page byte.

    The tracked manifest contains only portable ``page-NNN.png`` names.  Its
    PNG payload may live beside the manifest or under an explicit ``page_root``;
    relocation never changes the frozen manifest projection.
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Math12SourceError("Math12 render manifest cannot be loaded") from exc
    if not isinstance(raw, dict):
        raise Math12SourceError("Math12 render manifest root is not an object")
    frozen_pin = raw.get("render_manifest_projection_sha256")
    projection = {
        key: value
        for key, value in raw.items()
        if key != "render_manifest_projection_sha256"
    }
    if canonical_json_sha256(projection) != frozen_pin:
        raise Math12SourceError("render manifest projection pin mismatch")
    if (
        raw.get("schema_version") != RENDER_MANIFEST_SCHEMA
        or raw.get("document_id") != inventory.document_id
        or raw.get("pdf_sha256") != inventory.pdf_sha256
        or raw.get("inventory_projection_sha256")
        != inventory.inventory_projection_sha256
    ):
        raise Math12SourceError("render manifest source identity mismatch")
    pages: list[Math12RenderedPage] = []
    for raw_page, item in raw.get("pages", {}).items():
        if not isinstance(item, dict):
            raise Math12SourceError("render manifest page is not an object")
        page_number = int(raw_page)
        manifest_path = str(item.get("path") or "")
        raw_path = Path(manifest_path)
        if raw_path.is_absolute() or raw_path.name != manifest_path:
            raise Math12SourceError("render manifest page path is not portable")
        page_path = (page_root.resolve() if page_root is not None else path.parent) / raw_path
        page_path = page_path.resolve()
        page = Math12RenderedPage(
            page_number=page_number,
            manifest_path=manifest_path,
            path=page_path,
            sha256=str(item.get("sha256") or ""),
            size_bytes=int(item.get("size_bytes") or 0),
        )
        if (
            not page.path.is_file()
            or page.path.stat().st_size != page.size_bytes
            or sha256_file(page.path) != page.sha256
        ):
            raise Math12SourceError("rendered page bytes differ from their manifest pin")
        pages.append(page)
    expected = tuple(range(inventory.content_page_start, inventory.content_page_end + 1))
    if tuple(sorted(item.page_number for item in pages)) != expected or len(pages) != len(
        expected
    ):
        raise Math12SourceError("render manifest must pin every content page exactly once")
    manifest = Math12RenderManifest(
        schema_version=raw["schema_version"],
        document_id=raw["document_id"],
        pdf_sha256=raw["pdf_sha256"],
        inventory_projection_sha256=raw["inventory_projection_sha256"],
        render_dpi=int(raw["render_dpi"]),
        color_mode=raw["color_mode"],
        poppler_version=raw["poppler_version"],
        pages=tuple(sorted(pages, key=lambda item: item.page_number)),
        render_manifest_projection_sha256=frozen_pin,
    )
    validate_math12_render_manifest(inventory, manifest)
    return manifest


def validate_math12_render_manifest(
    inventory: Math12Inventory, render_manifest: Math12RenderManifest
) -> None:
    """Replay manifest projection, source pins, profile and every page hash."""

    if canonical_json_sha256(inventory.projection()) != inventory.inventory_projection_sha256:
        raise Math12SourceError("Math12 inventory projection pin mismatch")
    if (
        render_manifest.document_id != inventory.document_id
        or render_manifest.pdf_sha256 != inventory.pdf_sha256
        or render_manifest.inventory_projection_sha256
        != inventory.inventory_projection_sha256
        or render_manifest.render_dpi != FROZEN_SIFT_RUNTIME_PROFILE.render_dpi
        or render_manifest.color_mode != "gray_png"
        or render_manifest.poppler_version != EXPECTED_POPPLER_VERSION
    ):
        raise Math12SourceError("render manifest and frozen source profile differ")
    if (
        canonical_json_sha256(render_manifest.projection())
        != render_manifest.render_manifest_projection_sha256
    ):
        raise Math12SourceError("render manifest object projection pin mismatch")
    pages = render_manifest.page_map()
    expected_pages = set(
        range(inventory.content_page_start, inventory.content_page_end + 1)
    )
    if set(pages) != expected_pages or len(render_manifest.pages) != len(expected_pages):
        raise Math12SourceError("rendered page map must contain every content page exactly once")
    for page_number, page in pages.items():
        if (
            page.page_number != page_number
            or page.manifest_path != f"page-{page_number:03d}.png"
            or not page.path.is_file()
            or page.path.stat().st_size != page.size_bytes
            or sha256_file(page.path) != page.sha256
        ):
            raise Math12SourceError("rendered page bytes differ from their manifest pin")


def _evidence_to_mapping(item: VisualPageEvidence) -> dict[str, Any]:
    return asdict(item)


def _decision_to_mapping(item: Math12BindingDecision) -> dict[str, Any]:
    return asdict(item)


def _decision_from_mapping(value: Mapping[str, Any]) -> Math12BindingDecision:
    key_start = value.get("key_start")
    key_end = value.get("key_end_exclusive")
    return Math12BindingDecision(
        accepted=bool(value["accepted"]),
        reason=str(value["reason"]),
        checks=tuple((str(name), bool(passed)) for name, passed in value["checks"]),
        selected_content_page=value.get("selected_content_page"),
        selected_activity_number=value.get("selected_activity_number"),
        key_page_start=value.get("key_page_start"),
        key_page_end=value.get("key_page_end"),
        key_start=KeyLocator(**key_start) if key_start is not None else None,
        key_end_exclusive=KeyLocator(**key_end) if key_end is not None else None,
        binding_projection_sha256=value.get("binding_projection_sha256"),
        best_rank_score=float(value.get("best_rank_score", 0.0)),
        runner_rank_score=float(value.get("runner_rank_score", 0.0)),
    )


def load_math12_source_certificate(path: Path) -> Math12SourceCertificate:
    """Parse a certificate and verify only its serialized self-pins.

    This loader does *not* establish that the saved decision follows from the
    evidence or belongs to a supplied inventory/render set.  Trust decisions
    only after :func:`verify_math12_source_certificate` succeeds.
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Math12SourceError("Math12 source certificate cannot be loaded") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != CERTIFICATE_SCHEMA:
        raise Math12SourceError("unknown Math12 source certificate schema")
    thresholds = VisualBindingThresholds(**raw["thresholds"])
    runtime_profile = SiftRuntimeProfile(**raw["runtime_profile"])
    evidences = tuple(
        visual_page_evidence_from_mapping(item) for item in raw["evidences"]
    )
    decision = _decision_from_mapping(raw["decision"])
    evidence_pin = canonical_json_sha256(
        [_evidence_to_mapping(item) for item in evidences]
    )
    if evidence_pin != raw.get("evidence_projection_sha256"):
        raise Math12SourceError("source certificate evidence projection pin mismatch")
    projection = {
        "schema_version": raw["schema_version"],
        "document_id": raw["document_id"],
        "pdf_sha256": raw["pdf_sha256"],
        "inventory_projection_sha256": raw["inventory_projection_sha256"],
        "render_manifest_projection_sha256": raw[
            "render_manifest_projection_sha256"
        ],
        "task_image_sha256": raw["task_image_sha256"],
        "thresholds": asdict(thresholds),
        "runtime_profile": asdict(runtime_profile),
        "decision": _decision_to_mapping(decision),
        "evidence_projection_sha256": evidence_pin,
    }
    certificate_pin = canonical_json_sha256(projection)
    if certificate_pin != raw.get("certificate_projection_sha256"):
        raise Math12SourceError("source certificate projection pin mismatch")
    return Math12SourceCertificate(
        schema_version=raw["schema_version"],
        document_id=raw["document_id"],
        pdf_sha256=raw["pdf_sha256"],
        inventory_projection_sha256=raw["inventory_projection_sha256"],
        render_manifest_projection_sha256=raw[
            "render_manifest_projection_sha256"
        ],
        task_image_sha256=raw["task_image_sha256"],
        thresholds=thresholds,
        runtime_profile=runtime_profile,
        evidences=evidences,
        decision=decision,
        evidence_projection_sha256=evidence_pin,
        certificate_projection_sha256=certificate_pin,
    )


def decide_math12_source_binding(
    inventory: Math12Inventory,
    evidences: Sequence[VisualPageEvidence],
    *,
    task_image_sha256: str,
    thresholds: VisualBindingThresholds = FROZEN_VISUAL_THRESHOLDS,
) -> Math12BindingDecision:
    """Resolve all-page visual evidence to a source address, or abstain."""

    _require_frozen_math12_profile(thresholds)
    expected_pages = tuple(
        range(inventory.content_page_start, inventory.content_page_end + 1)
    )
    if _HEX64.fullmatch(task_image_sha256) is None:
        raise Math12SourceError("task image hash is malformed")
    evidence_pages = tuple(sorted(item.page_number for item in evidences))
    complete_sweep = evidence_pages == expected_pages and len(evidences) == len(expected_pages)
    if not complete_sweep:
        return Math12BindingDecision(
            False, "incomplete_or_duplicate_all_page_sweep", (("all_content_pages_once", False),)
        )
    identity_ok = all(
        item.task_image_sha256 == task_image_sha256
        and item.document_id == inventory.document_id
        and item.pdf_sha256 == inventory.pdf_sha256
        for item in evidences
    )
    if not identity_ok:
        return Math12BindingDecision(
            False, "source_identity_mismatch", (("source_identity", False),)
        )
    ordered = sorted(
        evidences,
        key=lambda item: (
            -item.rank_score,
            -item.inliers,
            item.page_number,
            item.rendered_page_sha256,
        ),
    )
    best = ordered[0]
    runner_score = ordered[1].rank_score
    margin = best.rank_score - runner_score
    ratio = best.rank_score / max(runner_score, 1e-9)
    checks = list(geometry_checks(best, thresholds))
    checks.extend(
        (
            ("all_content_pages_once", True),
            ("source_identity", True),
            ("page_rank_margin", margin >= thresholds.min_rank_score_margin),
            ("page_rank_ratio", ratio >= thresholds.min_rank_score_ratio),
        )
    )
    if not all(passed for _, passed in checks):
        return Math12BindingDecision(
            False,
            "visual_geometry_or_margin_failed",
            tuple(checks),
            selected_content_page=best.page_number,
            best_rank_score=best.rank_score,
            runner_rank_score=runner_score,
        )
    record = inventory.activity_for_page(best.page_number)
    range_unique = record is not None
    pins_present = record is not None and all(
        _HEX64.fullmatch(value) is not None
        for value in (
            record.index_projection_sha256,
            record.content_projection_sha256,
            record.key_projection_sha256,
            record.binding_projection_sha256,
        )
    )
    checks.extend(
        (("one_activity_range_for_page", range_unique), ("source_projection_pins", pins_present))
    )
    accepted = all(passed for _, passed in checks)
    return Math12BindingDecision(
        accepted,
        "accepted" if accepted else "activity_source_range_failed",
        tuple(checks),
        selected_content_page=best.page_number,
        selected_activity_number=record.activity_number if accepted and record else None,
        key_page_start=record.key_page_start if accepted and record else None,
        key_page_end=record.key_page_end if accepted and record else None,
        key_start=record.key_start if accepted and record else None,
        key_end_exclusive=record.key_end_exclusive if accepted and record else None,
        binding_projection_sha256=(
            record.binding_projection_sha256 if accepted and record else None
        ),
        best_rank_score=best.rank_score,
        runner_rank_score=runner_score,
    )


def issue_math12_source_certificate(
    inventory: Math12Inventory,
    evidences: Sequence[VisualPageEvidence],
    *,
    task_image_sha256: str,
    render_manifest_projection_sha256: str,
    thresholds: VisualBindingThresholds = FROZEN_VISUAL_THRESHOLDS,
    runtime_profile: SiftRuntimeProfile = FROZEN_SIFT_RUNTIME_PROFILE,
) -> Math12SourceCertificate:
    if _HEX64.fullmatch(render_manifest_projection_sha256) is None:
        raise Math12SourceError("render manifest projection pin is malformed")
    _require_frozen_math12_profile(thresholds, runtime_profile)
    frozen_evidence = tuple(sorted(evidences, key=lambda item: item.page_number))
    decision = decide_math12_source_binding(
        inventory, frozen_evidence, task_image_sha256=task_image_sha256, thresholds=thresholds
    )
    evidence_pin = canonical_json_sha256(
        [_evidence_to_mapping(item) for item in frozen_evidence]
    )
    projection = {
        "schema_version": CERTIFICATE_SCHEMA,
        "document_id": inventory.document_id,
        "pdf_sha256": inventory.pdf_sha256,
        "inventory_projection_sha256": inventory.inventory_projection_sha256,
        "render_manifest_projection_sha256": render_manifest_projection_sha256,
        "task_image_sha256": task_image_sha256,
        "thresholds": asdict(thresholds),
        "runtime_profile": asdict(runtime_profile),
        "decision": _decision_to_mapping(decision),
        "evidence_projection_sha256": evidence_pin,
    }
    return Math12SourceCertificate(
        schema_version=CERTIFICATE_SCHEMA,
        document_id=inventory.document_id,
        pdf_sha256=inventory.pdf_sha256,
        inventory_projection_sha256=inventory.inventory_projection_sha256,
        render_manifest_projection_sha256=render_manifest_projection_sha256,
        task_image_sha256=task_image_sha256,
        thresholds=thresholds,
        runtime_profile=runtime_profile,
        evidences=frozen_evidence,
        decision=decision,
        evidence_projection_sha256=evidence_pin,
        certificate_projection_sha256=canonical_json_sha256(projection),
    )


def verify_math12_source_certificate(
    inventory: Math12Inventory,
    render_manifest: Math12RenderManifest,
    certificate: Math12SourceCertificate,
) -> Math12BindingDecision:
    """Strictly replay a certificate from all evidence and frozen source pins.

    Unlike :func:`load_math12_source_certificate`, this is the authoritative
    admission API.  It replays the decision, inventory and render projections,
    exact thresholds/runtime profile, every rendered page byte and every
    evidence-to-render hash binding.  Any mismatch raises instead of returning
    a partially trusted decision.
    """

    if certificate.schema_version != CERTIFICATE_SCHEMA:
        raise Math12SourceError("unknown Math12 source certificate schema")
    assert_math12_runtime()
    _require_frozen_math12_profile(certificate.thresholds, certificate.runtime_profile)
    validate_math12_render_manifest(inventory, render_manifest)
    if (
        certificate.document_id != inventory.document_id
        or certificate.pdf_sha256 != inventory.pdf_sha256
        or certificate.inventory_projection_sha256
        != inventory.inventory_projection_sha256
        or certificate.render_manifest_projection_sha256
        != render_manifest.render_manifest_projection_sha256
    ):
        raise Math12SourceError("certificate, inventory and render source identity differ")
    page_map = render_manifest.page_map()
    if any(
        item.page_number not in page_map
        or item.rendered_page_sha256 != page_map[item.page_number].sha256
        for item in certificate.evidences
    ):
        raise Math12SourceError("certificate evidence is not bound to rendered page bytes")
    evidence_pin = canonical_json_sha256(
        [_evidence_to_mapping(item) for item in certificate.evidences]
    )
    if evidence_pin != certificate.evidence_projection_sha256:
        raise Math12SourceError("source certificate evidence projection pin mismatch")
    replayed = decide_math12_source_binding(
        inventory,
        certificate.evidences,
        task_image_sha256=certificate.task_image_sha256,
        thresholds=certificate.thresholds,
    )
    projection = {
        "schema_version": certificate.schema_version,
        "document_id": certificate.document_id,
        "pdf_sha256": certificate.pdf_sha256,
        "inventory_projection_sha256": certificate.inventory_projection_sha256,
        "render_manifest_projection_sha256": (
            certificate.render_manifest_projection_sha256
        ),
        "task_image_sha256": certificate.task_image_sha256,
        "thresholds": asdict(certificate.thresholds),
        "runtime_profile": asdict(certificate.runtime_profile),
        "decision": _decision_to_mapping(certificate.decision),
        "evidence_projection_sha256": evidence_pin,
    }
    if canonical_json_sha256(projection) != certificate.certificate_projection_sha256:
        raise Math12SourceError("source certificate projection pin mismatch")
    if replayed != certificate.decision:
        raise Math12SourceError("source certificate decision does not replay from all evidence")
    return replayed


def _solution_text_from_projection(projection: Mapping[str, Any]) -> str:
    """Lay out only words inside a logical key span in deterministic order."""

    raw_words = projection.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        raise Math12SourceError("official solution span has no PDF-native words")
    ordered = sorted(
        raw_words,
        key=lambda item: (
            int(item["physical_page"]),
            0 if item["column"] == "left" else 1,
            float(item["top"]),
            float(item["x0"]),
            str(item["text"]),
        ),
    )
    lines: list[str] = []
    line_words: list[Mapping[str, Any]] = []
    line_address: tuple[int, str] | None = None
    line_top = 0.0
    for word in ordered:
        address = (int(word["physical_page"]), str(word["column"]))
        top = float(word["top"])
        if line_words and (address != line_address or abs(top - line_top) > 2.5):
            lines.append(
                " ".join(
                    str(item["text"]).strip()
                    for item in sorted(line_words, key=lambda item: float(item["x0"]))
                    if str(item["text"]).strip()
                )
            )
            line_words = []
        if not line_words:
            line_address = address
            line_top = top
        line_words.append(word)
    if line_words:
        lines.append(
            " ".join(
                str(item["text"]).strip()
                for item in sorted(line_words, key=lambda item: float(item["x0"]))
                if str(item["text"]).strip()
            )
        )
    text = "\n".join(line for line in lines if line).strip()
    if not text:
        raise Math12SourceError("official solution text is empty after layout")
    return text


def extract_official_solution(
    pdf_path: Path,
    inventory: Math12Inventory,
    render_manifest: Math12RenderManifest,
    accepted_certificate: Math12SourceCertificate,
) -> Math12OfficialSolutionRecord:
    """Extract only the accepted activity's pinned official key span.

    The function has no benchmark row, expected activity, gold answer, scorer,
    or task outcome input.  It fails closed unless the visual certificate is
    accepted and its selected source record exactly matches this inventory.
    """

    decision = accepted_certificate.decision
    if (
        not decision.accepted
        or not decision.checks
        or any(not passed for _, passed in decision.checks)
        or decision.selected_activity_number is None
        or decision.selected_content_page is None
        or decision.key_start is None
        or decision.key_end_exclusive is None
        or decision.binding_projection_sha256 is None
    ):
        raise Math12SourceError("official solution requires a fully accepted binding")
    replayed = verify_math12_source_certificate(
        inventory, render_manifest, accepted_certificate
    )
    if replayed != decision:  # defensive: verifier already enforces equality
        raise Math12SourceError("source certificate does not replay from its evidence")
    if sha256_file(pdf_path) != inventory.pdf_sha256:
        raise Math12SourceError("official solution PDF differs from the inventory pin")
    record = inventory.activities[decision.selected_activity_number - 1]
    if (
        not record.content_page_start
        <= decision.selected_content_page
        <= record.content_page_end
        or record.key_start != decision.key_start
        or record.key_end_exclusive != decision.key_end_exclusive
        or record.binding_projection_sha256 != decision.binding_projection_sha256
    ):
        raise Math12SourceError("accepted decision does not replay to one inventory record")
    assert_math12_runtime(require_pdfplumber=True)
    import pdfplumber  # type: ignore
    with pdfplumber.open(pdf_path) as document:
        projection = _key_section_projection(
            document,
            record.activity_number,
            record.key_start,
            record.key_end_exclusive,
        )
    if canonical_json_sha256(projection) != record.key_projection_sha256:
        raise Math12SourceError("official solution key projection differs from inventory pin")
    solution_text = _solution_text_from_projection(projection)
    solution_text_sha = hashlib.sha256(solution_text.encode("utf-8")).hexdigest()
    bound_projection = {
        "schema_version": "math12-answer-bound-official-solution-v1",
        "document_id": inventory.document_id,
        "pdf_sha256": inventory.pdf_sha256,
        "inventory_projection_sha256": inventory.inventory_projection_sha256,
        "source_certificate_projection_sha256": (
            accepted_certificate.certificate_projection_sha256
        ),
        "task_image_sha256": accepted_certificate.task_image_sha256,
        "selected_content_page": decision.selected_content_page,
        "activity_number": record.activity_number,
        "key_start": asdict(record.key_start),
        "key_end_exclusive": asdict(record.key_end_exclusive),
        "binding_projection_sha256": record.binding_projection_sha256,
        "key_projection_sha256": record.key_projection_sha256,
        "official_solution_text_sha256": solution_text_sha,
    }
    return Math12OfficialSolutionRecord(
        schema_version=bound_projection["schema_version"],
        component_scope="official_source_solution_text_no_gold_no_correctness",
        document_id=inventory.document_id,
        pdf_sha256=inventory.pdf_sha256,
        inventory_projection_sha256=inventory.inventory_projection_sha256,
        source_certificate_projection_sha256=(
            accepted_certificate.certificate_projection_sha256
        ),
        task_image_sha256=accepted_certificate.task_image_sha256,
        selected_content_page=decision.selected_content_page,
        activity_number=record.activity_number,
        key_page_start=record.key_page_start,
        key_page_end=record.key_page_end,
        key_start=record.key_start,
        key_end_exclusive=record.key_end_exclusive,
        binding_projection_sha256=record.binding_projection_sha256,
        key_projection_sha256=record.key_projection_sha256,
        official_solution_text=solution_text,
        official_solution_text_sha256=solution_text_sha,
        answer_bound_certificate_projection_sha256=canonical_json_sha256(
            bound_projection
        ),
    )


def resolve_math12_image_bytes(
    image_bytes: bytes,
    inventory: Math12Inventory,
    render_manifest: Math12RenderManifest,
    *,
    thresholds: VisualBindingThresholds = FROZEN_VISUAL_THRESHOLDS,
    runtime_profile: SiftRuntimeProfile = FROZEN_SIFT_RUNTIME_PROFILE,
) -> Math12SourceCertificate:
    """Generic ``image bytes -> source address + certificate`` API.

    Every content page is evaluated.  There is no task id, expected activity,
    benchmark answer, scorer output, or per-row routing input in this API.
    """

    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise Math12SourceError("input image bytes are empty")
    _require_frozen_math12_profile(thresholds, runtime_profile)
    assert_math12_runtime(require_visual=True)
    validate_math12_render_manifest(inventory, render_manifest)
    rendered_pages = render_manifest.page_map()
    task_sha = hashlib.sha256(image_bytes).hexdigest()
    with tempfile.TemporaryDirectory(prefix="math12_source_binding_") as directory:
        # OpenCV sniffs image content, while a conventional suffix also keeps
        # compatibility with builds that consult the filename first.
        task_path = Path(directory) / "input_image.png"
        task_path.write_bytes(image_bytes)
        evidences = tuple(
            compute_sift_page_evidence(
                task_path,
                rendered_pages[page_number].path,
                task_image_sha256=task_sha,
                document_id=inventory.document_id,
                pdf_sha256=inventory.pdf_sha256,
                page_number=page_number,
                profile=runtime_profile,
            )
            for page_number in sorted(rendered_pages)
        )
    return issue_math12_source_certificate(
        inventory,
        evidences,
        task_image_sha256=task_sha,
        render_manifest_projection_sha256=(
            render_manifest.render_manifest_projection_sha256
        ),
        thresholds=thresholds,
        runtime_profile=runtime_profile,
    )


def write_canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
