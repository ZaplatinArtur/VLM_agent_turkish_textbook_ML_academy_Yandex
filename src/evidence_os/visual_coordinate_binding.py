"""Fail-closed visual binding of a task crop to a pinned workbook record.

This module deliberately stops at source identity.  It does not read benchmark
answers, candidates, scores, task outcomes, or a scorer.  A caller supplies
SIFT/RANSAC evidence for pages of one already-pinned PDF plus PDF-native word
markers and reviewed source-index records.  Admission requires strong projective
geometry, a page margin, and exactly one indexed printed marker inside the mapped
task polygon.

OpenCV is an optional runtime dependency.  The policy/decision layer is pure
Python so its fail-closed behavior can be tested without OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


VISUAL_BINDING_VERIFIER = "public-workbook-sift-pdf-marker-binding-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class VisualCoordinateBindingError(ValueError):
    """Visual evidence is malformed or violates its pinned source identity."""


def load_activity_visual_artifact_json(path: Path) -> dict[str, Any]:
    """Load canonical JSON while rejecting duplicate keys and NaN constants."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise VisualCoordinateBindingError(
                    f"duplicate JSON key in visual artifact: {key}"
                )
            result[key] = item
        return result

    def reject_constant(value: str) -> Any:
        raise VisualCoordinateBindingError(
            f"non-finite JSON constant in visual artifact: {value}"
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VisualCoordinateBindingError("visual artifact JSON cannot be loaded") from exc
    if not isinstance(value, dict):
        raise VisualCoordinateBindingError("visual artifact JSON root is not an object")
    return value


@dataclass(frozen=True, slots=True)
class VisualBindingThresholds:
    """Source-only thresholds frozen before any benchmark scoring."""

    min_good_matches: int = 50
    min_inliers: int = 40
    min_inlier_ratio: float = 0.65
    min_task_hull_fraction: float = 0.30
    max_median_reprojection_error: float = 1.0
    min_mapped_inside_fraction: float = 0.98
    max_scale_anisotropy: float = 1.15
    min_rank_score_margin: float = 10.0
    min_rank_score_ratio: float = 5.0

    def __post_init__(self) -> None:
        if self.min_good_matches < 4 or self.min_inliers < 4:
            raise ValueError("visual match thresholds must allow a homography")
        for name in (
            "min_inlier_ratio",
            "min_task_hull_fraction",
            "min_mapped_inside_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.max_median_reprojection_error <= 0.0:
            raise ValueError("max_median_reprojection_error must be positive")
        if self.max_scale_anisotropy < 1.0:
            raise ValueError("max_scale_anisotropy cannot be below one")
        if self.min_rank_score_margin < 0.0 or self.min_rank_score_ratio < 1.0:
            raise ValueError("page-margin thresholds are invalid")


def require_strict_activity_visual_thresholds(
    thresholds: VisualBindingThresholds,
) -> None:
    """Reject callers that weaken the frozen production safety floor."""

    floor = VisualBindingThresholds()
    minimum_fields = (
        "min_good_matches",
        "min_inliers",
        "min_inlier_ratio",
        "min_task_hull_fraction",
        "min_mapped_inside_fraction",
        "min_rank_score_margin",
        "min_rank_score_ratio",
    )
    maximum_fields = (
        "max_median_reprojection_error",
        "max_scale_anisotropy",
    )
    if any(getattr(thresholds, name) < getattr(floor, name) for name in minimum_fields) or any(
        getattr(thresholds, name) > getattr(floor, name) for name in maximum_fields
    ):
        raise VisualCoordinateBindingError("activity visual thresholds weaken the safety floor")


@dataclass(frozen=True, slots=True)
class SiftRuntimeProfile:
    render_dpi: int = 144
    nfeatures: int = 12_000
    contrast_threshold: float = 0.02
    edge_threshold: float = 12.0
    ratio_test: float = 0.72
    ransac_reprojection_px: float = 4.0
    ransac_max_iters: int = 5_000
    ransac_confidence: float = 0.999
    rng_seed: int = 19_870_511
    expected_opencv_version: str | None = None

    def __post_init__(self) -> None:
        if self.render_dpi <= 0 or self.nfeatures < 4:
            raise ValueError("invalid render/SIFT size")
        if not 0.0 < self.ratio_test < 1.0:
            raise ValueError("ratio_test must be between zero and one")
        if self.ransac_reprojection_px <= 0.0 or self.ransac_max_iters < 1:
            raise ValueError("invalid RANSAC settings")
        if not 0.0 < self.ransac_confidence < 1.0:
            raise ValueError("ransac_confidence must be between zero and one")


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class VisualPageEvidence:
    task_image_sha256: str
    document_id: str
    pdf_sha256: str
    page_number: int
    rendered_page_sha256: str
    good_matches: int
    inliers: int
    inlier_ratio: float
    task_hull_fraction: float
    median_reprojection_error: float | None
    mapped_inside_fraction: float
    scale_anisotropy: float | None
    orientation_preserved: bool
    convex_mapping: bool
    mapped_polygon: tuple[Point, Point, Point, Point] | None

    def __post_init__(self) -> None:
        for name in ("task_image_sha256", "pdf_sha256", "rendered_page_sha256"):
            if _HEX64.fullmatch(str(getattr(self, name))) is None:
                raise VisualCoordinateBindingError(f"{name} is not SHA-256")
        if not self.document_id or self.page_number < 1:
            raise VisualCoordinateBindingError("page source address is malformed")
        if self.good_matches < 0 or self.inliers < 0 or self.inliers > self.good_matches:
            raise VisualCoordinateBindingError("match counts are inconsistent")
        for name in ("inlier_ratio", "task_hull_fraction", "mapped_inside_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise VisualCoordinateBindingError(f"{name} is outside [0, 1]")
        expected_ratio = self.inliers / self.good_matches if self.good_matches else 0.0
        if not math.isclose(
            self.inlier_ratio,
            expected_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise VisualCoordinateBindingError("inlier ratio does not match integer counts")
        if self.median_reprojection_error is not None and (
            not math.isfinite(self.median_reprojection_error)
            or self.median_reprojection_error < 0.0
        ):
            raise VisualCoordinateBindingError("reprojection error is invalid")
        if self.scale_anisotropy is not None and (
            not math.isfinite(self.scale_anisotropy) or self.scale_anisotropy < 0.0
        ):
            raise VisualCoordinateBindingError("scale anisotropy is invalid")
        if self.mapped_polygon is not None:
            if len(self.mapped_polygon) != 4 or any(
                len(point) != 2 or not all(math.isfinite(value) for value in point)
                for point in self.mapped_polygon
            ):
                raise VisualCoordinateBindingError("mapped polygon is malformed")

    @property
    def rank_score(self) -> float:
        if self.median_reprojection_error is None:
            return 0.0
        return (
            self.inliers
            * self.inlier_ratio
            * math.sqrt(max(self.task_hull_fraction, 1e-6))
            * self.mapped_inside_fraction
            / (1.0 + self.median_reprojection_error)
        )


@dataclass(frozen=True, slots=True)
class PdfQuestionMarker:
    page_number: int
    question_number: int
    center: Point
    token: str

    def __post_init__(self) -> None:
        if self.page_number < 1 or not 1 <= self.question_number <= 999:
            raise VisualCoordinateBindingError("PDF marker address is invalid")
        if not all(math.isfinite(value) for value in self.center):
            raise VisualCoordinateBindingError("PDF marker center is invalid")
        if self.token not in {f"{self.question_number}.", f"{self.question_number})"}:
            raise VisualCoordinateBindingError("PDF marker token does not attest its number")


@dataclass(frozen=True, slots=True)
class IndexedQuestionRef:
    document_id: str
    record_id: str
    content_page_number: int
    question_number: int
    visually_checked: bool
    answer_format: str
    key_binding_kind: str
    key_page_number: int
    key_bbox: tuple[float, float, float, float] | None
    key_projection_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.document_id or not self.record_id.startswith(f"{self.document_id}:p"):
            raise VisualCoordinateBindingError("record is not source-addressed")
        if self.content_page_number < 1 or self.key_page_number < 1:
            raise VisualCoordinateBindingError("record page number is invalid")
        if not 1 <= self.question_number <= 999:
            raise VisualCoordinateBindingError("record question number is invalid")
        if self.answer_format not in {"choice", "short_text"}:
            raise VisualCoordinateBindingError("unsupported source answer format")
        if self.key_bbox is not None and (
            len(self.key_bbox) != 4
            or not all(math.isfinite(value) for value in self.key_bbox)
            or not (self.key_bbox[0] < self.key_bbox[2] and self.key_bbox[1] < self.key_bbox[3])
        ):
            raise VisualCoordinateBindingError("key bbox is malformed")
        if self.key_projection_sha256 is not None and _HEX64.fullmatch(self.key_projection_sha256) is None:
            raise VisualCoordinateBindingError("key projection is not SHA-256")

    @property
    def key_is_certifiable(self) -> bool:
        if not self.visually_checked or self.key_bbox is None:
            return False
        if self.answer_format == "choice":
            return self.key_binding_kind in {"inline_solution", "answer_key_table"}
        return (
            self.key_binding_kind == "coordinate_answer_key"
            and self.key_projection_sha256 is not None
        )


@dataclass(frozen=True, slots=True)
class VisualBindingDecision:
    accepted: bool
    reason: str
    checks: tuple[tuple[str, bool], ...]
    selected_page_number: int | None = None
    selected_question_number: int | None = None
    selected_record_id: str | None = None
    best_rank_score: float = 0.0
    runner_rank_score: float = 0.0


@dataclass(frozen=True, slots=True)
class ActivityVisualRecordRef:
    """Answer-free source identity needed for visual activity-page selection."""

    document_id: str
    record_id: str
    content_page_number: int
    activity_number: int
    key_projection_sha256: str
    content_projection_sha256: str
    binding_projection_sha256: str
    visually_checked: bool
    content_bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if (
            not self.document_id
            or self.record_id
            != f"{self.document_id}:p{self.content_page_number}:q{self.activity_number}"
            or self.content_page_number < 1
            or self.activity_number < 1
        ):
            raise VisualCoordinateBindingError(
                "visual activity record is not source-addressed"
            )
        if any(
            _HEX64.fullmatch(value) is None
            for value in (
                self.key_projection_sha256,
                self.content_projection_sha256,
                self.binding_projection_sha256,
            )
        ):
            raise VisualCoordinateBindingError(
                "visual activity record lacks its three PDF projection pins"
            )
        if (
            len(self.content_bbox) != 4
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in self.content_bbox
            )
            or not (
                self.content_bbox[0] < self.content_bbox[2]
                and self.content_bbox[1] < self.content_bbox[3]
            )
        ):
            raise VisualCoordinateBindingError("visual activity content bbox is malformed")


@dataclass(frozen=True, slots=True)
class ActivityVisualObservationRef:
    """Parser/source alignment facts; ``task_id`` is never returned to policy."""

    task_id: str
    task_image_sha256: str
    width: int
    height: int
    parser_identity: str
    document_id: str
    pdf_sha256: str
    marker_kind: str
    marker_number: int

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or _HEX64.fullmatch(self.task_image_sha256) is None
            or self.width < 1
            or self.height < 1
            or not self.parser_identity
            or not self.document_id
            or _HEX64.fullmatch(self.pdf_sha256) is None
            or self.marker_kind != "activity_label"
            or self.marker_number < 1
        ):
            raise VisualCoordinateBindingError("visual activity observation is malformed")


@dataclass(frozen=True, slots=True)
class VerifiedActivityVisualBinding:
    """Answer-free visual page decision that has passed the pure policy replay."""

    task_image_sha256: str
    document_id: str
    pdf_sha256: str
    observed_activity_number: int
    evidences: tuple[VisualPageEvidence, ...]
    decision: VisualBindingDecision
    thresholds: VisualBindingThresholds
    verifier: str = VISUAL_BINDING_VERIFIER

    def __post_init__(self) -> None:
        if self.verifier != VISUAL_BINDING_VERIFIER:
            raise VisualCoordinateBindingError("visual activity verifier is not pinned")
        if (
            _HEX64.fullmatch(self.task_image_sha256) is None
            or _HEX64.fullmatch(self.pdf_sha256) is None
            or not self.document_id
            or self.observed_activity_number < 1
            or not self.evidences
        ):
            raise VisualCoordinateBindingError("verified visual activity identity is malformed")
        if any(
            evidence.task_image_sha256 != self.task_image_sha256
            or evidence.document_id != self.document_id
            or evidence.pdf_sha256 != self.pdf_sha256
            for evidence in self.evidences
        ):
            raise VisualCoordinateBindingError("verified visual activity evidence identity differs")
        if (
            not self.decision.accepted
            or not self.decision.selected_record_id
            or self.decision.selected_page_number is None
            or self.decision.selected_question_number != self.observed_activity_number
            or not self.decision.checks
            or any(not passed for _, passed in self.decision.checks)
        ):
            raise VisualCoordinateBindingError("visual activity decision is not fully accepted")
        require_strict_activity_visual_thresholds(self.thresholds)

    def trace(self) -> dict[str, Any]:
        return {
            "verifier": self.verifier,
            "task_image_sha256": self.task_image_sha256,
            "document_id": self.document_id,
            "pdf_sha256": self.pdf_sha256,
            "observed_activity_number": self.observed_activity_number,
            "selected_page_number": self.decision.selected_page_number,
            "selected_record_id": self.decision.selected_record_id,
            "best_rank_score": self.decision.best_rank_score,
            "runner_rank_score": self.decision.runner_rank_score,
            "checks": dict(self.decision.checks),
            "thresholds": {
                "min_good_matches": self.thresholds.min_good_matches,
                "min_inliers": self.thresholds.min_inliers,
                "min_inlier_ratio": self.thresholds.min_inlier_ratio,
                "min_task_hull_fraction": self.thresholds.min_task_hull_fraction,
                "max_median_reprojection_error": self.thresholds.max_median_reprojection_error,
                "min_mapped_inside_fraction": self.thresholds.min_mapped_inside_fraction,
                "max_scale_anisotropy": self.thresholds.max_scale_anisotropy,
                "min_rank_score_margin": self.thresholds.min_rank_score_margin,
                "min_rank_score_ratio": self.thresholds.min_rank_score_ratio,
            },
        }


def verify_activity_visual_binding(
    evidences: Sequence[VisualPageEvidence],
    records: Sequence[ActivityVisualRecordRef],
    *,
    expected_task_image_sha256: str,
    expected_document_id: str,
    expected_pdf_sha256: str,
    observed_activity_number: int,
    thresholds: VisualBindingThresholds = VisualBindingThresholds(),
) -> VerifiedActivityVisualBinding | None:
    """Replay the answer-free activity policy and return only an accepted binding."""

    require_strict_activity_visual_thresholds(thresholds)
    frozen_evidences = tuple(evidences)
    decision = decide_visual_activity_binding(
        frozen_evidences,
        records,
        expected_task_image_sha256=expected_task_image_sha256,
        expected_document_id=expected_document_id,
        expected_pdf_sha256=expected_pdf_sha256,
        observed_activity_number=observed_activity_number,
        thresholds=thresholds,
    )
    if not decision.accepted:
        return None
    return VerifiedActivityVisualBinding(
        task_image_sha256=expected_task_image_sha256,
        document_id=expected_document_id,
        pdf_sha256=expected_pdf_sha256,
        observed_activity_number=observed_activity_number,
        evidences=frozen_evidences,
        decision=decision,
        thresholds=thresholds,
    )


def visual_page_evidence_from_mapping(value: Any) -> VisualPageEvidence:
    """Materialize an exact allowlisted evidence object from frozen JSON."""

    if not isinstance(value, dict):
        raise VisualCoordinateBindingError("visual page evidence is not an object")
    expected = {
        "task_image_sha256",
        "document_id",
        "pdf_sha256",
        "page_number",
        "rendered_page_sha256",
        "good_matches",
        "inliers",
        "inlier_ratio",
        "task_hull_fraction",
        "median_reprojection_error",
        "mapped_inside_fraction",
        "scale_anisotropy",
        "orientation_preserved",
        "convex_mapping",
        "mapped_polygon",
    }
    if set(value) != expected:
        raise VisualCoordinateBindingError(
            "visual page evidence fields are not on the exact allowlist"
        )
    for name in ("page_number", "good_matches", "inliers"):
        if type(value.get(name)) is not int:
            raise VisualCoordinateBindingError(f"{name} is not a canonical integer")
    for name in (
        "inlier_ratio",
        "task_hull_fraction",
        "mapped_inside_fraction",
    ):
        if (
            not isinstance(value.get(name), (int, float))
            or isinstance(value.get(name), bool)
            or not math.isfinite(float(value[name]))
        ):
            raise VisualCoordinateBindingError(f"{name} is not a finite number")
    for name in ("median_reprojection_error", "scale_anisotropy"):
        if value.get(name) is not None and (
            not isinstance(value.get(name), (int, float))
            or isinstance(value.get(name), bool)
            or not math.isfinite(float(value[name]))
        ):
            raise VisualCoordinateBindingError(f"{name} is not a finite number or null")
    for name in ("orientation_preserved", "convex_mapping"):
        if type(value.get(name)) is not bool:
            raise VisualCoordinateBindingError(f"{name} is not a canonical boolean")
    for name in (
        "task_image_sha256",
        "document_id",
        "pdf_sha256",
        "rendered_page_sha256",
    ):
        if type(value.get(name)) is not str:
            raise VisualCoordinateBindingError(f"{name} is not a canonical string")
    polygon_raw = value.get("mapped_polygon")
    if polygon_raw is None:
        polygon = None
    elif (
        isinstance(polygon_raw, list)
        and len(polygon_raw) == 4
        and all(
            isinstance(point, list)
            and len(point) == 2
            and all(
                isinstance(coordinate, (int, float))
                and not isinstance(coordinate, bool)
                and math.isfinite(float(coordinate))
                for coordinate in point
            )
            for point in polygon_raw
        )
    ):
        polygon = tuple(
            (float(point[0]), float(point[1])) for point in polygon_raw
        )
    else:
        raise VisualCoordinateBindingError("mapped_polygon is not canonical")
    return VisualPageEvidence(
        task_image_sha256=str(value["task_image_sha256"]),
        document_id=str(value["document_id"]),
        pdf_sha256=str(value["pdf_sha256"]),
        page_number=int(value["page_number"]),
        rendered_page_sha256=str(value["rendered_page_sha256"]),
        good_matches=int(value["good_matches"]),
        inliers=int(value["inliers"]),
        inlier_ratio=float(value["inlier_ratio"]),
        task_hull_fraction=float(value["task_hull_fraction"]),
        median_reprojection_error=(
            float(value["median_reprojection_error"])
            if value["median_reprojection_error"] is not None
            else None
        ),
        mapped_inside_fraction=float(value["mapped_inside_fraction"]),
        scale_anisotropy=(
            float(value["scale_anisotropy"])
            if value["scale_anisotropy"] is not None
            else None
        ),
        orientation_preserved=value["orientation_preserved"] is True,
        convex_mapping=value["convex_mapping"] is True,
        mapped_polygon=polygon,  # type: ignore[arg-type]
    )


def verified_activity_bindings_from_artifact(
    value: Any,
    *,
    repo_root: Path,
    expected_parser_sha256: str,
    expected_source_locators_sha256: str,
    expected_source_index_sha256: str,
    observations_by_task_id: Mapping[str, ActivityVisualObservationRef],
    records: Sequence[ActivityVisualRecordRef],
    document_pdf_paths: Mapping[str, Path],
    thresholds: VisualBindingThresholds = VisualBindingThresholds(),
) -> dict[str, VerifiedActivityVisualBinding]:
    """Strictly replay a frozen, answer-free visual-evidence artifact."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "generator",
        "inputs",
        "profiles",
        "runtime",
        "source_only_guards",
        "summary",
        "activity_records",
        "bindings_by_task_image_sha256",
    }:
        raise VisualCoordinateBindingError("visual activity artifact root is malformed")
    if (
        value.get("schema_version")
        != "maxim-activity-visual-binding-source-evidence-v1"
        or value.get("generator") != "build-maxim-activity-visual-binding-v1"
    ):
        raise VisualCoordinateBindingError("visual activity artifact schema is unsupported")
    guards = value.get("source_only_guards")
    expected_guards = {
        "benchmark_answer_candidate_outcome_artifacts_read": False,
        "source_answer_value_access": False,
        "task_id_is_policy_feature": False,
        "task_id_role": "alignment_audit_only",
        "parser_observation_filter": "observed_source_question_marker=activity_label",
        "source_index_record_filter": (
            "key_binding_kind=activity_answer_key AND "
            "question_marker_kind=activity_label"
        ),
        "render_scope": (
            "all indexed physical content pages expanded from each activity "
            "document content_page_ranges"
        ),
    }
    if guards != expected_guards:
        raise VisualCoordinateBindingError("visual activity source-only guards changed")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "documents",
        "parser_observations",
        "source_index",
        "source_locators",
        "task_image_dir",
    }:
        raise VisualCoordinateBindingError("visual activity input pins are malformed")
    for name, expected_sha in (
        ("parser_observations", expected_parser_sha256),
        ("source_index", expected_source_index_sha256),
        ("source_locators", expected_source_locators_sha256),
    ):
        spec = inputs.get(name)
        expected_fields = (
            {"path", "sha256"}
            if name == "source_index"
            else {"path", "sha256", "rows"}
        )
        if (
            not isinstance(spec, dict)
            or set(spec) != expected_fields
            or type(spec.get("path")) is not str
            or spec.get("sha256") != expected_sha
            or (
                "rows" in expected_fields
                and (type(spec.get("rows")) is not int or spec["rows"] < 1)
            )
        ):
            raise VisualCoordinateBindingError(f"visual activity {name} pin changed")
    if type(inputs.get("task_image_dir")) is not str:
        raise VisualCoordinateBindingError("visual activity task image directory is malformed")
    profiles = value.get("profiles")
    sift = profiles.get("sift") if isinstance(profiles, dict) else None
    expected_sift = {
        "render_dpi": 144,
        "nfeatures": 12_000,
        "contrast_threshold": 0.02,
        "edge_threshold": 12.0,
        "ratio_test": 0.72,
        "ransac_reprojection_px": 4.0,
        "ransac_max_iters": 5_000,
        "ransac_confidence": 0.999,
        "rng_seed": 19_870_511,
        "expected_opencv_version": "5.0.0",
    }
    if not isinstance(profiles, dict) or set(profiles) != {"sift"} or sift != expected_sift:
        raise VisualCoordinateBindingError("visual activity SIFT profile changed")
    runtime = value.get("runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime)
        != {
            "python",
            "opencv",
            "numpy",
            "poppler",
            "package_root",
            "sift_process_workers",
        }
        or runtime.get("package_root")
        != "tmp/portfolio_official_sources/python_pkgs"
        or type(runtime.get("sift_process_workers")) is not int
        or runtime.get("sift_process_workers") != 4
        or not isinstance(runtime.get("python"), dict)
        or set(runtime["python"]) != {"executable", "version"}
        or type(runtime["python"].get("executable")) is not str
        or runtime["python"].get("version") != "3.12.13"
        or not isinstance(runtime.get("opencv"), dict)
        or set(runtime["opencv"])
        != {"module_path", "opencl_enabled", "threads", "version"}
        or type(runtime["opencv"].get("module_path")) is not str
        or runtime["opencv"].get("version") != "5.0.0"
        or runtime["opencv"].get("opencl_enabled") is not False
        or type(runtime["opencv"].get("threads")) is not int
        or runtime["opencv"].get("threads") != 1
        or not isinstance(runtime.get("numpy"), dict)
        or set(runtime["numpy"]) != {"module_path", "version"}
        or type(runtime["numpy"].get("module_path")) is not str
        or runtime["numpy"].get("version") != "2.5.1"
        or not isinstance(runtime.get("poppler"), dict)
        or set(runtime["poppler"]) != {"pdfinfo", "pdftoppm"}
    ):
        raise VisualCoordinateBindingError("visual activity generation runtime changed")
    for tool_name in ("pdfinfo", "pdftoppm"):
        tool = runtime["poppler"].get(tool_name)
        if (
            not isinstance(tool, dict)
            or set(tool) != {"path", "sha256", "version", "version_line"}
            or type(tool.get("path")) is not str
            or _HEX64.fullmatch(str(tool.get("sha256") or "")) is None
            or type(tool.get("version")) is not str
            or type(tool.get("version_line")) is not str
        ):
            raise VisualCoordinateBindingError(
                f"visual activity {tool_name} runtime pin changed"
            )

    raw_records = value.get("activity_records")
    if not isinstance(raw_records, list) or not raw_records:
        raise VisualCoordinateBindingError("visual activity records are absent")
    expected_record_fields = {
        "document_id",
        "record_id",
        "content_page_number",
        "activity_number",
        "key_binding_kind",
        "question_marker_kind",
        "content_bbox",
        "key_projection_sha256",
        "content_projection_sha256",
        "binding_projection_sha256",
        "visually_checked",
    }
    artifact_refs: list[ActivityVisualRecordRef] = []
    for raw in raw_records:
        if not isinstance(raw, dict) or set(raw) != expected_record_fields:
            raise VisualCoordinateBindingError("visual activity record fields changed")
        bbox = raw.get("content_bbox")
        if (
            type(raw.get("document_id")) is not str
            or type(raw.get("record_id")) is not str
            or type(raw.get("content_page_number")) is not int
            or type(raw.get("activity_number")) is not int
            or type(raw.get("key_projection_sha256")) is not str
            or type(raw.get("content_projection_sha256")) is not str
            or type(raw.get("binding_projection_sha256")) is not str
            or type(raw.get("visually_checked")) is not bool
            or raw.get("key_binding_kind") != "activity_answer_key"
            or raw.get("question_marker_kind") != "activity_label"
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(item)
                for item in bbox
            )
        ):
            raise VisualCoordinateBindingError("visual activity record is not a PDF crop")
        artifact_refs.append(
            ActivityVisualRecordRef(
                document_id=str(raw["document_id"]),
                record_id=str(raw["record_id"]),
                content_page_number=int(raw["content_page_number"]),
                activity_number=int(raw["activity_number"]),
                key_projection_sha256=str(raw["key_projection_sha256"]),
                content_projection_sha256=str(raw["content_projection_sha256"]),
                binding_projection_sha256=str(raw["binding_projection_sha256"]),
                visually_checked=raw["visually_checked"] is True,
                content_bbox=tuple(float(item) for item in bbox),
            )
        )
    record_key = lambda item: (
        item.document_id,
        item.record_id,
        item.content_page_number,
        item.activity_number,
        item.key_projection_sha256,
        item.content_projection_sha256,
        item.binding_projection_sha256,
        item.visually_checked,
        item.content_bbox,
    )
    if sorted(map(record_key, artifact_refs)) != sorted(map(record_key, records)):
        raise VisualCoordinateBindingError("visual activity records differ from source index")

    raw_documents = inputs.get("documents")
    if not isinstance(raw_documents, dict) or not raw_documents:
        raise VisualCoordinateBindingError("visual activity document pins are absent")
    pages_by_document: dict[str, set[int]] = {}
    render_hashes: dict[tuple[str, int], str] = {}
    refs_by_document: dict[str, list[ActivityVisualRecordRef]] = {}
    for record in records:
        refs_by_document.setdefault(record.document_id, []).append(record)
    if set(raw_documents) != set(refs_by_document):
        raise VisualCoordinateBindingError("visual activity document set changed")
    for document_id, spec in raw_documents.items():
        if not isinstance(spec, dict) or set(spec) != {
            "candidate_content_pages",
            "content_page_ranges",
            "page_count",
            "pdf_path",
            "pdf_sha256",
            "rendered_pages",
        }:
            raise VisualCoordinateBindingError("visual activity document pin is malformed")
        pdf_path = document_pdf_paths.get(document_id)
        if pdf_path is None or _sha256_file(pdf_path) != str(spec.get("pdf_sha256") or ""):
            raise VisualCoordinateBindingError("visual activity PDF bytes changed")
        page_count = spec.get("page_count")
        content_ranges = spec.get("content_page_ranges")
        if (
            type(page_count) is not int
            or page_count < 1
            or not isinstance(content_ranges, list)
            or not content_ranges
            or any(
                not isinstance(page_range, list)
                or len(page_range) != 2
                or any(type(bound) is not int for bound in page_range)
                or page_range[0] < 1
                or page_range[0] > page_range[1]
                or page_range[1] > page_count
                for page_range in content_ranges
            )
        ):
            raise VisualCoordinateBindingError("visual activity content ranges changed")
        expanded_pages = [
            page
            for start, end in content_ranges
            for page in range(start, end + 1)
        ]
        candidate_pages = spec.get("candidate_content_pages")
        if (
            not isinstance(candidate_pages, list)
            or any(type(page) is not int for page in candidate_pages)
            or candidate_pages != expanded_pages
            or candidate_pages != sorted(set(candidate_pages))
        ):
            raise VisualCoordinateBindingError("visual activity page inventory changed")
        pages = set(candidate_pages)
        if any(
            record.content_page_number not in pages
            for record in refs_by_document[document_id]
        ):
            raise VisualCoordinateBindingError(
                "visual activity record falls outside rendered content pages"
            )
        rendered_pages = spec.get("rendered_pages")
        if (
            not isinstance(rendered_pages, dict)
            or any(
                type(page) is not str
                or not page.isdigit()
                or str(int(page)) != page
                for page in rendered_pages
            )
            or {int(page) for page in rendered_pages} != pages
        ):
            raise VisualCoordinateBindingError("visual activity render inventory changed")
        for page, render_spec in rendered_pages.items():
            if (
                not isinstance(render_spec, dict)
                or set(render_spec) != {"width", "height", "rendered_page_sha256"}
                or type(render_spec.get("width")) is not int
                or type(render_spec.get("height")) is not int
                or render_spec["width"] < 1
                or render_spec["height"] < 1
                or _HEX64.fullmatch(str(render_spec.get("rendered_page_sha256") or "")) is None
            ):
                raise VisualCoordinateBindingError("visual activity render pin is malformed")
            render_hashes[(document_id, int(page))] = str(render_spec["rendered_page_sha256"])
        pages_by_document[document_id] = pages

    raw_bindings = value.get("bindings_by_task_image_sha256")
    if not isinstance(raw_bindings, dict) or not raw_bindings:
        raise VisualCoordinateBindingError("visual activity evidence bindings are absent")
    verified: dict[str, VerifiedActivityVisualBinding] = {}
    artifact_task_ids: set[str] = set()
    expected_binding_fields = {
        "alignment_audit",
        "document_id",
        "observed_source_marker",
        "parser_identity",
        "raw_page_evidences",
        "source_pins",
        "task_image",
    }
    for image_sha, raw in raw_bindings.items():
        if (
            not isinstance(raw, dict)
            or set(raw) != expected_binding_fields
            or _HEX64.fullmatch(str(image_sha)) is None
        ):
            raise VisualCoordinateBindingError("visual activity binding fields changed")
        alignment = raw.get("alignment_audit")
        if not isinstance(alignment, dict) or set(alignment) != {
            "task_id",
            "task_id_role",
            "task_id_used_as_page_or_record_feature",
        }:
            raise VisualCoordinateBindingError("visual activity alignment audit changed")
        task_id = str(alignment.get("task_id") or "")
        observation = observations_by_task_id.get(task_id)
        if (
            observation is None
            or task_id in artifact_task_ids
            or alignment.get("task_id_role")
            != "parser_to_public_source_locator_alignment_only"
            or alignment.get("task_id_used_as_page_or_record_feature") is not False
        ):
            raise VisualCoordinateBindingError("visual activity alignment is not source-only")
        artifact_task_ids.add(task_id)
        marker = raw.get("observed_source_marker")
        task_image = raw.get("task_image")
        source_pins = raw.get("source_pins")
        if (
            not isinstance(marker, dict)
            or set(marker) != {"kind", "number"}
            or not isinstance(task_image, dict)
            or set(task_image) != {"path", "image_basename", "image_sha256", "width", "height"}
            or not isinstance(source_pins, dict)
            or set(source_pins) != {
                "parser_artifact_sha256",
                "parser_projection_sha256",
                "pdf_sha256",
                "source_identity_projection_sha256",
                "source_index_sha256",
                "source_locators_artifact_sha256",
            }
        ):
            raise VisualCoordinateBindingError("visual activity binding pins are malformed")
        document_id = str(raw.get("document_id") or "")
        if (
            type(task_image.get("path")) is not str
            or type(task_image.get("image_basename")) is not str
            or type(task_image.get("image_sha256")) is not str
            or type(task_image.get("width")) is not int
            or type(task_image.get("height")) is not int
            or type(marker.get("kind")) is not str
            or type(marker.get("number")) is not int
            or any(type(source_pins.get(name)) is not str for name in source_pins)
            or str(image_sha) != observation.task_image_sha256
            or task_image.get("image_sha256") != image_sha
            or task_image.get("width") != observation.width
            or task_image.get("height") != observation.height
            or raw.get("parser_identity") != observation.parser_identity
            or document_id != observation.document_id
            or marker.get("kind") != observation.marker_kind
            or marker.get("number") != observation.marker_number
            or source_pins.get("parser_artifact_sha256") != expected_parser_sha256
            or source_pins.get("source_locators_artifact_sha256")
            != expected_source_locators_sha256
            or source_pins.get("source_index_sha256") != expected_source_index_sha256
            or source_pins.get("pdf_sha256") != observation.pdf_sha256
            or any(
                _HEX64.fullmatch(str(source_pins.get(name) or "")) is None
                for name in ("parser_projection_sha256", "source_identity_projection_sha256")
            )
        ):
            raise VisualCoordinateBindingError("visual activity observation/source pins changed")
        image_path_raw = str(task_image.get("path") or "")
        image_path = Path(image_path_raw)
        resolved_image_path = (repo_root / image_path).resolve()
        try:
            resolved_image_path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise VisualCoordinateBindingError("visual activity image escapes repository") from exc
        if (
            image_path.is_absolute()
            or resolved_image_path.name != task_image.get("image_basename")
            or _sha256_file(resolved_image_path) != image_sha
        ):
            raise VisualCoordinateBindingError("visual activity image bytes changed")
        raw_evidences = raw.get("raw_page_evidences")
        if not isinstance(raw_evidences, list):
            raise VisualCoordinateBindingError("visual activity page evidence is absent")
        evidences = tuple(visual_page_evidence_from_mapping(item) for item in raw_evidences)
        if (
            len(evidences) != len(pages_by_document.get(document_id, ()))
            or {item.page_number for item in evidences}
            != pages_by_document.get(document_id)
            or any(
                item.rendered_page_sha256
                != render_hashes.get((document_id, item.page_number))
                for item in evidences
            )
        ):
            raise VisualCoordinateBindingError("visual activity evidence page pins changed")
        binding = verify_activity_visual_binding(
            evidences,
            refs_by_document.get(document_id, ()),
            expected_task_image_sha256=image_sha,
            expected_document_id=document_id,
            expected_pdf_sha256=observation.pdf_sha256,
            observed_activity_number=observation.marker_number,
            thresholds=thresholds,
        )
        if binding is None or image_sha in verified:
            raise VisualCoordinateBindingError("visual activity evidence does not pass policy")
        verified[image_sha] = binding
    expected_task_ids = {
        task_id
        for task_id, observation in observations_by_task_id.items()
        if observation.document_id in refs_by_document
    }
    if artifact_task_ids != expected_task_ids:
        raise VisualCoordinateBindingError(
            "visual activity artifact does not cover every eligible observation"
        )
    summary = value.get("summary")
    if summary != {
        "activity_documents": len(refs_by_document),
        "activity_observations": len(verified),
        "activity_records": len(records),
        "decision_layer": "not_applied_raw_evidence_only",
        "raw_page_evidences": sum(len(item.evidences) for item in verified.values()),
    }:
        raise VisualCoordinateBindingError("visual activity summary counters changed")
    return verified


def decide_visual_activity_binding(
    evidences: Sequence[VisualPageEvidence],
    records: Sequence[ActivityVisualRecordRef],
    *,
    expected_task_image_sha256: str,
    expected_document_id: str,
    expected_pdf_sha256: str,
    observed_activity_number: int,
    thresholds: VisualBindingThresholds = VisualBindingThresholds(),
) -> VisualBindingDecision:
    """Select one PDF-attested activity record from strong visual geometry."""

    if not evidences:
        return VisualBindingDecision(False, "no_page_evidence", ())
    page_addresses = [
        (evidence.document_id, evidence.page_number) for evidence in evidences
    ]
    if len(page_addresses) != len(set(page_addresses)):
        return VisualBindingDecision(
            False,
            "duplicate_page_evidence",
            (("unique_page_evidence", False),),
        )
    if (
        _HEX64.fullmatch(expected_task_image_sha256) is None
        or _HEX64.fullmatch(expected_pdf_sha256) is None
        or observed_activity_number < 1
    ):
        raise VisualCoordinateBindingError(
            "expected visual activity identity is malformed"
        )
    identity_ok = all(
        evidence.task_image_sha256 == expected_task_image_sha256
        and evidence.document_id == expected_document_id
        and evidence.pdf_sha256 == expected_pdf_sha256
        for evidence in evidences
    )
    if not identity_ok:
        return VisualBindingDecision(
            False,
            "source_identity_mismatch",
            (("source_identity", False),),
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
    runner_score = ordered[1].rank_score if len(ordered) > 1 else 0.0
    score_margin = best.rank_score - runner_score
    score_ratio = best.rank_score / max(runner_score, 1e-9)
    checks = list(geometry_checks(best, thresholds))
    checks.extend(
        (
            ("source_identity", True),
            (
                "page_rank_margin",
                score_margin >= thresholds.min_rank_score_margin,
            ),
            (
                "page_rank_ratio",
                score_ratio >= thresholds.min_rank_score_ratio,
            ),
        )
    )
    if not all(passed for _, passed in checks):
        return VisualBindingDecision(
            False,
            "visual_geometry_or_margin_failed",
            tuple(checks),
            selected_page_number=best.page_number,
            best_rank_score=best.rank_score,
            runner_rank_score=runner_score,
        )
    page_activity_records = [
        record
        for record in records
        if record.document_id == expected_document_id
        and record.content_page_number == best.page_number
    ]
    matching_records = [
        record
        for record in page_activity_records
        if record.activity_number == observed_activity_number
        and record.visually_checked
    ]
    unique_page_activity = len(page_activity_records) == 1
    unique_record = len(matching_records) == 1
    crop_bbox_iou = (
        _mapped_crop_bbox_iou(best.mapped_polygon, matching_records[0].content_bbox)
        if unique_record
        else 0.0
    )
    checks.extend(
        (
            ("one_indexed_activity_on_page", unique_page_activity),
            ("observed_activity_marker_agrees", unique_record),
            (
                "pdf_projection_pins_present",
                unique_record
                and all(
                    _HEX64.fullmatch(value) is not None
                    for value in (
                        matching_records[0].key_projection_sha256,
                        matching_records[0].content_projection_sha256,
                        matching_records[0].binding_projection_sha256,
                    )
                ),
            ),
            ("mapped_crop_bbox_iou", crop_bbox_iou >= 0.80),
        )
    )
    accepted = all(passed for _, passed in checks)
    selected = matching_records[0] if accepted else None
    return VisualBindingDecision(
        accepted,
        "accepted" if accepted else "activity_record_binding_failed",
        tuple(checks),
        selected_page_number=best.page_number,
        selected_question_number=(
            observed_activity_number if accepted else None
        ),
        selected_record_id=selected.record_id if selected else None,
        best_rank_score=best.rank_score,
        runner_rank_score=runner_score,
    )


def decide_visual_activity_page_binding(
    evidences: Sequence[VisualPageEvidence],
    records: Sequence[ActivityVisualRecordRef],
    *,
    expected_task_image_sha256: str,
    expected_document_id: str,
    expected_pdf_sha256: str,
    thresholds: VisualBindingThresholds = VisualBindingThresholds(),
) -> VisualBindingDecision:
    """Bind an image-only parser observation to one activity by exact page.

    This is deliberately narrower than inferring an activity number from task
    identity or benchmark position.  The visual sweep must first select one
    source page with the normal strict geometry and rank-margin gates.  That
    page must then contain exactly one indexed, visually reviewed activity
    record, and the mapped task crop must overlap that record's PDF content
    crop.  Ambiguous pages abstain.
    """

    if not evidences:
        return VisualBindingDecision(False, "no_page_evidence", ())
    page_addresses = [
        (evidence.document_id, evidence.page_number) for evidence in evidences
    ]
    if len(page_addresses) != len(set(page_addresses)):
        return VisualBindingDecision(
            False,
            "duplicate_page_evidence",
            (("unique_page_evidence", False),),
        )
    if (
        _HEX64.fullmatch(expected_task_image_sha256) is None
        or _HEX64.fullmatch(expected_pdf_sha256) is None
    ):
        raise VisualCoordinateBindingError(
            "expected image-only visual activity identity is malformed"
        )
    identity_ok = all(
        evidence.task_image_sha256 == expected_task_image_sha256
        and evidence.document_id == expected_document_id
        and evidence.pdf_sha256 == expected_pdf_sha256
        for evidence in evidences
    )
    if not identity_ok:
        return VisualBindingDecision(
            False,
            "source_identity_mismatch",
            (("source_identity", False),),
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
    runner_score = ordered[1].rank_score if len(ordered) > 1 else 0.0
    score_margin = best.rank_score - runner_score
    score_ratio = best.rank_score / max(runner_score, 1e-9)
    checks = list(geometry_checks(best, thresholds))
    checks.extend(
        (
            ("source_identity", True),
            (
                "page_rank_margin",
                score_margin >= thresholds.min_rank_score_margin,
            ),
            (
                "page_rank_ratio",
                score_ratio >= thresholds.min_rank_score_ratio,
            ),
        )
    )
    if not all(passed for _, passed in checks):
        return VisualBindingDecision(
            False,
            "visual_geometry_or_margin_failed",
            tuple(checks),
            selected_page_number=best.page_number,
            best_rank_score=best.rank_score,
            runner_rank_score=runner_score,
        )

    page_records = [
        record
        for record in records
        if record.document_id == expected_document_id
        and record.content_page_number == best.page_number
    ]
    reviewed_records = [record for record in page_records if record.visually_checked]
    unique_page_activity = len(page_records) == 1
    unique_reviewed_record = len(reviewed_records) == 1 and unique_page_activity
    selected = reviewed_records[0] if unique_reviewed_record else None
    crop_bbox_iou = (
        _mapped_crop_bbox_iou(best.mapped_polygon, selected.content_bbox)
        if selected is not None
        else 0.0
    )
    checks.extend(
        (
            ("one_indexed_activity_on_page", unique_page_activity),
            ("one_visually_reviewed_activity_on_page", unique_reviewed_record),
            (
                "pdf_projection_pins_present",
                selected is not None
                and all(
                    _HEX64.fullmatch(value) is not None
                    for value in (
                        selected.key_projection_sha256,
                        selected.content_projection_sha256,
                        selected.binding_projection_sha256,
                    )
                ),
            ),
            ("mapped_crop_bbox_iou", crop_bbox_iou >= 0.80),
        )
    )
    accepted = all(passed for _, passed in checks)
    return VisualBindingDecision(
        accepted,
        "accepted" if accepted else "activity_record_binding_failed",
        tuple(checks),
        selected_page_number=best.page_number,
        selected_question_number=(selected.activity_number if accepted and selected else None),
        selected_record_id=(selected.record_id if accepted and selected else None),
        best_rank_score=best.rank_score,
        runner_rank_score=runner_score,
    )


def _mapped_crop_bbox_iou(
    polygon: Sequence[Point] | None,
    pdf_bbox: tuple[float, float, float, float],
    *,
    render_dpi: int = 144,
) -> float:
    """Conservative axis-aligned overlap between mapped crop and PDF crop."""

    if polygon is None or len(polygon) != 4 or render_dpi <= 0:
        return 0.0
    scale = render_dpi / 72.0
    mapped = (
        min(point[0] for point in polygon),
        min(point[1] for point in polygon),
        max(point[0] for point in polygon),
        max(point[1] for point in polygon),
    )
    source = tuple(float(value) * scale for value in pdf_bbox)
    intersection_width = max(0.0, min(mapped[2], source[2]) - max(mapped[0], source[0]))
    intersection_height = max(0.0, min(mapped[3], source[3]) - max(mapped[1], source[1]))
    intersection = intersection_width * intersection_height
    mapped_area = max(0.0, mapped[2] - mapped[0]) * max(0.0, mapped[3] - mapped[1])
    source_area = max(0.0, source[2] - source[0]) * max(0.0, source[3] - source[1])
    union = mapped_area + source_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _point_in_convex_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) != 4:
        return False
    signs: list[float] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        cross = (end[0] - start[0]) * (point[1] - start[1]) - (
            end[1] - start[1]
        ) * (point[0] - start[0])
        if abs(cross) > 1e-7:
            signs.append(cross)
    return bool(signs) and (all(value > 0.0 for value in signs) or all(value < 0.0 for value in signs))


def geometry_checks(
    evidence: VisualPageEvidence,
    thresholds: VisualBindingThresholds,
) -> tuple[tuple[str, bool], ...]:
    return (
        ("good_matches", evidence.good_matches >= thresholds.min_good_matches),
        ("inliers", evidence.inliers >= thresholds.min_inliers),
        ("inlier_ratio", evidence.inlier_ratio >= thresholds.min_inlier_ratio),
        (
            "task_hull_fraction",
            evidence.task_hull_fraction >= thresholds.min_task_hull_fraction,
        ),
        (
            "median_reprojection_error",
            evidence.median_reprojection_error is not None
            and evidence.median_reprojection_error
            <= thresholds.max_median_reprojection_error,
        ),
        (
            "mapped_inside_fraction",
            evidence.mapped_inside_fraction >= thresholds.min_mapped_inside_fraction,
        ),
        (
            "scale_anisotropy",
            evidence.scale_anisotropy is not None
            and evidence.scale_anisotropy <= thresholds.max_scale_anisotropy,
        ),
        ("orientation_preserved", evidence.orientation_preserved),
        ("convex_mapping", evidence.convex_mapping),
        ("mapped_polygon", evidence.mapped_polygon is not None),
    )


def decide_visual_binding(
    evidences: Sequence[VisualPageEvidence],
    markers: Sequence[PdfQuestionMarker],
    records: Sequence[IndexedQuestionRef],
    *,
    expected_task_image_sha256: str,
    expected_document_id: str,
    expected_pdf_sha256: str,
    observed_question_number: int | None,
    thresholds: VisualBindingThresholds = VisualBindingThresholds(),
) -> VisualBindingDecision:
    """Return one certifiable source record, or abstain without an answer."""

    if not evidences:
        return VisualBindingDecision(False, "no_page_evidence", ())
    if _HEX64.fullmatch(expected_task_image_sha256) is None or _HEX64.fullmatch(expected_pdf_sha256) is None:
        raise VisualCoordinateBindingError("expected source pins are malformed")
    if observed_question_number is not None and not 1 <= observed_question_number <= 999:
        raise VisualCoordinateBindingError("observed question number is invalid")
    identity_ok = all(
        item.task_image_sha256 == expected_task_image_sha256
        and item.document_id == expected_document_id
        and item.pdf_sha256 == expected_pdf_sha256
        for item in evidences
    )
    if not identity_ok:
        return VisualBindingDecision(False, "source_identity_mismatch", (("source_identity", False),))

    ordered = sorted(
        evidences,
        key=lambda item: (-item.rank_score, -item.inliers, item.page_number, item.rendered_page_sha256),
    )
    best = ordered[0]
    runner_score = ordered[1].rank_score if len(ordered) > 1 else 0.0
    score_margin = best.rank_score - runner_score
    score_ratio = best.rank_score / max(runner_score, 1e-9)
    checks = list(geometry_checks(best, thresholds))
    checks.extend(
        (
            ("source_identity", True),
            ("page_rank_margin", score_margin >= thresholds.min_rank_score_margin),
            ("page_rank_ratio", score_ratio >= thresholds.min_rank_score_ratio),
        )
    )
    if not all(passed for _, passed in checks):
        return VisualBindingDecision(
            False,
            "visual_geometry_or_margin_failed",
            tuple(checks),
            selected_page_number=best.page_number,
            best_rank_score=best.rank_score,
            runner_rank_score=runner_score,
        )

    page_records = [
        record
        for record in records
        if record.document_id == expected_document_id
        and record.content_page_number == best.page_number
    ]
    indexed_numbers = {record.question_number for record in page_records}
    polygon = best.mapped_polygon
    assert polygon is not None
    inside = [
        marker
        for marker in markers
        if marker.page_number == best.page_number
        and marker.question_number in indexed_numbers
        and _point_in_convex_polygon(marker.center, polygon)
    ]
    inside_numbers = {marker.question_number for marker in inside}
    unique_marker = len(inside) == 1 and len(inside_numbers) == 1
    checks.append(("exactly_one_indexed_pdf_marker_inside", unique_marker))
    if not unique_marker:
        return VisualBindingDecision(
            False,
            "indexed_pdf_marker_ambiguous_or_absent",
            tuple(checks),
            selected_page_number=best.page_number,
            best_rank_score=best.rank_score,
            runner_rank_score=runner_score,
        )

    selected_number = next(iter(inside_numbers))
    observed_agrees = observed_question_number is None or observed_question_number == selected_number
    checks.append(("observed_number_agrees", observed_agrees))
    selected_records = [record for record in page_records if record.question_number == selected_number]
    unique_record = len(selected_records) == 1
    checks.append(("unique_source_record", unique_record))
    certifiable_key = unique_record and selected_records[0].key_is_certifiable
    checks.append(("certifiable_reviewed_key", certifiable_key))
    accepted = observed_agrees and unique_record and certifiable_key
    selected_record = selected_records[0] if unique_record else None
    return VisualBindingDecision(
        accepted,
        "accepted" if accepted else "question_or_key_binding_failed",
        tuple(checks),
        selected_page_number=best.page_number,
        selected_question_number=selected_number,
        selected_record_id=selected_record.record_id if accepted and selected_record else None,
        best_rank_score=best.rank_score,
        runner_rank_score=runner_score,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_sift_page_evidence(
    task_image_path: Path,
    rendered_page_path: Path,
    *,
    task_image_sha256: str,
    document_id: str,
    pdf_sha256: str,
    page_number: int,
    profile: SiftRuntimeProfile = SiftRuntimeProfile(),
) -> VisualPageEvidence:
    """Compute pinned SIFT/RANSAC evidence for one task-image/page pair.

    Rendering is intentionally outside this function: the caller must pin the
    PDF hash and fixed-DPI Poppler render profile before presenting page bytes.
    """

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise VisualCoordinateBindingError(
            "OpenCV and NumPy are required to compute visual evidence"
        ) from exc
    if profile.expected_opencv_version is not None and cv2.__version__ != profile.expected_opencv_version:
        raise VisualCoordinateBindingError("OpenCV runtime does not match the pinned profile")
    if _sha256_file(task_image_path) != task_image_sha256:
        raise VisualCoordinateBindingError("task image bytes do not match their parser pin")
    if _HEX64.fullmatch(pdf_sha256) is None:
        raise VisualCoordinateBindingError("PDF pin is malformed")
    rendered_sha = _sha256_file(rendered_page_path)
    task_gray = cv2.imread(str(task_image_path), cv2.IMREAD_GRAYSCALE)
    page_gray = cv2.imread(str(rendered_page_path), cv2.IMREAD_GRAYSCALE)
    if task_gray is None or page_gray is None:
        raise VisualCoordinateBindingError("task/page image cannot be decoded")
    cv2.setRNGSeed(profile.rng_seed)
    sift = cv2.SIFT_create(
        nfeatures=profile.nfeatures,
        contrastThreshold=profile.contrast_threshold,
        edgeThreshold=profile.edge_threshold,
    )
    task_kp, task_desc = sift.detectAndCompute(task_gray, None)
    page_kp, page_desc = sift.detectAndCompute(page_gray, None)
    empty = dict(
        task_image_sha256=task_image_sha256,
        document_id=document_id,
        pdf_sha256=pdf_sha256,
        page_number=page_number,
        rendered_page_sha256=rendered_sha,
        good_matches=0,
        inliers=0,
        inlier_ratio=0.0,
        task_hull_fraction=0.0,
        median_reprojection_error=None,
        mapped_inside_fraction=0.0,
        scale_anisotropy=None,
        orientation_preserved=False,
        convex_mapping=False,
        mapped_polygon=None,
    )
    if task_desc is None or page_desc is None or len(task_desc) < 4 or len(page_desc) < 4:
        return VisualPageEvidence(**empty)
    pairs = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False).knnMatch(task_desc, page_desc, k=2)
    good = [first for first, second in pairs if first.distance < profile.ratio_test * second.distance]
    empty["good_matches"] = len(good)
    if len(good) < 4:
        return VisualPageEvidence(**empty)
    source_points = np.float32([task_kp[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    target_points = np.float32([page_kp[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        profile.ransac_reprojection_px,
        maxIters=profile.ransac_max_iters,
        confidence=profile.ransac_confidence,
    )
    if homography is None or mask is None or not np.isfinite(homography).all():
        return VisualPageEvidence(**empty)
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    empty["inliers"] = inliers
    empty["inlier_ratio"] = inliers / len(good)
    if inliers < 4:
        return VisualPageEvidence(**empty)
    task_h, task_w = task_gray.shape[:2]
    page_h, page_w = page_gray.shape[:2]
    inlier_source = source_points.reshape(-1, 2)[inlier_mask]
    source_projected = cv2.perspectiveTransform(source_points[inlier_mask], homography).reshape(-1, 2)
    target_actual = target_points.reshape(-1, 2)[inlier_mask]
    errors = np.linalg.norm(source_projected - target_actual, axis=1)
    corners = np.float32(
        [[[0.0, 0.0]], [[task_w - 1.0, 0.0]], [[task_w - 1.0, task_h - 1.0]], [[0.0, task_h - 1.0]]]
    )
    mapped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    mapped_area = abs(float(cv2.contourArea(mapped.astype(np.float32))))
    hull_area = abs(float(cv2.contourArea(cv2.convexHull(inlier_source.astype(np.float32)))))
    page_polygon = np.asarray(
        [[0.0, 0.0], [page_w - 1.0, 0.0], [page_w - 1.0, page_h - 1.0], [0.0, page_h - 1.0]],
        dtype=np.float32,
    )
    convex = bool(cv2.isContourConvex(mapped.astype(np.float32)))
    try:
        intersection_area, _ = cv2.intersectConvexConvex(mapped.astype(np.float32), page_polygon)
    except cv2.error:
        intersection_area = 0.0
    inside = float(intersection_area) / mapped_area if convex and mapped_area > 1.0 else 0.0
    top_edge = mapped[1] - mapped[0]
    left_edge = mapped[3] - mapped[0]
    orientation = float(top_edge[0] * left_edge[1] - top_edge[1] * left_edge[0]) > 0.0
    edge_widths = [np.linalg.norm(mapped[1] - mapped[0]), np.linalg.norm(mapped[2] - mapped[3])]
    edge_heights = [np.linalg.norm(mapped[3] - mapped[0]), np.linalg.norm(mapped[2] - mapped[1])]
    scale_x = float(np.median(edge_widths)) / task_w
    scale_y = float(np.median(edge_heights)) / task_h
    anisotropy = max(scale_x, scale_y) / max(min(scale_x, scale_y), 1e-9)
    return VisualPageEvidence(
        task_image_sha256=task_image_sha256,
        document_id=document_id,
        pdf_sha256=pdf_sha256,
        page_number=page_number,
        rendered_page_sha256=rendered_sha,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=inliers / len(good),
        task_hull_fraction=hull_area / float(task_w * task_h),
        median_reprojection_error=float(np.median(errors)),
        mapped_inside_fraction=max(0.0, min(1.0, inside)),
        scale_anisotropy=float(anisotropy),
        orientation_preserved=orientation,
        convex_mapping=convex,
        mapped_polygon=tuple((float(point[0]), float(point[1])) for point in mapped),
    )


def unique_indexed_markers(
    raw_words: Iterable[dict[str, Any]],
    *,
    page_number: int,
    indexed_question_numbers: set[int],
    render_width: int,
    render_height: int,
    pdf_width: float,
    pdf_height: float,
) -> tuple[PdfQuestionMarker, ...]:
    """Project exact ``N.``/``N)`` PDF words into rendered-page pixels."""

    if min(render_width, render_height) <= 0 or min(pdf_width, pdf_height) <= 0.0:
        raise VisualCoordinateBindingError("page dimensions are invalid")
    marker_pattern = re.compile(r"^(\d{1,3})[.)]$")
    scale_x = render_width / pdf_width
    scale_y = render_height / pdf_height
    markers: list[PdfQuestionMarker] = []
    for word in raw_words:
        token = str(word.get("text") or "").strip()
        match = marker_pattern.fullmatch(token)
        if match is None:
            continue
        number = int(match.group(1))
        if number not in indexed_question_numbers:
            continue
        center = (
            ((float(word["x0"]) + float(word["x1"])) / 2.0) * scale_x,
            ((float(word["top"]) + float(word["bottom"])) / 2.0) * scale_y,
        )
        markers.append(PdfQuestionMarker(page_number, number, center, token))
    return tuple(markers)
