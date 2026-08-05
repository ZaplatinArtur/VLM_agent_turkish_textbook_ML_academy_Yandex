"""Gold-blind projection for one parser image with no textual blocks.

This module exists for workbook activities whose source page is fully visible
in the benchmark image but whose frozen parser emitted only one image block.
It deliberately exposes no task answer, task ordinal, inferred page, or
activity number.  Those are resolved later from public-source identity and a
full visual page sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .activity_answer_key import activity_marker_inventory
from .contracts import ProblemInput
from .official_ogm import (
    MatchResult,
    OfficialSourceError,
    _scan_observation_keys,
    canonical_json_sha256,
)
from .visual_coordinate_binding import (
    ActivityVisualRecordRef,
    VisualBindingDecision,
    VisualBindingThresholds,
    VisualCoordinateBindingError,
    VisualPageEvidence,
    decide_visual_activity_page_binding,
    load_activity_visual_artifact_json,
    require_strict_activity_visual_thresholds,
    visual_page_evidence_from_mapping,
)


SCHEMA = "maxim-paddleocr-vl16-task-parse-v1"
OBSERVATION_KIND = "single_full_page_image_block_without_text_v1"
MIN_BLOCK_AREA_COVERAGE = 0.90
ARTIFACT_SCHEMA = "maxim-image-only-activity-visual-binding-source-evidence-v1"
ARTIFACT_GENERATOR = "build-maxim-image-only-activity-visual-binding-v1"
RECORD_SELECTION_POLICY = (
    "unique_reviewed_record_and_unique_pdf_activity_marker_on_visual_page_v1"
)
IMAGE_ONLY_ACTIVITY_ARTIFACT_ROLE = (
    "answer_free_sift_ransac_unique_activity_page_binding_only"
)
IMAGE_ONLY_ACTIVITY_BINDING_VERIFIER = (
    "public-workbook-image-only-activity-visual-binding-v1"
)
IMAGE_ONLY_ACTIVITY_TRACE_SCHEMA = "public-workbook-source-trace-v1"
IMAGE_ONLY_ACTIVITY_TRACE_CHECKS = frozenset(
    {
        "strict_public_document_identity",
        "image_only_parser_observation",
        "visual_page_binding",
        "unique_source_activity_record",
        "unique_pdf_activity_marker_on_selected_page",
        "source_pdf_record_verified",
        "reviewed_embedded_key",
        "valid_source_answer",
        "source_address_not_task_id",
    }
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HTML_TAG = re.compile(r"<[^>]*>")


class ImageOnlyActivityError(ValueError):
    """A parser row is not one safe image-only activity observation."""


@dataclass(frozen=True, slots=True)
class ImageOnlyActivityObservation:
    task_id: str
    image_basename: str
    image_sha256: str
    width: int
    height: int
    parser_identity: str
    block_bbox: tuple[float, float, float, float]
    block_area_coverage: float
    parser_projection_sha256: str


def _canonical_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ImageOnlyActivityError(f"{label} is not a canonical integer")
    return value


def _plain_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImageOnlyActivityError(f"{label} is missing")
    return value.strip()


def _block_bbox(
    value: Any,
    *,
    width: int,
    height: int,
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
        raise ImageOnlyActivityError("image-only block bbox is malformed")
    bbox = tuple(float(item) for item in value)
    if not (
        0.0 <= bbox[0] < bbox[2] <= float(width)
        and 0.0 <= bbox[1] < bbox[3] <= float(height)
    ):
        raise ImageOnlyActivityError("image-only block bbox is outside the image")
    return bbox  # type: ignore[return-value]


def project_image_only_activity_observation(
    value: Mapping[str, Any],
) -> ImageOnlyActivityObservation:
    """Project one exact full-page image block, or reject it fail-closed."""

    try:
        _scan_observation_keys(value)
    except OfficialSourceError as exc:
        raise ImageOnlyActivityError(str(exc)) from exc
    if set(value) != {"schema_version", "task_id", "parser", "images"}:
        raise ImageOnlyActivityError("parser row fields changed")
    if value.get("schema_version") != SCHEMA:
        raise ImageOnlyActivityError("parser row schema changed")
    task_id = _plain_string(value.get("task_id"), "alignment task id")

    parser = value.get("parser")
    expected_parser_fields = {
        "pipeline_version",
        "layout_model",
        "recognition_model",
        "recognition_backend",
        "max_new_tokens",
        "gold_access",
    }
    if not isinstance(parser, Mapping) or set(parser) != expected_parser_fields:
        raise ImageOnlyActivityError("parser provenance fields changed")
    if parser.get("gold_access") is not False:
        raise ImageOnlyActivityError("parser provenance is not gold-blind")
    _canonical_integer(parser.get("max_new_tokens"), "max_new_tokens", minimum=1)
    parser_parts = tuple(
        _plain_string(parser.get(name), f"parser {name}")
        for name in ("pipeline_version", "layout_model", "recognition_model")
    )
    _plain_string(parser.get("recognition_backend"), "recognition backend")
    parser_identity = "/".join(parser_parts)

    images = value.get("images")
    if (
        not isinstance(images, Sequence)
        or isinstance(images, (str, bytes, bytearray))
        or len(images) != 1
        or not isinstance(images[0], Mapping)
    ):
        raise ImageOnlyActivityError("image-only projection requires exactly one image")
    image = images[0]
    expected_image_fields = {
        "image_index",
        "image_basename",
        "image_sha256",
        "width",
        "height",
        "input_decode",
        "parsing_res_list",
    }
    if set(image) != expected_image_fields:
        raise ImageOnlyActivityError("parser image fields changed")
    if _canonical_integer(image.get("image_index"), "image_index") != 0:
        raise ImageOnlyActivityError("the only parser image must have index zero")
    image_basename = _plain_string(image.get("image_basename"), "image basename")
    if (
        "/" in image_basename
        or "\\" in image_basename
        or not image_basename.casefold().endswith(".png")
    ):
        raise ImageOnlyActivityError("image basename is unsafe")
    image_sha256 = _plain_string(image.get("image_sha256"), "image SHA-256")
    if _HEX64.fullmatch(image_sha256) is None:
        raise ImageOnlyActivityError("parser image is not SHA-256 pinned")
    width = _canonical_integer(image.get("width"), "image width", minimum=1)
    height = _canonical_integer(image.get("height"), "image height", minimum=1)
    if image.get("input_decode") != {"kind": "path"}:
        raise ImageOnlyActivityError("parser input decode changed")

    blocks = image.get("parsing_res_list")
    if (
        not isinstance(blocks, Sequence)
        or isinstance(blocks, (str, bytes, bytearray))
        or len(blocks) != 1
        or not isinstance(blocks[0], Mapping)
    ):
        raise ImageOnlyActivityError(
            "parser row is not exactly one image block without text"
        )
    block = blocks[0]
    expected_block_fields = {
        "block_label",
        "block_content",
        "block_bbox",
        "block_id",
        "block_order",
        "group_id",
        "block_polygon_points",
    }
    if set(block) != expected_block_fields or block.get("block_label") != "image":
        raise ImageOnlyActivityError("the only parser block is not an image")
    if (
        _canonical_integer(block.get("block_id"), "block_id") != 0
        or _canonical_integer(block.get("group_id"), "group_id") != 0
        or block.get("block_order") is not None
    ):
        raise ImageOnlyActivityError("image-only block identity changed")
    block_content = _plain_string(block.get("block_content"), "image block content")
    if (
        len(re.findall(r"<img\b", block_content, flags=re.IGNORECASE)) != 1
        or _HTML_TAG.sub("", html.unescape(block_content)).strip()
    ):
        raise ImageOnlyActivityError("image block contains non-image text")
    bbox = _block_bbox(block.get("block_bbox"), width=width, height=height)
    coverage = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(width * height)
    if coverage < MIN_BLOCK_AREA_COVERAGE:
        raise ImageOnlyActivityError("image block does not cover the full task image")

    polygon = block.get("block_polygon_points")
    if (
        not isinstance(polygon, Sequence)
        or isinstance(polygon, (str, bytes, bytearray))
        or len(polygon) != 4
        or any(
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes, bytearray))
            or len(point) != 2
            or any(
                not isinstance(coordinate, (int, float))
                or isinstance(coordinate, bool)
                or not math.isfinite(float(coordinate))
                for coordinate in point
            )
            for point in polygon
        )
    ):
        raise ImageOnlyActivityError("image block polygon is malformed")

    projection = {
        "schema_version": SCHEMA,
        "observation_kind": OBSERVATION_KIND,
        "parser_identity": parser_identity,
        "image_sha256": image_sha256,
        "width": width,
        "height": height,
        "image_block": {
            "block_content_sha256": canonical_json_sha256(
                {"block_content": block_content}
            ),
            "block_bbox": [round(item, 3) for item in bbox],
            "block_polygon_points": [
                [round(float(coordinate), 3) for coordinate in point]
                for point in polygon
            ],
            "block_area_coverage": round(coverage, 12),
        },
    }
    return ImageOnlyActivityObservation(
        task_id=task_id,
        image_basename=image_basename,
        image_sha256=image_sha256,
        width=width,
        height=height,
        parser_identity=parser_identity,
        block_bbox=bbox,
        block_area_coverage=coverage,
        parser_projection_sha256=canonical_json_sha256(projection),
    )


@dataclass(frozen=True, slots=True)
class VerifiedImageOnlyActivityBinding:
    """Answer-free page/record decision replayed from frozen source evidence.

    ``alignment_task_id`` is retained only so callers can audit the parser to
    public-locator join.  It is deliberately omitted from :meth:`trace` and is
    never consulted by the page/record decision.
    """

    alignment_task_id: str
    observation: ImageOnlyActivityObservation
    document_id: str
    pdf_sha256: str
    source_identity_projection_sha256: str
    evidences: tuple[VisualPageEvidence, ...]
    decision: VisualBindingDecision
    marker_inventory: tuple[int, ...]
    marker_inventory_projection_sha256: str
    thresholds: VisualBindingThresholds
    verifier: str = IMAGE_ONLY_ACTIVITY_BINDING_VERIFIER

    def __post_init__(self) -> None:
        if (
            not self.alignment_task_id
            or self.alignment_task_id != self.observation.task_id
            or not self.document_id
            or _HEX64.fullmatch(self.pdf_sha256) is None
            or _HEX64.fullmatch(self.source_identity_projection_sha256) is None
            or _HEX64.fullmatch(self.marker_inventory_projection_sha256) is None
            or self.verifier != IMAGE_ONLY_ACTIVITY_BINDING_VERIFIER
            or not self.evidences
        ):
            raise ImageOnlyActivityError(
                "verified image-only activity identity is malformed"
            )
        if any(
            evidence.task_image_sha256 != self.observation.image_sha256
            or evidence.document_id != self.document_id
            or evidence.pdf_sha256 != self.pdf_sha256
            for evidence in self.evidences
        ):
            raise ImageOnlyActivityError(
                "image-only activity evidence identity differs"
            )
        if (
            not self.decision.accepted
            or not self.decision.selected_record_id
            or self.decision.selected_page_number is None
            or self.decision.selected_question_number is None
            or self.marker_inventory != (self.decision.selected_question_number,)
            or not self.decision.checks
            or any(not passed for _, passed in self.decision.checks)
        ):
            raise ImageOnlyActivityError(
                "image-only activity decision is not fully source-bound"
            )
        try:
            require_strict_activity_visual_thresholds(self.thresholds)
        except VisualCoordinateBindingError as exc:
            raise ImageOnlyActivityError(str(exc)) from exc

    def trace(self) -> dict[str, Any]:
        """Return the policy trace, intentionally excluding the task ID."""

        return {
            "verifier": self.verifier,
            "observation_kind": OBSERVATION_KIND,
            "parser_projection_sha256": self.observation.parser_projection_sha256,
            "task_image_sha256": self.observation.image_sha256,
            "document_id": self.document_id,
            "pdf_sha256": self.pdf_sha256,
            "source_identity_projection_sha256": (
                self.source_identity_projection_sha256
            ),
            "selected_page_number": self.decision.selected_page_number,
            "selected_question_number": self.decision.selected_question_number,
            "selected_record_id": self.decision.selected_record_id,
            "source_page_activity_marker_numbers": list(self.marker_inventory),
            "source_page_activity_marker_projection_sha256": (
                self.marker_inventory_projection_sha256
            ),
            "best_rank_score": self.decision.best_rank_score,
            "runner_rank_score": self.decision.runner_rank_score,
            "checks": {name: passed for name, passed in self.decision.checks},
            "thresholds": _threshold_projection(self.thresholds),
            "task_id_is_policy_feature": False,
        }


def load_image_only_activity_visual_artifact_json(path: Path) -> dict[str, Any]:
    """Load canonical JSON with the shared duplicate-key/NaN rejection."""

    try:
        return load_activity_visual_artifact_json(path)
    except VisualCoordinateBindingError as exc:
        raise ImageOnlyActivityError(str(exc)) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(repo_root: Path, raw_value: Any, label: str) -> Path:
    if type(raw_value) is not str or not raw_value:
        raise ImageOnlyActivityError(f"{label} path is malformed")
    raw = Path(raw_value)
    if raw.is_absolute():
        raise ImageOnlyActivityError(f"{label} path must be repository-relative")
    resolved = (repo_root / raw).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ImageOnlyActivityError(f"{label} path escapes the repository") from exc
    return resolved


def _threshold_projection(
    thresholds: VisualBindingThresholds,
) -> dict[str, Any]:
    return {
        "min_good_matches": thresholds.min_good_matches,
        "min_inliers": thresholds.min_inliers,
        "min_inlier_ratio": thresholds.min_inlier_ratio,
        "min_task_hull_fraction": thresholds.min_task_hull_fraction,
        "max_median_reprojection_error": (
            thresholds.max_median_reprojection_error
        ),
        "min_mapped_inside_fraction": thresholds.min_mapped_inside_fraction,
        "max_scale_anisotropy": thresholds.max_scale_anisotropy,
        "min_rank_score_margin": thresholds.min_rank_score_margin,
        "min_rank_score_ratio": thresholds.min_rank_score_ratio,
    }


def _sift_projection() -> dict[str, Any]:
    return {
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


def _decision_projection(decision: VisualBindingDecision) -> dict[str, Any]:
    return {
        "accepted": decision.accepted,
        "reason": decision.reason,
        "checks": [[name, passed] for name, passed in decision.checks],
        "selected_page_number": decision.selected_page_number,
        "selected_question_number": decision.selected_question_number,
        "selected_record_id": decision.selected_record_id,
        "best_rank_score": decision.best_rank_score,
        "runner_rank_score": decision.runner_rank_score,
    }


def _record_ref_from_mapping(value: Any) -> ActivityVisualRecordRef:
    expected_fields = {
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
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ImageOnlyActivityError("image-only activity record fields changed")
    bbox = value.get("content_bbox")
    if (
        type(value.get("document_id")) is not str
        or type(value.get("record_id")) is not str
        or type(value.get("content_page_number")) is not int
        or type(value.get("activity_number")) is not int
        or value.get("key_binding_kind") != "activity_answer_key"
        or value.get("question_marker_kind") != "activity_label"
        or type(value.get("key_projection_sha256")) is not str
        or type(value.get("content_projection_sha256")) is not str
        or type(value.get("binding_projection_sha256")) is not str
        or type(value.get("visually_checked")) is not bool
        or not isinstance(bbox, list)
        or len(bbox) != 4
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in bbox
        )
    ):
        raise ImageOnlyActivityError("image-only activity record is malformed")
    try:
        return ActivityVisualRecordRef(
            document_id=value["document_id"],
            record_id=value["record_id"],
            content_page_number=value["content_page_number"],
            activity_number=value["activity_number"],
            key_projection_sha256=value["key_projection_sha256"],
            content_projection_sha256=value["content_projection_sha256"],
            binding_projection_sha256=value["binding_projection_sha256"],
            visually_checked=value["visually_checked"],
            content_bbox=tuple(float(item) for item in bbox),
        )
    except VisualCoordinateBindingError as exc:
        raise ImageOnlyActivityError(str(exc)) from exc


def _record_key(record: ActivityVisualRecordRef) -> tuple[Any, ...]:
    return (
        record.document_id,
        record.record_id,
        record.content_page_number,
        record.activity_number,
        record.key_projection_sha256,
        record.content_projection_sha256,
        record.binding_projection_sha256,
        record.visually_checked,
        record.content_bbox,
    )


def verified_image_only_activity_bindings_from_artifact(
    value: Any,
    *,
    repo_root: Path,
    expected_parser_sha256: str,
    expected_source_locators_sha256: str,
    observations_by_task_id: Mapping[str, ImageOnlyActivityObservation],
    source_urls_by_task_id: Mapping[str, str],
    documents_by_id: Mapping[str, Any],
    records: Sequence[ActivityVisualRecordRef],
    document_pdf_paths: Mapping[str, Path],
    thresholds: VisualBindingThresholds = VisualBindingThresholds(),
) -> dict[str, VerifiedImageOnlyActivityBinding]:
    """Strictly replay the frozen page-only Activity evidence.

    The alignment key joins one parser row to its public source locator only.
    Page, activity number, and record selection are recomputed without it from
    the image hash, the complete page-evidence inventory, the official PDF,
    and answer-free source record projections.
    """

    repo_root = repo_root.resolve()
    try:
        require_strict_activity_visual_thresholds(thresholds)
    except VisualCoordinateBindingError as exc:
        raise ImageOnlyActivityError(str(exc)) from exc
    if thresholds != VisualBindingThresholds():
        raise ImageOnlyActivityError(
            "image-only activity replay requires frozen default thresholds"
        )
    expected_root_fields = {
        "activity_records",
        "bindings_by_task_image_sha256",
        "generator",
        "inputs",
        "profiles",
        "runtime",
        "schema_version",
        "source_only_guards",
        "summary",
    }
    if not isinstance(value, dict) or set(value) != expected_root_fields:
        raise ImageOnlyActivityError("image-only activity artifact root changed")
    if (
        value.get("schema_version") != ARTIFACT_SCHEMA
        or value.get("generator") != ARTIFACT_GENERATOR
    ):
        raise ImageOnlyActivityError("image-only activity artifact schema changed")
    if value.get("source_only_guards") != {
        "benchmark_answer_candidate_outcome_artifacts_read": False,
        "parser_observation_filter": OBSERVATION_KIND,
        "record_selection": RECORD_SELECTION_POLICY,
        "render_scope": (
            "all indexed physical content pages expanded from each activity "
            "document content_page_ranges"
        ),
        "source_answer_value_access": False,
        "source_index_record_filter": (
            "key_binding_kind=activity_answer_key AND "
            "question_marker_kind=activity_label"
        ),
        "task_id_is_policy_feature": False,
        "task_id_role": "alignment_audit_only",
    }:
        raise ImageOnlyActivityError("image-only source-only guards changed")

    inputs = value.get("inputs")
    expected_input_fields = {
        "profile",
        "parser_observations",
        "source_locators",
        "source_index",
        "task_image_dir",
        "documents",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_input_fields:
        raise ImageOnlyActivityError("image-only activity input pins changed")
    profile_spec = inputs.get("profile")
    parser_spec = inputs.get("parser_observations")
    locator_spec = inputs.get("source_locators")
    source_index_spec = inputs.get("source_index")
    if (
        not isinstance(profile_spec, dict)
        or set(profile_spec) != {"path", "sha256"}
        or not isinstance(parser_spec, dict)
        or set(parser_spec) != {"path", "sha256", "rows"}
        or not isinstance(locator_spec, dict)
        or set(locator_spec) != {"path", "sha256", "rows"}
        or not isinstance(source_index_spec, dict)
        or set(source_index_spec) != {"path", "sha256"}
        or parser_spec.get("sha256") != expected_parser_sha256
        or locator_spec.get("sha256") != expected_source_locators_sha256
        or _HEX64.fullmatch(str(source_index_spec.get("sha256") or "")) is None
        or type(parser_spec.get("rows")) is not int
        or type(locator_spec.get("rows")) is not int
        or parser_spec["rows"] < 1
        or locator_spec["rows"] < 1
    ):
        raise ImageOnlyActivityError("image-only activity primary pins changed")
    pinned_paths: dict[str, Path] = {}
    for name, spec in (
        ("profile", profile_spec),
        ("parser observations", parser_spec),
        ("source locators", locator_spec),
        ("source index", source_index_spec),
    ):
        path = _repo_path(repo_root, spec.get("path"), name)
        if not path.is_file() or _sha256_file(path) != spec.get("sha256"):
            raise ImageOnlyActivityError(f"image-only activity {name} bytes changed")
        pinned_paths[name] = path
    task_image_dir = _repo_path(
        repo_root, inputs.get("task_image_dir"), "task image directory"
    )
    if not task_image_dir.is_dir():
        raise ImageOnlyActivityError("image-only activity task image directory is absent")

    isolated_profile = load_image_only_activity_visual_artifact_json(
        pinned_paths["profile"]
    )
    profile_inputs = isolated_profile.get("inputs")
    profile_policy = isolated_profile.get("policy")
    profile_documents = isolated_profile.get("documents")
    if (
        isolated_profile.get("schema_version")
        != "maxim-public-workbook-profile-v1"
        or not isinstance(profile_inputs, dict)
        or not isinstance(profile_policy, dict)
        or not isinstance(profile_documents, list)
        or profile_inputs.get("parser_observations", {}).get("sha256")
        != expected_parser_sha256
        or profile_inputs.get("source_locators", {}).get("sha256")
        != expected_source_locators_sha256
        or profile_inputs.get("source_index", {}).get("sha256")
        != source_index_spec.get("sha256")
        or profile_policy.get("image_only_activity_observation_projection")
        != OBSERVATION_KIND
        or profile_policy.get("image_only_activity_record_projection")
        != RECORD_SELECTION_POLICY
        or profile_policy.get("task_id_is_policy_feature") is not False
        or profile_policy.get("benchmark_candidate_or_outcome_access") is not False
    ):
        raise ImageOnlyActivityError("isolated image-only source profile changed")
    expected_observation_count = profile_policy.get(
        "expected_image_only_activity_observations"
    )
    if type(expected_observation_count) is not int or expected_observation_count < 1:
        raise ImageOnlyActivityError(
            "isolated image-only observation count is not pinned"
        )
    for field, expected in _threshold_projection(thresholds).items():
        if profile_policy.get(f"visual_{field}") != expected:
            raise ImageOnlyActivityError(
                "isolated image-only visual thresholds changed"
            )

    try:
        from .official_workbook import (
            parse_workbook_index,
            strict_public_document_identity,
        )

        isolated_index = parse_workbook_index(
            load_image_only_activity_visual_artifact_json(
                pinned_paths["source index"]
            )
        )
    except OfficialSourceError as exc:
        raise ImageOnlyActivityError(str(exc)) from exc
    isolated_documents = {
        document.document_id: document for document in isolated_index.documents
    }
    profile_documents_by_id = {
        str(item.get("document_id") or ""): item
        for item in profile_documents
        if isinstance(item, dict)
    }
    if (
        len(profile_documents_by_id) != len(profile_documents)
        or set(profile_documents_by_id) != set(isolated_documents)
    ):
        raise ImageOnlyActivityError("isolated image-only document set changed")
    for document_id, isolated_document in isolated_documents.items():
        profile_document = profile_documents_by_id[document_id]
        live_document = documents_by_id.get(document_id)
        if (
            live_document is None
            or set(profile_document)
            != {"document_id", "pdf_sha256", "page_count"}
            or profile_document.get("pdf_sha256") != isolated_document.pdf_sha256
            or profile_document.get("page_count") != isolated_document.page_count
            or live_document.pdf_sha256 != isolated_document.pdf_sha256
            or live_document.page_count != isolated_document.page_count
            or live_document.identity != isolated_document.identity
        ):
            raise ImageOnlyActivityError(
                "isolated/final image-only source document differs"
            )

    raw_records = value.get("activity_records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ImageOnlyActivityError("image-only activity records are absent")
    artifact_records = tuple(_record_ref_from_mapping(item) for item in raw_records)
    if len({_record_key(item) for item in artifact_records}) != len(artifact_records):
        raise ImageOnlyActivityError("image-only activity records are duplicated")
    isolated_records = tuple(
        ActivityVisualRecordRef(
            document_id=document.document_id,
            record_id=question.record_id,
            content_page_number=question.content_page_number,
            activity_number=question.question_number,
            key_projection_sha256=question.key_projection_sha256,
            content_projection_sha256=question.content_projection_sha256,
            binding_projection_sha256=question.binding_projection_sha256,
            visually_checked=question.visually_checked,
            content_bbox=question.content_bbox,
        )
        for document in isolated_index.documents
        for question in document.questions
        if question.key_binding_kind == "activity_answer_key"
        and question.question_marker_kind == "activity_label"
        and question.key_projection_sha256 is not None
        and question.content_projection_sha256 is not None
        and question.binding_projection_sha256 is not None
        and question.content_bbox is not None
    )
    if sorted(map(_record_key, artifact_records)) != sorted(
        map(_record_key, isolated_records)
    ):
        raise ImageOnlyActivityError(
            "artifact records differ from the isolated source index"
        )
    live_record_keys = [_record_key(record) for record in records]
    if len(set(live_record_keys)) != len(live_record_keys) or any(
        live_record_keys.count(_record_key(record)) != 1
        for record in artifact_records
    ):
        raise ImageOnlyActivityError(
            "image-only records do not occur exactly once in the final source index"
        )
    artifact_pages = {
        (record.document_id, record.content_page_number)
        for record in artifact_records
    }
    for document_id, content_page_number in artifact_pages:
        artifact_page_records = sorted(
            _record_key(record)
            for record in artifact_records
            if record.document_id == document_id
            and record.content_page_number == content_page_number
        )
        live_page_records = sorted(
            _record_key(record)
            for record in records
            if record.document_id == document_id
            and record.content_page_number == content_page_number
        )
        if live_page_records != artifact_page_records:
            raise ImageOnlyActivityError(
                "final source index changes an image-only selected page record set"
            )

    profiles = value.get("profiles")
    if (
        not isinstance(profiles, dict)
        or set(profiles) != {"sift", "thresholds"}
        or profiles.get("sift") != _sift_projection()
        or profiles.get("thresholds") != _threshold_projection(thresholds)
    ):
        raise ImageOnlyActivityError("image-only activity visual profile changed")
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
        or runtime.get("sift_process_workers") != 4
        or not isinstance(runtime.get("python"), dict)
        or set(runtime["python"]) != {"executable", "version"}
        or type(runtime["python"].get("executable")) is not str
        or runtime["python"].get("version") != "3.12.13"
        or not isinstance(runtime.get("opencv"), dict)
        or set(runtime["opencv"])
        != {"module_path", "opencl_enabled", "threads", "version"}
        or runtime["opencv"].get("version") != "5.0.0"
        or runtime["opencv"].get("opencl_enabled") is not False
        or runtime["opencv"].get("threads") != 1
        or type(runtime["opencv"].get("module_path")) is not str
        or not isinstance(runtime.get("numpy"), dict)
        or set(runtime["numpy"]) != {"module_path", "version"}
        or runtime["numpy"].get("version") != "2.5.1"
        or type(runtime["numpy"].get("module_path")) is not str
        or not isinstance(runtime.get("poppler"), dict)
        or set(runtime["poppler"]) != {"pdfinfo", "pdftoppm"}
    ):
        raise ImageOnlyActivityError("image-only generation runtime changed")
    for tool_name in ("pdfinfo", "pdftoppm"):
        tool = runtime["poppler"].get(tool_name)
        if (
            not isinstance(tool, dict)
            or set(tool) != {"path", "sha256", "version", "version_line"}
            or type(tool.get("path")) is not str
            or _HEX64.fullmatch(str(tool.get("sha256") or "")) is None
            or tool.get("version") != "26.05.0"
            or type(tool.get("version_line")) is not str
        ):
            raise ImageOnlyActivityError(
                f"image-only activity {tool_name} runtime changed"
            )

    raw_documents = inputs.get("documents")
    if not isinstance(raw_documents, dict) or set(raw_documents) != set(
        isolated_documents
    ):
        raise ImageOnlyActivityError("image-only document render inventory changed")
    pages_by_document: dict[str, set[int]] = {}
    render_hashes: dict[tuple[str, int], str] = {}
    records_by_document: dict[str, tuple[ActivityVisualRecordRef, ...]] = {}
    for document_id in isolated_documents:
        records_by_document[document_id] = tuple(
            record for record in artifact_records if record.document_id == document_id
        )
    for document_id, spec in raw_documents.items():
        if not isinstance(spec, dict) or set(spec) != {
            "candidate_content_pages",
            "content_page_ranges",
            "page_count",
            "pdf_path",
            "pdf_sha256",
            "rendered_pages",
        }:
            raise ImageOnlyActivityError("image-only document pin changed")
        document = isolated_documents[document_id]
        pdf_path = document_pdf_paths.get(document_id)
        content_ranges = spec.get("content_page_ranges")
        if (
            pdf_path is None
            or not pdf_path.is_file()
            or _sha256_file(pdf_path) != document.pdf_sha256
            or spec.get("pdf_sha256") != document.pdf_sha256
            or spec.get("page_count") != document.page_count
            or type(spec.get("pdf_path")) is not str
            or not isinstance(content_ranges, list)
            or content_ranges
            != [list(page_range) for page_range in document.content_page_ranges]
        ):
            raise ImageOnlyActivityError("image-only official PDF pin changed")
        expanded_pages = [
            page
            for start, end in document.content_page_ranges
            for page in range(start, end + 1)
        ]
        candidate_pages = spec.get("candidate_content_pages")
        if (
            not isinstance(candidate_pages, list)
            or candidate_pages != expanded_pages
            or candidate_pages != sorted(set(candidate_pages))
        ):
            raise ImageOnlyActivityError("image-only full page inventory changed")
        rendered = spec.get("rendered_pages")
        if (
            not isinstance(rendered, dict)
            or {str(page) for page in candidate_pages} != set(rendered)
        ):
            raise ImageOnlyActivityError("image-only rendered page set changed")
        for raw_page, render in rendered.items():
            if (
                not isinstance(render, dict)
                or set(render) != {"width", "height", "rendered_page_sha256"}
                or type(render.get("width")) is not int
                or type(render.get("height")) is not int
                or render["width"] < 1
                or render["height"] < 1
                or _HEX64.fullmatch(
                    str(render.get("rendered_page_sha256") or "")
                )
                is None
            ):
                raise ImageOnlyActivityError("image-only rendered page pin changed")
            render_hashes[(document_id, int(raw_page))] = render[
                "rendered_page_sha256"
            ]
        pages_by_document[document_id] = set(candidate_pages)

    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise ImageOnlyActivityError(
            "image-only activity marker replay requires pdfplumber"
        ) from exc
    if str(pdfplumber.__version__) != "0.11.9":
        raise ImageOnlyActivityError(
            "image-only activity marker replay requires pdfplumber 0.11.9"
        )
    marker_inventories: dict[tuple[str, int], tuple[int, ...]] = {}
    for document_id, document_records in records_by_document.items():
        pdf_path = document_pdf_paths[document_id]
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) != isolated_documents[document_id].page_count:
                raise ImageOnlyActivityError(
                    "image-only official PDF page count changed"
                )
            for record in document_records:
                address = (document_id, record.content_page_number)
                if address in marker_inventories:
                    raise ImageOnlyActivityError(
                        "multiple image-only records share one source page"
                    )
                inventory = activity_marker_inventory(
                    pdf.pages[record.content_page_number - 1]
                )
                if inventory != (record.activity_number,):
                    raise ImageOnlyActivityError(
                        "full PDF page does not expose exactly the selected activity"
                    )
                marker_inventories[address] = inventory

    raw_bindings = value.get("bindings_by_task_image_sha256")
    if not isinstance(raw_bindings, dict) or not raw_bindings:
        raise ImageOnlyActivityError("image-only activity bindings are absent")
    expected_binding_fields = {
        "alignment_audit",
        "task_image",
        "parser_identity",
        "image_only_observation",
        "source_pins",
        "document_id",
        "source_page_activity_marker_inventory",
        "raw_page_evidences",
        "page_binding_decision",
    }
    verified: dict[str, VerifiedImageOnlyActivityBinding] = {}
    artifact_task_ids: set[str] = set()
    for image_sha, raw in raw_bindings.items():
        if (
            _HEX64.fullmatch(str(image_sha)) is None
            or not isinstance(raw, dict)
            or set(raw) != expected_binding_fields
        ):
            raise ImageOnlyActivityError("image-only binding fields changed")
        alignment = raw.get("alignment_audit")
        if not isinstance(alignment, dict) or set(alignment) != {
            "task_id",
            "task_id_role",
            "task_id_used_as_page_or_record_feature",
        }:
            raise ImageOnlyActivityError("image-only alignment audit changed")
        task_id = str(alignment.get("task_id") or "")
        observation = observations_by_task_id.get(task_id)
        source_url = source_urls_by_task_id.get(task_id)
        if (
            observation is None
            or source_url is None
            or task_id in artifact_task_ids
            or observation.image_sha256 != image_sha
            or alignment.get("task_id_role")
            != "parser_to_public_source_locator_alignment_only"
            or alignment.get("task_id_used_as_page_or_record_feature") is not False
        ):
            raise ImageOnlyActivityError("image-only alignment is not source-only")
        artifact_task_ids.add(task_id)
        try:
            identity = strict_public_document_identity(
                source_url,
                allow_missing_nosw=True,
            )
        except OfficialSourceError as exc:
            raise ImageOnlyActivityError(str(exc)) from exc
        document_matches = [
            document
            for document in isolated_documents.values()
            if document.identity == identity
        ]
        if len(document_matches) != 1:
            raise ImageOnlyActivityError(
                "image-only source locator does not select one isolated document"
            )
        document = document_matches[0]
        source_identity_projection_sha256 = canonical_json_sha256(
            {
                "document_id": document.document_id,
                "kind": document.identity.kind,
                "public_locator": document.identity.public_locator,
                "name": document.identity.name,
            }
        )
        task_image = raw.get("task_image")
        image_only = raw.get("image_only_observation")
        source_pins = raw.get("source_pins")
        if (
            not isinstance(task_image, dict)
            or set(task_image)
            != {"path", "image_basename", "image_sha256", "width", "height"}
            or not isinstance(image_only, dict)
            or set(image_only) != {"kind", "block_bbox", "block_area_coverage"}
            or not isinstance(source_pins, dict)
            or set(source_pins)
            != {
                "parser_artifact_sha256",
                "parser_projection_sha256",
                "source_locators_artifact_sha256",
                "source_identity_projection_sha256",
                "source_index_sha256",
                "pdf_sha256",
            }
        ):
            raise ImageOnlyActivityError("image-only observation pins changed")
        image_path = _repo_path(repo_root, task_image.get("path"), "task image")
        if (
            not image_path.is_file()
            or image_path.parent != task_image_dir
            or image_path.name != observation.image_basename
            or _sha256_file(image_path) != observation.image_sha256
            or task_image.get("image_basename") != observation.image_basename
            or task_image.get("image_sha256") != observation.image_sha256
            or task_image.get("width") != observation.width
            or task_image.get("height") != observation.height
            or raw.get("parser_identity") != observation.parser_identity
            or raw.get("document_id") != document.document_id
            or image_only.get("kind") != OBSERVATION_KIND
            or image_only.get("block_bbox") != list(observation.block_bbox)
            or image_only.get("block_area_coverage")
            != observation.block_area_coverage
            or source_pins.get("parser_artifact_sha256")
            != expected_parser_sha256
            or source_pins.get("parser_projection_sha256")
            != observation.parser_projection_sha256
            or source_pins.get("source_locators_artifact_sha256")
            != expected_source_locators_sha256
            or source_pins.get("source_identity_projection_sha256")
            != source_identity_projection_sha256
            or source_pins.get("source_index_sha256")
            != source_index_spec.get("sha256")
            or source_pins.get("pdf_sha256") != document.pdf_sha256
        ):
            raise ImageOnlyActivityError(
                "image-only parser/source/image projection changed"
            )
        raw_evidences = raw.get("raw_page_evidences")
        if not isinstance(raw_evidences, list):
            raise ImageOnlyActivityError("image-only page evidence is absent")
        try:
            evidences = tuple(
                visual_page_evidence_from_mapping(item) for item in raw_evidences
            )
        except VisualCoordinateBindingError as exc:
            raise ImageOnlyActivityError(str(exc)) from exc
        expected_pages = pages_by_document[document.document_id]
        if (
            len(evidences) != len(expected_pages)
            or {item.page_number for item in evidences} != expected_pages
            or any(
                item.rendered_page_sha256
                != render_hashes[(document.document_id, item.page_number)]
                for item in evidences
            )
        ):
            raise ImageOnlyActivityError(
                "image-only evidence is not the complete rendered page inventory"
            )
        try:
            decision = decide_visual_activity_page_binding(
                evidences,
                records_by_document[document.document_id],
                expected_task_image_sha256=observation.image_sha256,
                expected_document_id=document.document_id,
                expected_pdf_sha256=document.pdf_sha256,
                thresholds=thresholds,
            )
        except VisualCoordinateBindingError as exc:
            raise ImageOnlyActivityError(str(exc)) from exc
        if (
            not decision.accepted
            or raw.get("page_binding_decision") != _decision_projection(decision)
            or image_sha in verified
        ):
            raise ImageOnlyActivityError(
                "image-only page/record decision does not replay exactly"
            )
        selected_page = int(decision.selected_page_number or 0)
        inventory = marker_inventories.get((document.document_id, selected_page))
        marker_projection = {
            "pdf_sha256": document.pdf_sha256,
            "physical_page_number": selected_page,
            "canonical_activity_marker_numbers": list(inventory or ()),
        }
        marker_projection_sha256 = canonical_json_sha256(marker_projection)
        expected_marker_payload = {
            **marker_projection,
            "projection_sha256": marker_projection_sha256,
        }
        if (
            inventory != (decision.selected_question_number,)
            or raw.get("source_page_activity_marker_inventory")
            != expected_marker_payload
        ):
            raise ImageOnlyActivityError(
                "image-only full-page activity marker inventory changed"
            )
        verified[image_sha] = VerifiedImageOnlyActivityBinding(
            alignment_task_id=task_id,
            observation=observation,
            document_id=document.document_id,
            pdf_sha256=document.pdf_sha256,
            source_identity_projection_sha256=(
                source_identity_projection_sha256
            ),
            evidences=evidences,
            decision=decision,
            marker_inventory=inventory,
            marker_inventory_projection_sha256=marker_projection_sha256,
            thresholds=thresholds,
        )

    eligible_task_ids: set[str] = set()
    for task_id, observation in observations_by_task_id.items():
        source_url = source_urls_by_task_id.get(task_id)
        if source_url is None:
            raise ImageOnlyActivityError(
                "image-only parser observation lacks its public locator"
            )
        try:
            identity = strict_public_document_identity(
                source_url,
                allow_missing_nosw=True,
            )
        except OfficialSourceError:
            continue
        if any(document.identity == identity for document in isolated_documents.values()):
            eligible_task_ids.add(task_id)
    if (
        artifact_task_ids != eligible_task_ids
        or len(verified) != expected_observation_count
    ):
        raise ImageOnlyActivityError(
            "image-only artifact does not cover every eligible observation"
        )
    if value.get("summary") != {
        "activity_documents": len(isolated_documents),
        "activity_records": len(artifact_records),
        "image_only_activity_observations": len(verified),
        "accepted_page_bindings": len(verified),
        "raw_page_evidences": sum(len(item.evidences) for item in verified.values()),
        "decision_layer": "strict_unique_page_activity_applied",
    }:
        raise ImageOnlyActivityError("image-only activity summary changed")
    return verified


def problem_for_image_only_activity(
    observation: ImageOnlyActivityObservation,
    source_url: str,
    *,
    answer_format: str,
) -> ProblemInput:
    """Build the certificate input without inventing OCR text or a marker."""

    return ProblemInput(
        statement="",
        image_fingerprints=(observation.image_sha256,),
        constraints=(
            f"official_source={source_url}",
            f"parser={observation.parser_identity}",
            f"observation_kind={OBSERVATION_KIND}",
            f"parser_projection={observation.parser_projection_sha256}",
        ),
        answer_format=answer_format,
    )


def resolve_image_only_activity_question(
    observation: ImageOnlyActivityObservation,
    source_url: str,
    document: Any,
    binding: VerifiedImageOnlyActivityBinding,
    *,
    verified_content_marker_counts: Mapping[str, int] | None,
    allow_missing_nosw: bool = False,
) -> MatchResult:
    """Resolve one verified image-only page to its unique official record.

    No source activity number is projected into the parser observation.  The
    visual decision independently selects a page; exactly one reviewed record
    on that page supplies the source address and only then its verified answer.
    """

    try:
        from .official_workbook import strict_public_document_identity

        observed_identity = strict_public_document_identity(
            source_url,
            allow_missing_nosw=allow_missing_nosw,
        )
    except OfficialSourceError:
        observed_identity = None
    selected_matches = [
        question
        for question in document.questions
        if question.record_id == binding.decision.selected_record_id
        and question.content_page_number == binding.decision.selected_page_number
        and question.question_number == binding.decision.selected_question_number
        and question.key_binding_kind == "activity_answer_key"
        and question.question_marker_kind == "activity_label"
    ]
    selected = selected_matches[0] if len(selected_matches) == 1 else None
    parser_projection_bound = (
        binding.observation.image_sha256 == observation.image_sha256
        and binding.observation.parser_projection_sha256
        == observation.parser_projection_sha256
        and binding.observation.parser_identity == observation.parser_identity
        and binding.observation.width == observation.width
        and binding.observation.height == observation.height
        and binding.observation.block_bbox == observation.block_bbox
        and math.isclose(
            binding.observation.block_area_coverage,
            observation.block_area_coverage,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    )
    visual_binding = (
        binding.document_id == document.document_id
        and binding.pdf_sha256 == document.pdf_sha256
        and binding.decision.accepted
        and bool(binding.decision.checks)
        and all(passed for _, passed in binding.decision.checks)
    )
    unique_marker = (
        selected is not None
        and binding.marker_inventory == (selected.question_number,)
    )
    source_pdf_record_verified = (
        selected is not None
        and verified_content_marker_counts is not None
        and verified_content_marker_counts.get(selected.record_id) == 1
        and _HEX64.fullmatch(selected.key_projection_sha256) is not None
        and _HEX64.fullmatch(selected.content_projection_sha256) is not None
        and _HEX64.fullmatch(selected.binding_projection_sha256) is not None
        and selected.content_bbox is not None
    )
    checks = (
        ("strict_public_document_identity", observed_identity == document.identity),
        ("image_only_parser_observation", parser_projection_bound),
        ("visual_page_binding", visual_binding),
        ("unique_source_activity_record", selected is not None),
        ("unique_pdf_activity_marker_on_selected_page", unique_marker),
        ("source_pdf_record_verified", source_pdf_record_verified),
        ("reviewed_embedded_key", selected is not None and selected.visually_checked),
        (
            "valid_source_answer",
            selected is not None
            and (
                (
                    selected.answer_format == "choice"
                    and selected.answer in frozenset("ABCDE")
                )
                or (
                    selected.answer_format == "short_text"
                    and bool(selected.answer.strip())
                )
            ),
        ),
        (
            "source_address_not_task_id",
            selected is not None
            and selected.record_id.startswith(f"{document.document_id}:p"),
        ),
    )
    accepted = all(passed for _, passed in checks)
    answer = selected.answer if accepted and selected is not None else None
    problem = problem_for_image_only_activity(
        observation,
        source_url,
        answer_format=selected.answer_format if selected else "source_answer",
    )
    ordered_evidence = sorted(
        binding.evidences,
        key=lambda item: (
            -item.rank_score,
            -item.inliers,
            item.page_number,
            item.rendered_page_sha256,
        ),
    )
    runner_page = ordered_evidence[1].page_number if len(ordered_evidence) > 1 else None
    trace: dict[str, Any] = {
        "schema_version": IMAGE_ONLY_ACTIVITY_TRACE_SCHEMA,
        # Keep the public-workbook certificate verifier stable.  The nested
        # binding trace names the distinct image-only algorithm.
        "verifier": "public-workbook-ocr-page-key-binding-v1",
        "source": {
            "document_id": document.document_id,
            "public_locator": document.identity.public_locator,
            "name": document.identity.name,
            "pdf_sha256": document.pdf_sha256,
            "matched_page_number": (
                selected.content_page_number if selected is not None else None
            ),
            "runner_up_page_number": runner_page,
            "record_id": selected.record_id if selected is not None else None,
            "question_number": (
                selected.question_number if selected is not None else None
            ),
            "answer_format": selected.answer_format if selected is not None else None,
            "question_marker_kind": (
                selected.question_marker_kind if selected is not None else None
            ),
            "key_binding_kind": (
                selected.key_binding_kind if selected is not None else None
            ),
            "key_projection_sha256": (
                selected.key_projection_sha256 if selected is not None else None
            ),
            "content_projection_sha256": (
                selected.content_projection_sha256 if selected is not None else None
            ),
            "key_page_number": (
                selected.key_page_number if selected is not None else None
            ),
            "key_context_page_number": (
                selected.key_context_page_number if selected is not None else None
            ),
            "key_bbox": list(selected.key_bbox) if selected is not None else None,
            "content_bbox": (
                list(selected.content_bbox)
                if selected is not None and selected.content_bbox is not None
                else None
            ),
            "binding_projection_sha256": (
                selected.binding_projection_sha256 if selected is not None else None
            ),
            "source_answer_format": (
                selected.source_answer_format if selected is not None else None
            ),
            "source_unit_number": (
                selected.source_unit_number if selected is not None else None
            ),
            "test_variant": selected.test_variant if selected is not None else None,
        },
        "observation": {
            "image_sha256": observation.image_sha256,
            "image_size": [observation.width, observation.height],
            "parser_identity": observation.parser_identity,
            "image_only_observation_kind": OBSERVATION_KIND,
            "parser_projection_sha256": observation.parser_projection_sha256,
            "block_bbox": list(observation.block_bbox),
            "block_area_coverage": observation.block_area_coverage,
            "observed_source_marker_kind": None,
            "observed_source_marker_number": None,
        },
        "match": {
            "question_binding_method": (
                "image_only_full_page_visual_unique_activity_v1"
            ),
            "page_binding_decision": _decision_projection(binding.decision),
            "source_page_activity_marker_inventory": {
                "pdf_sha256": binding.pdf_sha256,
                "physical_page_number": binding.decision.selected_page_number,
                "canonical_activity_marker_numbers": list(
                    binding.marker_inventory
                ),
                "projection_sha256": (
                    binding.marker_inventory_projection_sha256
                ),
            },
        },
        "thresholds": _threshold_projection(binding.thresholds),
        "checks": {name: passed for name, passed in checks},
        "accepted": accepted,
        "image_only_activity_binding": binding.trace(),
    }
    return MatchResult(
        accepted=accepted,
        answer=answer,
        problem=problem,
        checks=checks,
        trace=trace,
    )
