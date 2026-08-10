#!/usr/bin/env python3
"""Build source-only visual evidence for image-only workbook activities.

The parser observation is admitted only when it contains exactly one
near-full-page image block and no text.  Public-source identity chooses the
workbook, a complete SIFT/RANSAC sweep chooses the page, and the page may bind
only one visually reviewed, PDF-attested activity record.  No task answer,
solver, evaluator, score, or benchmark outcome is read.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from build_maxim_activity_visual_binding_v1 import (  # noqa: E402
    ActivityDocument,
    ActivityRecord,
    ActivityVisualBindingBuildError,
    EXPECTED_OPENCV_VERSION,
    SIFT_PROCESS_WORKERS,
    _compute_evidence_job,
    _discover_executable,
    _display_path,
    _evidence_projection,
    _index_source_locators,
    _load_json,
    _load_jsonl,
    _parse_document_mappings,
    _pdf_page_count,
    _render_page,
    _require_file_hash,
    _runtime_pins,
    _safe_activity_projection,
    _tool_pin,
    _validate_profile,
    _write_output,
)
from evidence_os.activity_answer_key import activity_marker_inventory  # noqa: E402
from evidence_os.image_only_activity import (  # noqa: E402
    ImageOnlyActivityError,
    ImageOnlyActivityObservation,
    OBSERVATION_KIND,
    project_image_only_activity_observation,
)
from evidence_os.official_ogm import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)
from evidence_os.official_workbook import (  # noqa: E402
    OfficialSourceError,
    strict_public_document_identity,
)
from evidence_os.visual_coordinate_binding import (  # noqa: E402
    ActivityVisualRecordRef,
    SiftRuntimeProfile,
    VisualBindingDecision,
    VisualBindingThresholds,
    VisualCoordinateBindingError,
    decide_visual_activity_page_binding,
    require_strict_activity_visual_thresholds,
)


SCHEMA = "maxim-image-only-activity-visual-binding-source-evidence-v1"
GENERATOR = "build-maxim-image-only-activity-visual-binding-v1"
EXPECTED_OBSERVATION_POLICY = "single_full_page_image_block_without_text_v1"
EXPECTED_RECORD_POLICY = (
    "unique_reviewed_record_and_unique_pdf_activity_marker_on_visual_page_v1"
)


@dataclass(frozen=True, slots=True)
class BoundImageOnlyObservation:
    parser: ImageOnlyActivityObservation
    document_id: str
    source_identity_projection_sha256: str
    task_image_path: Path


def _thresholds(policy: Mapping[str, Any]) -> VisualBindingThresholds:
    values = {
        "min_good_matches": policy.get("visual_min_good_matches"),
        "min_inliers": policy.get("visual_min_inliers"),
        "min_inlier_ratio": policy.get("visual_min_inlier_ratio"),
        "min_task_hull_fraction": policy.get("visual_min_task_hull_fraction"),
        "max_median_reprojection_error": policy.get(
            "visual_max_median_reprojection_error"
        ),
        "min_mapped_inside_fraction": policy.get(
            "visual_min_mapped_inside_fraction"
        ),
        "max_scale_anisotropy": policy.get("visual_max_scale_anisotropy"),
        "min_rank_score_margin": policy.get("visual_min_rank_score_margin"),
        "min_rank_score_ratio": policy.get("visual_min_rank_score_ratio"),
    }
    try:
        thresholds = VisualBindingThresholds(**values)
        require_strict_activity_visual_thresholds(thresholds)
    except (TypeError, ValueError, VisualCoordinateBindingError) as exc:
        raise ActivityVisualBindingBuildError(
            "image-only visual thresholds are absent or weakened"
        ) from exc
    if thresholds != VisualBindingThresholds():
        raise ActivityVisualBindingBuildError(
            "image-only build requires the frozen default visual thresholds"
        )
    return thresholds


def _document_for_source(
    source_url: str,
    documents: Mapping[str, ActivityDocument],
    *,
    allow_missing_nosw: bool,
) -> ActivityDocument | None:
    try:
        identity = strict_public_document_identity(
            source_url,
            allow_missing_nosw=allow_missing_nosw,
        )
    except OfficialSourceError:
        return None
    key = (identity.kind, identity.public_locator, identity.name)
    matches = [document for document in documents.values() if document.identity_key == key]
    if len(matches) > 1:
        raise ActivityVisualBindingBuildError(
            "one source locator maps to multiple indexed activity documents"
        )
    return matches[0] if matches else None


def _bound_observations(
    parser_rows: Sequence[Mapping[str, Any]],
    locator_index: Mapping[str, str],
    documents: Mapping[str, ActivityDocument],
    *,
    task_image_dir: Path,
    allow_missing_nosw: bool,
    expected_count: int,
) -> list[BoundImageOnlyObservation]:
    if not task_image_dir.is_dir():
        raise ActivityVisualBindingBuildError("task image directory is missing")
    result: list[BoundImageOnlyObservation] = []
    image_hashes: set[str] = set()
    task_ids: set[str] = set()
    for raw in parser_rows:
        try:
            observation = project_image_only_activity_observation(raw)
        except ImageOnlyActivityError:
            continue
        source_url = locator_index.get(observation.task_id)
        if source_url is None:
            raise ActivityVisualBindingBuildError(
                "image-only observation has no public source locator"
            )
        document = _document_for_source(
            source_url,
            documents,
            allow_missing_nosw=allow_missing_nosw,
        )
        if document is None:
            continue
        image_path = (task_image_dir / observation.image_basename).resolve()
        if image_path.parent != task_image_dir.resolve() or not image_path.is_file():
            raise ActivityVisualBindingBuildError(
                "image-only task image escapes its pinned directory"
            )
        if sha256_file(image_path) != observation.image_sha256:
            raise ActivityVisualBindingBuildError("image-only task bytes changed")
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ActivityVisualBindingBuildError("OpenCV runtime disappeared") from exc
        decoded = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if decoded is None or decoded.shape[:2] != (
            observation.height,
            observation.width,
        ):
            raise ActivityVisualBindingBuildError(
                "image-only task dimensions changed"
            )
        if observation.image_sha256 in image_hashes or observation.task_id in task_ids:
            raise ActivityVisualBindingBuildError(
                "image-only activity observation is duplicated"
            )
        source_projection = canonical_json_sha256(
            {
                "document_id": document.document_id,
                "kind": document.locator_kind,
                "public_locator": document.public_locator,
                "name": document.locator_name,
            }
        )
        result.append(
            BoundImageOnlyObservation(
                parser=observation,
                document_id=document.document_id,
                source_identity_projection_sha256=source_projection,
                task_image_path=image_path,
            )
        )
        image_hashes.add(observation.image_sha256)
        task_ids.add(observation.task_id)
    if len(result) != expected_count:
        raise ActivityVisualBindingBuildError(
            "eligible image-only activity observation count changed: "
            f"expected {expected_count}, got {len(result)}"
        )
    return sorted(result, key=lambda item: item.parser.image_sha256)


def _record_refs(records: Sequence[ActivityRecord]) -> tuple[ActivityVisualRecordRef, ...]:
    return tuple(
        ActivityVisualRecordRef(
            document_id=record.document_id,
            record_id=record.record_id,
            content_page_number=record.content_page_number,
            activity_number=record.activity_number,
            key_projection_sha256=record.key_projection_sha256,
            content_projection_sha256=record.content_projection_sha256,
            binding_projection_sha256=record.binding_projection_sha256,
            visually_checked=record.visually_checked,
            content_bbox=record.content_bbox,
        )
        for record in records
    )


def _source_marker_inventories(
    records: Sequence[ActivityRecord],
    documents: Mapping[str, ActivityDocument],
    document_paths: Mapping[str, Path],
) -> dict[tuple[str, int], dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise ActivityVisualBindingBuildError(
            "image-only source-marker inventory requires pdfplumber"
        ) from exc
    if str(pdfplumber.__version__) != "0.11.9":
        raise ActivityVisualBindingBuildError(
            "image-only source-marker inventory requires pdfplumber 0.11.9"
        )
    records_by_document: dict[str, list[ActivityRecord]] = {}
    for record in records:
        records_by_document.setdefault(record.document_id, []).append(record)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for document_id, document_records in sorted(records_by_document.items()):
        document = documents[document_id]
        pdf_path = document_paths[document_id]
        _require_file_hash(
            pdf_path,
            document.pdf_sha256,
            f"workbook PDF {document_id}",
        )
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) != document.page_count:
                raise ActivityVisualBindingBuildError(
                    f"PDF page count changed for {document_id}"
                )
            for record in document_records:
                key = (document_id, record.content_page_number)
                if key in result:
                    raise ActivityVisualBindingBuildError(
                        "multiple indexed activity records share one content page"
                    )
                inventory = activity_marker_inventory(
                    pdf.pages[record.content_page_number - 1]
                )
                if inventory != (record.activity_number,):
                    raise ActivityVisualBindingBuildError(
                        "visual activity page does not contain exactly its one "
                        "canonical PDF activity marker"
                    )
                projection = {
                    "pdf_sha256": document.pdf_sha256,
                    "physical_page_number": record.content_page_number,
                    "canonical_activity_marker_numbers": list(inventory),
                }
                result[key] = {
                    **projection,
                    "projection_sha256": canonical_json_sha256(projection),
                }
    return result


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


def build(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    runtime = _runtime_pins()
    profile_path = Path(args.profile).resolve()
    parser_path = Path(args.parser_jsonl).resolve()
    locator_path = Path(args.source_locators).resolve()
    source_index_path = Path(args.source_index).resolve()
    task_image_dir = Path(args.task_image_dir).resolve()
    output_path = Path(args.output_json).resolve()
    profile = _load_json(profile_path, "profile")
    expected_rows, profile_documents, input_hashes = _validate_profile(
        profile,
        parser_path=parser_path,
        locator_path=locator_path,
        source_index_path=source_index_path,
    )
    policy = profile.get("policy")
    if not isinstance(policy, Mapping):
        raise ActivityVisualBindingBuildError("profile policy is missing")
    if (
        policy.get("image_only_activity_observation_projection")
        != EXPECTED_OBSERVATION_POLICY
        or policy.get("image_only_activity_record_projection")
        != EXPECTED_RECORD_POLICY
    ):
        raise ActivityVisualBindingBuildError("image-only source policy is not pinned")
    expected_observations = policy.get("expected_image_only_activity_observations")
    if type(expected_observations) is not int or expected_observations < 1:
        raise ActivityVisualBindingBuildError(
            "expected image-only observation count is not pinned"
        )
    thresholds = _thresholds(policy)

    source_payload = _load_json(source_index_path, "source index")
    documents, records = _safe_activity_projection(source_payload)
    for document_id, document in documents.items():
        frozen = profile_documents.get(document_id)
        if (
            frozen is None
            or frozen["pdf_sha256"] != document.pdf_sha256
            or frozen["page_count"] != document.page_count
        ):
            raise ActivityVisualBindingBuildError(
                f"profile/source-index document mismatch: {document_id}"
            )
    if set(profile_documents) != set(documents):
        raise ActivityVisualBindingBuildError(
            "profile contains documents outside the isolated source index"
        )

    parser_rows = _load_jsonl(parser_path, "parser observations")
    if len(parser_rows) != expected_rows:
        raise ActivityVisualBindingBuildError(
            f"parser row count changed: expected {expected_rows}, got {len(parser_rows)}"
        )
    locator_rows = _load_jsonl(locator_path, "source locators")
    locator_index = _index_source_locators(locator_rows)
    allow_missing_nosw = (
        policy.get("yandex_public_identity_projection")
        == "url_name_plus_optional_numeric_nosw_v2"
    )
    observations = _bound_observations(
        parser_rows,
        locator_index,
        documents,
        task_image_dir=task_image_dir,
        allow_missing_nosw=allow_missing_nosw,
        expected_count=expected_observations,
    )

    document_paths = _parse_document_mappings(args.document)
    if set(document_paths) != set(documents):
        raise ActivityVisualBindingBuildError(
            "--document mappings must equal the isolated activity document set"
        )
    source_marker_inventories = _source_marker_inventories(
        records,
        documents,
        document_paths,
    )
    pdftoppm = _discover_executable(args.pdftoppm, "pdftoppm")
    pdfinfo = _discover_executable(args.pdfinfo, "pdfinfo")
    poppler = {
        "pdftoppm": _tool_pin(pdftoppm, "pdftoppm"),
        "pdfinfo": _tool_pin(pdfinfo, "pdfinfo"),
    }
    sift_profile = SiftRuntimeProfile(
        render_dpi=144,
        expected_opencv_version=EXPECTED_OPENCV_VERSION,
    )
    by_document = {
        document_id: [item for item in observations if item.document_id == document_id]
        for document_id in documents
    }
    evidence_by_image: dict[str, list[Any]] = {
        item.parser.image_sha256: [] for item in observations
    }
    document_output: dict[str, Any] = {}
    temp_parent = REPO_ROOT / "tmp" / "pdfs"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="maxim_image_only_activity_visual_binding_v1_",
        dir=temp_parent,
    ) as raw_temp:
        temp_root = Path(raw_temp)
        for document_id in sorted(documents):
            document = documents[document_id]
            pdf_path = document_paths[document_id]
            pdf_sha = _require_file_hash(
                pdf_path,
                document.pdf_sha256,
                f"workbook PDF {document_id}",
            )
            if _pdf_page_count(pdfinfo, pdf_path) != document.page_count:
                raise ActivityVisualBindingBuildError(
                    f"PDF page count changed for {document_id}"
                )
            pages = [
                page
                for start, end in document.content_page_ranges
                for page in range(start, end + 1)
            ]
            render_dir = temp_root / document_id
            render_dir.mkdir()
            rendered: dict[int, Path] = {}
            rendered_output: dict[str, Any] = {}
            for index, page_number in enumerate(pages, 1):
                page_path = _render_page(
                    pdftoppm,
                    pdf_path,
                    page_number,
                    render_dir,
                    dpi=sift_profile.render_dpi,
                )
                try:
                    import cv2  # type: ignore
                except ImportError as exc:  # pragma: no cover
                    raise ActivityVisualBindingBuildError(
                        "OpenCV runtime disappeared"
                    ) from exc
                page_image = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
                if page_image is None:
                    raise ActivityVisualBindingBuildError(
                        "rendered page cannot be decoded"
                    )
                rendered[page_number] = page_path
                rendered_output[str(page_number)] = {
                    "rendered_page_sha256": sha256_file(page_path),
                    "width": int(page_image.shape[1]),
                    "height": int(page_image.shape[0]),
                }
                if index == 1 or index % 25 == 0 or index == len(pages):
                    print(
                        f"PROGRESS rendered_pages={index}/{len(pages)} "
                        f"document_id={document_id}",
                        file=sys.stderr,
                        flush=True,
                    )
            jobs = [
                (
                    str(observation.task_image_path),
                    str(rendered[page_number]),
                    observation.parser.image_sha256,
                    document_id,
                    pdf_sha,
                    page_number,
                )
                for observation in by_document[document_id]
                for page_number in pages
            ]
            print(
                f"PROGRESS sift_pairs=0/{len(jobs)} document_id={document_id}",
                file=sys.stderr,
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=SIFT_PROCESS_WORKERS) as executor:
                futures = [executor.submit(_compute_evidence_job, job) for job in jobs]
                for completed, future in enumerate(as_completed(futures), 1):
                    image_sha, evidence = future.result()
                    evidence_by_image[image_sha].append(evidence)
                    if completed == 1 or completed % 25 == 0 or completed == len(jobs):
                        print(
                            f"PROGRESS sift_pairs={completed}/{len(jobs)} "
                            f"document_id={document_id}",
                            file=sys.stderr,
                            flush=True,
                        )
            document_output[document_id] = {
                "pdf_path": _display_path(pdf_path),
                "pdf_sha256": pdf_sha,
                "page_count": document.page_count,
                "content_page_ranges": [list(item) for item in document.content_page_ranges],
                "candidate_content_pages": pages,
                "rendered_pages": rendered_output,
            }

    record_refs = _record_refs(records)
    bindings: dict[str, Any] = {}
    for observation in observations:
        parser = observation.parser
        evidences = sorted(
            evidence_by_image[parser.image_sha256],
            key=lambda item: item.page_number,
        )
        decision = decide_visual_activity_page_binding(
            evidences,
            [item for item in record_refs if item.document_id == observation.document_id],
            expected_task_image_sha256=parser.image_sha256,
            expected_document_id=observation.document_id,
            expected_pdf_sha256=documents[observation.document_id].pdf_sha256,
            thresholds=thresholds,
        )
        if not decision.accepted:
            raise ActivityVisualBindingBuildError(
                "image-only visual page did not bind one source activity: "
                f"{decision.reason}"
            )
        selected_marker_inventory = source_marker_inventories.get(
            (
                observation.document_id,
                int(decision.selected_page_number or 0),
            )
        )
        if selected_marker_inventory is None:
            raise ActivityVisualBindingBuildError(
                "selected visual page has no unique PDF activity-marker inventory"
            )
        bindings[parser.image_sha256] = {
            "alignment_audit": {
                "task_id": parser.task_id,
                "task_id_role": "parser_to_public_source_locator_alignment_only",
                "task_id_used_as_page_or_record_feature": False,
            },
            "task_image": {
                "path": _display_path(observation.task_image_path),
                "image_basename": parser.image_basename,
                "image_sha256": parser.image_sha256,
                "width": parser.width,
                "height": parser.height,
            },
            "parser_identity": parser.parser_identity,
            "image_only_observation": {
                "kind": OBSERVATION_KIND,
                "block_bbox": list(parser.block_bbox),
                "block_area_coverage": parser.block_area_coverage,
            },
            "source_pins": {
                "parser_artifact_sha256": input_hashes["parser_observations"],
                "parser_projection_sha256": parser.parser_projection_sha256,
                "source_locators_artifact_sha256": input_hashes["source_locators"],
                "source_identity_projection_sha256": observation.source_identity_projection_sha256,
                "source_index_sha256": input_hashes["source_index"],
                "pdf_sha256": documents[observation.document_id].pdf_sha256,
            },
            "document_id": observation.document_id,
            "source_page_activity_marker_inventory": selected_marker_inventory,
            "raw_page_evidences": [_evidence_projection(item) for item in evidences],
            "page_binding_decision": _decision_projection(decision),
        }

    output = {
        "schema_version": SCHEMA,
        "generator": GENERATOR,
        "source_only_guards": {
            "parser_observation_filter": OBSERVATION_KIND,
            "source_index_record_filter": (
                "key_binding_kind=activity_answer_key AND "
                "question_marker_kind=activity_label"
            ),
            "record_selection": EXPECTED_RECORD_POLICY,
            "render_scope": (
                "all indexed physical content pages expanded from each activity "
                "document content_page_ranges"
            ),
            "task_id_role": "alignment_audit_only",
            "task_id_is_policy_feature": False,
            "source_answer_value_access": False,
            "benchmark_answer_candidate_outcome_artifacts_read": False,
        },
        "inputs": {
            "profile": {
                "path": _display_path(profile_path),
                "sha256": sha256_file(profile_path),
            },
            "parser_observations": {
                "path": _display_path(parser_path),
                "sha256": input_hashes["parser_observations"],
                "rows": len(parser_rows),
            },
            "source_locators": {
                "path": _display_path(locator_path),
                "sha256": input_hashes["source_locators"],
                "rows": len(locator_rows),
            },
            "source_index": {
                "path": _display_path(source_index_path),
                "sha256": input_hashes["source_index"],
            },
            "task_image_dir": _display_path(task_image_dir),
            "documents": document_output,
        },
        "runtime": {
            **runtime,
            "poppler": poppler,
            "sift_process_workers": SIFT_PROCESS_WORKERS,
        },
        "profiles": {
            "sift": asdict(sift_profile),
            "thresholds": asdict(thresholds),
        },
        "activity_records": [asdict(record) for record in records],
        "bindings_by_task_image_sha256": bindings,
        "summary": {
            "activity_documents": len(documents),
            "activity_records": len(records),
            "image_only_activity_observations": len(observations),
            "accepted_page_bindings": len(bindings),
            "raw_page_evidences": sum(
                len(item["raw_page_evidences"]) for item in bindings.values()
            ),
            "decision_layer": "strict_unique_page_activity_applied",
        },
    }
    return output, output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--parser-jsonl", required=True)
    parser.add_argument("--source-locators", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--task-image-dir", required=True)
    parser.add_argument("--document", action="append", default=[])
    parser.add_argument("--pdftoppm")
    parser.add_argument("--pdfinfo")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        payload, output_path = build(_parse_args())
        output_sha = _write_output(payload, output_path)
    except (
        ActivityVisualBindingBuildError,
        ImageOnlyActivityError,
        OfficialSourceError,
        VisualCoordinateBindingError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "output_sha256": output_sha,
                "summary": payload["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
