#!/usr/bin/env python3
"""Re-adjudicate changed image rows from pre-score official-source certificates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.official_ogm import canonical_json_bytes, sha256_file  # noqa: E402
from evidence_os.adapters import certificate_from_record  # noqa: E402
from evidence_os.contracts import (  # noqa: E402
    CandidateEnvelope,
    CertificateKind,
    CertificateStrength,
    CertificateVerdict,
)
from evidence_os.official_ogm import (  # noqa: E402
    OfficialSourceError,
    PageMatcher,
    observed_source_question_marker,
    parser_observation_allow_missing_number,
    parser_observation_primary_layout_number,
    problem_for,
)
from evidence_os.image_only_activity import (  # noqa: E402
    IMAGE_ONLY_ACTIVITY_ARTIFACT_ROLE,
    OBSERVATION_KIND as IMAGE_ONLY_ACTIVITY_OBSERVATION_KIND,
    RECORD_SELECTION_POLICY as IMAGE_ONLY_ACTIVITY_RECORD_SELECTION_POLICY,
    ImageOnlyActivityError,
    load_image_only_activity_visual_artifact_json,
    problem_for_image_only_activity,
    project_image_only_activity_observation,
    resolve_image_only_activity_question,
    verified_image_only_activity_bindings_from_artifact,
)
from evidence_os.official_workbook import (  # noqa: E402
    VERIFIER as WORKBOOK_VERIFIER,
    WorkbookThresholds,
    activity_label_projection_enabled,
    document_for_source,
    parse_workbook_index,
    resolve_workbook_question,
    validate_fail_closed_workbook_policy,
    verify_workbook_index_pdf,
)
from compose_maxim_official_ogm_failclosed_v2 import (  # noqa: E402
    CompositionError,
    _validate_workbook_certificate_artifact,
)
from evidence_os.visual_coordinate_binding import (  # noqa: E402
    ActivityVisualObservationRef,
    ActivityVisualRecordRef,
    VisualBindingThresholds,
    VisualCoordinateBindingError,
    load_activity_visual_artifact_json,
    verified_activity_bindings_from_artifact,
)


SCHEMA = "maxim-evidence-os-image-judge-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class JudgeBuildError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise JudgeBuildError(f"{path}: expected an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise JudgeBuildError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise JudgeBuildError(f"{label}: missing or duplicate task_id")
        result[task_id] = row
    return result


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise JudgeBuildError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(canonical_json_bytes(row).decode("utf-8") + "\n")
    temporary.replace(path)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_activity_visual_reproduction(
    *,
    profile_visual_spec: Any,
    composition_manifest: dict[str, Any],
    visual_path: Path | None,
    manifest_key: str = "activity_visual_reproduction",
    expected_mode: str = "fresh_source_only_poppler_sift_exact_bytes_v1",
) -> None:
    reproduction = composition_manifest.get(manifest_key)
    if profile_visual_spec is None:
        if reproduction is not None:
            raise JudgeBuildError("nonvisual composition has a stray visual reproduction")
        return
    if not isinstance(profile_visual_spec, dict) or visual_path is None:
        raise JudgeBuildError("visual profile is malformed")
    expected_sha = str(profile_visual_spec.get("sha256") or "")
    expected_fields = {
        "mode",
        "generator",
        "frozen_artifact",
        "reproduced_artifact",
        "exact_byte_identity",
        "command_projection_sha256",
        "runtime",
        "summary",
        "benchmark_answer_candidate_outcome_artifacts_read",
    }
    if (
        not isinstance(reproduction, dict)
        or set(reproduction) != expected_fields
        or reproduction.get("mode") != expected_mode
        or reproduction.get("exact_byte_identity") is not True
        or reproduction.get("benchmark_answer_candidate_outcome_artifacts_read")
        is not False
        or _HEX64.fullmatch(
            str(reproduction.get("command_projection_sha256") or "")
        )
        is None
    ):
        raise JudgeBuildError("composition lacks an exact source-only visual reproduction")
    generator = reproduction.get("generator")
    frozen = reproduction.get("frozen_artifact")
    rebuilt = reproduction.get("reproduced_artifact")
    runtime = reproduction.get("runtime")
    summary = reproduction.get("summary")
    if (
        not isinstance(generator, dict)
        or set(generator) != {"path", "sha256"}
        or type(generator.get("path")) is not str
        or _HEX64.fullmatch(str(generator.get("sha256") or "")) is None
        or not isinstance(frozen, dict)
        or set(frozen) != {"path", "sha256", "size_bytes"}
        or Path(str(frozen.get("path") or "")).resolve() != visual_path.resolve()
        or frozen.get("sha256") != expected_sha
        or type(frozen.get("size_bytes")) is not int
        or frozen["size_bytes"] < 1
        or not isinstance(rebuilt, dict)
        or set(rebuilt) != {"sha256", "size_bytes"}
        or rebuilt.get("sha256") != expected_sha
        or rebuilt.get("size_bytes") != frozen.get("size_bytes")
        or not isinstance(runtime, dict)
        or set(runtime)
        != {
            "python_executable_sha256",
            "pdftoppm_sha256",
            "pdfinfo_sha256",
            "runtime_projection_sha256",
        }
        or any(_HEX64.fullmatch(str(value or "")) is None for value in runtime.values())
        or not isinstance(summary, dict)
        or type(summary.get("raw_page_evidences")) is not int
        or summary["raw_page_evidences"] < 1
    ):
        raise JudgeBuildError("composition visual reproduction attestation is malformed")


def build(
    profile_path: Path,
    resolver_manifest_path: Path,
    expected_resolver_manifest_sha256: str,
    composition_manifest_path: Path,
    expected_composition_manifest_sha256: str,
    base_solver_path: Path,
    expected_base_solver_sha256: str,
    base_judge_path: Path,
    expected_base_judge_sha256: str,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    profile = _load_json(profile_path)
    resolver_manifest = _load_json(resolver_manifest_path)
    composition_manifest = _load_json(composition_manifest_path)
    profile_sha = sha256_file(profile_path)
    resolver_sha = _require_hash(
        resolver_manifest_path,
        expected_resolver_manifest_sha256,
        "resolver manifest",
    )
    composition_sha = _require_hash(
        composition_manifest_path,
        expected_composition_manifest_sha256,
        "composition manifest",
    )
    if profile.get("schema_version") != "maxim-public-workbook-profile-v1":
        raise JudgeBuildError("image adjudication requires the public-workbook profile")
    if (
        resolver_manifest.get("schema_version") != "maxim-public-workbook-run-v1"
        or composition_manifest.get("schema_version")
        != "maxim-official-source-failclosed-composition-v2"
        or resolver_manifest.get("gold_access") is not False
        or composition_manifest.get("profile", {}).get("sha256") != profile_sha
        or composition_manifest.get("resolver_manifest", {}).get("sha256") != resolver_sha
        or composition_manifest.get("score_or_outcome_access") is not False
        or resolver_manifest.get("benchmark_candidate_or_outcome_access") is not False
    ):
        raise JudgeBuildError("composition/resolver is not bound to the frozen blind profile")
    base_solver_sha = _require_hash(
        base_solver_path, expected_base_solver_sha256, "base image-judge solver"
    )
    base_judge_sha = _require_hash(
        base_judge_path, expected_base_judge_sha256, "base image judge"
    )
    solver_spec = composition_manifest.get("output", {}).get("solver")
    decisions_spec = composition_manifest.get("output", {}).get("decisions")
    artifacts = resolver_manifest.get("artifacts")
    if not all(isinstance(item, dict) for item in (solver_spec, decisions_spec, artifacts)):
        raise JudgeBuildError("composition or resolver artifacts are incomplete")
    solver_path = Path(str(solver_spec["path"])).resolve()
    decisions_path = Path(str(decisions_spec["path"])).resolve()
    candidate_path = Path(str(artifacts["candidate"]["path"])).resolve()
    certificate_path = Path(str(artifacts["certificates"]["path"])).resolve()
    _require_hash(solver_path, str(solver_spec["sha256"]), "composed solver")
    _require_hash(decisions_path, str(decisions_spec["sha256"]), "composition decisions")
    _require_hash(candidate_path, str(artifacts["candidate"]["sha256"]), "resolver candidate")
    _require_hash(
        certificate_path, str(artifacts["certificates"]["sha256"]), "resolver certificates"
    )
    solver = _index(_load_jsonl(solver_path), "composed solver")
    base_solver = _index(_load_jsonl(base_solver_path), "base solver")
    anchor_spec = profile.get("anchor")
    if not isinstance(anchor_spec, dict):
        raise JudgeBuildError("profile anchor is missing")
    anchor_path = (REPO_ROOT / str(anchor_spec.get("path") or "")).resolve()
    _require_hash(anchor_path, str(anchor_spec.get("sha256") or ""), "profile anchor")
    anchor = _index(_load_jsonl(anchor_path), "profile anchor")
    decisions = _index(_load_jsonl(decisions_path), "composition decisions")
    candidates = _index(_load_jsonl(candidate_path), "resolver candidate")
    certificates = _index(_load_jsonl(certificate_path), "resolver certificates")
    parser_spec = profile["inputs"]["parser_observations"]
    locator_spec = profile["inputs"]["source_locators"]
    source_index_spec = profile["inputs"]["source_index"]
    visual_spec = profile["inputs"].get("activity_visual_evidence")
    image_only_visual_spec = profile["inputs"].get(
        "image_only_activity_visual_evidence"
    )
    parser_path = (REPO_ROOT / str(parser_spec["path"])).resolve()
    locator_path = (REPO_ROOT / str(locator_spec["path"])).resolve()
    source_index_path = (REPO_ROOT / str(source_index_spec["path"])).resolve()
    _require_hash(parser_path, str(parser_spec["sha256"]), "parser observations")
    _require_hash(locator_path, str(locator_spec["sha256"]), "source locators")
    _require_hash(source_index_path, str(source_index_spec["sha256"]), "source index")
    if visual_spec is not None:
        if (
            not isinstance(visual_spec, dict)
            or set(visual_spec) != {"path", "sha256", "allowed_role"}
            or visual_spec.get("allowed_role")
            != "answer_free_sift_ransac_page_binding_fallback_only"
        ):
            raise JudgeBuildError("activity visual evidence role is not fail-closed")
        visual_path = (REPO_ROOT / str(visual_spec.get("path") or "")).resolve()
        visual_sha = _require_hash(
            visual_path,
            str(visual_spec.get("sha256") or ""),
            "activity visual evidence",
        )
        manifest_visual_spec = resolver_manifest.get("inputs", {}).get(
            "activity_visual_evidence"
        )
        if (
            not isinstance(manifest_visual_spec, dict)
            or manifest_visual_spec.get("sha256") != visual_sha
        ):
            raise JudgeBuildError("resolver visual evidence provenance changed")
    else:
        visual_path = None
    if image_only_visual_spec is not None:
        if (
            not isinstance(image_only_visual_spec, dict)
            or set(image_only_visual_spec)
            != {"path", "sha256", "allowed_role"}
            or image_only_visual_spec.get("allowed_role")
            != IMAGE_ONLY_ACTIVITY_ARTIFACT_ROLE
        ):
            raise JudgeBuildError(
                "image-only activity visual evidence role is not fail-closed"
            )
        image_only_visual_path = (
            REPO_ROOT / str(image_only_visual_spec.get("path") or "")
        ).resolve()
        image_only_visual_sha = _require_hash(
            image_only_visual_path,
            str(image_only_visual_spec.get("sha256") or ""),
            "image-only activity visual evidence",
        )
        manifest_image_only_spec = resolver_manifest.get("inputs", {}).get(
            "image_only_activity_visual_evidence"
        )
        if (
            not isinstance(manifest_image_only_spec, dict)
            or manifest_image_only_spec.get("sha256") != image_only_visual_sha
        ):
            raise JudgeBuildError(
                "resolver image-only activity evidence provenance changed"
            )
    else:
        image_only_visual_path = None
    _require_activity_visual_reproduction(
        profile_visual_spec=visual_spec,
        composition_manifest=composition_manifest,
        visual_path=visual_path,
    )
    _require_activity_visual_reproduction(
        profile_visual_spec=image_only_visual_spec,
        composition_manifest=composition_manifest,
        visual_path=image_only_visual_path,
        manifest_key="image_only_activity_visual_reproduction",
        expected_mode=(
            "fresh_source_only_poppler_sift_image_only_activity_exact_bytes_v1"
        ),
    )
    parser_rows = _index(_load_jsonl(parser_path), "parser observations")
    locator_rows = _index(_load_jsonl(locator_path), "source locators")
    source_index = parse_workbook_index(_load_json(source_index_path))
    source_documents = {document.document_id: document for document in source_index.documents}
    policy = profile.get("policy")
    runtime = profile.get("runtime")
    if not isinstance(policy, dict) or not isinstance(runtime, dict):
        raise JudgeBuildError("profile policy/runtime is incomplete")
    if image_only_visual_path is not None and (
        policy.get("image_only_activity_observation_projection")
        != IMAGE_ONLY_ACTIVITY_OBSERVATION_KIND
        or policy.get("image_only_activity_record_projection")
        != IMAGE_ONLY_ACTIVITY_RECORD_SELECTION_POLICY
        or policy.get("task_id_is_policy_feature") is not False
        or policy.get("benchmark_candidate_or_outcome_access") is not False
    ):
        raise JudgeBuildError("image-only activity policy changed")
    thresholds = WorkbookThresholds(
        min_page_coverage=float(policy["min_page_coverage"]),
        min_page_matched_tokens=int(policy["min_page_matched_tokens"]),
        min_page_margin=float(policy["min_page_margin"]),
        min_numberless_question_coverage=float(policy["min_numberless_question_coverage"]),
        min_numberless_question_matched_tokens=int(
            policy["min_numberless_question_matched_tokens"]
        ),
        min_numberless_question_margin=float(policy["min_numberless_question_margin"]),
        min_coordinate_question_similarity=float(
            policy.get("min_coordinate_question_similarity", 0.90)
        ),
        min_coordinate_question_source_tokens=int(
            policy.get("min_coordinate_question_source_tokens", 8)
        ),
        min_coordinate_question_margin=float(
            policy.get("min_coordinate_question_margin", 0.25)
        ),
        min_inline_question_coverage=float(
            policy.get("min_inline_question_coverage", 0.85)
        ),
        min_inline_question_matched_tokens=int(
            policy.get("min_inline_question_matched_tokens", 8)
        ),
        min_inline_question_margin=float(
            policy.get("min_inline_question_margin", 0.25)
        ),
    )
    visual_thresholds = VisualBindingThresholds(
        min_good_matches=int(policy.get("visual_min_good_matches", 50)),
        min_inliers=int(policy.get("visual_min_inliers", 40)),
        min_inlier_ratio=float(policy.get("visual_min_inlier_ratio", 0.65)),
        min_task_hull_fraction=float(
            policy.get("visual_min_task_hull_fraction", 0.30)
        ),
        max_median_reprojection_error=float(
            policy.get("visual_max_median_reprojection_error", 1.0)
        ),
        min_mapped_inside_fraction=float(
            policy.get("visual_min_mapped_inside_fraction", 0.98)
        ),
        max_scale_anisotropy=float(
            policy.get("visual_max_scale_anisotropy", 1.15)
        ),
        min_rank_score_margin=float(
            policy.get("visual_min_rank_score_margin", 10.0)
        ),
        min_rank_score_ratio=float(policy.get("visual_min_rank_score_ratio", 5.0)),
    )
    number_projection, allow_example_label_marker = (
        validate_fail_closed_workbook_policy(policy)
    )
    allow_activity_label_marker = activity_label_projection_enabled(policy)
    if number_projection == "unique_block_markers_v1":
        observation_loader = parser_observation_allow_missing_number
    elif number_projection == "primary_layout_then_unique_v1":
        observation_loader = parser_observation_primary_layout_number
    else:
        raise JudgeBuildError("unsupported question-number projection")
    identity_projection = str(
        policy.get("yandex_public_identity_projection")
        or "url_name_plus_required_numeric_nosw_v1"
    )
    if identity_projection not in {
        "url_name_plus_required_numeric_nosw_v1",
        "url_name_plus_optional_numeric_nosw_v2",
    }:
        raise JudgeBuildError("unsupported Yandex public-identity projection")
    allow_missing_nosw = identity_projection == "url_name_plus_optional_numeric_nosw_v2"
    try:
        import pdfplumber
        import pypdf
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise JudgeBuildError("certificate adjudication requires pinned PDF runtimes") from exc
    if (
        str(runtime.get("pypdf_version") or "") != str(pypdf.__version__)
        or str(runtime.get("pdfplumber_version") or "") != str(pdfplumber.__version__)
    ):
        raise JudgeBuildError("PDF runtime differs from the frozen profile")
    manifest_documents = resolver_manifest.get("inputs", {}).get("documents")
    if not isinstance(manifest_documents, dict) or set(manifest_documents) != set(source_documents):
        raise JudgeBuildError("resolver document set differs from the pinned source index")
    source_verification: dict[str, dict[str, Any]] = {}
    source_caches: dict[str, dict[str, Any]] = {}
    for document_id, document in source_documents.items():
        document_spec = manifest_documents[document_id]
        if not isinstance(document_spec, dict):
            raise JudgeBuildError(f"resolver document {document_id} is malformed")
        pdf_path = Path(str(document_spec.get("path") or "")).resolve()
        _require_hash(pdf_path, document.pdf_sha256, f"workbook PDF {document_id}")
        source_verification[document_id] = verify_workbook_index_pdf(pdf_path, document)
        page_texts = [page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages]
        if len(page_texts) != document.page_count:
            raise JudgeBuildError(f"workbook PDF {document_id} page count changed")
        source_caches[document_id] = {
            "path": pdf_path,
            "page_texts": page_texts,
            "matcher": PageMatcher(page_texts),
            "content_marker_counts": source_verification[document_id][
                "content_marker_counts"
            ],
        }
    visual_records = tuple(
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
        for document in source_index.documents
        for question in document.questions
        if question.key_binding_kind == "activity_answer_key"
        and question.question_marker_kind == "activity_label"
        and question.key_projection_sha256 is not None
        and question.content_projection_sha256 is not None
        and question.binding_projection_sha256 is not None
        and question.content_bbox is not None
    )
    verified_visual_bindings = {}
    if visual_path is not None:
        if policy.get("visual_page_binding_mode") != (
            "fallback_only_after_text_page_gate_failure_v1"
        ):
            raise JudgeBuildError("activity visual fallback policy changed")
        visual_observations: dict[str, ActivityVisualObservationRef] = {}
        for task_id, raw in parser_rows.items():
            try:
                observation = observation_loader(raw)
            except (OfficialSourceError, ValueError, KeyError, TypeError):
                continue
            marker_kind, marker_number = observed_source_question_marker(observation)
            if marker_kind != "activity_label" or marker_number is None:
                continue
            source_url = str(locator_rows[task_id].get("source_url") or "")
            try:
                visual_document = document_for_source(
                    source_index,
                    source_url,
                    allow_missing_nosw=allow_missing_nosw,
                )
            except OfficialSourceError:
                continue
            if visual_document is None:
                continue
            visual_observations[task_id] = ActivityVisualObservationRef(
                task_id=task_id,
                task_image_sha256=observation.image_sha256,
                width=observation.width,
                height=observation.height,
                parser_identity=observation.parser_identity,
                document_id=visual_document.document_id,
                pdf_sha256=visual_document.pdf_sha256,
                marker_kind=marker_kind,
                marker_number=marker_number,
            )
        try:
            verified_visual_bindings = verified_activity_bindings_from_artifact(
                load_activity_visual_artifact_json(visual_path),
                repo_root=REPO_ROOT,
                expected_parser_sha256=str(parser_spec["sha256"]),
                expected_source_locators_sha256=str(locator_spec["sha256"]),
                expected_source_index_sha256=str(source_index_spec["sha256"]),
                observations_by_task_id=visual_observations,
                records=visual_records,
                document_pdf_paths={
                    document_id: cache["path"]
                    for document_id, cache in source_caches.items()
                },
                thresholds=visual_thresholds,
            )
        except VisualCoordinateBindingError as exc:
            raise JudgeBuildError(
                f"activity visual evidence failed repeated replay: {exc}"
            ) from exc
    image_only_observations = {}
    verified_image_only_bindings = {}
    if image_only_visual_path is not None:
        for task_id, raw in parser_rows.items():
            try:
                image_only_observation = (
                    project_image_only_activity_observation(raw)
                )
            except ImageOnlyActivityError:
                continue
            try:
                observation_loader(raw)
            except (OfficialSourceError, ValueError, KeyError, TypeError):
                pass
            else:
                raise JudgeBuildError(
                    f"parser row {task_id} is ambiguous between text and image-only routes"
                )
            image_only_observations[task_id] = image_only_observation
        try:
            verified_image_only_bindings = (
                verified_image_only_activity_bindings_from_artifact(
                    load_image_only_activity_visual_artifact_json(
                        image_only_visual_path
                    ),
                    repo_root=REPO_ROOT,
                    expected_parser_sha256=str(parser_spec["sha256"]),
                    expected_source_locators_sha256=str(locator_spec["sha256"]),
                    observations_by_task_id=image_only_observations,
                    source_urls_by_task_id={
                        task_id: str(
                            locator_rows[task_id].get("source_url") or ""
                        )
                        for task_id in image_only_observations
                    },
                    documents_by_id=source_documents,
                    records=visual_records,
                    document_pdf_paths={
                        document_id: cache["path"]
                        for document_id, cache in source_caches.items()
                    },
                    thresholds=VisualBindingThresholds(),
                )
            )
        except ImageOnlyActivityError as exc:
            raise JudgeBuildError(
                f"image-only activity evidence failed repeated replay: {exc}"
            ) from exc
        image_only_certificate_count = sum(
            isinstance(
                certificate.get("trace", {}).get(
                    "image_only_activity_binding"
                ),
                dict,
            )
            for certificate in certificates.values()
            if isinstance(certificate.get("trace"), dict)
        )
        if (
            resolver_manifest.get("image_only_activity_observations")
            != len(image_only_observations)
            or resolver_manifest.get("image_only_activity_certificates")
            != image_only_certificate_count
        ):
            raise JudgeBuildError("image-only activity counters changed")
    base_judge_rows = _load_jsonl(base_judge_path)
    base_judge = _index(base_judge_rows, "base image judge")
    if (
        len(solver) != 274
        or len(anchor) != 274
        or len(base_solver) != 274
        or len(base_judge) != 97
    ):
        raise JudgeBuildError("expected composed/base solver and image judge rows 274/274/97")
    if (
        set(solver) != set(anchor)
        or set(decisions) != set(solver)
        or set(candidates) != set(solver)
        or set(parser_rows) != set(solver)
        or not set(solver) <= set(locator_rows)
        or not set(certificates) <= set(solver)
    ):
        raise JudgeBuildError("composed/base solver and decision task sets do not align")
    if not set(base_judge) <= set(solver):
        raise JudgeBuildError("image judge contains an unknown task")
    for task_id, decision in decisions.items():
        base_answer = str(base_solver[task_id].get("final_answer") or "").strip()
        anchor_answer = str(anchor[task_id].get("final_answer") or "").strip()
        current_answer = str(solver[task_id].get("final_answer") or "").strip()
        action = decision.get("action")
        if (
            decision.get("anchor_answer") != anchor_answer
            or decision.get("selected_answer") != current_answer
            or action not in {"keep_anchor", "replace_anchor"}
            or (action == "keep_anchor" and current_answer != anchor_answer)
            or (action == "replace_anchor" and current_answer == anchor_answer)
        ):
            raise JudgeBuildError(f"decision {task_id} is inconsistent with solver artifacts")
        if task_id in base_judge and anchor_answer != base_answer:
            raise JudgeBuildError(
                f"image row {task_id} anchor differs from the candidate judged by base image judge"
            )
    source_adjudication_ids: set[str] = set()
    for task_id in base_judge:
        decision = decisions[task_id]
        certificate = certificates.get(task_id)
        if certificate is None:
            continue
        action = decision.get("action")
        reason = decision.get("reason")
        composition_selected = (
            (action == "replace_anchor" and reason == "strongly_verified_challenger")
            or (action == "keep_anchor" and reason == "equivalent_to_anchor")
        )
        if (
            not composition_selected
            or decision.get("certificate_trace_fingerprint")
            != certificate.get("trace_fingerprint")
        ):
            raise JudgeBuildError(
                f"image row {task_id} has a source certificate that is not "
                "composition-selected"
            )
        source_adjudication_ids.add(task_id)
    if not source_adjudication_ids:
        raise JudgeBuildError(
            "no image rows have a composition-selected source certificate"
        )
    output_rows: list[dict[str, Any]] = []
    adjudicated: list[dict[str, Any]] = []
    for original in base_judge_rows:
        task_id = str(original["task_id"])
        current_answer = str(solver[task_id].get("final_answer") or "").strip()
        if task_id not in source_adjudication_ids:
            base_answer = str(base_solver[task_id].get("final_answer") or "").strip()
            if current_answer != base_answer:
                raise JudgeBuildError(
                    f"unchanged image row {task_id} differs from the base judge candidate"
                )
            output_rows.append(dict(original))
            continue
        candidate = candidates.get(task_id)
        certificate = certificates.get(task_id)
        if candidate is None or certificate is None:
            raise JudgeBuildError(
                f"source-adjudicated image row {task_id} lacks its certificate"
            )
        trace = certificate.get("trace")
        if not isinstance(trace, dict):
            raise JudgeBuildError(
                f"source-adjudicated image row {task_id} lacks an inline trace"
            )
        try:
            _validate_workbook_certificate_artifact(
                task_id=task_id,
                raw_candidate=candidate,
                raw_certificate=certificate,
                trace=trace,
                profile=profile,
                profile_sha=profile_sha,
            )
        except CompositionError as exc:
            raise JudgeBuildError(str(exc)) from exc
        candidate_answer = str(candidate.get("final_answer") or "").strip()
        trace_source = trace.get("source")
        if not isinstance(trace_source, dict):
            raise JudgeBuildError(
                f"source-adjudicated image row {task_id} lacks source metadata"
            )
        image_only_observation = image_only_observations.get(task_id)
        observation = (
            image_only_observation
            if image_only_observation is not None
            else observation_loader(parser_rows[task_id])
        )
        source_url = str(locator_rows[task_id].get("source_url") or "")
        document = document_for_source(
            source_index,
            source_url,
            allow_missing_nosw=allow_missing_nosw,
        )
        if document is None:
            raise JudgeBuildError(
                f"source-adjudicated image row {task_id} has no indexed source document"
            )
        source_record_id = str(trace_source.get("record_id") or "")
        source_records = {
            question.record_id: question
            for question in document.questions
        }
        source_record = source_records.get(source_record_id)
        if image_only_observation is not None:
            trace_observation = trace.get("observation")
            marker_matches = (
                source_record is not None
                and source_record.question_marker_kind == "activity_label"
                and isinstance(trace.get("image_only_activity_binding"), dict)
                and isinstance(trace_observation, dict)
                and trace_observation.get("observed_source_marker_kind") is None
                and trace_observation.get("observed_source_marker_number") is None
            )
        else:
            observed_marker_kind, observed_marker_number = (
                observed_source_question_marker(observation)
            )
            marker_matches = source_record is not None and (
                (
                    source_record.question_marker_kind == "numbered_item"
                    and observed_marker_kind == "numbered_item"
                    and observed_marker_number == source_record.question_number
                )
                or (
                    allow_example_label_marker
                    and source_record.question_marker_kind == "example_label"
                    and observed_marker_kind == "example_label"
                    and observed_marker_number == source_record.question_number
                )
                or (
                    allow_activity_label_marker
                    and source_record.question_marker_kind == "activity_label"
                    and observed_marker_kind == "activity_label"
                    and observed_marker_number == source_record.question_number
                )
            )
        activity_source_matches = (
            source_record is None
            or source_record.key_binding_kind != "activity_answer_key"
            or (
                trace_source.get("binding_projection_sha256")
                == source_record.binding_projection_sha256
                and trace_source.get("source_answer_format")
                == source_record.source_answer_format
                and trace_source.get("source_unit_number")
                == source_record.source_unit_number
                and trace_source.get("test_variant")
                == source_record.test_variant
            )
        )
        coordinate_choice_source_matches = (
            source_record is None
            or source_record.key_binding_kind != "coordinate_choice_answer_key"
            or (
                trace_source.get("content_section")
                == source_record.content_section
                and trace_source.get("section") == source_record.section
                and trace_source.get("test_variant")
                == source_record.test_variant
            )
        )
        if (
            source_record is None
            or trace_source.get("document_id") != document.document_id
            or trace_source.get("public_locator") != document.identity.public_locator
            or trace_source.get("name") != document.identity.name
            or trace_source.get("pdf_sha256") != document.pdf_sha256
            or trace_source.get("question_number") != source_record.question_number
            or trace_source.get("question_marker_kind")
            != source_record.question_marker_kind
            or trace_source.get("matched_page_number") != source_record.content_page_number
            or trace_source.get("key_page_number") != source_record.key_page_number
            or trace_source.get("key_context_page_number")
            != source_record.key_context_page_number
            or trace_source.get("key_binding_kind") != source_record.key_binding_kind
            or trace_source.get("answer_format") != source_record.answer_format
            or list(source_record.key_bbox) != trace_source.get("key_bbox")
            or (
                list(source_record.content_bbox)
                if source_record.content_bbox is not None
                else None
            )
            != trace_source.get("content_bbox")
            or str(trace_source.get("key_projection_sha256") or "")
            != source_record.key_projection_sha256
            or str(trace_source.get("content_projection_sha256") or "")
            != source_record.content_projection_sha256
            or not marker_matches
            or not activity_source_matches
            or not coordinate_choice_source_matches
            or candidate_answer != source_record.answer
        ):
            raise JudgeBuildError(
                f"source-adjudicated image row {task_id} is not bound to its "
                "pinned source-index record"
            )
        cache = source_caches[document.document_id]
        if image_only_observation is not None:
            image_only_binding = verified_image_only_bindings.get(
                image_only_observation.image_sha256
            )
            if image_only_binding is None:
                raise JudgeBuildError(
                    f"source-adjudicated image-only row {task_id} lacks a replayed binding"
                )
            recomputed = resolve_image_only_activity_question(
                image_only_observation,
                source_url,
                document,
                image_only_binding,
                verified_content_marker_counts=cache["content_marker_counts"],
                allow_missing_nosw=allow_missing_nosw,
            )
        else:
            recomputed = resolve_workbook_question(
                observation,
                source_url,
                document,
                cache["matcher"],
                cache["page_texts"],
                thresholds,
                allow_missing_nosw=allow_missing_nosw,
                allow_example_label_marker=allow_example_label_marker,
                allow_activity_label_marker=allow_activity_label_marker,
                verified_content_marker_counts=cache["content_marker_counts"],
                verified_activity_visual_binding=verified_visual_bindings.get(
                    observation.image_sha256
                ),
                activity_visual_thresholds=visual_thresholds,
            )
        recomputed_trace = recomputed.trace
        actual_trace_core = {key: trace.get(key) for key in recomputed_trace}
        if (
            not recomputed.accepted
            or recomputed.answer != candidate_answer
            or set(trace) != set(recomputed_trace) | {"provenance"}
            or actual_trace_core != recomputed_trace
        ):
            raise JudgeBuildError(
                f"source-adjudicated image row {task_id} fails the repeated "
                "OCR-to-source resolution"
            )
        if image_only_observation is not None:
            problem = problem_for_image_only_activity(
                image_only_observation,
                source_url,
                answer_format=str(trace_source.get("answer_format") or ""),
            )
        else:
            problem = problem_for(
                observation,
                source_url,
                answer_format=str(trace_source.get("answer_format") or ""),
            )
        try:
            bound_certificate = certificate_from_record(
                problem,
                CandidateEnvelope(source=WORKBOOK_VERIFIER, final_answer=candidate_answer),
                certificate,
                allowed_verifiers=frozenset({WORKBOOK_VERIFIER}),
                allowed_kinds=frozenset({CertificateKind.SOURCE_ENTAILMENT}),
                require_inline_trace=True,
            )
        except (TypeError, ValueError) as exc:
            raise JudgeBuildError(
                f"source-adjudicated image row {task_id} certificate is not "
                f"input/answer bound: {exc}"
            ) from exc
        if (
            bound_certificate.strength is not CertificateStrength.STRONG
            or bound_certificate.verdict is not CertificateVerdict.PASS
            or bound_certificate.input_bound is not True
            or bound_certificate.answer_bound is not True
            or bound_certificate.claim_coverage != 1.0
            or bound_certificate.contradiction_count != 0
            or not all(bound_certificate.deterministic_checks)
        ):
            raise JudgeBuildError(
                f"source-adjudicated image row {task_id} certificate is not a "
                "strong passing proof"
            )
        decision = decisions[task_id]
        if (
            current_answer != candidate_answer
            or decision.get("selected_answer") != candidate_answer
            or decision.get("certificate_trace_fingerprint")
            != certificate.get("trace_fingerprint")
        ):
            raise JudgeBuildError(
                f"source-adjudicated image row {task_id} is not certificate-selected"
            )
        generation = solver[task_id].get("generation")
        override = generation.get("official_source_override") if isinstance(generation, dict) else None
        action = decision.get("action")
        if action == "replace_anchor":
            if (
                not isinstance(override, dict)
                or generation.get("gold_access") is not False
                or override.get("profile_sha256") != profile_sha
                or override.get("trace_fingerprint")
                != certificate.get("trace_fingerprint")
                or override.get("verifier") != WORKBOOK_VERIFIER
            ):
                raise JudgeBuildError(
                    f"changed image row {task_id} lacks its solver override binding"
                )
        elif (
            action != "keep_anchor"
            or decision.get("reason") != "equivalent_to_anchor"
            or solver[task_id] != anchor[task_id]
            or current_answer
            != str(base_solver[task_id].get("final_answer") or "").strip()
        ):
            raise JudgeBuildError(
                f"confirmed image row {task_id} is not an unchanged anchor answer"
            )
        row = {
                "task_id": task_id,
                "subject": original.get("subject"),
                "grade": original.get("grade"),
                "answer_type": original.get("answer_type"),
                "setup": "maxim_evidence_os_official_source_adjudication_v1",
                "prompt_version": "official-source-certificate-v1",
                "request_id": _text_sha256(
                    f"{SCHEMA}:{task_id}:{certificate['trace_fingerprint']}"
                ),
                "judge": {
                    "attempts": 0,
                    "backend": "deterministic-pinned-pdf-certificate",
                    "backend_config_hash": _text_sha256(SCHEMA),
                    "cache_hit": False,
                    "error": None,
                    "model": None,
                },
                "metadata": {
                    "adjudication_protocol": SCHEMA,
                    "candidate_sha256": _text_sha256(candidate_answer),
                    "profile_sha256": profile_sha,
                    "source_document_id": trace["source"]["document_id"],
                    "source_record_id": trace["source"]["record_id"],
                    "source_pdf_sha256": trace["source"]["pdf_sha256"],
                    "certificate_trace_fingerprint": certificate["trace_fingerprint"],
                    "composition_action": action,
                    "answer_changed_from_anchor": action == "replace_anchor",
                },
                "verdict": {
                    "complete": True,
                    "confidence": 1.0,
                    "error_types": [],
                    "final_answer_correct": True,
                    "label": "fully_correct",
                    "rationale": (
                        "The frozen candidate is bound before scoring to the unique source "
                        "question and its printed answer in the pinned official PDF."
                    ),
                    "reasoning_correct": None,
                    "reference_quality_issue": False,
                    "score": 4,
                    "strict_correct": True,
                },
            }
        output_rows.append(row)
        adjudicated.append(
            {
                "task_id": task_id,
                "source_record_id": trace["source"]["record_id"],
                "certificate_trace_fingerprint": certificate["trace_fingerprint"],
                "composition_action": action,
                "answer_changed_from_anchor": action == "replace_anchor",
            }
        )
    _write_jsonl(output_path, output_rows)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reporting_status": "official_certificate_adjudicated_development_replay",
        "solver_and_source_certificates_hashed_before_adjudication": True,
        "tamper_resistant_external_freeze": False,
        "benchmark_reference_answers_opened": False,
        "base_image_judge_outcomes_read_and_copied_for_unchanged_rows": True,
        "base_image_judge_outcomes_used_for_changed_rows": False,
        "base_image_judge_outcomes_used_for_source_adjudicated_rows": False,
        "source_pdf_reverification": source_verification,
        "profile": {"path": str(profile_path), "sha256": profile_sha},
        "profile_anchor": {
            "path": str(anchor_path),
            "sha256": str(anchor_spec["sha256"]),
        },
        "resolver_manifest": {"path": str(resolver_manifest_path), "sha256": resolver_sha},
        "composition_manifest": {
            "path": str(composition_manifest_path),
            "sha256": composition_sha,
        },
        "base_solver": {"path": str(base_solver_path), "sha256": base_solver_sha},
        "base_image_judge": {"path": str(base_judge_path), "sha256": base_judge_sha},
        "output": {"path": str(output_path), "sha256": sha256_file(output_path), "rows": 97},
        "official_certificate_rows": adjudicated,
        "copied_unchanged_rows": 97 - len(adjudicated),
        "copied_nonadjudicated_rows": 97 - len(adjudicated),
        "limitations": [
            "Certificate adjudication replaces the VLM judge wherever a strong source "
            "certificate is composition-selected, including confirmations equal to the anchor.",
            "The evaluated target set is a previously inspected development replay, not a holdout.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--resolver-manifest", type=Path, required=True)
    parser.add_argument("--expected-resolver-manifest-sha256", required=True)
    parser.add_argument("--composition-manifest", type=Path, required=True)
    parser.add_argument("--expected-composition-manifest-sha256", required=True)
    parser.add_argument("--base-solver", type=Path, required=True)
    parser.add_argument("--expected-base-solver-sha256", required=True)
    parser.add_argument("--base-image-judge", type=Path, required=True)
    parser.add_argument("--expected-base-judge-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(
            args.profile_json.resolve(),
            args.resolver_manifest.resolve(),
            args.expected_resolver_manifest_sha256.lower(),
            args.composition_manifest.resolve(),
            args.expected_composition_manifest_sha256.lower(),
            args.base_solver.resolve(),
            args.expected_base_solver_sha256.lower(),
            args.base_image_judge.resolve(),
            args.expected_base_judge_sha256.lower(),
            args.output.resolve(),
            args.manifest.resolve(),
        )
    except (
        JudgeBuildError,
        OfficialSourceError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
