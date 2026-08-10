from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import compositor_v1_1 as legacy
import selector_v1_2 as selector


MODEL = "Qwen/Qwen3.5-9B"
COMPOSITOR_ID = "maxim_9b_baseline_selector_compositor_v1_2"
PROFILE_SCHEMA = "maxim-9b-baseline-selector-compositor-profile-v1.2"
PREREG_FREEZE_SCHEMA = "maxim-9b-baseline-selector-compositor-preregistered-freeze-v1.2"
MANIFEST_SCHEMA = "maxim-9b-baseline-selector-composition-manifest-v1.2"
DECISION_SCHEMA = "maxim-9b-baseline-selector-composition-decision-v1.2"
OUTPUT_FREEZE_SCHEMA = "maxim-9b-baseline-selector-composition-output-freeze-v1.2"
ROW_PROVENANCE_SCHEMA = "maxim-9b-baseline-selector-row-composition-provenance-v1.2"
ROW_COUNT = 274

SELECTOR_FREEZE_SHA256 = "55dd839d9c29201b2a66e63662491d85c74ee5cfc202f0b972e100dbe76331d9"
SELECTOR_MANIFEST_SHA256 = "2c3fa0cf20d3984aab062bf9db4e6c461218775b86e78026b552873fdc8845d2"
PROPOSALS_SHA256 = "51a5ed6a8cb76677d17eb1e8a55319b57503069f0660735d2c7273624ba598ec"
INPUT_PACKAGE_SHA256 = "12d73ab5cd5955dffee921c97b84fb8bc6c99e3b0152c295e74aa287ad3666e8"
CANDIDATE_POOL_SHA256 = "bfc4c2000cefb6ddbe2efb03364d1f2702a3d8075f089689bd3731dbb46c853d"
BASE_SOLVER_SHA256 = "9d26067064ee07fe480391759782c86d66adbb76dbc0da0d86ccc1b3f035211e"
STRUCTURAL_FULL_ROWS_SHA256 = "4c73b6eb326e5790b19e14c01a79df853be23bb5f55b498ce4c58b78ebc3dff5"
BENCHMARK_SHA256 = "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"

BASE_SOLVER_REPOSITORY_PATH = (
    "reports/maxim_9b_source_replay_v1_20260809/active_crop/"
    "fill_composed/solver.jsonl"
)
STRUCTURAL_RELATIVE_PATH = "input/v1_2/frozen/upstream/structural_strict_9b.jsonl"
SELECTOR_OUTPUT_RELATIVE_PATH = "output_v1_2/selector_manifest_v1_2.json"
INPUT_PACKAGE_RELATIVE_PATH = "input/v1_2/frozen/input_package_v1_2.json"

EXPECTED_PROFILE = {
    "schema_version": PROFILE_SCHEMA,
    "compositor_id": COMPOSITOR_ID,
    "status": "preregistered_before_composition_and_before_any_evaluation_of_v1_2_composited_outputs",
    "chronology": {
        "historical_benchmark_aggregate_score_and_prior_task_outcomes_were_known": True,
        "selector_v1_2_output_was_frozen_unscored_before_composition": True,
        "compositor_rules_frozen_before_composition": True,
        "compositor_runtime_does_not_read_gold_reference_correctness_outcomes_or_judge_verdicts": True,
    },
    "model_closure": [MODEL],
    "inputs": {
        "selector_freeze_sha256": SELECTOR_FREEZE_SHA256,
        "selector_output_manifest_sha256": SELECTOR_MANIFEST_SHA256,
        "selector_proposals_sha256": PROPOSALS_SHA256,
        "selector_input_package_sha256": INPUT_PACKAGE_SHA256,
        "base_source_solver_repository_relative_path": BASE_SOLVER_REPOSITORY_PATH,
        "base_source_solver_sha256": BASE_SOLVER_SHA256,
        "structural_full_rows_relative_path": STRUCTURAL_RELATIVE_PATH,
        "structural_full_rows_sha256": STRUCTURAL_FULL_ROWS_SHA256,
        "benchmark_sha256": BENCHMARK_SHA256,
        "rows": ROW_COUNT,
    },
    "rules": {
        "arms": ["primary", "exploratory"],
        "base": "frozen_final_activecrop9b_source_solver",
        "replacement_source": "bound_structural_strict_9b_full_row",
        "replace_only_when_selector_action": "propose_challenger",
        "replacement_gates": [
            "authoritative_route_is_deterministic",
            "task_is_outside_pinned_source_union",
            "proposal_selected_answer_equals_structural_challenger",
            "proposal_selected_answer_equals_normalized_structural_answer",
            "proposal_selected_answer_equals_full_structural_row_final_answer",
            "base_structural_proposal_and_authoritative_task_identity_match",
            "structural_full_row_is_qwen35_9b_and_has_coherent_answer_payload",
        ],
        "source_union_policy": "preserve_all_156_base_rows_as_exact_original_line_bytes",
        "image_judge_policy": "preserve_all_97_base_rows_as_exact_original_line_bytes",
        "other_passthrough_policy": "preserve_every_nonproposed_base_row_as_exact_original_line_bytes",
        "changed_row_policy": "copy_coherent_full_structural_row_and_add_explicit_selector_composition_provenance",
        "structural_source_access_disclosure": "structural_generation_may_declare_source_access_but_source_union_veto_remains_absolute",
        "no_fallback_repair": True,
        "fail_closed": True,
    },
    "expected_selector_actions": {
        "primary_proposals": 2,
        "exploratory_proposals": 3,
    },
}


class CompositorV12Error(RuntimeError):
    pass


def _translate_legacy_error(callable_: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return callable_(*args, **kwargs)
    except legacy.CompositorError as exc:
        raise CompositorV12Error(str(exc)) from exc


def _canonical_json(value: Any) -> bytes:
    return legacy._canonical_json(value)


def _sha256(path: Path) -> str:
    return _translate_legacy_error(legacy._sha256, path)


def _bytes_sha256(value: bytes) -> str:
    return legacy._bytes_sha256(value)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    return _translate_legacy_error(legacy._read_json, path, label)


def _strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    _translate_legacy_error(legacy._strict_keys, value, expected, label)


def _read_jsonl_raw(path: Path, label: str, expected_sha256: str) -> list[legacy.RawRow]:
    return _translate_legacy_error(legacy._read_jsonl_raw, path, label, expected_sha256)


def load_profile(root: Path) -> dict[str, Any]:
    profile = _read_json(root / "compositor_profile_v1_2.json", "v1.2 compositor profile")
    if profile != EXPECTED_PROFILE:
        raise CompositorV12Error("v1.2 compositor profile differs from exact contract")
    return profile


def verify_preregistered_freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    freeze_path = root / "COMPOSITOR_PREREGISTERED_FREEZE_v1_2.json"
    freeze = _read_json(freeze_path, "v1.2 compositor preregistered freeze")
    _strict_keys(
        freeze,
        {
            "schema_version",
            "compositor_id",
            "status",
            "historical_outcomes_known_before_freeze",
            "frozen_before_composition",
            "frozen_before_any_evaluation_of_v1_2_composited_outputs",
            "runtime_outcome_access",
            "selector_freeze_sha256",
            "selector_output_manifest_sha256",
            "artifacts",
        },
        "v1.2 compositor preregistered freeze",
    )
    if freeze["schema_version"] != PREREG_FREEZE_SCHEMA or freeze["compositor_id"] != COMPOSITOR_ID:
        raise CompositorV12Error("v1.2 compositor freeze schema/identity mismatch")
    if freeze["status"] != "preregistered_not_composed_not_evaluated":
        raise CompositorV12Error("v1.2 compositor freeze status mismatch")
    if freeze["historical_outcomes_known_before_freeze"] is not True:
        raise CompositorV12Error("v1.2 compositor historical context disclosure missing")
    if freeze["frozen_before_composition"] is not True:
        raise CompositorV12Error("v1.2 compositor was not frozen before composition")
    if freeze["frozen_before_any_evaluation_of_v1_2_composited_outputs"] is not True:
        raise CompositorV12Error("v1.2 compositor freeze does not predate evaluation")
    if freeze["runtime_outcome_access"] is not False:
        raise CompositorV12Error("v1.2 compositor runtime outcome access is not false")
    if freeze["selector_freeze_sha256"] != SELECTOR_FREEZE_SHA256:
        raise CompositorV12Error("v1.2 compositor selector freeze pin mismatch")
    if freeze["selector_output_manifest_sha256"] != SELECTOR_MANIFEST_SHA256:
        raise CompositorV12Error("v1.2 compositor selector manifest pin mismatch")
    artifacts = freeze["artifacts"]
    if not isinstance(artifacts, dict):
        raise CompositorV12Error("v1.2 compositor freeze artifacts must be object")
    expected_names = {
        "profile": "compositor_profile_v1_2.json",
        "code": "compositor_v1_2.py",
        "tests": "test_compositor_v1_2.py",
        "selector_dependency": "selector_v1_2.py",
        "legacy_helper_dependency": "compositor_v1_1.py",
    }
    _strict_keys(artifacts, set(expected_names), "v1.2 compositor freeze artifacts")
    verified: dict[str, dict[str, str]] = {}
    for role, expected_name in expected_names.items():
        descriptor = artifacts[role]
        if not isinstance(descriptor, dict):
            raise CompositorV12Error(f"v1.2 compositor freeze {role} descriptor must be object")
        _strict_keys(descriptor, {"path", "sha256"}, f"v1.2 compositor freeze {role}")
        if descriptor["path"] != expected_name:
            raise CompositorV12Error(f"v1.2 compositor freeze {role} path mismatch")
        digest = descriptor["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise CompositorV12Error(f"v1.2 compositor freeze {role} SHA malformed")
        path = (root / expected_name).resolve()
        if _sha256(path) != digest:
            raise CompositorV12Error(f"v1.2 compositor freeze {role} SHA mismatch")
        verified[role] = {"path": str(path), "sha256": digest}
    load_profile(root)
    return {
        "status": "compositor_v1_2_preregistered_freeze_verified",
        "freeze_path": str(freeze_path),
        "freeze_sha256": _sha256(freeze_path),
        "artifacts": verified,
    }


def _validate_solver_row(row: dict[str, Any], task_id: str, label: str) -> None:
    try:
        selector.base._assert_no_excluded_runtime_fields(row, label)
    except selector.base.SelectorError as exc:
        raise CompositorV12Error(str(exc)) from exc
    if row.get("task_id") != task_id:
        raise CompositorV12Error(f"{label} task identity mismatch")
    if row.get("model") != MODEL:
        raise CompositorV12Error(f"{label} model closure is not pure Qwen3.5-9B")
    if not isinstance(row.get("final_answer"), str) or not row["final_answer"].strip():
        raise CompositorV12Error(f"{label} final_answer is empty or missing")


def _validate_structural_payload(row: dict[str, Any], task_id: str, expected_answer: str) -> None:
    _validate_solver_row(row, task_id, "bound Structural full row")
    answer = str(row["final_answer"]).strip().upper()
    if not selector.base.CHOICE_RE.fullmatch(answer):
        raise CompositorV12Error("bound Structural full row final_answer is not A-E")
    if answer != expected_answer:
        raise CompositorV12Error("bound Structural full row answer differs from selector challenger")
    for field in ("solution_steps", "raw_response"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise CompositorV12Error(f"bound Structural full row.{field} is missing")
    try:
        raw_payload = json.loads(row["raw_response"])
    except json.JSONDecodeError as exc:
        raise CompositorV12Error("bound Structural raw_response is not coherent JSON") from exc
    if not isinstance(raw_payload, dict):
        raise CompositorV12Error("bound Structural raw_response must be an object")
    raw_answer = raw_payload.get("final_answer")
    if not isinstance(raw_answer, str) or raw_answer.strip().upper() != expected_answer:
        raise CompositorV12Error("bound Structural raw_response final_answer is inconsistent")
    generation = row.get("generation")
    if not isinstance(generation, dict) or generation.get("gold_access") is not False:
        raise CompositorV12Error("bound Structural generation must attest gold_access=false")


def _load_selector_manifest(root: Path) -> tuple[dict[str, Any], Path]:
    path = (root / SELECTOR_OUTPUT_RELATIVE_PATH).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise CompositorV12Error("v1.2 selector output path escaped experiment root") from exc
    if _sha256(path) != SELECTOR_MANIFEST_SHA256:
        raise CompositorV12Error("v1.2 selector output manifest SHA mismatch")
    manifest = _read_json(path, "v1.2 selector output manifest")
    _strict_keys(
        manifest,
        {
            "schema_version",
            "selector_id",
            "status",
            "artifact_kind",
            "rows",
            "benchmark_sha256",
            "candidate_pool_sha256",
            "freeze_sha256",
            "input_package_sha256",
            "model_closure",
            "primary_action_counts",
            "exploratory_action_counts",
            "source_union_changes",
            "image_judge_changes",
            "selection_proposals",
            "runtime_outcome_access",
        },
        "v1.2 selector output manifest",
    )
    expected = {
        "schema_version": selector.OUTPUT_MANIFEST_SCHEMA,
        "selector_id": selector.SELECTOR_ID,
        "status": "v1_2_arm_outputs_frozen_not_evaluated",
        "artifact_kind": "patch_proposals_not_a_scored_solver",
        "rows": ROW_COUNT,
        "benchmark_sha256": BENCHMARK_SHA256,
        "candidate_pool_sha256": CANDIDATE_POOL_SHA256,
        "freeze_sha256": SELECTOR_FREEZE_SHA256,
        "input_package_sha256": INPUT_PACKAGE_SHA256,
        "model_closure": [MODEL],
        "primary_action_counts": {"preserve_anchor": 272, "propose_challenger": 2},
        "exploratory_action_counts": {"preserve_anchor": 271, "propose_challenger": 3},
        "source_union_changes": 0,
        "image_judge_changes": 0,
        "runtime_outcome_access": False,
    }
    for key, value in expected.items():
        if manifest[key] != value:
            raise CompositorV12Error(f"v1.2 selector manifest {key} mismatch")
    descriptor = manifest["selection_proposals"]
    if descriptor != {"path": "selection_proposals_v1_2.jsonl", "sha256": PROPOSALS_SHA256}:
        raise CompositorV12Error("v1.2 selector proposals descriptor mismatch")
    proposal_path = (path.parent / descriptor["path"]).resolve()
    if proposal_path.parent != path.parent or _sha256(proposal_path) != PROPOSALS_SHA256:
        raise CompositorV12Error("v1.2 selector proposals SHA/path mismatch")
    return manifest, proposal_path


def _validate_proposal_row(
    proposal: dict[str, Any],
    *,
    index: int,
    task_id: str,
    route: str,
    protected: bool,
    normalized_row: dict[str, Any],
) -> None:
    _strict_keys(
        proposal,
        {
            "schema_version",
            "row_index",
            "task_id",
            "authoritative_evaluation_route",
            "protected_by_source_union",
            "anchor_answer",
            "structural_challenger",
            "native_group_answer",
            "native_member_answers",
            "parallel_group_answer",
            "primary",
            "exploratory",
        },
        f"v1.2 selector proposal[{index}]",
    )
    try:
        selector.base._assert_no_excluded_runtime_fields(proposal, f"v1.2 selector proposal[{index}]")
    except selector.base.SelectorError as exc:
        raise CompositorV12Error(str(exc)) from exc
    if proposal["schema_version"] != selector.OUTPUT_ROW_SCHEMA:
        raise CompositorV12Error("v1.2 selector proposal row schema mismatch")
    if proposal["row_index"] != index or proposal["task_id"] != task_id:
        raise CompositorV12Error("v1.2 selector proposal identity/order mismatch")
    if proposal["authoritative_evaluation_route"] != route:
        raise CompositorV12Error("v1.2 selector proposal route differs from independent authority")
    if proposal["protected_by_source_union"] is not protected:
        raise CompositorV12Error("v1.2 selector proposal protection differs from membership authority")
    if proposal["anchor_answer"] != normalized_row["anchor"]["final_answer"]:
        raise CompositorV12Error("v1.2 selector proposal anchor differs from normalized input")
    normalized_structural = normalized_row["structural"]
    normalized_structural_choice = selector._choice(normalized_structural)
    if proposal["structural_challenger"] != normalized_structural_choice:
        raise CompositorV12Error("v1.2 selector Structural challenger differs from normalized input")
    for arm in ("primary", "exploratory"):
        arm_value = proposal[arm]
        if not isinstance(arm_value, dict):
            raise CompositorV12Error(f"v1.2 selector proposal {arm} must be object")
        _strict_keys(
            arm_value,
            {"arm", "action", "selected_answer", "selected_choice", "reason"},
            f"v1.2 selector proposal[{index}].{arm}",
        )
        if arm_value["action"] not in {"preserve_anchor", "propose_challenger"}:
            raise CompositorV12Error("v1.2 selector proposal action is invalid")
        if arm_value["action"] == "preserve_anchor":
            if arm_value["selected_answer"] != proposal["anchor_answer"]:
                raise CompositorV12Error("v1.2 selector preserve action changes anchor")
        else:
            if protected or route != "deterministic":
                raise CompositorV12Error("v1.2 selector proposed a protected-slice change")
            if arm_value["selected_answer"] != proposal["structural_challenger"]:
                raise CompositorV12Error("v1.2 selector proposed answer differs from Structural challenger")
            if arm_value["selected_answer"] != normalized_structural_choice:
                raise CompositorV12Error("v1.2 selector proposed answer differs from normalized Structural row")


def load_bound_inputs(root: Path) -> dict[str, Any]:
    root = root.resolve()
    selector_freeze = selector.verify_preregistered_freeze(root)
    if selector_freeze["freeze_sha256"] != SELECTOR_FREEZE_SHA256:
        raise CompositorV12Error("active v1.2 selector freeze SHA mismatch")
    selector_manifest, proposal_path = _load_selector_manifest(root)
    proposals = _read_jsonl_raw(proposal_path, "v1.2 selector proposals", PROPOSALS_SHA256)
    selector_profile = selector.load_profile(root / "profile_v1_2.json", require_ready=True)
    package_path = (root / INPUT_PACKAGE_RELATIVE_PATH).resolve()
    if _sha256(package_path) != INPUT_PACKAGE_SHA256:
        raise CompositorV12Error("v1.2 selector input package SHA mismatch")
    normalized_rows, ordered_ids, routes, protected_ids, _ = selector.load_input_package(
        package_path, selector_profile
    )
    base_path = _translate_legacy_error(
        legacy._find_repository_file,
        root,
        BASE_SOLVER_REPOSITORY_PATH,
        BASE_SOLVER_SHA256,
    )
    base_rows = _read_jsonl_raw(base_path, "base source solver", BASE_SOLVER_SHA256)
    structural_path = (root / STRUCTURAL_RELATIVE_PATH).resolve()
    try:
        structural_path.relative_to(root)
    except ValueError as exc:
        raise CompositorV12Error("Structural input path escaped experiment root") from exc
    structural_rows = _read_jsonl_raw(
        structural_path, "bound Structural full rows", STRUCTURAL_FULL_ROWS_SHA256
    )
    for index, task_id in enumerate(ordered_ids):
        _validate_solver_row(base_rows[index].value, task_id, f"base solver row {index}")
        _validate_solver_row(structural_rows[index].value, task_id, f"Structural full row {index}")
        _validate_proposal_row(
            proposals[index].value,
            index=index,
            task_id=task_id,
            route=routes[index],
            protected=task_id in protected_ids,
            normalized_row=normalized_rows[index],
        )
    return {
        "selector_manifest": selector_manifest,
        "proposals": proposals,
        "normalized_rows": normalized_rows,
        "ordered_ids": ordered_ids,
        "routes": routes,
        "protected_ids": protected_ids,
        "base_rows": base_rows,
        "structural_rows": structural_rows,
        "base_path": base_path,
        "structural_path": structural_path,
    }


def _compose_arm(
    *,
    arm: str,
    bound: dict[str, Any],
    selector_manifest_sha256: str,
) -> tuple[list[bytes], list[dict[str, Any]], Counter[str]]:
    if arm not in {"primary", "exploratory"}:
        raise CompositorV12Error("unknown v1.2 composition arm")
    output_lines: list[bytes] = []
    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, task_id in enumerate(bound["ordered_ids"]):
        proposal_row = bound["proposals"][index]
        proposal = proposal_row.value
        arm_proposal = proposal[arm]
        base = bound["base_rows"][index]
        structural = bound["structural_rows"][index]
        normalized = bound["normalized_rows"][index]
        route = bound["routes"][index]
        protected = task_id in bound["protected_ids"]
        if arm_proposal["action"] == "preserve_anchor":
            output_raw = base.raw_line
            action = "base_passthrough_exact_bytes"
            source_row_sha = _bytes_sha256(base.raw_line)
        else:
            if protected or route != "deterministic":
                raise CompositorV12Error("v1.2 replacement reached protected slice")
            selected_answer = str(arm_proposal["selected_answer"] or "").strip().upper()
            _validate_structural_payload(structural.value, task_id, selected_answer)
            if selected_answer != selector._choice(normalized["structural"]):
                raise CompositorV12Error("v1.2 full and normalized Structural answers differ")
            if selected_answer == str(base.value["final_answer"]).strip().upper():
                raise CompositorV12Error("v1.2 selector replacement does not differ from base")
            changed = copy.deepcopy(structural.value)
            changed["task_id"] = base.value["task_id"]
            generation = changed.get("generation")
            if not isinstance(generation, dict):
                raise CompositorV12Error("bound Structural row generation must be object")
            structural_source_access = normalized["structural"]["generation"]["source_access"]
            if not isinstance(structural_source_access, bool):
                raise CompositorV12Error("normalized Structural source_access is not boolean")
            generation["baseline_selector_composition_v1_2"] = {
                "schema_version": ROW_PROVENANCE_SCHEMA,
                "compositor_id": COMPOSITOR_ID,
                "arm": arm,
                "selector_id": selector.SELECTOR_ID,
                "selector_freeze_sha256": SELECTOR_FREEZE_SHA256,
                "selector_output_manifest_sha256": selector_manifest_sha256,
                "selector_proposal_row_sha256": _bytes_sha256(proposal_row.raw_line),
                "base_row_sha256": _bytes_sha256(base.raw_line),
                "bound_structural_full_row_sha256": _bytes_sha256(structural.raw_line),
                "authoritative_route": route,
                "protected_by_source_union": False,
                "structural_source_access": structural_source_access,
                "action": "replace_with_bound_structural_full_row",
                "selected_answer": selected_answer,
                "runtime_outcome_access": False,
            }
            _validate_solver_row(changed, task_id, f"v1.2 composited {arm} row {index}")
            if changed["final_answer"].strip().upper() != selected_answer:
                raise CompositorV12Error("v1.2 composited row answer consistency failed")
            output_raw = _canonical_json(changed)
            action = "bound_structural_selector_replacement"
            source_row_sha = _bytes_sha256(structural.raw_line)
        output_lines.append(output_raw)
        counts[action] += 1
        decisions.append(
            {
                "schema_version": DECISION_SCHEMA,
                "row_index": index,
                "task_id": task_id,
                "arm": arm,
                "selector_action": arm_proposal["action"],
                "composition_action": action,
                "authoritative_route": route,
                "protected_by_source_union": protected,
                "base_row_sha256": _bytes_sha256(base.raw_line),
                "source_row_sha256": source_row_sha,
                "selector_proposal_row_sha256": _bytes_sha256(proposal_row.raw_line),
                "output_row_sha256": _bytes_sha256(output_raw),
                "runtime_outcome_access": False,
            }
        )
    return output_lines, decisions, counts


def run_compositor(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    prereg = verify_preregistered_freeze(root)
    bound = load_bound_inputs(root)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, dict[str, Any]] = {}
    counts_by_arm: dict[str, dict[str, int]] = {}
    expected = {"primary": 2, "exploratory": 3}
    for arm in ("primary", "exploratory"):
        solver_lines, decisions, counts = _compose_arm(
            arm=arm,
            bound=bound,
            selector_manifest_sha256=SELECTOR_MANIFEST_SHA256,
        )
        replacements = expected[arm]
        if counts != Counter(
            {
                "base_passthrough_exact_bytes": ROW_COUNT - replacements,
                "bound_structural_selector_replacement": replacements,
            }
        ):
            raise CompositorV12Error(f"v1.2 {arm} composition counts mismatch")
        solver_path = output_dir / f"{arm}_solver.jsonl"
        decisions_path = output_dir / f"{arm}_decisions.jsonl"
        solver_path.write_bytes(b"".join(solver_lines))
        decisions_path.write_bytes(b"".join(_canonical_json(row) for row in decisions))
        artifacts[arm] = {
            "solver": {"path": solver_path.name, "rows": ROW_COUNT, "sha256": _sha256(solver_path)},
            "decisions": {
                "path": decisions_path.name,
                "rows": ROW_COUNT,
                "sha256": _sha256(decisions_path),
            },
        }
        counts_by_arm[arm] = dict(sorted(counts.items()))
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "compositor_id": COMPOSITOR_ID,
        "status": "composited_frozen_before_evaluation",
        "artifact_kind": "two_unscored_9b_solver_variants",
        "rows_per_solver": ROW_COUNT,
        "model_closure": [MODEL],
        "benchmark_sha256": BENCHMARK_SHA256,
        "runtime_outcome_access": False,
        "inputs": {
            "compositor_preregistered_freeze_sha256": prereg["freeze_sha256"],
            "selector_freeze_sha256": SELECTOR_FREEZE_SHA256,
            "selector_output_manifest_sha256": SELECTOR_MANIFEST_SHA256,
            "selector_proposals_sha256": PROPOSALS_SHA256,
            "selector_input_package_sha256": INPUT_PACKAGE_SHA256,
            "base_source_solver_sha256": BASE_SOLVER_SHA256,
            "bound_structural_full_rows_sha256": STRUCTURAL_FULL_ROWS_SHA256,
        },
        "preservation": {
            "source_union_rows": 156,
            "source_union_changes_primary": 0,
            "source_union_changes_exploratory": 0,
            "image_judge_rows": 97,
            "image_judge_changes_primary": 0,
            "image_judge_changes_exploratory": 0,
            "passthrough_representation": "exact_original_base_jsonl_line_bytes",
        },
        "composition_counts": counts_by_arm,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "composition_manifest_v1_2.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    output_freeze = {
        "schema_version": OUTPUT_FREEZE_SCHEMA,
        "compositor_id": COMPOSITOR_ID,
        "status": "output_frozen_unscored_not_evaluated",
        "runtime_outcome_access": False,
        "compositor_preregistered_freeze_sha256": prereg["freeze_sha256"],
        "composition_manifest": {"path": manifest_path.name, "sha256": _sha256(manifest_path)},
        "artifacts": artifacts,
    }
    freeze_path = output_dir / "COMPOSITION_OUTPUT_FREEZE_v1_2.json"
    freeze_path.write_bytes(_canonical_json(output_freeze))
    return {
        "status": output_freeze["status"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "output_freeze_path": str(freeze_path),
        "output_freeze_sha256": _sha256(freeze_path),
        "artifacts": artifacts,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed unscored selector compositor v1.2")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-freeze", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.output_dir is not None:
            report = run_compositor(args.root, args.output_dir)
        else:
            if not args.verify_freeze:
                raise CompositorV12Error("use --verify-freeze or provide --output-dir")
            report = verify_preregistered_freeze(args.root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (CompositorV12Error, selector.SelectorV12Error, selector.base.SelectorError) as exc:
        print(f"compositor v1.2 error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
