from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import selector_v1_1 as selector


MODEL = "Qwen/Qwen3.5-9B"
COMPOSITOR_ID = "maxim_9b_baseline_selector_compositor_v1_1"
PROFILE_SCHEMA = "maxim-9b-baseline-selector-compositor-profile-v1.1"
PREREG_FREEZE_SCHEMA = "maxim-9b-baseline-selector-compositor-preregistered-freeze-v1.1"
MANIFEST_SCHEMA = "maxim-9b-baseline-selector-composition-manifest-v1.1"
DECISION_SCHEMA = "maxim-9b-baseline-selector-composition-decision-v1.1"
OUTPUT_FREEZE_SCHEMA = "maxim-9b-baseline-selector-composition-output-freeze-v1.1"
ROW_COUNT = 274

SELECTOR_FREEZE_SHA256 = "858ef54c3bb558bdd31f8d0ead605bf3a3bcdb8816f97c0d0d93f86b1eaf4193"
SELECTOR_MANIFEST_SHA256 = "c978351b5d482b6af87c8731e7f3f1cb3b0f208397bca0d8ed3ae2043f7b77ae"
PROPOSALS_SHA256 = "ce73875f63496553bdbd2a8ed69bba63a15f1822cf2c438239236b8b350a8ec7"
INPUT_PACKAGE_SHA256 = "f2e7bdf8ea0cd8d44d073c3cc3f7a6933a98de2b032d44e1e5625e98eb869f0e"
BASE_SOLVER_SHA256 = "9d26067064ee07fe480391759782c86d66adbb76dbc0da0d86ccc1b3f035211e"
V4_FULL_ROWS_SHA256 = "0fd0e6fef6b220749faa015e7a163cdf596e070afaaad407e4d36bc9b1337307"
BENCHMARK_SHA256 = "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
BASE_SOLVER_REPOSITORY_PATH = (
    "reports/maxim_9b_source_replay_v1_20260809/active_crop/"
    "fill_composed/solver.jsonl"
)
V4_RELATIVE_PATH = "input/frozen/upstream/native_thinking_math_router_v4.jsonl"
SELECTOR_OUTPUT_RELATIVE_PATH = "output_v1_1/selector_manifest.json"
INPUT_PACKAGE_RELATIVE_PATH = "input/frozen/input_package_v1_1.json"

EXPECTED_PROFILE = {
    "schema_version": PROFILE_SCHEMA,
    "compositor_id": COMPOSITOR_ID,
    "status": "preregistered_before_any_gold_score_correctness_or_judge_evaluation_of_composited_outputs",
    "chronology": {
        "historical_benchmark_aggregate_score_and_prior_task_outcomes_were_known": True,
        "selector_output_was_frozen_unscored_before_composition": True,
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
        "native_v4_full_rows_relative_path": V4_RELATIVE_PATH,
        "native_v4_full_rows_sha256": V4_FULL_ROWS_SHA256,
        "benchmark_sha256": BENCHMARK_SHA256,
        "rows": ROW_COUNT,
    },
    "rules": {
        "arms": ["primary", "secondary"],
        "base": "frozen_final_activecrop9b_source_solver",
        "replacement_source": "bound_native_thinking_math_router_v4_full_row",
        "replace_only_when_selector_action": "propose_challenger",
        "replacement_gates": [
            "authoritative_route_is_deterministic",
            "task_is_outside_pinned_source_union",
            "proposal_selected_answer_equals_router_challenger",
            "proposal_selected_answer_equals_normalized_v4_answer",
            "proposal_selected_answer_equals_full_v4_row_final_answer",
            "base_v4_proposal_and_authoritative_task_identity_match",
            "v4_full_row_is_pure_qwen35_9b_and_has_coherent_answer_payload",
        ],
        "source_union_policy": "preserve_all_156_base_rows_as_exact_original_line_bytes",
        "image_judge_policy": "preserve_all_97_base_rows_as_exact_original_line_bytes",
        "other_passthrough_policy": "preserve_every_nonproposed_base_row_as_exact_original_line_bytes",
        "changed_row_policy": "copy_coherent_full_v4_row_and_add_explicit_selector_composition_provenance",
        "no_fallback_repair": True,
        "fail_closed": True,
    },
    "expected_selector_actions": {"primary_proposals": 3, "secondary_proposals": 5},
}


class CompositorError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawRow:
    value: dict[str, Any]
    raw_line: bytes


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CompositorError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompositorError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompositorError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompositorError(f"{label} must be a JSON object")
    return value


def _strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise CompositorError(
            f"{label} schema mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _read_jsonl_raw(path: Path, label: str, expected_sha256: str) -> list[RawRow]:
    if _sha256(path) != expected_sha256:
        raise CompositorError(f"{label} SHA-256 mismatch")
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        raise CompositorError(f"cannot read {label}: {exc}") from exc
    rows: list[RawRow] = []
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        if not raw.endswith(b"\n"):
            raise CompositorError(f"{label}:{line_number} lacks a terminating newline")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompositorError(f"{label}:{line_number} is invalid UTF-8 JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise CompositorError(f"{label}:{line_number} must be an object")
        rows.append(RawRow(value=value, raw_line=raw))
    if len(rows) != ROW_COUNT:
        raise CompositorError(f"{label} must contain exactly {ROW_COUNT} rows")
    return rows


def _find_repository_file(start: Path, relative_path: str, expected_sha256: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise CompositorError("repository-relative input path is unsafe")
    candidates: list[Path] = []
    for ancestor in (start.resolve(), *start.resolve().parents):
        candidate = (ancestor / relative).resolve()
        try:
            candidate.relative_to(ancestor)
        except ValueError:
            continue
        if candidate.is_file() and _sha256(candidate) == expected_sha256:
            candidates.append(candidate)
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise CompositorError(
            f"repository input must resolve uniquely by SHA; found {len(unique)} candidates"
        )
    return unique[0]


def load_profile(root: Path) -> dict[str, Any]:
    profile = _read_json(root / "compositor_profile_v1_1.json", "compositor profile")
    if profile != EXPECTED_PROFILE:
        raise CompositorError("compositor profile differs from exact v1.1 contract")
    return profile


def verify_preregistered_freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    freeze_path = root / "COMPOSITOR_PREREGISTERED_FREEZE.json"
    freeze = _read_json(freeze_path, "compositor preregistered freeze")
    _strict_keys(
        freeze,
        {
            "schema_version",
            "compositor_id",
            "status",
            "frozen_before_composition",
            "frozen_before_any_gold_score_correctness_or_judge_evaluation_of_composited_outputs",
            "selector_freeze_sha256",
            "selector_output_manifest_sha256",
            "artifacts",
        },
        "compositor preregistered freeze",
    )
    if freeze["schema_version"] != PREREG_FREEZE_SCHEMA:
        raise CompositorError("compositor freeze schema mismatch")
    if freeze["compositor_id"] != COMPOSITOR_ID:
        raise CompositorError("compositor freeze identity mismatch")
    if freeze["status"] != "preregistered_not_composed_not_evaluated":
        raise CompositorError("compositor freeze status mismatch")
    if freeze["frozen_before_composition"] is not True:
        raise CompositorError("compositor was not frozen before composition")
    if freeze[
        "frozen_before_any_gold_score_correctness_or_judge_evaluation_of_composited_outputs"
    ] is not True:
        raise CompositorError("compositor freeze does not predate evaluation")
    if freeze["selector_freeze_sha256"] != SELECTOR_FREEZE_SHA256:
        raise CompositorError("compositor freeze selector pin mismatch")
    if freeze["selector_output_manifest_sha256"] != SELECTOR_MANIFEST_SHA256:
        raise CompositorError("compositor freeze selector output pin mismatch")
    artifacts = freeze["artifacts"]
    if not isinstance(artifacts, dict):
        raise CompositorError("compositor freeze artifacts must be object")
    _strict_keys(artifacts, {"profile", "code", "tests"}, "compositor freeze artifacts")
    expected_names = {
        "profile": "compositor_profile_v1_1.json",
        "code": "compositor_v1_1.py",
        "tests": "test_compositor_v1_1.py",
    }
    verified: dict[str, dict[str, str]] = {}
    for role, expected_name in expected_names.items():
        descriptor = artifacts[role]
        if not isinstance(descriptor, dict):
            raise CompositorError(f"compositor freeze {role} descriptor must be object")
        _strict_keys(descriptor, {"path", "sha256"}, f"compositor freeze {role}")
        if descriptor["path"] != expected_name:
            raise CompositorError(f"compositor freeze {role} path mismatch")
        digest = descriptor["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise CompositorError(f"compositor freeze {role} SHA malformed")
        path = (root / expected_name).resolve()
        if _sha256(path) != digest:
            raise CompositorError(f"compositor freeze {role} SHA mismatch")
        verified[role] = {"path": str(path), "sha256": digest}
    load_profile(root)
    return {
        "status": "compositor_preregistered_freeze_verified",
        "freeze_path": str(freeze_path),
        "freeze_sha256": _sha256(freeze_path),
        "artifacts": verified,
    }


def _validate_solver_row(row: dict[str, Any], task_id: str, label: str) -> None:
    selector._assert_no_excluded_runtime_fields(row, label)
    if row.get("task_id") != task_id:
        raise CompositorError(f"{label} task identity mismatch")
    if row.get("model") != MODEL:
        raise CompositorError(f"{label} model closure is not pure Qwen3.5-9B")
    answer = row.get("final_answer")
    if not isinstance(answer, str) or not answer.strip():
        raise CompositorError(f"{label} final_answer is empty or missing")


def _validate_v4_payload(row: dict[str, Any], task_id: str, expected_answer: str) -> None:
    _validate_solver_row(row, task_id, "bound V4 full row")
    if not selector.CHOICE_RE.fullmatch(str(row["final_answer"])):
        raise CompositorError("bound V4 full row final_answer is not A-E")
    if row.get("reasoning") is not None and not isinstance(row.get("reasoning"), str):
        raise CompositorError("bound V4 full row.reasoning must be string or null")
    for field in ("solution_steps", "raw_response"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise CompositorError(f"bound V4 full row.{field} is missing")
    if row["final_answer"].strip().upper() != expected_answer:
        raise CompositorError("bound V4 full row answer differs from selector challenger")
    try:
        raw_payload = json.loads(row["raw_response"])
    except json.JSONDecodeError as exc:
        raise CompositorError("bound V4 raw_response is not coherent JSON") from exc
    if not isinstance(raw_payload, dict):
        raise CompositorError("bound V4 raw_response must be an object")
    raw_answer = raw_payload.get("final_answer")
    if not isinstance(raw_answer, str) or raw_answer.strip().upper() != expected_answer:
        raise CompositorError("bound V4 raw_response final_answer is inconsistent")


def _load_selector_manifest(root: Path) -> tuple[dict[str, Any], Path]:
    path = (root / SELECTOR_OUTPUT_RELATIVE_PATH).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise CompositorError("selector output path escaped experiment root") from exc
    if _sha256(path) != SELECTOR_MANIFEST_SHA256:
        raise CompositorError("selector output manifest SHA mismatch")
    manifest = _read_json(path, "selector output manifest")
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
            "profile_sha256",
            "primary_action_counts",
            "secondary_action_counts",
            "source_union_changes",
            "image_judge_changes",
            "selection_proposals",
            "runtime_outcome_access",
        },
        "selector output manifest",
    )
    if manifest["status"] != "new_selector_arm_outputs_frozen_not_evaluated":
        raise CompositorError("selector output is not frozen and unscored")
    if manifest["artifact_kind"] != "patch_proposals_not_a_scored_solver":
        raise CompositorError("selector output artifact kind mismatch")
    if manifest["rows"] != ROW_COUNT or manifest["benchmark_sha256"] != BENCHMARK_SHA256:
        raise CompositorError("selector output benchmark/row mismatch")
    if manifest["freeze_sha256"] != SELECTOR_FREEZE_SHA256:
        raise CompositorError("selector output freeze pin mismatch")
    if manifest["runtime_outcome_access"] is not False:
        raise CompositorError("selector output runtime outcome access is not false")
    if manifest["source_union_changes"] != 0 or manifest["image_judge_changes"] != 0:
        raise CompositorError("selector output violates protected-slice contract")
    if manifest["primary_action_counts"] != {"preserve_anchor": 271, "propose_challenger": 3}:
        raise CompositorError("selector primary action counts mismatch")
    if manifest["secondary_action_counts"] != {"preserve_anchor": 269, "propose_challenger": 5}:
        raise CompositorError("selector secondary action counts mismatch")
    descriptor = manifest["selection_proposals"]
    if descriptor != {"path": "selection_proposals.jsonl", "sha256": PROPOSALS_SHA256}:
        raise CompositorError("selector proposals descriptor mismatch")
    proposal_path = (path.parent / descriptor["path"]).resolve()
    if proposal_path.parent != path.parent or _sha256(proposal_path) != PROPOSALS_SHA256:
        raise CompositorError("selector proposals SHA/path mismatch")
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
            "router_challenger",
            "primary",
            "secondary",
        },
        f"selector proposal[{index}]",
    )
    selector._assert_no_excluded_runtime_fields(proposal, f"selector proposal[{index}]")
    if proposal["schema_version"] != selector.OUTPUT_ROW_SCHEMA:
        raise CompositorError("selector proposal row schema mismatch")
    if proposal["row_index"] != index or proposal["task_id"] != task_id:
        raise CompositorError("selector proposal identity/order mismatch")
    if proposal["authoritative_evaluation_route"] != route:
        raise CompositorError("selector proposal route differs from independent route authority")
    if proposal["protected_by_source_union"] is not protected:
        raise CompositorError("selector proposal protection flag differs from membership authority")
    normalized_anchor = normalized_row["anchor"]["final_answer"]
    if proposal["anchor_answer"] != normalized_anchor:
        raise CompositorError("selector proposal anchor answer differs from bound input row")
    for arm in ("primary", "secondary"):
        arm_value = proposal[arm]
        if not isinstance(arm_value, dict):
            raise CompositorError(f"selector proposal {arm} must be object")
        _strict_keys(
            arm_value,
            {
                "arm",
                "action",
                "selected_answer",
                "selected_choice",
                "reason",
                "raw_parallel_support",
            },
            f"selector proposal[{index}].{arm}",
        )
        if arm_value["action"] not in {"preserve_anchor", "propose_challenger"}:
            raise CompositorError("selector proposal action is invalid")
        if arm_value["action"] == "preserve_anchor":
            if arm_value["selected_answer"] != proposal["anchor_answer"]:
                raise CompositorError("selector preserve action does not preserve anchor answer")
        else:
            if protected or route != "deterministic":
                raise CompositorError("selector proposed a change on a protected slice")
            if arm_value["selected_answer"] != proposal["router_challenger"]:
                raise CompositorError("selector proposed answer differs from router challenger")
            normalized_v4 = normalized_row["routers"]["v4"]
            if not normalized_v4["available"]:
                raise CompositorError("selector proposed from unavailable normalized V4")
            if arm_value["selected_answer"] != normalized_v4["final_answer"]:
                raise CompositorError("selector proposed answer differs from normalized V4")


def load_bound_inputs(root: Path) -> dict[str, Any]:
    root = root.resolve()
    selector_freeze = selector.verify_preregistered_freeze(root)
    if selector_freeze["freeze_sha256"] != SELECTOR_FREEZE_SHA256:
        raise CompositorError("active selector freeze SHA mismatch")
    selector_manifest, proposal_path = _load_selector_manifest(root)
    proposals = _read_jsonl_raw(proposal_path, "selector proposals", PROPOSALS_SHA256)

    selector_profile = selector.load_profile(root / "profile_v1_1.json", require_ready=True)
    package_path = (root / INPUT_PACKAGE_RELATIVE_PATH).resolve()
    if _sha256(package_path) != INPUT_PACKAGE_SHA256:
        raise CompositorError("selector input package SHA mismatch")
    normalized_rows, ordered_ids, routes, protected_ids, _ = selector.load_input_package(
        package_path,
        selector_profile["authority_pins"],
    )
    base_path = _find_repository_file(root, BASE_SOLVER_REPOSITORY_PATH, BASE_SOLVER_SHA256)
    base_rows = _read_jsonl_raw(base_path, "base source solver", BASE_SOLVER_SHA256)
    v4_path = (root / V4_RELATIVE_PATH).resolve()
    try:
        v4_path.relative_to(root)
    except ValueError as exc:
        raise CompositorError("V4 input path escaped experiment root") from exc
    v4_rows = _read_jsonl_raw(v4_path, "bound V4 full rows", V4_FULL_ROWS_SHA256)

    for index, task_id in enumerate(ordered_ids):
        _validate_solver_row(base_rows[index].value, task_id, f"base solver row {index}")
        _validate_solver_row(v4_rows[index].value, task_id, f"V4 full row {index}")
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
        "v4_rows": v4_rows,
        "base_path": base_path,
        "v4_path": v4_path,
    }


def _compose_arm(
    *,
    arm: str,
    bound: dict[str, Any],
    selector_manifest_sha256: str,
) -> tuple[list[bytes], list[dict[str, Any]], Counter[str]]:
    output_lines: list[bytes] = []
    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, task_id in enumerate(bound["ordered_ids"]):
        proposal_row = bound["proposals"][index]
        proposal = proposal_row.value
        arm_proposal = proposal[arm]
        base = bound["base_rows"][index]
        v4 = bound["v4_rows"][index]
        route = bound["routes"][index]
        protected = task_id in bound["protected_ids"]
        if arm_proposal["action"] == "preserve_anchor":
            output_raw = base.raw_line
            action = "base_passthrough_exact_bytes"
            source_row_sha = _bytes_sha256(base.raw_line)
        else:
            if protected or route != "deterministic":
                raise CompositorError("replacement reached protected slice")
            selected_answer = str(arm_proposal["selected_answer"] or "").strip().upper()
            _validate_v4_payload(v4.value, task_id, selected_answer)
            if selected_answer == str(base.value["final_answer"]).strip().upper():
                raise CompositorError("selector replacement does not differ from base answer")
            changed = copy.deepcopy(v4.value)
            changed["task_id"] = base.value["task_id"]
            generation = changed.get("generation")
            if not isinstance(generation, dict):
                raise CompositorError("bound V4 row generation must be object")
            generation["baseline_selector_composition_v1_1"] = {
                "schema_version": "maxim-9b-baseline-selector-row-composition-provenance-v1.1",
                "compositor_id": COMPOSITOR_ID,
                "arm": arm,
                "selector_id": selector.SELECTOR_ID,
                "selector_freeze_sha256": SELECTOR_FREEZE_SHA256,
                "selector_output_manifest_sha256": selector_manifest_sha256,
                "selector_proposal_row_sha256": _bytes_sha256(proposal_row.raw_line),
                "base_row_sha256": _bytes_sha256(base.raw_line),
                "bound_v4_full_row_sha256": _bytes_sha256(v4.raw_line),
                "authoritative_route": route,
                "protected_by_source_union": False,
                "action": "replace_with_bound_v4_full_row",
                "selected_answer": selected_answer,
                "runtime_outcome_access": False,
            }
            _validate_solver_row(changed, task_id, f"composited {arm} row {index}")
            if changed["final_answer"].strip().upper() != selected_answer:
                raise CompositorError("composited row answer consistency failed")
            output_raw = _canonical_json(changed)
            action = "bound_v4_selector_replacement"
            source_row_sha = _bytes_sha256(v4.raw_line)
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
    for arm in ("primary", "secondary"):
        solver_lines, decisions, counts = _compose_arm(
            arm=arm,
            bound=bound,
            selector_manifest_sha256=SELECTOR_MANIFEST_SHA256,
        )
        expected_replacements = 3 if arm == "primary" else 5
        if counts != Counter(
            {
                "base_passthrough_exact_bytes": ROW_COUNT - expected_replacements,
                "bound_v4_selector_replacement": expected_replacements,
            }
        ):
            raise CompositorError(f"{arm} composition counts mismatch")
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
            "bound_v4_full_rows_sha256": V4_FULL_ROWS_SHA256,
        },
        "preservation": {
            "source_union_rows": 156,
            "source_union_changes_primary": 0,
            "source_union_changes_secondary": 0,
            "image_judge_rows": 97,
            "image_judge_changes_primary": 0,
            "image_judge_changes_secondary": 0,
            "passthrough_representation": "exact_original_base_jsonl_line_bytes",
        },
        "composition_counts": counts_by_arm,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "composition_manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    output_freeze = {
        "schema_version": OUTPUT_FREEZE_SCHEMA,
        "compositor_id": COMPOSITOR_ID,
        "status": "output_frozen_unscored_not_evaluated",
        "runtime_outcome_access": False,
        "compositor_preregistered_freeze_sha256": prereg["freeze_sha256"],
        "composition_manifest": {
            "path": manifest_path.name,
            "sha256": _sha256(manifest_path),
        },
        "artifacts": artifacts,
    }
    freeze_path = output_dir / "COMPOSITION_OUTPUT_FREEZE.json"
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
    parser = argparse.ArgumentParser(description="Fail-closed unscored selector compositor v1.1")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-freeze", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.output_dir is not None:
            report = run_compositor(args.root, args.output_dir)
        else:
            if not args.verify_freeze:
                raise CompositorError("use --verify-freeze or provide --output-dir")
            report = verify_preregistered_freeze(args.root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (CompositorError, selector.SelectorError) as exc:
        print(f"compositor v1.1 error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
