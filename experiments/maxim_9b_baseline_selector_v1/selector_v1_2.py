from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import selector_v1_1 as base


MODEL = base.MODEL
SELECTOR_ID = "maxim_9b_baseline_selector_v1_2"
PROFILE_SCHEMA = "maxim-9b-baseline-selector-profile-v1.2"
FREEZE_SCHEMA = "maxim-9b-baseline-selector-preregistered-freeze-v1.2"
PACKAGE_SCHEMA = "maxim-9b-baseline-selector-input-package-v1.2"
POOL_ROW_SCHEMA = "maxim-9b-baseline-selector-pool-row-v1.2"
BINDING_SCHEMA = "maxim-9b-baseline-selector-combined-row-bindings-v1.2"
OUTPUT_ROW_SCHEMA = "maxim-9b-baseline-selector-output-row-v1.2"
OUTPUT_MANIFEST_SCHEMA = "maxim-9b-baseline-selector-output-manifest-v1.2"
ROW_COUNT = 274
STRUCTURAL_ROLE = "structural_strict_9b"
UPSTREAM_ROLES = (*base.UPSTREAM_ROLES, STRUCTURAL_ROLE)
INPUT_PACKAGE_RELATIVE_PATH = "input/v1_2/frozen/input_package_v1_2.json"

CHRONOLOGY = {
    "historical_benchmark_aggregate_score_and_prior_task_outcomes_were_known_before_freeze": True,
    "rules_and_profile_frozen_before_generation_and_any_gold_score_correctness_or_judge_evaluation_of_v1_2_arm_outputs": True,
    "selector_runtime_inputs_and_algorithm_do_not_read_gold_reference_correctness_outcomes_or_judge_verdicts": True,
}

ALGORITHM = {
    "anchor": "active_crop_v2",
    "shared_safety_gates": [
        "authoritative_route_is_deterministic",
        "authoritative_task_id_is_outside_pinned_source_union",
        "anchor_and_structural_answers_are_valid_A_to_E",
        "structural_challenger_differs_from_anchor",
        "all_candidate_projections_match_pinned_per_upstream_row_bindings",
    ],
    "primary_arm": {
        "name": "three_group_unanimity",
        "role": "primary_conservative",
        "groups": [
            "structural_strict_9b_single_member_group",
            "native_group_only_when_v4_equals_v5",
            "parallel_group_only_when_parallel8_v1_final_equals_parallel8_reasoning_first_v2_final",
        ],
        "rule": "all_three_group_answers_equal_the_same_structural_challenger_not_equal_to_anchor",
    },
    "exploratory_arm": {
        "name": "structural_parallel_plus_one_native",
        "role": "preregistered_exploratory",
        "rule": "structural_equals_parallel_group_and_at_least_one_of_v4_or_v5_equals_same_challenger_not_equal_to_anchor",
    },
    "parallel_raw_votes_used": False,
    "fallback": "preserve_active_crop_v2_anchor",
    "source_union_policy": "always_preserve_anchor",
    "image_judge_policy": "always_preserve_anchor",
    "no_post_evaluation_rule_change": True,
}


class SelectorV12Error(RuntimeError):
    pass


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
        raise SelectorV12Error(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise SelectorV12Error(
            f"{label} schema mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _hex64(value: Any, label: str) -> str:
    digest = str(value or "").casefold()
    if not base.SHA256_RE.fullmatch(digest):
        raise SelectorV12Error(f"{label} must be lowercase SHA-256")
    return digest


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return base._read_json(path, label)
    except base.SelectorError as exc:
        raise SelectorV12Error(str(exc)) from exc


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        return base._read_jsonl(path, label)
    except base.SelectorError as exc:
        raise SelectorV12Error(str(exc)) from exc


def _relative_descriptor(
    base_dir: Path, value: Any, label: str, expected_sha256: str
) -> Path:
    try:
        return base._relative_descriptor(
            base_dir, value, label, expected_sha256=expected_sha256
        )
    except base.SelectorError as exc:
        raise SelectorV12Error(str(exc)) from exc


def _validate_authority_pins(value: Any, require_ready: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectorV12Error("authority_pins must be object")
    _strict_keys(
        value,
        {
            "status",
            "input_package_manifest_sha256",
            "benchmark_order_sha256",
            "route_map_sha256",
            "source_union_membership_sha256",
            "candidate_pool_sha256",
            "row_bindings_sha256",
        },
        "authority_pins",
    )
    if value["status"] not in {"awaiting_structural_input_lock", "locked_before_v1_2_evaluation"}:
        raise SelectorV12Error("authority_pins status invalid")
    for key in set(value) - {"status"}:
        if value[key] is not None:
            _hex64(value[key], f"authority_pins.{key}")
    ready = value["status"] == "locked_before_v1_2_evaluation" and all(
        value[key] is not None for key in value if key != "status"
    )
    if require_ready and not ready:
        raise SelectorV12Error("v1.2 input authority is not locked")
    return value


def validate_profile(profile: dict[str, Any], require_ready: bool) -> dict[str, Any]:
    _strict_keys(
        profile,
        {
            "schema_version",
            "selector_id",
            "status",
            "model_closure",
            "chronology",
            "group_provenance",
            "benchmark",
            "upstream_artifact_sha256",
            "protected_scope",
            "algorithm",
            "runtime_input_contract",
            "authority_pins",
        },
        "selector v1.2 profile",
    )
    if profile["schema_version"] != PROFILE_SCHEMA or profile["selector_id"] != SELECTOR_ID:
        raise SelectorV12Error("selector v1.2 profile identity mismatch")
    allowed_status = {
        "draft_awaiting_structural_input_lock_not_frozen",
        "preregistered_after_historical_outcomes_known_before_v1_2_evaluation",
    }
    if profile["status"] not in allowed_status:
        raise SelectorV12Error("selector v1.2 profile status invalid")
    if require_ready and profile["status"] != (
        "preregistered_after_historical_outcomes_known_before_v1_2_evaluation"
    ):
        raise SelectorV12Error("selector v1.2 profile is draft")
    if profile["model_closure"] != [MODEL] or profile["chronology"] != CHRONOLOGY:
        raise SelectorV12Error("selector v1.2 model/chronology mismatch")
    expected_group_provenance = {
        "structural_group": "separate_structural_strict_9b_generation_family_with_declared_source_access",
        "native_group": "two_correlated_native_thinking_ablations_group_vote_exists_only_on_v4_v5_agreement",
        "parallel_group": "two_separately_executed_batches_group_vote_exists_only_on_final_answer_agreement",
        "independence_scope": "three_distinct_preregistered_evidence_groups_not_an_iid_or_statistical_independence_claim",
    }
    if profile["group_provenance"] != expected_group_provenance:
        raise SelectorV12Error("selector v1.2 group provenance mismatch")
    if profile["benchmark"] != {"sha256": base.BENCHMARK_SHA256, "ordered_rows": ROW_COUNT}:
        raise SelectorV12Error("selector v1.2 benchmark mismatch")
    upstream = profile["upstream_artifact_sha256"]
    if not isinstance(upstream, dict):
        raise SelectorV12Error("selector v1.2 upstream pins must be object")
    _strict_keys(upstream, set(UPSTREAM_ROLES), "selector v1.2 upstream pins")
    for role in base.UPSTREAM_ROLES:
        if upstream[role] != base.UPSTREAM_SHA256[role]:
            raise SelectorV12Error(f"selector v1.2 inherited upstream pin mismatch for {role}")
    if upstream[STRUCTURAL_ROLE] is not None:
        _hex64(upstream[STRUCTURAL_ROLE], "structural upstream pin")
    if require_ready and upstream[STRUCTURAL_ROLE] is None:
        raise SelectorV12Error("structural upstream pin is not locked")
    expected_protected = {
        "aggregate_file_sha256": base.SOURCE_AGGREGATE_SHA256,
        "source_union_projection_sha256": base.SOURCE_UNION_SHA256,
        "source_union_size": base.SOURCE_UNION_SIZE,
        "runtime_membership_schema": base.MEMBERSHIP_SCHEMA,
        "semantics": "safety_veto_only_not_a_quality_group",
    }
    if profile["protected_scope"] != expected_protected:
        raise SelectorV12Error("selector v1.2 protected scope mismatch")
    if profile["algorithm"] != ALGORITHM:
        raise SelectorV12Error("selector v1.2 algorithm contract mismatch")
    expected_runtime = {
        "pool_contains_no_task_id_or_evaluation_route": True,
        "identity_comes_only_from_pinned_benchmark_order": True,
        "route_comes_only_from_pinned_outcome_free_route_map": True,
        "all_six_role_projections_match_pinned_row_bindings": True,
        "runtime_excludes_gold_reference_correctness_outcomes_and_judge_verdicts": True,
        "structural_source_access_is_explicit_per_row_and_not_an_outcome_signal": True,
    }
    if profile["runtime_input_contract"] != expected_runtime:
        raise SelectorV12Error("selector v1.2 runtime contract mismatch")
    _validate_authority_pins(profile["authority_pins"], require_ready)
    return profile


def load_profile(path: Path, require_ready: bool) -> dict[str, Any]:
    return validate_profile(_read_json(path, "selector v1.2 profile"), require_ready)


def verify_preregistered_freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    path = root / "PREREGISTERED_FREEZE_v1_2.json"
    freeze = _read_json(path, "selector v1.2 freeze")
    _strict_keys(
        freeze,
        {
            "schema_version",
            "selector_id",
            "status",
            "historical_outcomes_known_before_freeze",
            "frozen_before_any_evaluation_of_v1_2_outputs",
            "runtime_outcome_access",
            "artifacts",
            "input_package_sha256",
        },
        "selector v1.2 freeze",
    )
    if freeze["schema_version"] != FREEZE_SCHEMA or freeze["selector_id"] != SELECTOR_ID:
        raise SelectorV12Error("selector v1.2 freeze identity mismatch")
    if freeze["status"] != "preregistered_locked_not_generated_not_evaluated":
        raise SelectorV12Error("selector v1.2 freeze status mismatch")
    if freeze["historical_outcomes_known_before_freeze"] is not True:
        raise SelectorV12Error("selector v1.2 chronology disclosure missing")
    if freeze["frozen_before_any_evaluation_of_v1_2_outputs"] is not True:
        raise SelectorV12Error("selector v1.2 freeze does not predate evaluation")
    if freeze["runtime_outcome_access"] is not False:
        raise SelectorV12Error("selector v1.2 runtime outcome access is not false")
    profile = load_profile(root / "profile_v1_2.json", require_ready=True)
    if freeze["input_package_sha256"] != profile["authority_pins"]["input_package_manifest_sha256"]:
        raise SelectorV12Error("selector v1.2 freeze input package mismatch")
    artifacts = freeze["artifacts"]
    if not isinstance(artifacts, dict):
        raise SelectorV12Error("selector v1.2 freeze artifacts must be object")
    expected = {
        "profile": "profile_v1_2.json",
        "code": "selector_v1_2.py",
        "tests": "test_selector_v1_2.py",
    }
    _strict_keys(artifacts, set(expected), "selector v1.2 freeze artifacts")
    verified: dict[str, dict[str, str]] = {}
    for role, filename in expected.items():
        descriptor = artifacts[role]
        if not isinstance(descriptor, dict):
            raise SelectorV12Error(f"selector v1.2 freeze {role} descriptor invalid")
        _strict_keys(descriptor, {"path", "sha256"}, f"selector v1.2 freeze {role}")
        if descriptor["path"] != filename or _sha256(root / filename) != descriptor["sha256"]:
            raise SelectorV12Error(f"selector v1.2 freeze {role} pin mismatch")
        verified[role] = {"path": str(root / filename), "sha256": descriptor["sha256"]}
    return {
        "status": "selector_v1_2_freeze_verified",
        "freeze_path": str(path),
        "freeze_sha256": _sha256(path),
        "artifacts": verified,
        "profile": profile,
    }


def _validate_generation(value: Any, label: str, role: str, upstream: dict[str, str]) -> None:
    if not isinstance(value, dict):
        raise SelectorV12Error(f"{label} must be object")
    _strict_keys(
        value,
        {
            "finish_reason",
            "error",
            "forced_answer",
            "input_tokens",
            "output_tokens",
            "call_count",
            "temperature",
            "seed",
            "prompt_version",
            "upstream_artifact_sha256",
            "new_arm_gold_reference_correctness_outcome_or_judge_access",
            "source_access",
        },
        label,
    )
    if value["finish_reason"] not in {"stop", "length", "error", "missing"}:
        raise SelectorV12Error(f"{label}.finish_reason invalid")
    if value["error"] is not None and not isinstance(value["error"], str):
        raise SelectorV12Error(f"{label}.error invalid")
    if not isinstance(value["forced_answer"], bool):
        raise SelectorV12Error(f"{label}.forced_answer invalid")
    for key in ("input_tokens", "output_tokens", "call_count"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0:
            raise SelectorV12Error(f"{label}.{key} invalid")
    temperature = value["temperature"]
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) < 0
    ):
        raise SelectorV12Error(f"{label}.temperature invalid")
    if value["seed"] is not None and (
        isinstance(value["seed"], bool) or not isinstance(value["seed"], int)
    ):
        raise SelectorV12Error(f"{label}.seed invalid")
    if not isinstance(value["prompt_version"], str) or not value["prompt_version"]:
        raise SelectorV12Error(f"{label}.prompt_version invalid")
    if value["upstream_artifact_sha256"] != upstream[role]:
        raise SelectorV12Error(f"{label} upstream pin mismatch")
    if value["new_arm_gold_reference_correctness_outcome_or_judge_access"] is not False:
        raise SelectorV12Error(f"{label} new-arm outcome access is not false")
    if role == STRUCTURAL_ROLE:
        if not isinstance(value["source_access"], bool):
            raise SelectorV12Error(f"{label} structural source access must be boolean")
    elif value["source_access"] is not False:
        raise SelectorV12Error(f"{label} non-structural source access is not false")


def _validate_candidate(value: Any, label: str, role: str, upstream: dict[str, str]) -> None:
    if not isinstance(value, dict):
        raise SelectorV12Error(f"{label} must be object")
    _strict_keys(value, {"available", "model", "final_answer", "generation"}, label)
    try:
        base._assert_no_excluded_runtime_fields(value, label)
    except base.SelectorError as exc:
        raise SelectorV12Error(str(exc)) from exc
    if not isinstance(value["available"], bool) or value["model"] != MODEL:
        raise SelectorV12Error(f"{label} availability/model invalid")
    if value["available"] and not isinstance(value["final_answer"], str):
        raise SelectorV12Error(f"{label}.final_answer missing")
    if not value["available"] and value["final_answer"] is not None:
        raise SelectorV12Error(f"{label}.final_answer must be null when unavailable")
    _validate_generation(value["generation"], f"{label}.generation", role, upstream)


def _validate_batch(value: Any, label: str, role: str, upstream: dict[str, str]) -> None:
    if not isinstance(value, dict):
        raise SelectorV12Error(f"{label} must be object")
    _strict_keys(value, {"final", "raw_votes"}, label)
    _validate_candidate(value["final"], f"{label}.final", role, upstream)
    if not isinstance(value["raw_votes"], list) or len(value["raw_votes"]) != 8:
        raise SelectorV12Error(f"{label}.raw_votes must contain 8 rows")
    for index, candidate in enumerate(value["raw_votes"]):
        _validate_candidate(candidate, f"{label}.raw_votes[{index}]", role, upstream)


def validate_pool_row(row: Any, index: int, upstream: dict[str, str]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise SelectorV12Error("v1.2 pool row must be object")
    _strict_keys(
        row,
        {"schema_version", "row_index", "anchor", "structural", "routers", "parallel_batches"},
        "v1.2 pool row",
    )
    try:
        base._assert_no_excluded_runtime_fields(row, "v1.2 pool row")
    except base.SelectorError as exc:
        raise SelectorV12Error(str(exc)) from exc
    if row["schema_version"] != POOL_ROW_SCHEMA or row["row_index"] != index:
        raise SelectorV12Error("v1.2 pool row schema/order mismatch")
    _validate_candidate(row["anchor"], "anchor", "active_crop_v2", upstream)
    _validate_candidate(row["structural"], "structural", STRUCTURAL_ROLE, upstream)
    routers = row["routers"]
    if not isinstance(routers, dict):
        raise SelectorV12Error("routers must be object")
    _strict_keys(routers, {"v4", "v5"}, "routers")
    _validate_candidate(routers["v4"], "routers.v4", "native_thinking_math_router_v4", upstream)
    _validate_candidate(routers["v5"], "routers.v5", "native_thinking_math_router_v5", upstream)
    batches = row["parallel_batches"]
    if not isinstance(batches, dict):
        raise SelectorV12Error("parallel_batches must be object")
    _strict_keys(batches, {"parallel8_v1", "parallel8_reasoning_first_v2"}, "parallel_batches")
    _validate_batch(batches["parallel8_v1"], "parallel8_v1", "parallel8_v1", upstream)
    _validate_batch(
        batches["parallel8_reasoning_first_v2"],
        "parallel8_reasoning_first_v2",
        "parallel8_reasoning_first_v2",
        upstream,
    )
    return row


def _role_projection(row: dict[str, Any], role: str) -> dict[str, Any]:
    if role == "active_crop_v2":
        return row["anchor"]
    if role == STRUCTURAL_ROLE:
        return row["structural"]
    if role == "native_thinking_math_router_v4":
        return row["routers"]["v4"]
    if role == "native_thinking_math_router_v5":
        return row["routers"]["v5"]
    return row["parallel_batches"][role]


def _load_bindings(
    path: Path,
    ordered_ids: tuple[str, ...],
    order_sha: str,
    upstream: dict[str, str],
) -> dict[str, tuple[str, ...]]:
    value = _read_json(path, "v1.2 combined row bindings")
    _strict_keys(
        value,
        {
            "schema_version",
            "benchmark_sha256",
            "benchmark_order_sha256",
            "projection_contract",
            "upstream_artifact_sha256",
            "rows",
        },
        "v1.2 combined row bindings",
    )
    if value["schema_version"] != BINDING_SCHEMA:
        raise SelectorV12Error("v1.2 binding schema mismatch")
    if value["benchmark_sha256"] != base.BENCHMARK_SHA256 or value["benchmark_order_sha256"] != order_sha:
        raise SelectorV12Error("v1.2 binding benchmark mismatch")
    if value["projection_contract"] != "sha256_of_utf8_json_sort_keys_compact_role_projection":
        raise SelectorV12Error("v1.2 binding projection contract mismatch")
    if value["upstream_artifact_sha256"] != upstream:
        raise SelectorV12Error("v1.2 binding upstream map mismatch")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise SelectorV12Error("v1.2 bindings must contain 274 rows")
    result: dict[str, list[str]] = {role: [] for role in UPSTREAM_ROLES}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SelectorV12Error(f"v1.2 binding row {index} invalid")
        _strict_keys(
            row,
            {"row_index", "task_id", "role_projection_sha256"},
            f"v1.2 binding row {index}",
        )
        if row["row_index"] != index or row["task_id"] != ordered_ids[index]:
            raise SelectorV12Error("v1.2 binding identity/order mismatch")
        role_map = row["role_projection_sha256"]
        if not isinstance(role_map, dict):
            raise SelectorV12Error("v1.2 binding role map invalid")
        _strict_keys(role_map, set(UPSTREAM_ROLES), f"v1.2 binding role map {index}")
        for role in UPSTREAM_ROLES:
            result[role].append(_hex64(role_map[role], f"v1.2 binding {index}.{role}"))
    return {role: tuple(values) for role, values in result.items()}


def load_input_package(
    path: Path, profile: dict[str, Any]
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], frozenset[str], str]:
    pins = _validate_authority_pins(profile["authority_pins"], True)
    upstream = profile["upstream_artifact_sha256"]
    path = path.resolve()
    if _sha256(path) != pins["input_package_manifest_sha256"]:
        raise SelectorV12Error("v1.2 input package SHA mismatch")
    package = _read_json(path, "v1.2 input package")
    try:
        base._assert_no_excluded_runtime_fields(package, "v1.2 input package")
    except base.SelectorError as exc:
        raise SelectorV12Error(str(exc)) from exc
    _strict_keys(
        package,
        {
            "schema_version",
            "rows",
            "benchmark_sha256",
            "created_before_new_arm_evaluation",
            "runtime_outcome_access",
            "benchmark_order",
            "route_map",
            "source_union_membership",
            "upstream_artifacts",
            "row_bindings",
            "candidate_pool",
        },
        "v1.2 input package",
    )
    if package["schema_version"] != PACKAGE_SCHEMA or package["rows"] != ROW_COUNT:
        raise SelectorV12Error("v1.2 package schema/row mismatch")
    if package["benchmark_sha256"] != base.BENCHMARK_SHA256:
        raise SelectorV12Error("v1.2 package benchmark mismatch")
    if package["created_before_new_arm_evaluation"] is not True or package["runtime_outcome_access"] is not False:
        raise SelectorV12Error("v1.2 package chronology/access mismatch")
    root = path.parent
    order_path = _relative_descriptor(root, package["benchmark_order"], "v1.2 benchmark_order", pins["benchmark_order_sha256"])
    route_path = _relative_descriptor(root, package["route_map"], "v1.2 route_map", pins["route_map_sha256"])
    membership_path = _relative_descriptor(
        root,
        package["source_union_membership"],
        "v1.2 source_union_membership",
        pins["source_union_membership_sha256"],
    )
    pool_path = _relative_descriptor(root, package["candidate_pool"], "v1.2 candidate_pool", pins["candidate_pool_sha256"])
    binding_path = _relative_descriptor(root, package["row_bindings"], "v1.2 row_bindings", pins["row_bindings_sha256"])
    descriptors = package["upstream_artifacts"]
    if not isinstance(descriptors, dict):
        raise SelectorV12Error("v1.2 upstream descriptors must be object")
    _strict_keys(descriptors, set(UPSTREAM_ROLES), "v1.2 upstream descriptors")
    for role in UPSTREAM_ROLES:
        _relative_descriptor(root, descriptors[role], f"v1.2 upstream {role}", upstream[role])
    try:
        ordered_ids = base._load_order(order_path)
        routes = base._load_routes(route_path, ordered_ids, pins["benchmark_order_sha256"])
        protected = base._load_source_union_membership(membership_path)
    except base.SelectorError as exc:
        raise SelectorV12Error(str(exc)) from exc
    if not protected.issubset(set(ordered_ids)):
        raise SelectorV12Error("v1.2 protected IDs absent from order")
    bindings = _load_bindings(binding_path, ordered_ids, pins["benchmark_order_sha256"], upstream)
    rows = _read_jsonl(pool_path, "v1.2 candidate pool")
    if len(rows) != ROW_COUNT:
        raise SelectorV12Error("v1.2 candidate pool missing/extra rows")
    for index, row in enumerate(rows):
        validate_pool_row(row, index, upstream)
        for role in UPSTREAM_ROLES:
            digest = hashlib.sha256(
                json.dumps(
                    _role_projection(row, role),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if digest != bindings[role][index]:
                raise SelectorV12Error(f"v1.2 row {index} role {role} binding mismatch")
    return rows, ordered_ids, routes, protected, _sha256(pool_path)


def _choice(candidate: dict[str, Any]) -> str | None:
    if not candidate["available"]:
        return None
    generation = candidate["generation"]
    if generation["finish_reason"] in {"error", "missing"} or generation["error"]:
        return None
    answer = candidate["final_answer"]
    if not isinstance(answer, str):
        return None
    match = base.CHOICE_RE.fullmatch(answer)
    return match.group(1).upper() if match else None


def _result(name: str, anchor: str | None, challenger: str | None, accepted: bool, reason: str) -> dict[str, Any]:
    return {
        "arm": name,
        "action": "propose_challenger" if accepted else "preserve_anchor",
        "selected_answer": challenger if accepted else anchor,
        "selected_choice": challenger if accepted else (_choice_value(anchor)),
        "reason": reason,
    }


def _choice_value(answer: Any) -> str | None:
    if not isinstance(answer, str):
        return None
    match = base.CHOICE_RE.fullmatch(answer)
    return match.group(1).upper() if match else None


def select_bound_row(
    row: dict[str, Any],
    *,
    task_id: str,
    route: str,
    protected: bool,
    upstream: dict[str, str],
) -> dict[str, Any]:
    validate_pool_row(row, row["row_index"], upstream)
    anchor = _choice(row["anchor"])
    structural = _choice(row["structural"])
    v4 = _choice(row["routers"]["v4"])
    v5 = _choice(row["routers"]["v5"])
    p1 = _choice(row["parallel_batches"]["parallel8_v1"]["final"])
    p2 = _choice(row["parallel_batches"]["parallel8_reasoning_first_v2"]["final"])
    native_group = v4 if v4 is not None and v4 == v5 else None
    parallel_group = p1 if p1 is not None and p1 == p2 else None
    common_reason: str | None = None
    if protected:
        common_reason = "protected_by_source_union"
    elif route != "deterministic":
        common_reason = "image_judge_route_preserved"
    elif anchor is None or structural is None:
        common_reason = "anchor_or_structural_answer_invalid"
    elif structural == anchor:
        common_reason = "structural_agrees_with_anchor_no_change"

    if common_reason is not None:
        primary = _result(ALGORITHM["primary_arm"]["name"], row["anchor"]["final_answer"], structural, False, common_reason)
        exploratory = _result(ALGORITHM["exploratory_arm"]["name"], row["anchor"]["final_answer"], structural, False, common_reason)
    else:
        assert structural is not None
        primary_accept = native_group == structural and parallel_group == structural
        if native_group is None:
            primary_reason = "native_group_has_no_v4_v5_consensus"
        elif parallel_group is None:
            primary_reason = "parallel_group_has_no_final_consensus"
        elif not primary_accept:
            primary_reason = "three_group_answers_do_not_match"
        else:
            primary_reason = "all_three_preregistered_groups_agree"
        primary = _result(
            ALGORITHM["primary_arm"]["name"],
            row["anchor"]["final_answer"],
            structural,
            primary_accept,
            primary_reason,
        )
        exploratory_accept = parallel_group == structural and (v4 == structural or v5 == structural)
        if parallel_group is None:
            exploratory_reason = "parallel_group_has_no_final_consensus"
        elif parallel_group != structural:
            exploratory_reason = "structural_and_parallel_group_disagree"
        elif v4 != structural and v5 != structural:
            exploratory_reason = "neither_native_member_supports_structural_challenger"
        else:
            exploratory_reason = "structural_parallel_and_one_native_support_challenger"
        exploratory = _result(
            ALGORITHM["exploratory_arm"]["name"],
            row["anchor"]["final_answer"],
            structural,
            exploratory_accept,
            exploratory_reason,
        )
    if (protected or route == "image_judge") and (
        primary["action"] != "preserve_anchor" or exploratory["action"] != "preserve_anchor"
    ):
        raise SelectorV12Error("v1.2 protected-slice invariant failed")
    return {
        "schema_version": OUTPUT_ROW_SCHEMA,
        "row_index": row["row_index"],
        "task_id": task_id,
        "authoritative_evaluation_route": route,
        "protected_by_source_union": protected,
        "anchor_answer": row["anchor"]["final_answer"],
        "structural_challenger": structural,
        "native_group_answer": native_group,
        "parallel_group_answer": parallel_group,
        "native_member_answers": {"v4": v4, "v5": v5},
        "primary": primary,
        "exploratory": exploratory,
    }


def run_selector(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    freeze = verify_preregistered_freeze(root)
    package_path = (root / INPUT_PACKAGE_RELATIVE_PATH).resolve()
    rows, ids, routes, protected_ids, pool_sha = load_input_package(package_path, freeze["profile"])
    outputs = [
        select_bound_row(
            row,
            task_id=ids[index],
            route=routes[index],
            protected=ids[index] in protected_ids,
            upstream=freeze["profile"]["upstream_artifact_sha256"],
        )
        for index, row in enumerate(rows)
    ]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "selection_proposals_v1_2.jsonl"
    rows_path.write_bytes(b"".join(_canonical_json(row) for row in outputs))
    primary_counts = Counter(row["primary"]["action"] for row in outputs)
    exploratory_counts = Counter(row["exploratory"]["action"] for row in outputs)
    manifest = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA,
        "selector_id": SELECTOR_ID,
        "status": "v1_2_arm_outputs_frozen_not_evaluated",
        "artifact_kind": "patch_proposals_not_a_scored_solver",
        "rows": ROW_COUNT,
        "benchmark_sha256": base.BENCHMARK_SHA256,
        "model_closure": [MODEL],
        "runtime_outcome_access": False,
        "freeze_sha256": freeze["freeze_sha256"],
        "input_package_sha256": freeze["profile"]["authority_pins"]["input_package_manifest_sha256"],
        "candidate_pool_sha256": pool_sha,
        "source_union_changes": 0,
        "image_judge_changes": 0,
        "primary_action_counts": dict(sorted(primary_counts.items())),
        "exploratory_action_counts": dict(sorted(exploratory_counts.items())),
        "selection_proposals": {"path": rows_path.name, "sha256": _sha256(rows_path)},
    }
    manifest_path = output_dir / "selector_manifest_v1_2.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    return {
        "status": manifest["status"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "action_counts": {
            "primary": dict(sorted(primary_counts.items())),
            "exploratory": dict(sorted(exploratory_counts.items())),
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Outcome-blind three-group selector v1.2")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-freeze", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.output_dir is not None:
            report = run_selector(args.root, args.output_dir)
        else:
            if not args.verify_freeze:
                raise SelectorV12Error("use --verify-freeze or provide --output-dir")
            report = verify_preregistered_freeze(args.root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (SelectorV12Error, base.SelectorError) as exc:
        print(f"selector v1.2 error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
