from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MODEL = "Qwen/Qwen3.5-9B"
SELECTOR_ID = "maxim_9b_baseline_selector_v1_1"
PROFILE_SCHEMA = "maxim-9b-baseline-selector-profile-v1.1"
FREEZE_SCHEMA = "maxim-9b-baseline-selector-preregistered-freeze-v1.1"
PACKAGE_SCHEMA = "maxim-9b-baseline-selector-input-package-v1.1"
ORDER_SCHEMA = "maxim-9b-baseline-selector-benchmark-order-v1.1"
ROUTE_SCHEMA = "maxim-9b-baseline-selector-route-map-v1.1"
BINDING_SCHEMA = "maxim-9b-baseline-selector-row-bindings-v1.1"
COMBINED_BINDING_SCHEMA = "maxim-9b-baseline-selector-combined-row-bindings-v1.1"
MEMBERSHIP_SCHEMA = "maxim-9b-baseline-selector-source-union-membership-v1.1"
POOL_ROW_SCHEMA = "maxim-9b-baseline-selector-pool-row-v1.1"
OUTPUT_ROW_SCHEMA = "maxim-9b-baseline-selector-output-row-v1.1"
OUTPUT_MANIFEST_SCHEMA = "maxim-9b-baseline-selector-output-manifest-v1.1"
SOURCE_AGGREGATE_SCHEMA = "maxim-9b-source-replay-aggregate-v1"
ROW_COUNT = 274
BENCHMARK_SHA256 = "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
CHOICE_RE = re.compile(r"^\s*([A-E])\s*[.)]?\s*$", flags=re.IGNORECASE)

UPSTREAM_SHA256 = {
    "active_crop_v2": "6697c043f3142a736b817ead5da494eea334f5349e0db833bd72f23fe35cb17c",
    "native_thinking_math_router_v4": "0fd0e6fef6b220749faa015e7a163cdf596e070afaaad407e4d36bc9b1337307",
    "native_thinking_math_router_v5": "45dc8c16f834d27e5d114f9162b7984c8baec4037fef011730e91ba13f192845",
    "parallel8_v1": "b1b7a1b785a9a3fc076c04c37fa72f4a82169f137651ff57b40e984901b0d645",
    "parallel8_reasoning_first_v2": "6115effd03d7eac3e11e9726ecd9822802a04f235a0b361e1863b9bc7e221023",
}
UPSTREAM_ROLES = tuple(UPSTREAM_SHA256)

SOURCE_AGGREGATE_REPOSITORY_PATH = (
    "reports/maxim_9b_source_replay_v1_20260809/active_crop/"
    "source_v7_aggregate/aggregate.json"
)
SOURCE_AGGREGATE_SHA256 = (
    "3de5129dee80d2f2fda544bdf7eecfa7d0f467d56bb7e43afc8eac89a6a5dacd"
)
SOURCE_UNION_SHA256 = (
    "7e5e0c972e82d87d164cb3ef03b13fbb4c8084bf07c2512129958882871508cd"
)
SOURCE_UNION_SIZE = 156

CHRONOLOGY_CONTRACT = {
    "historical_benchmark_aggregate_score_and_prior_task_outcomes_were_known_before_freeze": True,
    "rules_and_profile_frozen_before_generation_and_any_gold_score_correctness_or_judge_evaluation_of_new_selector_arm_outputs": True,
    "selector_runtime_inputs_and_algorithm_do_not_read_gold_reference_correctness_outcomes_or_judge_verdicts": True,
}

CANDIDATE_PROVENANCE = {
    "native_router_pair": "two_correlated_native_thinking_ablations_with_same_routed86_model_and_seeds_but_different_max_tokens_and_treatment",
    "parallel_pair": "two_separately_executed_8_route_batches_with_same_routes_and_model_v2_core_and_v1_diversity_donor",
    "independence_claim": "not_four_fully_independent_systems",
    "endpoint_revision_caveat": "parallel_batches_have_no_immutable_endpoint_revision_manifest",
}

ALGORITHM_CONTRACT = {
    "anchor": "active_crop_v2",
    "authoritative_identity": "benchmark_order_row_index_not_pool_opaque_id",
    "authoritative_route": "independent_outcome_free_route_map_not_pool_metadata",
    "shared_gates": [
        "authoritative_route_is_deterministic",
        "authoritative_task_id_is_outside_pinned_source_union",
        "anchor_is_valid_A_to_E",
        "native_router_v4_and_v5_are_valid_and_agree",
        "router_challenger_differs_from_anchor",
        "every_candidate_projection_matches_its_pinned_upstream_row_binding",
    ],
    "primary_arm": {
        "name": "router_agreement_plus_parallel16_supermajority",
        "role": "primary_conservative",
        "rule": "V4_equals_V5_and_challenger_has_at_least_13_of_16_raw_parallel_votes",
        "raw_vote_threshold": 13,
        "raw_vote_denominator": 16,
        "threshold_rationale": "fixed_81.25_percent_supermajority_with_iid_fair_null_tail_697_of_65536_but_no_independence_claim",
    },
    "secondary_arm": {
        "name": "four_final_answers_unanimous",
        "role": "preregistered_exploratory",
        "rule": "V4_equals_V5_equals_parallel8_v1_final_equals_parallel8_reasoning_first_v2_final",
    },
    "fallback": "emit_no_patch_and_preserve_active_crop_v2_anchor",
    "image_judge_policy": "always_preserve_anchor_byte_for_byte",
    "source_stack_policy": "emit_proposals_only_never_mutate_frozen_source_stack",
    "no_post_evaluation_rule_or_threshold_change": True,
}

RUNTIME_INPUT_EXCLUDED_FIELDS = {
    "gold",
    "gold_answer",
    "reference_answer",
    "reference_solution",
    "correct",
    "correctness",
    "is_correct",
    "strict_correct",
    "outcome",
    "outcomes",
    "judge",
    "judge_verdict",
    "verdict",
    "score",
    "accuracy",
}


class SelectorError(RuntimeError):
    """Raised when a v1.1 selector artifact violates its locked contract."""


def _canonical_json(value: Any, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + suffix
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SelectorError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SelectorError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectorError(f"{label} must be a JSON object")
    return value


def _strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise SelectorError(
            f"{label} schema mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _hex64(value: Any, label: str) -> str:
    digest = str(value or "").casefold()
    if not SHA256_RE.fullmatch(digest):
        raise SelectorError(f"{label} must be a lowercase SHA-256")
    return digest


def _safe_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value):
        raise SelectorError(f"{label} must be a stable non-semantic identifier")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SelectorError(f"{label} must be a non-negative integer")
    return value


def _false(value: Any, label: str) -> None:
    if value is not False:
        raise SelectorError(f"{label} must be exactly false")


def _assert_no_excluded_runtime_fields(value: Any, label: str) -> None:
    if isinstance(value, dict):
        forbidden = RUNTIME_INPUT_EXCLUDED_FIELDS.intersection(value)
        if forbidden:
            raise SelectorError(f"{label} contains excluded runtime fields {sorted(forbidden)}")
        for key, item in value.items():
            _assert_no_excluded_runtime_fields(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_excluded_runtime_fields(item, f"{label}[{index}]")


def _relative_descriptor(
    base: Path,
    value: Any,
    label: str,
    *,
    expected_sha256: str,
) -> Path:
    if not isinstance(value, dict):
        raise SelectorError(f"{label} descriptor must be an object")
    _strict_keys(value, {"path", "sha256"}, f"{label} descriptor")
    locator = value["path"]
    if not isinstance(locator, str) or not locator:
        raise SelectorError(f"{label}.path must be non-empty")
    unresolved = Path(locator)
    if unresolved.is_absolute() or ".." in unresolved.parts:
        raise SelectorError(f"{label}.path must remain relative and traversal-free")
    root = base.resolve()
    path = (root / unresolved).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SelectorError(f"{label}.path escaped its input-package root") from exc
    descriptor_sha = _hex64(value["sha256"], f"{label}.sha256")
    if descriptor_sha != expected_sha256:
        raise SelectorError(f"{label} differs from preregistered SHA-256")
    if _sha256(path) != expected_sha256:
        raise SelectorError(f"{label} file SHA-256 mismatch")
    return path


def _validate_authority_pins(value: Any, *, require_ready: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectorError("profile.authority_pins must be an object")
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
        "profile.authority_pins",
    )
    if value["status"] not in {"awaiting_nineb_input_lock", "locked_before_new_arm_evaluation"}:
        raise SelectorError("profile.authority_pins.status is invalid")
    digest_fields = (
        "input_package_manifest_sha256",
        "benchmark_order_sha256",
        "route_map_sha256",
        "source_union_membership_sha256",
        "candidate_pool_sha256",
        "row_bindings_sha256",
    )
    for key in digest_fields:
        if value[key] is not None:
            _hex64(value[key], f"profile.authority_pins.{key}")
    ready = (
        value["status"] == "locked_before_new_arm_evaluation"
        and all(value[key] is not None for key in digest_fields)
    )
    if require_ready and not ready:
        raise SelectorError("v1.1 authority pins are not locked; selection is disabled")
    return value


def validate_profile(profile: dict[str, Any], *, require_ready: bool) -> dict[str, Any]:
    _strict_keys(
        profile,
        {
            "schema_version",
            "selector_id",
            "status",
            "model_closure",
            "chronology",
            "candidate_provenance",
            "benchmark",
            "upstream_artifact_sha256",
            "protected_scope",
            "algorithm",
            "runtime_input_contract",
            "authority_pins",
        },
        "selector profile v1.1",
    )
    if profile["schema_version"] != PROFILE_SCHEMA or profile["selector_id"] != SELECTOR_ID:
        raise SelectorError("selector profile identity mismatch")
    allowed_status = {
        "draft_awaiting_nineb_input_lock_not_frozen",
        "preregistered_after_historical_outcomes_known_before_new_arm_evaluation",
    }
    if profile["status"] not in allowed_status:
        raise SelectorError("selector profile status is invalid")
    if require_ready and profile["status"] != (
        "preregistered_after_historical_outcomes_known_before_new_arm_evaluation"
    ):
        raise SelectorError("selector profile is a non-executable draft")
    if profile["model_closure"] != [MODEL]:
        raise SelectorError("selector profile model closure mismatch")
    if profile["chronology"] != CHRONOLOGY_CONTRACT:
        raise SelectorError("selector chronology contract mismatch")
    if profile["candidate_provenance"] != CANDIDATE_PROVENANCE:
        raise SelectorError("candidate provenance contract mismatch")
    if profile["benchmark"] != {"sha256": BENCHMARK_SHA256, "ordered_rows": ROW_COUNT}:
        raise SelectorError("benchmark contract mismatch")
    if profile["upstream_artifact_sha256"] != UPSTREAM_SHA256:
        raise SelectorError("upstream artifact pins mismatch")
    expected_protected = {
        "aggregate_repository_relative_path": SOURCE_AGGREGATE_REPOSITORY_PATH,
        "aggregate_schema": SOURCE_AGGREGATE_SCHEMA,
        "aggregate_file_sha256": SOURCE_AGGREGATE_SHA256,
        "source_union_size": SOURCE_UNION_SIZE,
        "source_union_projection_sha256": SOURCE_UNION_SHA256,
        "runtime_membership_schema": MEMBERSHIP_SCHEMA,
        "semantics": "membership_is_a_safety_veto_only_not_a_quality_feature",
    }
    if profile["protected_scope"] != expected_protected:
        raise SelectorError("protected scope contract mismatch")
    if profile["algorithm"] != ALGORITHM_CONTRACT:
        raise SelectorError("selector algorithm differs from preregistered v1.1 rules")
    if profile["runtime_input_contract"] != {
        "pool_contains_no_task_id_or_evaluation_route": True,
        "identity_comes_only_from_pinned_benchmark_order": True,
        "route_comes_only_from_pinned_outcome_free_route_map": True,
        "candidate_projection_must_match_pinned_per_upstream_row_binding": True,
        "missing_extra_reordered_or_relabelled_rows_fail_closed": True,
        "excluded_field_names": sorted(RUNTIME_INPUT_EXCLUDED_FIELDS),
    }:
        raise SelectorError("runtime input contract mismatch")
    _validate_authority_pins(profile["authority_pins"], require_ready=require_ready)
    return profile


def load_profile(path: Path, *, require_ready: bool) -> dict[str, Any]:
    return validate_profile(_read_json(path, "selector profile v1.1"), require_ready=require_ready)


def verify_preregistered_freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    freeze_path = root / "PREREGISTERED_FREEZE.json"
    freeze = _read_json(freeze_path, "preregistered freeze v1.1")
    _strict_keys(
        freeze,
        {
            "schema_version",
            "selector_id",
            "status",
            "historical_benchmark_aggregate_score_and_prior_task_outcomes_were_known_before_freeze",
            "rules_and_profile_frozen_before_generation_and_any_gold_score_correctness_or_judge_evaluation_of_new_selector_arm_outputs",
            "selector_runtime_inputs_and_algorithm_do_not_read_gold_reference_correctness_outcomes_or_judge_verdicts",
            "supersedes",
            "artifacts",
        },
        "preregistered freeze v1.1",
    )
    if freeze["schema_version"] != FREEZE_SCHEMA or freeze["selector_id"] != SELECTOR_ID:
        raise SelectorError("freeze v1.1 identity mismatch")
    if freeze["status"] != "preregistered_locked_not_evaluated":
        raise SelectorError("freeze v1.1 is not active")
    for key, required in CHRONOLOGY_CONTRACT.items():
        if freeze[key] is not required:
            raise SelectorError(f"freeze chronology mismatch for {key}")
    supersedes = freeze["supersedes"]
    if supersedes != {
        "selector_id": "maxim_9b_baseline_selector_v1",
        "freeze_sha256": "001b4d5664c44e7ba686a4457a838e6f50991611751e7cb6fb8ad70f1a5619b4",
        "status": "superseded_not_executed",
    }:
        raise SelectorError("freeze supersession record mismatch")
    artifacts = freeze["artifacts"]
    if not isinstance(artifacts, dict):
        raise SelectorError("freeze.artifacts must be an object")
    _strict_keys(
        artifacts,
        {"profile", "selector_code", "tests", "superseded_freeze_record"},
        "freeze.artifacts",
    )
    expected_names = {
        "profile": "profile_v1_1.json",
        "selector_code": "selector_v1_1.py",
        "tests": "test_selector_v1_1.py",
        "superseded_freeze_record": "SUPERSEDED_FREEZE_v1.json",
    }
    verified: dict[str, dict[str, str]] = {}
    for role, filename in expected_names.items():
        descriptor = artifacts[role]
        digest = _hex64(descriptor.get("sha256") if isinstance(descriptor, dict) else None, f"freeze.{role}.sha256")
        path = _relative_descriptor(root, descriptor, f"freeze.{role}", expected_sha256=digest)
        if path.name != filename:
            raise SelectorError(f"freeze.{role} points to the wrong file")
        verified[role] = {"path": str(path), "sha256": digest}
    profile = load_profile(Path(verified["profile"]["path"]), require_ready=True)
    return {
        "status": "preregistered_freeze_v1_1_verified",
        "freeze_path": str(freeze_path),
        "freeze_sha256": _sha256(freeze_path),
        "artifacts": verified,
        "profile": profile,
    }


def _load_source_union_membership(path: Path) -> frozenset[str]:
    value = _read_json(path, "outcome-free source-union membership")
    _strict_keys(
        value,
        {"schema_version", "authority", "derivation", "task_ids"},
        "outcome-free source-union membership",
    )
    if value["schema_version"] != MEMBERSHIP_SCHEMA:
        raise SelectorError("source-union membership schema mismatch")
    authority = value["authority"]
    if not isinstance(authority, dict):
        raise SelectorError("source-union membership authority must be an object")
    _strict_keys(
        authority,
        {"aggregate_sha256", "source_union_projection_sha256", "source_union_size"},
        "source-union membership authority",
    )
    if authority != {
        "aggregate_sha256": SOURCE_AGGREGATE_SHA256,
        "source_union_projection_sha256": SOURCE_UNION_SHA256,
        "source_union_size": SOURCE_UNION_SIZE,
    }:
        raise SelectorError("source-union membership authority pin mismatch")
    if value["derivation"] != "projection_task_id_only_no_answer_score_correctness_outcome_or_judge":
        raise SelectorError("source-union membership derivation is not outcome-free")
    task_ids = value["task_ids"]
    if not isinstance(task_ids, list) or len(task_ids) != SOURCE_UNION_SIZE:
        raise SelectorError("source-union membership size mismatch")
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise SelectorError("source-union membership contains invalid task ID")
    if len(set(task_ids)) != SOURCE_UNION_SIZE:
        raise SelectorError("source-union membership contains duplicate task ID")
    if task_ids != sorted(task_ids):
        raise SelectorError("source-union membership is not in canonical sorted order")
    return frozenset(task_ids)


def _validate_generation(value: Any, label: str, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectorError(f"{label} must be an object")
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
        raise SelectorError(f"{label}.finish_reason is invalid")
    if value["error"] is not None and not isinstance(value["error"], str):
        raise SelectorError(f"{label}.error must be string or null")
    if not isinstance(value["forced_answer"], bool):
        raise SelectorError(f"{label}.forced_answer must be boolean")
    for key in ("input_tokens", "output_tokens", "call_count"):
        _nonnegative_int(value[key], f"{label}.{key}")
    temperature = value["temperature"]
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) < 0
    ):
        raise SelectorError(f"{label}.temperature is invalid")
    if value["seed"] is not None and (
        isinstance(value["seed"], bool) or not isinstance(value["seed"], int)
    ):
        raise SelectorError(f"{label}.seed must be integer or null")
    _safe_identifier(value["prompt_version"], f"{label}.prompt_version")
    if value["upstream_artifact_sha256"] != UPSTREAM_SHA256[role]:
        raise SelectorError(f"{label} upstream artifact pin mismatch")
    _false(
        value["new_arm_gold_reference_correctness_outcome_or_judge_access"],
        f"{label}.new_arm_gold_reference_correctness_outcome_or_judge_access",
    )
    _false(value["source_access"], f"{label}.source_access")
    return value


def _validate_candidate(value: Any, label: str, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectorError(f"{label} must be an object")
    _strict_keys(value, {"available", "model", "final_answer", "generation"}, label)
    _assert_no_excluded_runtime_fields(value, label)
    if not isinstance(value["available"], bool):
        raise SelectorError(f"{label}.available must be boolean")
    if value["model"] != MODEL:
        raise SelectorError(f"{label} model closure is not pure Qwen3.5-9B")
    if value["available"]:
        if not isinstance(value["final_answer"], str):
            raise SelectorError(f"{label}.final_answer must be string when available")
    elif value["final_answer"] is not None:
        raise SelectorError(f"{label}.final_answer must be null when unavailable")
    _validate_generation(value["generation"], f"{label}.generation", role)
    return value


def _validate_parallel_batch(value: Any, label: str, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectorError(f"{label} must be an object")
    _strict_keys(value, {"final", "raw_votes"}, label)
    _validate_candidate(value["final"], f"{label}.final", role)
    votes = value["raw_votes"]
    if not isinstance(votes, list) or len(votes) != 8:
        raise SelectorError(f"{label}.raw_votes must contain exactly 8 candidates")
    for index, candidate in enumerate(votes):
        _validate_candidate(candidate, f"{label}.raw_votes[{index}]", role)
    return value


def validate_pool_row(row: Any, expected_index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise SelectorError("candidate pool row must be an object")
    _strict_keys(
        row,
        {"schema_version", "row_index", "anchor", "routers", "parallel_batches"},
        "candidate pool row",
    )
    _assert_no_excluded_runtime_fields(row, "candidate pool row")
    if row["schema_version"] != POOL_ROW_SCHEMA:
        raise SelectorError("candidate pool row schema mismatch")
    if row["row_index"] != expected_index:
        raise SelectorError("candidate pool has missing, extra, or reordered row index")
    _validate_candidate(row["anchor"], "anchor", "active_crop_v2")
    routers = row["routers"]
    if not isinstance(routers, dict):
        raise SelectorError("routers must be an object")
    _strict_keys(routers, {"v4", "v5"}, "routers")
    _validate_candidate(routers["v4"], "routers.v4", "native_thinking_math_router_v4")
    _validate_candidate(routers["v5"], "routers.v5", "native_thinking_math_router_v5")
    batches = row["parallel_batches"]
    if not isinstance(batches, dict):
        raise SelectorError("parallel_batches must be an object")
    _strict_keys(batches, {"parallel8_v1", "parallel8_reasoning_first_v2"}, "parallel_batches")
    _validate_parallel_batch(batches["parallel8_v1"], "parallel8_v1", "parallel8_v1")
    _validate_parallel_batch(
        batches["parallel8_reasoning_first_v2"],
        "parallel8_reasoning_first_v2",
        "parallel8_reasoning_first_v2",
    )
    return row


def _role_projection(row: dict[str, Any], role: str) -> dict[str, Any]:
    if role == "active_crop_v2":
        return row["anchor"]
    if role == "native_thinking_math_router_v4":
        return row["routers"]["v4"]
    if role == "native_thinking_math_router_v5":
        return row["routers"]["v5"]
    return row["parallel_batches"][role]


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SelectorError(f"{label}:{line_number} is not an object")
                values.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorError(f"cannot read {label} {path}: {exc}") from exc
    return values


def _load_order(path: Path) -> tuple[str, ...]:
    value = _read_json(path, "benchmark order")
    _strict_keys(value, {"schema_version", "benchmark_sha256", "rows"}, "benchmark order")
    if value["schema_version"] != ORDER_SCHEMA or value["benchmark_sha256"] != BENCHMARK_SHA256:
        raise SelectorError("benchmark order authority mismatch")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise SelectorError("benchmark order must contain exactly 274 IDs")
    if any(not isinstance(task_id, str) or not task_id for task_id in rows):
        raise SelectorError("benchmark order contains invalid task ID")
    if len(set(rows)) != ROW_COUNT:
        raise SelectorError("benchmark order contains duplicate task ID")
    return tuple(rows)


def _load_routes(path: Path, ordered_ids: tuple[str, ...], order_sha256: str) -> tuple[str, ...]:
    value = _read_json(path, "outcome-free route map")
    _strict_keys(
        value,
        {"schema_version", "benchmark_sha256", "benchmark_order_sha256", "derivation", "rows"},
        "outcome-free route map",
    )
    if value["schema_version"] != ROUTE_SCHEMA or value["benchmark_sha256"] != BENCHMARK_SHA256:
        raise SelectorError("route map authority mismatch")
    if value["benchmark_order_sha256"] != order_sha256:
        raise SelectorError("route map benchmark-order pin mismatch")
    if value["derivation"] != "question_image_format_metadata_only_no_gold_score_correctness_outcome_or_judge":
        raise SelectorError("route map derivation is not outcome-free")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise SelectorError("route map must contain exactly 274 rows")
    routes: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SelectorError(f"route map row {index} is not object")
        _strict_keys(row, {"row_index", "task_id", "evaluation_route"}, f"route map row {index}")
        if row["row_index"] != index or row["task_id"] != ordered_ids[index]:
            raise SelectorError("route map row identity/order mismatch")
        if row["evaluation_route"] not in {"deterministic", "image_judge"}:
            raise SelectorError("route map contains invalid evaluation route")
        routes.append(row["evaluation_route"])
    return tuple(routes)


def _load_combined_bindings(
    path: Path,
    *,
    ordered_ids: tuple[str, ...],
    order_sha256: str,
) -> dict[str, tuple[str, ...]]:
    value = _read_json(path, "combined row bindings")
    _strict_keys(
        value,
        {
            "schema_version",
            "upstream_artifact_sha256",
            "benchmark_sha256",
            "benchmark_order_sha256",
            "projection_contract",
            "rows",
        },
        "combined row bindings",
    )
    if value["schema_version"] != COMBINED_BINDING_SCHEMA:
        raise SelectorError("combined row bindings schema mismatch")
    if value["upstream_artifact_sha256"] != UPSTREAM_SHA256:
        raise SelectorError("combined row bindings upstream artifact map mismatch")
    if value["benchmark_sha256"] != BENCHMARK_SHA256 or value["benchmark_order_sha256"] != order_sha256:
        raise SelectorError("combined row bindings benchmark pin mismatch")
    if value["projection_contract"] != "sha256_of_utf8_json_sort_keys_compact_role_projection":
        raise SelectorError("combined row bindings projection contract mismatch")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise SelectorError("combined row bindings must contain exactly 274 rows")
    digests: dict[str, list[str]] = {role: [] for role in UPSTREAM_ROLES}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SelectorError(f"combined row bindings[{index}] is not object")
        _strict_keys(
            row,
            {"row_index", "task_id", "role_projection_sha256"},
            f"combined row bindings[{index}]",
        )
        if row["row_index"] != index or row["task_id"] != ordered_ids[index]:
            raise SelectorError("combined row bindings identity/order mismatch")
        role_map = row["role_projection_sha256"]
        if not isinstance(role_map, dict):
            raise SelectorError(f"combined row bindings[{index}] role map is not object")
        _strict_keys(role_map, set(UPSTREAM_ROLES), f"combined row bindings[{index}] role map")
        for role in UPSTREAM_ROLES:
            digests[role].append(
                _hex64(
                    role_map[role],
                    f"combined row bindings[{index}].role_projection_sha256.{role}",
                )
            )
    return {role: tuple(values) for role, values in digests.items()}


def load_input_package(
    package_path: Path,
    authority_pins: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    tuple[str, ...],
    tuple[str, ...],
    frozenset[str],
    str,
]:
    pins = _validate_authority_pins(authority_pins, require_ready=True)
    package_path = package_path.resolve()
    if _sha256(package_path) != pins["input_package_manifest_sha256"]:
        raise SelectorError("input package manifest differs from preregistered SHA-256")
    package = _read_json(package_path, "input package manifest")
    _assert_no_excluded_runtime_fields(package, "input package manifest")
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
        "input package manifest",
    )
    if package["schema_version"] != PACKAGE_SCHEMA or package["rows"] != ROW_COUNT:
        raise SelectorError("input package schema/row count mismatch")
    if package["benchmark_sha256"] != BENCHMARK_SHA256:
        raise SelectorError("input package benchmark SHA mismatch")
    if package["created_before_new_arm_evaluation"] is not True:
        raise SelectorError("input package was not frozen before new-arm evaluation")
    _false(package["runtime_outcome_access"], "input package runtime_outcome_access")
    base = package_path.parent
    order_path = _relative_descriptor(
        base,
        package["benchmark_order"],
        "benchmark_order",
        expected_sha256=pins["benchmark_order_sha256"],
    )
    route_path = _relative_descriptor(
        base,
        package["route_map"],
        "route_map",
        expected_sha256=pins["route_map_sha256"],
    )
    membership_path = _relative_descriptor(
        base,
        package["source_union_membership"],
        "source_union_membership",
        expected_sha256=pins["source_union_membership_sha256"],
    )
    upstream_artifacts = package["upstream_artifacts"]
    if not isinstance(upstream_artifacts, dict):
        raise SelectorError("input package upstream descriptor map must be object")
    _strict_keys(upstream_artifacts, set(UPSTREAM_ROLES), "upstream_artifacts")
    for role in UPSTREAM_ROLES:
        _relative_descriptor(
            base,
            upstream_artifacts[role],
            f"upstream_artifacts.{role}",
            expected_sha256=UPSTREAM_SHA256[role],
        )
    pool_path = _relative_descriptor(
        base,
        package["candidate_pool"],
        "candidate_pool",
        expected_sha256=pins["candidate_pool_sha256"],
    )
    ordered_ids = _load_order(order_path)
    routes = _load_routes(route_path, ordered_ids, pins["benchmark_order_sha256"])
    protected_source_union = _load_source_union_membership(membership_path)
    if not protected_source_union.issubset(set(ordered_ids)):
        raise SelectorError("benchmark order omits protected source-union task IDs")
    binding_path = _relative_descriptor(
        base,
        package["row_bindings"],
        "row_bindings",
        expected_sha256=pins["row_bindings_sha256"],
    )
    bindings = _load_combined_bindings(
        binding_path,
        ordered_ids=ordered_ids,
        order_sha256=pins["benchmark_order_sha256"],
    )
    rows = _read_jsonl(pool_path, "candidate pool")
    if len(rows) != ROW_COUNT:
        raise SelectorError("candidate pool has missing or extra rows")
    for index, row in enumerate(rows):
        validate_pool_row(row, index)
        for role in UPSTREAM_ROLES:
            projection = _role_projection(row, role)
            digest = hashlib.sha256(_canonical_json(projection, newline=False)).hexdigest()
            if digest != bindings[role][index]:
                raise SelectorError(
                    f"candidate pool row {index} role {role} violates pinned row binding"
                )
    return rows, ordered_ids, routes, protected_source_union, _sha256(pool_path)


def _trusted_choice(candidate: dict[str, Any]) -> str | None:
    if not candidate["available"]:
        return None
    generation = candidate["generation"]
    if generation["finish_reason"] in {"error", "missing"} or generation["error"]:
        return None
    answer = candidate["final_answer"]
    if not isinstance(answer, str):
        return None
    match = CHOICE_RE.fullmatch(answer)
    return match.group(1).upper() if match else None


def _arm_result(
    *,
    arm: str,
    anchor_answer: str | None,
    anchor_choice: str | None,
    challenger: str | None,
    accepted: bool,
    reason: str,
    raw_support: int | None,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "action": "propose_challenger" if accepted else "preserve_anchor",
        "selected_answer": challenger if accepted else anchor_answer,
        "selected_choice": challenger if accepted else anchor_choice,
        "reason": reason,
        "raw_parallel_support": raw_support,
    }


def select_bound_row(
    row: dict[str, Any],
    *,
    task_id: str,
    authoritative_route: str,
    protected_source_union: frozenset[str],
) -> dict[str, Any]:
    validate_pool_row(row, row["row_index"])
    if authoritative_route not in {"deterministic", "image_judge"}:
        raise SelectorError("authoritative route is invalid")
    anchor_answer = row["anchor"]["final_answer"]
    anchor_choice = _trusted_choice(row["anchor"])
    v4 = _trusted_choice(row["routers"]["v4"])
    v5 = _trusted_choice(row["routers"]["v5"])
    protected = task_id in protected_source_union
    common_reason: str | None = None
    challenger: str | None = None
    if protected:
        common_reason = "protected_by_pinned_source_union"
    elif authoritative_route != "deterministic":
        common_reason = "authoritative_image_judge_route_byte_preserved"
    elif anchor_choice is None:
        common_reason = "anchor_is_not_valid_A_to_E"
    elif v4 is None or v5 is None:
        common_reason = "router_answer_missing_or_invalid"
    elif v4 != v5:
        common_reason = "router_disagreement"
    elif v4 == anchor_choice:
        common_reason = "router_agrees_with_anchor_no_change"
    else:
        challenger = v4

    votes = [
        *row["parallel_batches"]["parallel8_v1"]["raw_votes"],
        *row["parallel_batches"]["parallel8_reasoning_first_v2"]["raw_votes"],
    ]
    raw_support = (
        sum(_trusted_choice(candidate) == challenger for candidate in votes)
        if challenger is not None
        else None
    )
    primary_name = ALGORITHM_CONTRACT["primary_arm"]["name"]
    secondary_name = ALGORITHM_CONTRACT["secondary_arm"]["name"]
    if common_reason is not None:
        primary = _arm_result(
            arm=primary_name,
            anchor_answer=anchor_answer,
            anchor_choice=anchor_choice,
            challenger=challenger,
            accepted=False,
            reason=common_reason,
            raw_support=raw_support,
        )
        secondary = _arm_result(
            arm=secondary_name,
            anchor_answer=anchor_answer,
            anchor_choice=anchor_choice,
            challenger=challenger,
            accepted=False,
            reason=common_reason,
            raw_support=raw_support,
        )
    else:
        assert challenger is not None
        primary_accept = raw_support is not None and raw_support >= 13
        primary = _arm_result(
            arm=primary_name,
            anchor_answer=anchor_answer,
            anchor_choice=anchor_choice,
            challenger=challenger,
            accepted=primary_accept,
            reason=(
                "all_preregistered_gates_passed"
                if primary_accept
                else "raw_parallel_support_below_13_of_16"
            ),
            raw_support=raw_support,
        )
        p8_final = _trusted_choice(row["parallel_batches"]["parallel8_v1"]["final"])
        p8_reasoning_final = _trusted_choice(
            row["parallel_batches"]["parallel8_reasoning_first_v2"]["final"]
        )
        secondary_accept = p8_final == challenger and p8_reasoning_final == challenger
        secondary = _arm_result(
            arm=secondary_name,
            anchor_answer=anchor_answer,
            anchor_choice=anchor_choice,
            challenger=challenger,
            accepted=secondary_accept,
            reason=(
                "all_preregistered_gates_passed"
                if secondary_accept
                else "four_final_answers_not_unanimous"
            ),
            raw_support=raw_support,
        )
    if (protected or authoritative_route == "image_judge") and (
        primary["selected_answer"] != anchor_answer
        or secondary["selected_answer"] != anchor_answer
    ):
        raise SelectorError("safety-vetoed row changed despite anchor preservation")
    return {
        "schema_version": OUTPUT_ROW_SCHEMA,
        "row_index": row["row_index"],
        "task_id": task_id,
        "authoritative_evaluation_route": authoritative_route,
        "protected_by_source_union": protected,
        "anchor_answer": anchor_answer,
        "router_challenger": challenger,
        "primary": primary,
        "secondary": secondary,
    }


def run_selector(
    root: Path,
    input_package_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    freeze = verify_preregistered_freeze(root)
    rows, ordered_ids, routes, protected, pool_sha = load_input_package(
        input_package_path,
        freeze["profile"]["authority_pins"],
    )
    outputs = [
        select_bound_row(
            row,
            task_id=ordered_ids[index],
            authoritative_route=routes[index],
            protected_source_union=protected,
        )
        for index, row in enumerate(rows)
    ]
    if any(
        output["protected_by_source_union"]
        and (
            output["primary"]["action"] != "preserve_anchor"
            or output["secondary"]["action"] != "preserve_anchor"
        )
        for output in outputs
    ):
        raise SelectorError("protected source-union invariant failed")
    if any(
        output["authoritative_evaluation_route"] == "image_judge"
        and (
            output["primary"]["action"] != "preserve_anchor"
            or output["secondary"]["action"] != "preserve_anchor"
        )
        for output in outputs
    ):
        raise SelectorError("image-judge byte-preservation invariant failed")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "selection_proposals.jsonl"
    rows_path.write_bytes(b"".join(_canonical_json(row) for row in outputs))
    primary_counts = Counter(row["primary"]["action"] for row in outputs)
    secondary_counts = Counter(row["secondary"]["action"] for row in outputs)
    manifest = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA,
        "selector_id": SELECTOR_ID,
        "status": "new_selector_arm_outputs_frozen_not_evaluated",
        "artifact_kind": "patch_proposals_not_a_scored_solver",
        "rows": ROW_COUNT,
        "benchmark_sha256": BENCHMARK_SHA256,
        "candidate_pool_sha256": pool_sha,
        "freeze_sha256": freeze["freeze_sha256"],
        "profile_sha256": freeze["artifacts"]["profile"]["sha256"],
        "primary_action_counts": dict(sorted(primary_counts.items())),
        "secondary_action_counts": dict(sorted(secondary_counts.items())),
        "source_union_changes": 0,
        "image_judge_changes": 0,
        "selection_proposals": {
            "path": "selection_proposals.jsonl",
            "sha256": _sha256(rows_path),
        },
        "runtime_outcome_access": False,
    }
    manifest_path = output_dir / "selector_manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    return {
        "status": manifest["status"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed preregistered baseline selector v1.1")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-freeze", action="store_true")
    parser.add_argument("--input-package", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.input_package:
            if args.output_dir is None:
                raise SelectorError("--output-dir is required with --input-package")
            report = run_selector(args.root, args.input_package, args.output_dir)
        else:
            if not args.verify_freeze:
                raise SelectorError("use --verify-freeze or provide --input-package")
            report = verify_preregistered_freeze(args.root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except SelectorError as exc:
        print(f"selector v1.1 error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
