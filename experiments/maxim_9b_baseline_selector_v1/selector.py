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
PROFILE_SCHEMA = "maxim-9b-baseline-selector-profile-v1"
FREEZE_SCHEMA = "maxim-9b-baseline-selector-preregistered-freeze-v1"
POOL_MANIFEST_SCHEMA = "maxim-9b-baseline-candidate-pool-manifest-v1"
POOL_ROW_SCHEMA = "maxim-9b-baseline-candidate-pool-row-v1"
OUTPUT_ROW_SCHEMA = "maxim-9b-baseline-selector-output-row-v1"
OUTPUT_MANIFEST_SCHEMA = "maxim-9b-baseline-selector-output-manifest-v1"
SOURCE_AGGREGATE_SCHEMA = "maxim-9b-source-replay-aggregate-v1"
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


class SelectorError(RuntimeError):
    """Raised when a selector artifact violates the preregistered contract."""


EXPECTED_PROFILE: dict[str, Any] = {
    "schema_version": PROFILE_SCHEMA,
    "selector_id": "maxim_9b_baseline_selector_v1",
    "status": "preregistered_after_candidate_pool_structural_audit_before_any_gold_score_or_judge_outcome_access",
    "model_closure": [MODEL],
    "allowed_signal_families": [
        "observable_question_image_and_evaluator_format_metadata",
        "answer_format_validity_and_completion",
        "candidate_answer_agreement",
        "generation_metadata",
        "source_independent_deterministic_confidence",
    ],
    "forbidden_signal_families": [
        "task_id_quality_rules_or_task_specific_overrides",
        "gold_or_reference_answers",
        "benchmark_score_correctness_or_outcomes",
        "judge_verdicts",
        "known_error_memory",
        "official_or_web_source_evidence_as_a_quality_signal",
        "trained_or_fitted_quality_selector",
    ],
    "upstream_artifact_sha256": dict(UPSTREAM_SHA256),
    "candidate_provenance": {
        "native_router_pair": "two_correlated_native_thinking_ablations_with_same_routed86_model_and_seeds_but_different_max_tokens_and_treatment",
        "parallel_pair": "two_separately_executed_8_route_batches_with_same_routes_and_model_v2_core_and_v1_diversity_donor",
        "independence_claim": "not_four_fully_independent_systems",
        "endpoint_revision_caveat": "parallel_batches_have_no_immutable_endpoint_revision_manifest",
    },
    "input_contract": {
        "candidate_model": MODEL,
        "opaque_row_id_is_join_only_and_never_a_quality_feature": True,
        "answer_alphabet": ["A", "B", "C", "D", "E"],
        "parallel_votes_per_batch": 8,
        "parallel_batches": ["parallel8_v1", "parallel8_reasoning_first_v2"],
        "all_candidates_are_source_independent": True,
    },
    "protected_scope": {
        "purpose": "safety_veto_only_never_challenger_selection",
        "aggregate_repository_relative_path": SOURCE_AGGREGATE_REPOSITORY_PATH,
        "aggregate_schema": SOURCE_AGGREGATE_SCHEMA,
        "aggregate_file_sha256": SOURCE_AGGREGATE_SHA256,
        "source_union_projection_field": "source_union.latest_stage_owner_projection",
        "source_union_size": SOURCE_UNION_SIZE,
        "source_union_projection_sha256": SOURCE_UNION_SHA256,
        "policy": "both_arms_preserve_anchor_inside_frozen_source_union",
    },
    "algorithm": {
        "shared_gates": [
            "observable_evaluation_route_is_deterministic",
            "opaque_id_is_outside_pinned_source_union",
            "anchor_is_valid_A_to_E",
            "native_router_v4_and_v5_are_valid_and_agree",
            "router_challenger_differs_from_anchor",
        ],
        "primary_arm": {
            "name": "router_agreement_plus_parallel16_supermajority",
            "role": "primary_conservative",
            "rule": "V4_equals_V5_and_challenger_has_at_least_13_of_16_raw_parallel_votes",
            "raw_vote_threshold": 13,
            "raw_vote_denominator": 16,
            "threshold_rationale": "fixed_81.25_percent_supermajority_with_iid_fair_null_tail_697_of_65536_but_no_independence_claim",
            "fallback": "preserve_active_crop_v2_anchor",
        },
        "secondary_arm": {
            "name": "four_final_answers_unanimous",
            "role": "preregistered_exploratory",
            "rule": "V4_equals_V5_equals_parallel8_v1_final_equals_parallel8_reasoning_first_v2_final",
            "fallback": "preserve_active_crop_v2_anchor",
        },
        "image_judge_policy": "always_preserve_anchor_byte_for_byte",
        "source_stack_policy": "selector_emits_proposals_only_and_never_mutates_the_frozen_source_stack",
        "missing_invalid_tie_or_profile_mismatch_policy": "emit_no_patch_and_preserve_anchor",
        "no_training_calibration_or_post_score_threshold_change": True,
    },
    "access_attestations": {
        "candidate_pool_access_before_profile_freeze": True,
        "candidate_answer_agreement_counts_access": True,
        "source_union_membership_for_safety_veto": True,
        "evaluation_route_format_metadata_access": True,
        "upstream_provenance_summary_access_before_freeze": True,
        "gold_access": False,
        "reference_answer_access": False,
        "benchmark_score_access": False,
        "benchmark_correctness_access": False,
        "benchmark_outcome_access": False,
        "judge_access": False,
        "known_error_memory_access": False,
        "official_or_web_source_access_for_selection": False,
    },
}


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


def _relative_descriptor(
    base: Path,
    value: Any,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
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
        raise SelectorError(f"{label}.path escaped its artifact root") from exc
    digest = str(value["sha256"] or "").casefold()
    if not SHA256_RE.fullmatch(digest):
        raise SelectorError(f"{label}.sha256 is malformed")
    if expected_sha256 is not None and digest != expected_sha256:
        raise SelectorError(f"{label} does not match its preregistered SHA-256")
    if _sha256(path) != digest:
        raise SelectorError(f"{label} SHA-256 mismatch")
    return path, digest


def load_profile(path: Path) -> dict[str, Any]:
    profile = _read_json(path, "selector profile")
    if profile != EXPECTED_PROFILE:
        raise SelectorError("selector profile differs from the preregistered v1 contract")
    return profile


def verify_preregistered_freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    freeze_path = root / "PREREGISTERED_FREEZE.json"
    freeze = _read_json(freeze_path, "preregistered freeze")
    _strict_keys(
        freeze,
        {
            "schema_version",
            "selector_id",
            "status",
            "frozen_after_candidate_pool_structural_audit",
            "frozen_before_gold_score_correctness_or_judge_outcome_access",
            "artifacts",
        },
        "preregistered freeze",
    )
    if freeze["schema_version"] != FREEZE_SCHEMA:
        raise SelectorError("unexpected freeze schema")
    if freeze["selector_id"] != EXPECTED_PROFILE["selector_id"]:
        raise SelectorError("freeze selector_id mismatch")
    if freeze["status"] != "preregistered_after_structural_audit_before_evaluation":
        raise SelectorError("freeze status is not preregistered")
    if freeze["frozen_after_candidate_pool_structural_audit"] is not True:
        raise SelectorError("freeze chronology omits the prior candidate-pool structural audit")
    if freeze["frozen_before_gold_score_correctness_or_judge_outcome_access"] is not True:
        raise SelectorError("freeze does not predate evaluation access")
    artifacts = freeze["artifacts"]
    if not isinstance(artifacts, dict):
        raise SelectorError("freeze.artifacts must be an object")
    _strict_keys(artifacts, {"profile", "selector_code", "tests"}, "freeze.artifacts")
    expected_paths = {
        "profile": "profile.json",
        "selector_code": "selector.py",
        "tests": "test_selector.py",
    }
    verified: dict[str, dict[str, str]] = {}
    for role, expected_name in expected_paths.items():
        path, digest = _relative_descriptor(root, artifacts[role], f"freeze.{role}")
        if path.name != expected_name:
            raise SelectorError(f"freeze.{role} points to the wrong file")
        verified[role] = {"path": str(path), "sha256": digest}
    load_profile(Path(verified["profile"]["path"]))
    return {
        "status": "preregistered_freeze_verified",
        "freeze_path": str(freeze_path),
        "freeze_sha256": _sha256(freeze_path),
        "artifacts": verified,
    }


def _find_repository_artifact(start: Path, repository_relative_path: str) -> Path:
    relative = Path(repository_relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SelectorError("protected scope repository path is unsafe")
    candidates: list[Path] = []
    start = start.resolve()
    for ancestor in (start, *start.parents):
        candidate = (ancestor / relative).resolve()
        try:
            candidate.relative_to(ancestor)
        except ValueError:
            continue
        if candidate.is_file():
            candidates.append(candidate)
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise SelectorError(
            f"protected scope authority must resolve uniquely; found {len(unique)} candidates"
        )
    return unique[0]


def _validate_source_union_object(aggregate: dict[str, Any]) -> frozenset[str]:
    if aggregate.get("schema_version") != SOURCE_AGGREGATE_SCHEMA:
        raise SelectorError("protected scope aggregate schema mismatch")
    source_union = aggregate.get("source_union")
    if not isinstance(source_union, dict):
        raise SelectorError("protected scope source_union is missing")
    _strict_keys(
        source_union,
        {"size", "sha256", "latest_stage_owner_projection", "answer_conflicts"},
        "protected scope source_union",
    )
    if source_union["size"] != SOURCE_UNION_SIZE:
        raise SelectorError("protected scope source_union size mismatch")
    if source_union["sha256"] != SOURCE_UNION_SHA256:
        raise SelectorError("protected scope source_union internal SHA mismatch")
    if source_union["answer_conflicts"] != 0:
        raise SelectorError("protected scope source_union has answer conflicts")
    projection = source_union["latest_stage_owner_projection"]
    if not isinstance(projection, list) or len(projection) != SOURCE_UNION_SIZE:
        raise SelectorError("protected scope projection row count mismatch")
    computed = hashlib.sha256(_canonical_json(projection, newline=False)).hexdigest()
    if computed != SOURCE_UNION_SHA256:
        raise SelectorError("protected scope projection bytes do not match internal SHA")
    opaque_ids: list[str] = []
    for index, record in enumerate(projection):
        if not isinstance(record, dict):
            raise SelectorError(f"protected scope projection[{index}] is not an object")
        _strict_keys(
            record,
            {"task_id", "answer_sha256", "owner_stage"},
            f"protected scope projection[{index}]",
        )
        task_id = record["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise SelectorError(f"protected scope projection[{index}].task_id is invalid")
        if not SHA256_RE.fullmatch(str(record["answer_sha256"] or "")):
            raise SelectorError(
                f"protected scope projection[{index}].answer_sha256 is malformed"
            )
        _safe_identifier(record["owner_stage"], f"protected scope projection[{index}].owner_stage")
        opaque_ids.append(task_id)
    if len(set(opaque_ids)) != SOURCE_UNION_SIZE:
        raise SelectorError("protected scope projection has duplicate task_id values")
    return frozenset(opaque_ids)


def load_protected_source_union(root: Path) -> tuple[frozenset[str], dict[str, str]]:
    path = _find_repository_artifact(root, SOURCE_AGGREGATE_REPOSITORY_PATH)
    digest = _sha256(path)
    if digest != SOURCE_AGGREGATE_SHA256:
        raise SelectorError("protected scope aggregate file SHA-256 mismatch")
    aggregate = _read_json(path, "protected scope aggregate")
    opaque_ids = _validate_source_union_object(aggregate)
    return opaque_ids, {
        "aggregate_path": str(path),
        "aggregate_sha256": digest,
        "source_union_sha256": SOURCE_UNION_SHA256,
    }


def _validate_observable(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectorError("observable must be an object")
    _strict_keys(
        value,
        {
            "answer_type",
            "option_count",
            "evaluation_route",
            "subject",
            "grade",
            "question_char_count",
            "ocr_char_count",
            "image_count",
            "image_widths",
            "image_heights",
        },
        "observable",
    )
    if value["answer_type"] not in {"choice", "other"}:
        raise SelectorError("observable.answer_type is invalid")
    option_count = value["option_count"]
    if option_count is not None and (
        isinstance(option_count, bool)
        or not isinstance(option_count, int)
        or not 2 <= option_count <= 5
    ):
        raise SelectorError("observable.option_count is outside 2..5")
    if value["evaluation_route"] not in {"deterministic", "image_judge"}:
        raise SelectorError("observable.evaluation_route is invalid")
    for key in ("subject", "grade"):
        if value[key] is not None and (
            not isinstance(value[key], str) or len(value[key]) > 128
        ):
            raise SelectorError(f"observable.{key} is invalid")
    for key in ("question_char_count", "image_count"):
        _nonnegative_int(value[key], f"observable.{key}")
    if value["ocr_char_count"] is not None:
        _nonnegative_int(value["ocr_char_count"], "observable.ocr_char_count")
    for key in ("image_widths", "image_heights"):
        dimensions = value[key]
        if not isinstance(dimensions, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in dimensions
        ):
            raise SelectorError(f"observable.{key} must be positive integer dimensions")
        if len(dimensions) != value["image_count"]:
            raise SelectorError(f"observable.{key} length differs from image_count")
    return value


def _validate_generation(value: Any, label: str, upstream_role: str) -> dict[str, Any]:
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
            "gold_access",
            "score_or_outcome_access",
            "known_error_memory_access",
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
    if value["upstream_artifact_sha256"] != UPSTREAM_SHA256[upstream_role]:
        raise SelectorError(f"{label} upstream profile mismatch")
    for key in (
        "gold_access",
        "score_or_outcome_access",
        "known_error_memory_access",
        "source_access",
    ):
        _false(value[key], f"{label}.{key}")
    return value


def _validate_candidate(value: Any, label: str, upstream_role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectorError(f"{label} must be an object")
    _strict_keys(value, {"available", "model", "final_answer", "generation"}, label)
    if not isinstance(value["available"], bool):
        raise SelectorError(f"{label}.available must be boolean")
    if value["model"] != MODEL:
        raise SelectorError(f"{label} model closure is not pure Qwen3.5-9B")
    if value["available"]:
        if not isinstance(value["final_answer"], str):
            raise SelectorError(f"{label}.final_answer must be string when available")
    elif value["final_answer"] is not None:
        raise SelectorError(f"{label}.final_answer must be null when unavailable")
    _validate_generation(value["generation"], f"{label}.generation", upstream_role)
    return value


def _validate_parallel_batch(value: Any, label: str, upstream_role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectorError(f"{label} must be an object")
    _strict_keys(value, {"final", "raw_votes"}, label)
    _validate_candidate(value["final"], f"{label}.final", upstream_role)
    raw_votes = value["raw_votes"]
    if not isinstance(raw_votes, list) or len(raw_votes) != 8:
        raise SelectorError(f"{label}.raw_votes must contain exactly 8 candidates")
    for index, candidate in enumerate(raw_votes):
        _validate_candidate(candidate, f"{label}.raw_votes[{index}]", upstream_role)
    return value


def validate_pool_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise SelectorError("candidate pool row must be an object")
    _strict_keys(
        row,
        {"schema_version", "opaque_id", "observable", "anchor", "routers", "parallel_batches"},
        "pool row",
    )
    if row["schema_version"] != POOL_ROW_SCHEMA:
        raise SelectorError("unexpected candidate pool row schema")
    opaque_id = row["opaque_id"]
    if not isinstance(opaque_id, str) or not opaque_id or len(opaque_id) > 256:
        raise SelectorError("opaque_id must be a non-empty join key")
    _validate_observable(row["observable"])
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
    _strict_keys(
        batches,
        {"parallel8_v1", "parallel8_reasoning_first_v2"},
        "parallel_batches",
    )
    _validate_parallel_batch(batches["parallel8_v1"], "parallel8_v1", "parallel8_v1")
    _validate_parallel_batch(
        batches["parallel8_reasoning_first_v2"],
        "parallel8_reasoning_first_v2",
        "parallel8_reasoning_first_v2",
    )
    return row


def _trusted_choice(candidate: dict[str, Any], observable: dict[str, Any]) -> str | None:
    if not candidate["available"]:
        return None
    generation = candidate["generation"]
    if generation["finish_reason"] in {"error", "missing"} or generation["error"]:
        return None
    answer = candidate["final_answer"]
    if not isinstance(answer, str):
        return None
    match = CHOICE_RE.fullmatch(answer)
    if not match:
        return None
    choice = match.group(1).upper()
    option_count = observable["option_count"]
    if option_count is not None and ord(choice) - ord("A") >= option_count:
        return None
    return choice


def _anchor_result(
    *,
    name: str,
    anchor: dict[str, Any],
    anchor_choice: str | None,
    reason: str,
    raw_support: int | None,
) -> dict[str, Any]:
    return {
        "arm": name,
        "action": "preserve_anchor",
        "selected_answer": anchor["final_answer"],
        "selected_choice": anchor_choice,
        "reason": reason,
        "raw_parallel_support": raw_support,
    }


def _replacement_result(
    *,
    name: str,
    challenger: str,
    raw_support: int | None,
) -> dict[str, Any]:
    return {
        "arm": name,
        "action": "propose_challenger",
        "selected_answer": challenger,
        "selected_choice": challenger,
        "reason": "all_preregistered_gates_passed",
        "raw_parallel_support": raw_support,
    }


def select_row(row: dict[str, Any], protected_source_union: frozenset[str]) -> dict[str, Any]:
    validate_pool_row(row)
    observable = row["observable"]
    anchor = row["anchor"]
    anchor_choice = _trusted_choice(anchor, observable)
    primary_name = EXPECTED_PROFILE["algorithm"]["primary_arm"]["name"]
    secondary_name = EXPECTED_PROFILE["algorithm"]["secondary_arm"]["name"]

    common_reason: str | None = None
    if row["opaque_id"] in protected_source_union:
        common_reason = "protected_by_pinned_source_union"
    elif observable["evaluation_route"] != "deterministic":
        common_reason = "image_judge_route_is_byte_preserved"
    elif observable["answer_type"] != "choice" or anchor_choice is None:
        common_reason = "anchor_or_answer_format_is_not_valid_A_to_E"

    v4 = _trusted_choice(row["routers"]["v4"], observable)
    v5 = _trusted_choice(row["routers"]["v5"], observable)
    challenger: str | None = None
    if common_reason is None:
        if v4 is None or v5 is None:
            common_reason = "router_answer_missing_or_invalid"
        elif v4 != v5:
            common_reason = "router_disagreement"
        elif v4 == anchor_choice:
            common_reason = "router_agrees_with_anchor_no_change"
        else:
            challenger = v4

    raw_votes = [
        *row["parallel_batches"]["parallel8_v1"]["raw_votes"],
        *row["parallel_batches"]["parallel8_reasoning_first_v2"]["raw_votes"],
    ]
    raw_support = (
        sum(_trusted_choice(candidate, observable) == challenger for candidate in raw_votes)
        if challenger is not None
        else None
    )

    if common_reason is not None:
        primary = _anchor_result(
            name=primary_name,
            anchor=anchor,
            anchor_choice=anchor_choice,
            reason=common_reason,
            raw_support=raw_support,
        )
        secondary = _anchor_result(
            name=secondary_name,
            anchor=anchor,
            anchor_choice=anchor_choice,
            reason=common_reason,
            raw_support=raw_support,
        )
    else:
        assert challenger is not None
        threshold = EXPECTED_PROFILE["algorithm"]["primary_arm"]["raw_vote_threshold"]
        if raw_support is not None and raw_support >= threshold:
            primary = _replacement_result(
                name=primary_name,
                challenger=challenger,
                raw_support=raw_support,
            )
        else:
            primary = _anchor_result(
                name=primary_name,
                anchor=anchor,
                anchor_choice=anchor_choice,
                reason="raw_parallel_support_below_13_of_16",
                raw_support=raw_support,
            )

        p8_final = _trusted_choice(
            row["parallel_batches"]["parallel8_v1"]["final"], observable
        )
        p8_reasoning_final = _trusted_choice(
            row["parallel_batches"]["parallel8_reasoning_first_v2"]["final"],
            observable,
        )
        if p8_final == challenger and p8_reasoning_final == challenger:
            secondary = _replacement_result(
                name=secondary_name,
                challenger=challenger,
                raw_support=raw_support,
            )
        else:
            secondary = _anchor_result(
                name=secondary_name,
                anchor=anchor,
                anchor_choice=anchor_choice,
                reason="four_final_answers_not_unanimous",
                raw_support=raw_support,
            )

    protected = row["opaque_id"] in protected_source_union
    if protected and (
        primary["selected_answer"] != anchor["final_answer"]
        or secondary["selected_answer"] != anchor["final_answer"]
    ):
        raise SelectorError("protected source-union row changed despite safety veto")
    if observable["evaluation_route"] == "image_judge" and (
        primary["selected_answer"] != anchor["final_answer"]
        or secondary["selected_answer"] != anchor["final_answer"]
    ):
        raise SelectorError("image-judge row changed despite byte-preservation policy")
    return {
        "schema_version": OUTPUT_ROW_SCHEMA,
        "opaque_id": row["opaque_id"],
        "anchor_answer": anchor["final_answer"],
        "protected_by_source_union": protected,
        "evaluation_route": observable["evaluation_route"],
        "router_challenger": challenger,
        "primary": primary,
        "secondary": secondary,
    }


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SelectorError(f"{label}:{line_number} is not an object")
                opaque_id = value.get("opaque_id")
                if not isinstance(opaque_id, str) or not opaque_id or opaque_id in seen:
                    raise SelectorError(f"{label}:{line_number} has missing/duplicate opaque_id")
                seen.add(opaque_id)
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorError(f"cannot read {label} {path}: {exc}") from exc
    return rows


def load_candidate_pool(
    manifest_path: Path,
    *,
    freeze_sha256: str,
    profile_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path, "candidate pool manifest")
    _strict_keys(
        manifest,
        {
            "schema_version",
            "pool_id",
            "created_after_selector_freeze",
            "created_before_evaluation",
            "rows",
            "model_closure",
            "selector_freeze_sha256",
            "selector_profile_sha256",
            "source_union_authority",
            "upstream_artifacts",
            "candidate_pool",
            "access_attestations",
        },
        "candidate pool manifest",
    )
    if manifest["schema_version"] != POOL_MANIFEST_SCHEMA:
        raise SelectorError("unexpected candidate pool manifest schema")
    _safe_identifier(manifest["pool_id"], "pool_id")
    if manifest["created_after_selector_freeze"] is not True:
        raise SelectorError("candidate pool does not attest creation after selector freeze")
    if manifest["created_before_evaluation"] is not True:
        raise SelectorError("candidate pool was not frozen before evaluation")
    if manifest["selector_freeze_sha256"] != freeze_sha256:
        raise SelectorError("candidate pool selector freeze pin mismatch")
    if manifest["selector_profile_sha256"] != profile_sha256:
        raise SelectorError("candidate pool selector profile pin mismatch")
    if manifest["model_closure"] != [MODEL]:
        raise SelectorError("candidate pool model closure is not pure Qwen3.5-9B")
    rows_count = _nonnegative_int(manifest["rows"], "candidate pool rows")

    source_authority = manifest["source_union_authority"]
    if not isinstance(source_authority, dict):
        raise SelectorError("candidate pool source_union_authority must be an object")
    _strict_keys(
        source_authority,
        {"aggregate_sha256", "source_union_sha256", "source_union_size"},
        "candidate pool source_union_authority",
    )
    if source_authority != {
        "aggregate_sha256": SOURCE_AGGREGATE_SHA256,
        "source_union_sha256": SOURCE_UNION_SHA256,
        "source_union_size": SOURCE_UNION_SIZE,
    }:
        raise SelectorError("candidate pool protected source-union authority mismatch")

    upstream = manifest["upstream_artifacts"]
    if not isinstance(upstream, dict):
        raise SelectorError("candidate pool upstream_artifacts must be an object")
    _strict_keys(upstream, set(UPSTREAM_SHA256), "candidate pool upstream_artifacts")
    for role, expected_digest in UPSTREAM_SHA256.items():
        _relative_descriptor(
            manifest_path.parent,
            upstream[role],
            f"upstream_artifacts.{role}",
            expected_sha256=expected_digest,
        )

    attestations = manifest["access_attestations"]
    if not isinstance(attestations, dict):
        raise SelectorError("candidate pool access_attestations must be an object")
    expected_attestations = {
        "gold_access",
        "reference_answer_access",
        "benchmark_score_access",
        "benchmark_correctness_access",
        "benchmark_outcome_access",
        "judge_access",
        "known_error_memory_access",
        "official_or_web_source_access_for_candidate_generation",
    }
    _strict_keys(attestations, expected_attestations, "candidate pool access_attestations")
    for key in expected_attestations:
        _false(attestations[key], f"candidate pool access_attestations.{key}")

    pool_path, pool_sha = _relative_descriptor(
        manifest_path.parent, manifest["candidate_pool"], "candidate_pool"
    )
    rows = _read_jsonl(pool_path, "candidate pool")
    if len(rows) != rows_count:
        raise SelectorError("candidate pool row count differs from manifest")
    for row in rows:
        validate_pool_row(row)
    return manifest, rows, pool_sha


def _write_outputs(
    *,
    outputs: list[dict[str, Any]],
    output_dir: Path,
    freeze_report: dict[str, Any],
    pool_sha256: str,
    source_authority: dict[str, str],
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "selections.jsonl"
    output_path.write_bytes(b"".join(_canonical_json(value) for value in outputs))
    primary_counts = Counter(value["primary"]["action"] for value in outputs)
    secondary_counts = Counter(value["secondary"]["action"] for value in outputs)
    protected_rows = sum(value["protected_by_source_union"] for value in outputs)
    image_rows = sum(value["evaluation_route"] == "image_judge" for value in outputs)
    if any(
        value["protected_by_source_union"]
        and (
            value["primary"]["action"] != "preserve_anchor"
            or value["secondary"]["action"] != "preserve_anchor"
        )
        for value in outputs
    ):
        raise SelectorError("output violates protected source-union invariant")
    if any(
        value["evaluation_route"] == "image_judge"
        and (
            value["primary"]["action"] != "preserve_anchor"
            or value["secondary"]["action"] != "preserve_anchor"
        )
        for value in outputs
    ):
        raise SelectorError("output violates image-judge byte-preservation invariant")
    manifest = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA,
        "selector_id": EXPECTED_PROFILE["selector_id"],
        "evaluation_status": "not_evaluated_before_preregistered_selection_freeze",
        "artifact_kind": "gold_blind_patch_proposals_not_a_scored_solver",
        "rows": len(outputs),
        "candidate_pool_sha256": pool_sha256,
        "profile_sha256": freeze_report["artifacts"]["profile"]["sha256"],
        "selector_code_sha256": freeze_report["artifacts"]["selector_code"]["sha256"],
        "tests_sha256": freeze_report["artifacts"]["tests"]["sha256"],
        "freeze_sha256": freeze_report["freeze_sha256"],
        "protected_source_union": {
            **source_authority,
            "protected_rows_in_pool": protected_rows,
            "primary_changes_inside_union": 0,
            "secondary_changes_inside_union": 0,
        },
        "image_judge_preservation": {
            "image_rows_in_pool": image_rows,
            "primary_changes": 0,
            "secondary_changes": 0,
            "composition_requirement": "reuse_anchor_image_judge_bytes_unchanged",
        },
        "primary_action_counts": dict(sorted(primary_counts.items())),
        "secondary_action_counts": dict(sorted(secondary_counts.items())),
        "selections": {"path": "selections.jsonl", "sha256": _sha256(output_path)},
        "access_attestations": dict(EXPECTED_PROFILE["access_attestations"]),
    }
    manifest_path = output_dir / "selector_manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    return {
        "status": "selection_proposals_frozen_without_evaluation",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "rows": len(outputs),
        "primary_action_counts": dict(sorted(primary_counts.items())),
        "secondary_action_counts": dict(sorted(secondary_counts.items())),
    }


def run_selector(root: Path, pool_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    freeze_report = verify_preregistered_freeze(root)
    protected_union, source_authority = load_protected_source_union(root)
    _, rows, pool_sha = load_candidate_pool(
        pool_manifest_path,
        freeze_sha256=freeze_report["freeze_sha256"],
        profile_sha256=freeze_report["artifacts"]["profile"]["sha256"],
    )
    row_ids = {row["opaque_id"] for row in rows}
    missing_protected = protected_union - row_ids
    if missing_protected:
        raise SelectorError(
            f"candidate pool omits {len(missing_protected)} protected source-union rows"
        )
    outputs = [select_row(row, protected_union) for row in rows]
    return _write_outputs(
        outputs=outputs,
        output_dir=output_dir,
        freeze_report=freeze_report,
        pool_sha256=pool_sha,
        source_authority=source_authority,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preregistered gold-blind conservative selector for pure Qwen3.5-9B candidates"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-freeze", action="store_true")
    parser.add_argument("--verify-source-union", action="store_true")
    parser.add_argument("--pool-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.pool_manifest:
            if args.output_dir is None:
                raise SelectorError("--output-dir is required with --pool-manifest")
            report = run_selector(args.root, args.pool_manifest, args.output_dir)
        elif args.verify_source_union:
            verify_preregistered_freeze(args.root)
            opaque_ids, authority = load_protected_source_union(args.root)
            report = {
                "status": "protected_source_union_verified_after_selector_freeze",
                "rows": len(opaque_ids),
                **authority,
            }
        else:
            if not args.verify_freeze:
                raise SelectorError(
                    "use --verify-freeze, --verify-source-union, or provide --pool-manifest"
                )
            report = verify_preregistered_freeze(args.root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except SelectorError as exc:
        print(f"selector error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
