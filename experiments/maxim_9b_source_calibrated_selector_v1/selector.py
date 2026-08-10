from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable
import unicodedata


MODEL = "Qwen/Qwen3.5-9B"
EXPERIMENT_ID = "maxim_9b_source_calibrated_selector_v1"
PREREG_SCHEMA = "maxim-9b-source-calibrated-selector-preregistration-v1"
CALIBRATION_SCHEMA = "maxim-9b-source-calibration-v1"
DECISION_SCHEMA = "maxim-9b-source-calibrated-decision-v1"
MANIFEST_SCHEMA = "maxim-9b-source-calibrated-candidate-manifest-v1"
FREEZE_SCHEMA = "maxim-9b-source-calibrated-freeze-v1"
PINS_SCHEMA = "maxim-9b-source-calibrated-freeze-pins-v1"
SOURCE_AGGREGATE_SCHEMA = "maxim-9b-source-replay-aggregate-v1"
SOURCE_FREEZE_SCHEMA = "maxim-9b-source-replay-freeze-v1"
INPUT_PACKAGE_SCHEMA = "maxim-9b-baseline-selector-input-package-v1.1"
BENCHMARK_ORDER_SCHEMA = "maxim-9b-baseline-selector-benchmark-order-v1.1"
ROUTE_MAP_SCHEMA = "maxim-9b-baseline-selector-route-map-v1.1"
ROW_BINDINGS_SCHEMA = "maxim-9b-baseline-selector-combined-row-bindings-v1.1"
SOURCE_MEMBERSHIP_SCHEMA = "maxim-9b-baseline-selector-source-union-membership-v1.1"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CHOICE_RE = re.compile(r"^\s*[\[(]?\s*([a-eA-E])\s*[\])]?\s*[.)]?\s*$")

ALPHA = 1.0
BETA = 1.0
MIN_FOLD_VALID = 20
MIN_FOLD_POSTERIOR = 0.50
MIN_GLOBAL_DECISIVE = 3
MAX_LOGIT_WEIGHT = 3.0
ANCHOR_BONUS = 0.25
MIN_MARGIN = 0.10
MIN_TOP_SHARE = 0.50
MIN_SUPPORT_GROUPS = 2

ROLE_ORDER = (
    "active_crop_v2",
    "no_tools_v1",
    "native_thinking_v4",
    "native_thinking_v5",
    "parallel8_v1",
    "parallel8_reasoning_first_v2",
)
DONOR_PRIORITY = (
    "active_crop_v2",
    "no_tools_v1",
    "native_thinking_v5",
    "native_thinking_v4",
    "parallel8_reasoning_first_v2",
    "parallel8_v1",
)
ROLE_GROUP = {
    "active_crop_v2": "active_crop_anchor",
    "no_tools_v1": "direct_no_tools",
    "native_thinking_v4": "native_thinking_pair",
    "native_thinking_v5": "native_thinking_pair",
    "parallel8_v1": "parallel8_pair",
    "parallel8_reasoning_first_v2": "parallel8_pair",
}
UPSTREAM = {
    "active_crop_v2": {
        "path": "reports/maxim_query_active_crop_v2_20260803/solver.jsonl",
        "sha256": "6697c043f3142a736b817ead5da494eea334f5349e0db833bd72f23fe35cb17c",
    },
    "no_tools_v1": {
        "path": "artifacts/baselines/no_tools_v1/b0_no_tools_raw.jsonl",
        "sha256": "496236da966ed68aa81af3d33da1c40b85c5a11b342de253ada244f97320de8f",
    },
    "native_thinking_v4": {
        "path": "reports/maxim_native_thinking_math_router_v4_20260803/solver.jsonl",
        "sha256": "0fd0e6fef6b220749faa015e7a163cdf596e070afaaad407e4d36bc9b1337307",
    },
    "native_thinking_v5": {
        "path": "reports/maxim_native_thinking_math_router_v5_20260803/solver.jsonl",
        "sha256": "45dc8c16f834d27e5d114f9162b7984c8baec4037fef011730e91ba13f192845",
    },
    "parallel8_v1": {
        "path": "reports/maxim_ideas_full274_20260731/parallel8_v1/solver.jsonl",
        "sha256": "b1b7a1b785a9a3fc076c04c37fa72f4a82169f137651ff57b40e984901b0d645",
    },
    "parallel8_reasoning_first_v2": {
        "path": "reports/maxim_ideas_full274_20260731/parallel8_reasoning_first_v2/solver.jsonl",
        "sha256": "6115effd03d7eac3e11e9726ecd9822802a04f235a0b361e1863b9bc7e221023",
    },
}
SOURCE = {
    "aggregate": {
        "path": "reports/maxim_9b_source_replay_v1_20260809/active_crop/source_v7_aggregate/aggregate.json",
        "sha256": "3de5129dee80d2f2fda544bdf7eecfa7d0f467d56bb7e43afc8eac89a6a5dacd",
    },
    "freeze_manifest": {
        "path": "reports/maxim_9b_source_replay_v1_20260809/active_crop/source_v7_aggregate/freeze_manifest.json",
        "sha256": "68b11fd254ce698a7da6f13c154e3d291a43701f065211c0a29c60854098c773",
    },
    "final_solver": {
        "path": "reports/maxim_9b_source_replay_v1_20260809/active_crop/fill_composed/solver.jsonl",
        "sha256": "9d26067064ee07fe480391759782c86d66adbb76dbc0da0d86ccc1b3f035211e",
    },
    "union_size": 156,
    "union_sha256": "7e5e0c972e82d87d164cb3ef03b13fbb4c8084bf07c2512129958882871508cd",
}
INPUT_AUTHORITY = {
    "package": {
        "path": "experiments/maxim_9b_baseline_selector_v1/input/frozen/input_package_v1_1.json",
        "sha256": "f2e7bdf8ea0cd8d44d073c3cc3f7a6933a98de2b032d44e1e5625e98eb869f0e",
    },
    "benchmark_sha256": "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9",
    "benchmark_order": {
        "path": "benchmark_order_v1_1.json",
        "sha256": "7140c7c01b48053f6a15a3b0113f68cad37bbb887744828b570a6eaa0447d62b",
    },
    "candidate_pool": {
        "path": "candidate_pool_v1_1.jsonl",
        "sha256": "b755e730c0841fc154ce83f3f82ec8a136cbb177c54b0c0233bb3e118926c1b3",
    },
    "route_map": {
        "path": "evaluator_route_map_v1_1.json",
        "sha256": "f89ef00f95b9d83610b66948fcb11667dc927f2452b000ef62e031a1a0de26f6",
    },
    "row_bindings": {
        "path": "row_bindings_v1_1.json",
        "sha256": "ababaf1ff19b48275f5a6718177a4391984460e475ba475a40620dffab2e9aa6",
    },
    "source_union_membership": {
        "path": "source_union_membership_v1_1.json",
        "sha256": "93a1018a63e2b9dfeef841541df3b566d6bd6275471accb9f167c7c60c44416a",
    },
    "route_counts": {"deterministic": 177, "image_judge": 97},
}

PACKAGE_ROLE_MAP = {
    "active_crop_v2": "active_crop_v2",
    "native_thinking_v4": "native_thinking_math_router_v4",
    "native_thinking_v5": "native_thinking_math_router_v5",
    "parallel8_v1": "parallel8_v1",
    "parallel8_reasoning_first_v2": "parallel8_reasoning_first_v2",
}


class SelectorError(RuntimeError):
    pass


def canonical_bytes(value: Any, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + suffix
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SelectorError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectorError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectorError(f"{label} root must be an object")
    return value


def normalize_answer(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", value)
    text = " ".join(text.strip().casefold().split())
    if not text:
        return None
    choice = CHOICE_RE.fullmatch(text)
    if choice:
        return choice.group(1).upper()
    return text


_SALVAGE_KEYS = {"final_answer", "answer", "choice", "result", "output"}


def _strict_json_loads(text: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelectorError("duplicate JSON key in strict salvage")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise SelectorError(f"non-finite JSON constant in strict salvage: {value}")

    return json.loads(text, object_pairs_hook=pairs_hook, parse_constant=reject_constant)


def strict_scalar_salvage(value: Any, *, depth: int = 0) -> str | None:
    if depth > 3:
        return None
    if isinstance(value, dict):
        if len(value) != 1:
            return None
        key, child = next(iter(value.items()))
        if key not in _SALVAGE_KEYS:
            return None
        return strict_scalar_salvage(child, depth=depth + 1)
    if isinstance(value, bool) or value is None or isinstance(value, (list, tuple)):
        return None
    if isinstance(value, (str, int, float)):
        return normalize_answer(str(value))
    return None


def _salvage_text(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1)
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        parsed = _strict_json_loads(stripped)
    except (json.JSONDecodeError, SelectorError):
        return None
    return strict_scalar_salvage(parsed)


def candidate_answer(row: dict[str, Any]) -> str | None:
    final = row.get("final_answer")
    if isinstance(final, str) and final.strip():
        salvaged = _salvage_text(final)
        if salvaged is not None:
            return salvaged
        # Anything presented as structured output is malformed when it does
        # not satisfy the strict one-path schema.  Do not reinterpret broken
        # JSON or a rejected fenced object as an ordinary free-text answer.
        if final.lstrip().startswith(("{", "[", "```")):
            return _salvage_text(row.get("raw_response"))
        normalized = normalize_answer(final)
        if normalized is not None:
            return normalized
    return _salvage_text(row.get("raw_response"))


def _model_values(value: Any) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() == "model":
                found.append(child)
            found.extend(_model_values(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_model_values(child))
    return found


def validate_solver_row(row: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise SelectorError(f"{role} row is not an object")
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id or len(task_id) > 128:
        raise SelectorError(f"{role} row has unsafe task_id")
    if row.get("model") != MODEL:
        raise SelectorError(f"{role}:{task_id} top-level model closure failed")
    models = _model_values(row)
    if not models or any(value != MODEL for value in models):
        raise SelectorError(f"{role}:{task_id} nested model closure failed")
    if any("27b" in str(value).casefold() for value in models):
        raise SelectorError(f"{role}:{task_id} contains forbidden 27B model")
    answer = row.get("final_answer")
    if not (answer is None or isinstance(answer, str)) or (
        isinstance(answer, str) and len(answer) > 8192
    ):
        raise SelectorError(f"{role}:{task_id} final_answer is invalid")
    return row


def load_solver(path: Path, *, role: str, expected_sha256: str) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, bytes]]:
    if sha256_file(path) != expected_sha256:
        raise SelectorError(f"{role} SHA-256 mismatch")
    try:
        raw_lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        raise SelectorError(f"cannot read {role}: {exc}") from exc
    order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    raw: dict[str, bytes] = {}
    for number, line in enumerate(raw_lines, 1):
        if not line.strip():
            raise SelectorError(f"{role} contains blank line {number}")
        try:
            row = validate_solver_row(json.loads(line), role=role)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise SelectorError(f"{role} malformed JSONL line {number}") from exc
        task_id = row["task_id"]
        if task_id in rows:
            raise SelectorError(f"{role} duplicate task_id")
        order.append(task_id)
        rows[task_id] = row
        raw[task_id] = line.rstrip(b"\r\n") + b"\n"
    if len(order) != 274:
        raise SelectorError(f"{role} must contain exactly 274 rows")
    return order, rows, raw


def repo_root(experiment_root: Path) -> Path:
    root = experiment_root.resolve().parents[1]
    if not (root / "reports").is_dir() or not (root / "artifacts").is_dir():
        raise SelectorError("cannot resolve repository root")
    return root


def fixed_path(root: Path, descriptor: dict[str, Any], label: str) -> Path:
    relative = Path(descriptor["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise SelectorError(f"unsafe fixed path for {label}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise SelectorError(f"fixed path escaped root for {label}") from exc
    if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
        raise SelectorError(f"fixed artifact mismatch for {label}")
    return path


def verify_preregistration(experiment_root: Path) -> dict[str, Any]:
    profile = load_json(experiment_root / "PREREGISTRATION.json", "preregistration")
    if profile.get("schema_version") != PREREG_SCHEMA or profile.get("selector_id") != EXPERIMENT_ID:
        raise SelectorError("preregistration identity mismatch")
    if profile.get("model_closure") != [MODEL]:
        raise SelectorError("preregistration model closure mismatch")
    if profile.get("candidate_roles") != list(ROLE_ORDER):
        raise SelectorError("preregistered candidate roles differ from code")
    if profile.get("candidate_groups") != ROLE_GROUP:
        raise SelectorError("preregistered candidate groups differ from code")
    calibration = profile.get("calibration", {})
    if calibration.get("source_family_for_crossfit", profile.get("source_authority", {}).get("source_family_for_crossfit")) not in {None, "owner_stage"}:
        raise SelectorError("cross-fit family changed")
    if calibration.get("parameters") != {
        "beta_alpha": ALPHA,
        "beta_beta": BETA,
        "minimum_total_training_rows_per_fold": MIN_FOLD_VALID,
        "minimum_fold_posterior_conditional_fix_rate": MIN_FOLD_POSTERIOR,
        "minimum_global_decisive_fixes_plus_regressions": MIN_GLOBAL_DECISIVE,
        "maximum_nonnegative_logit_weight": MAX_LOGIT_WEIGHT,
    }:
        raise SelectorError("calibration parameters differ from code")
    selection = profile.get("selection", {})
    expected = {
        "anchor_prior_bonus": ANCHOR_BONUS,
        "replacement_min_normalized_margin_over_anchor": MIN_MARGIN,
        "replacement_min_top_share": MIN_TOP_SHARE,
        "replacement_min_independent_candidate_groups": MIN_SUPPORT_GROUPS,
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise SelectorError("selection thresholds differ from code")
    if profile.get("upstream_candidate_sha256") != {
        role: UPSTREAM[role]["sha256"] for role in ROLE_ORDER
    }:
        raise SelectorError("preregistered upstream pins differ from code")
    source = profile.get("source_authority", {})
    if (
        source.get("aggregate_path") != SOURCE["aggregate"]["path"]
        or source.get("aggregate_sha256") != SOURCE["aggregate"]["sha256"]
        or source.get("freeze_manifest_path") != SOURCE["freeze_manifest"]["path"]
        or source.get("freeze_manifest_sha256") != SOURCE["freeze_manifest"]["sha256"]
        or source.get("final_solver_path") != SOURCE["final_solver"]["path"]
        or source.get("final_solver_sha256") != SOURCE["final_solver"]["sha256"]
        or source.get("source_union_size") != SOURCE["union_size"]
        or source.get("source_union_projection_sha256") != SOURCE["union_sha256"]
    ):
        raise SelectorError("preregistered source authority differs from code")
    input_authority = profile.get("outcome_free_input_authority", {})
    if input_authority != {
        "package_path": INPUT_AUTHORITY["package"]["path"],
        "package_sha256": INPUT_AUTHORITY["package"]["sha256"],
        "benchmark_sha256": INPUT_AUTHORITY["benchmark_sha256"],
        "benchmark_order_sha256": INPUT_AUTHORITY["benchmark_order"]["sha256"],
        "route_map_sha256": INPUT_AUTHORITY["route_map"]["sha256"],
        "row_bindings_sha256": INPUT_AUTHORITY["row_bindings"]["sha256"],
        "source_union_membership_sha256": INPUT_AUTHORITY["source_union_membership"]["sha256"],
        "candidate_pool_sha256_integrity_checked_but_content_not_parsed": INPUT_AUTHORITY["candidate_pool"]["sha256"],
        "route_counts": INPUT_AUTHORITY["route_counts"],
        "route_policy": "safety_veto_only_not_quality_feature; all image_judge rows in primary candidate copy exact ActiveCrop row bytes",
    }:
        raise SelectorError("preregistered outcome-free input authority differs from code")
    forbidden = profile.get("forbidden", {})
    if not forbidden or not all(value is True for value in forbidden.values()):
        raise SelectorError("forbidden access policy weakened")
    access = profile.get("access_attestation", {})
    for key in (
        "runtime_benchmark_gold_accessed",
        "runtime_benchmark_score_accessed",
        "runtime_benchmark_correctness_accessed",
        "runtime_judge_outcomes_accessed",
        "task_specific_known_error_rules_used",
        "network_used",
        "gpu_used",
    ):
        if access.get(key) is not False:
            raise SelectorError(f"preregistration access attestation failed: {key}")
    if access.get("historical_aggregate_and_prior_project_outcomes_known_before_this_experiment") is not True:
        raise SelectorError("historical-outcomes chronology disclosure missing")
    if access.get("source_union_labels_allowed") is not True:
        raise SelectorError("source-label calibration authority missing")
    if profile.get("evaluation_status") != "must_remain_not_evaluated_until_candidate_solver_manifest_and_freeze_are_finalized":
        raise SelectorError("preregistration evaluation status changed")
    return profile


def load_source_authority(root: Path) -> tuple[dict[str, dict[str, str]], list[str], dict[str, dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    aggregate_path = fixed_path(root, SOURCE["aggregate"], "source aggregate")
    freeze_path = fixed_path(root, SOURCE["freeze_manifest"], "source freeze")
    solver_path = fixed_path(root, SOURCE["final_solver"], "source final solver")
    aggregate = load_json(aggregate_path, "source aggregate")
    if aggregate.get("schema_version") != SOURCE_AGGREGATE_SCHEMA:
        raise SelectorError("source aggregate schema mismatch")
    if aggregate.get("model_closure") != [MODEL] or aggregate.get("upstream_generation_model_closure") != [MODEL]:
        raise SelectorError("source aggregate contains foreign model closure")
    union = aggregate.get("source_union")
    if not isinstance(union, dict) or union.get("size") != SOURCE["union_size"] or union.get("sha256") != SOURCE["union_sha256"]:
        raise SelectorError("source union authority mismatch")
    projection = union.get("latest_stage_owner_projection")
    if not isinstance(projection, list) or len(projection) != SOURCE["union_size"]:
        raise SelectorError("source union projection length mismatch")
    if hashlib.sha256(canonical_bytes(projection, newline=False)).hexdigest() != SOURCE["union_sha256"]:
        raise SelectorError("source union projection bytes mismatch")
    freeze = load_json(freeze_path, "source freeze")
    if (
        freeze.get("schema_version") != SOURCE_FREEZE_SCHEMA
        or freeze.get("model_closure") != [MODEL]
        or freeze.get("upstream_generation_model_closure") != [MODEL]
        or freeze.get("inherited_27b_outputs") is not False
    ):
        raise SelectorError("source freeze model closure failed")
    if freeze.get("aggregate", {}).get("sha256") != SOURCE["aggregate"]["sha256"]:
        raise SelectorError("source freeze aggregate pin mismatch")
    if freeze.get("final_solver", {}).get("sha256") != SOURCE["final_solver"]["sha256"] or freeze.get("final_solver", {}).get("rows") != 274:
        raise SelectorError("source freeze final-solver pin mismatch")
    source_order, source_rows, source_raw = load_solver(
        solver_path, role="frozen_source_final_solver", expected_sha256=SOURCE["final_solver"]["sha256"]
    )
    labels: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for entry in projection:
        if not isinstance(entry, dict) or set(entry) != {"task_id", "owner_stage", "answer_sha256"}:
            raise SelectorError("source union entry schema mismatch")
        task_id = entry["task_id"]
        family = entry["owner_stage"]
        answer_sha = entry["answer_sha256"]
        if (
            not isinstance(task_id, str)
            or task_id in seen
            or not isinstance(family, str)
            or not family
            or not isinstance(answer_sha, str)
            or not SHA_RE.fullmatch(answer_sha)
            or task_id not in source_rows
        ):
            raise SelectorError("source union entry is unsafe")
        answer = source_rows[task_id]["final_answer"]
        if hashlib.sha256(answer.encode("utf-8")).hexdigest() != answer_sha:
            raise SelectorError("source label differs from frozen answer projection")
        if normalize_answer(answer) is None:
            raise SelectorError("source label is empty after normalization")
        seen.add(task_id)
        labels[task_id] = {"owner_stage": family, "answer": answer, "answer_sha256": answer_sha}
    return labels, source_order, source_rows, source_raw, {
        "aggregate_sha256": SOURCE["aggregate"]["sha256"],
        "freeze_manifest_sha256": SOURCE["freeze_manifest"]["sha256"],
        "final_solver_sha256": SOURCE["final_solver"]["sha256"],
        "source_union_size": SOURCE["union_size"],
        "source_union_projection_sha256": SOURCE["union_sha256"],
    }


def load_candidates(root: Path) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, bytes]]]:
    anchor_order: list[str] | None = None
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    raw: dict[str, dict[str, bytes]] = {}
    expected_ids: set[str] | None = None
    for role in ROLE_ORDER:
        path = fixed_path(root, UPSTREAM[role], role)
        order, rows, raw_rows = load_solver(path, role=role, expected_sha256=UPSTREAM[role]["sha256"])
        if anchor_order is None:
            anchor_order = order
            expected_ids = set(order)
        elif set(order) != expected_ids:
            raise SelectorError(f"{role} task set differs from anchor")
        candidates[role] = rows
        raw[role] = raw_rows
    assert anchor_order is not None
    return anchor_order, candidates, raw


def fixed_package_member(
    package_path: Path, descriptor: dict[str, str], label: str
) -> Path:
    relative = Path(descriptor["path"])
    package_directory = package_path.parent.resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise SelectorError(f"unsafe input-package member path for {label}")
    path = (package_directory / relative).resolve()
    try:
        path.relative_to(package_directory)
    except ValueError as exc:
        raise SelectorError(f"input-package member escaped package for {label}") from exc
    if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
        raise SelectorError(f"input-package member mismatch for {label}")
    return path


def load_input_authority(
    root: Path,
    *,
    expected_order: list[str],
    source_ids: set[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    package_path = fixed_path(root, INPUT_AUTHORITY["package"], "outcome-free input package")
    package = load_json(package_path, "outcome-free input package")
    expected_package_keys = {
        "schema_version",
        "rows",
        "benchmark_sha256",
        "created_before_new_arm_evaluation",
        "runtime_outcome_access",
        "benchmark_order",
        "candidate_pool",
        "route_map",
        "row_bindings",
        "source_union_membership",
        "upstream_artifacts",
    }
    if set(package) != expected_package_keys:
        raise SelectorError("input package exact shape mismatch")
    if (
        package.get("schema_version") != INPUT_PACKAGE_SCHEMA
        or package.get("rows") != 274
        or package.get("benchmark_sha256") != INPUT_AUTHORITY["benchmark_sha256"]
        or package.get("created_before_new_arm_evaluation") is not True
        or package.get("runtime_outcome_access") is not False
    ):
        raise SelectorError("input package identity/chronology mismatch")
    for key in (
        "benchmark_order",
        "candidate_pool",
        "route_map",
        "row_bindings",
        "source_union_membership",
    ):
        if package.get(key) != INPUT_AUTHORITY[key]:
            raise SelectorError(f"input package descriptor changed: {key}")

    expected_upstream = {
        package_role: {
            "path": f"upstream/{package_role}.jsonl",
            "sha256": UPSTREAM[role]["sha256"],
        }
        for role, package_role in PACKAGE_ROLE_MAP.items()
    }
    if package.get("upstream_artifacts") != expected_upstream:
        raise SelectorError("input package upstream closure mismatch")
    for package_role, descriptor in expected_upstream.items():
        fixed_package_member(package_path, descriptor, f"snapshot {package_role}")

    # The normalized candidate pool is pinned as part of the package closure,
    # but this selector deliberately does not read it or any selection output.
    fixed_package_member(package_path, INPUT_AUTHORITY["candidate_pool"], "candidate pool")

    order_path = fixed_package_member(
        package_path, INPUT_AUTHORITY["benchmark_order"], "benchmark order"
    )
    order_value = load_json(order_path, "benchmark order")
    if set(order_value) != {"schema_version", "benchmark_sha256", "rows"}:
        raise SelectorError("benchmark order exact shape mismatch")
    if (
        order_value.get("schema_version") != BENCHMARK_ORDER_SCHEMA
        or order_value.get("benchmark_sha256") != INPUT_AUTHORITY["benchmark_sha256"]
        or order_value.get("rows") != expected_order
        or len(set(expected_order)) != 274
    ):
        raise SelectorError("benchmark order differs from candidate task authority")

    route_path = fixed_package_member(
        package_path, INPUT_AUTHORITY["route_map"], "outcome-free route map"
    )
    route_value = load_json(route_path, "outcome-free route map")
    if set(route_value) != {
        "schema_version",
        "benchmark_sha256",
        "benchmark_order_sha256",
        "derivation",
        "rows",
    }:
        raise SelectorError("route map exact shape mismatch")
    if (
        route_value.get("schema_version") != ROUTE_MAP_SCHEMA
        or route_value.get("benchmark_sha256") != INPUT_AUTHORITY["benchmark_sha256"]
        or route_value.get("benchmark_order_sha256")
        != INPUT_AUTHORITY["benchmark_order"]["sha256"]
        or route_value.get("derivation")
        != "question_image_format_metadata_only_no_gold_score_correctness_outcome_or_judge"
    ):
        raise SelectorError("route map authority/derivation mismatch")
    route_rows = route_value.get("rows")
    if not isinstance(route_rows, list) or len(route_rows) != 274:
        raise SelectorError("route map must have exactly 274 rows")
    routes: dict[str, str] = {}
    route_counts: Counter[str] = Counter()
    for index, row in enumerate(route_rows):
        if not isinstance(row, dict) or set(row) != {
            "row_index",
            "task_id",
            "evaluation_route",
        }:
            raise SelectorError("route row exact shape mismatch")
        task_id = expected_order[index]
        route = row.get("evaluation_route")
        if (
            row.get("row_index") != index
            or row.get("task_id") != task_id
            or route not in {"deterministic", "image_judge"}
        ):
            raise SelectorError("route row identity/order mismatch")
        routes[task_id] = route
        route_counts[route] += 1
    if dict(route_counts) != INPUT_AUTHORITY["route_counts"]:
        raise SelectorError("route counts differ from frozen authority")

    binding_path = fixed_package_member(
        package_path, INPUT_AUTHORITY["row_bindings"], "combined row bindings"
    )
    binding_value = load_json(binding_path, "combined row bindings")
    if set(binding_value) != {
        "schema_version",
        "benchmark_sha256",
        "benchmark_order_sha256",
        "projection_contract",
        "upstream_artifact_sha256",
        "rows",
    }:
        raise SelectorError("combined bindings exact shape mismatch")
    expected_binding_upstream = {
        package_role: UPSTREAM[role]["sha256"]
        for role, package_role in PACKAGE_ROLE_MAP.items()
    }
    if (
        binding_value.get("schema_version") != ROW_BINDINGS_SCHEMA
        or binding_value.get("benchmark_sha256") != INPUT_AUTHORITY["benchmark_sha256"]
        or binding_value.get("benchmark_order_sha256")
        != INPUT_AUTHORITY["benchmark_order"]["sha256"]
        or binding_value.get("projection_contract")
        != "sha256_of_utf8_json_sort_keys_compact_role_projection"
        or binding_value.get("upstream_artifact_sha256") != expected_binding_upstream
    ):
        raise SelectorError("combined bindings authority mismatch")
    binding_rows = binding_value.get("rows")
    if not isinstance(binding_rows, list) or len(binding_rows) != 274:
        raise SelectorError("combined bindings must have exactly 274 rows")
    package_roles = set(expected_binding_upstream)
    for index, row in enumerate(binding_rows):
        if not isinstance(row, dict) or set(row) != {
            "row_index",
            "task_id",
            "role_projection_sha256",
        }:
            raise SelectorError("combined binding row shape mismatch")
        role_hashes = row.get("role_projection_sha256")
        if (
            row.get("row_index") != index
            or row.get("task_id") != expected_order[index]
            or not isinstance(role_hashes, dict)
            or set(role_hashes) != package_roles
            or any(not isinstance(value, str) or not SHA_RE.fullmatch(value) for value in role_hashes.values())
        ):
            raise SelectorError("combined binding row identity/hash mismatch")

    membership_path = fixed_package_member(
        package_path,
        INPUT_AUTHORITY["source_union_membership"],
        "source-union membership",
    )
    membership = load_json(membership_path, "source-union membership")
    if set(membership) != {"schema_version", "authority", "derivation", "task_ids"}:
        raise SelectorError("source-union membership exact shape mismatch")
    if (
        membership.get("schema_version") != SOURCE_MEMBERSHIP_SCHEMA
        or membership.get("derivation")
        != "projection_task_id_only_no_answer_score_correctness_outcome_or_judge"
        or membership.get("authority")
        != {
            "aggregate_sha256": SOURCE["aggregate"]["sha256"],
            "source_union_projection_sha256": SOURCE["union_sha256"],
            "source_union_size": SOURCE["union_size"],
        }
        or not isinstance(membership.get("task_ids"), list)
        or len(membership["task_ids"]) != 156
        or len(set(membership["task_ids"])) != 156
        or set(membership["task_ids"]) != source_ids
    ):
        raise SelectorError("source-union membership differs from source authority")

    report = {
        "package": dict(INPUT_AUTHORITY["package"]),
        "benchmark_sha256": INPUT_AUTHORITY["benchmark_sha256"],
        "benchmark_order": dict(INPUT_AUTHORITY["benchmark_order"]),
        "route_map": dict(INPUT_AUTHORITY["route_map"]),
        "row_bindings": dict(INPUT_AUTHORITY["row_bindings"]),
        "source_union_membership": dict(INPUT_AUTHORITY["source_union_membership"]),
        "candidate_pool_integrity_checked_but_content_not_parsed": dict(INPUT_AUTHORITY["candidate_pool"]),
        "route_counts": dict(INPUT_AUTHORITY["route_counts"]),
        "route_use": "safety_veto_only_not_quality_feature",
    }
    return routes, report


def posterior_precision(correct: int, valid: int) -> float:
    return (correct + ALPHA) / (valid + ALPHA + BETA)


def reliability_weight(precision: float) -> float:
    clipped = min(max(precision, 1e-9), 1 - 1e-9)
    return max(0.0, min(MAX_LOGIT_WEIGHT, math.log(clipped / (1 - clipped))))


def calibrate(labels: dict[str, dict[str, str]], candidates: dict[str, dict[str, dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, float]]:
    families = sorted({value["owner_stage"] for value in labels.values()})
    family_counts = Counter(value["owner_stage"] for value in labels.values())
    roles: dict[str, Any] = {}
    base_weights: dict[str, float] = {}
    for role in ROLE_ORDER:
        global_total = 0
        global_disagreements = 0
        global_fixes = 0
        global_regressions = 0
        global_unresolved = 0
        for task_id, label in labels.items():
            candidate = candidate_answer(candidates[role][task_id])
            anchor = candidate_answer(candidates["active_crop_v2"][task_id])
            target = normalize_answer(label["answer"])
            if candidate is None or anchor is None:
                continue
            global_total += 1
            if candidate == anchor:
                continue
            global_disagreements += 1
            if candidate == target and anchor != target:
                global_fixes += 1
            elif candidate != target and anchor == target:
                global_regressions += 1
            else:
                global_unresolved += 1
        global_decisive = global_fixes + global_regressions
        override_p = posterior_precision(global_fixes, global_decisive)
        folds: list[dict[str, Any]] = []
        gate = global_decisive >= MIN_GLOBAL_DECISIVE and role != "active_crop_v2"
        for held_out in families:
            training_rows = 0
            disagreements = 0
            fixes = 0
            regressions = 0
            unresolved = 0
            for task_id, label in labels.items():
                if label["owner_stage"] == held_out:
                    continue
                candidate = candidate_answer(candidates[role][task_id])
                anchor = candidate_answer(candidates["active_crop_v2"][task_id])
                target = normalize_answer(label["answer"])
                if candidate is None or anchor is None:
                    continue
                training_rows += 1
                if candidate == anchor:
                    continue
                disagreements += 1
                if candidate == target and anchor != target:
                    fixes += 1
                elif candidate != target and anchor == target:
                    regressions += 1
                else:
                    unresolved += 1
            decisive = fixes + regressions
            precision = posterior_precision(fixes, decisive)
            passed = training_rows >= MIN_FOLD_VALID and precision >= MIN_FOLD_POSTERIOR
            gate = gate and passed
            folds.append(
                {
                    "held_out_owner_stage": held_out,
                    "held_out_rows": family_counts[held_out],
                    "training_rows": training_rows,
                    "candidate_anchor_disagreements": disagreements,
                    "decisive_fixes_plus_regressions": decisive,
                    "fixes": fixes,
                    "regressions": regressions,
                    "unresolved_both_wrong": unresolved,
                    "posterior_conditional_fix_rate": precision,
                    "gate_passed": passed,
                }
            )
        global_weight = reliability_weight(override_p)
        base_weight = global_weight if gate else 0.0
        base_weights[role] = base_weight
        roles[role] = {
            "global_training_rows": global_total,
            "global_candidate_anchor_disagreements": global_disagreements,
            "global_fixes": global_fixes,
            "global_regressions": global_regressions,
            "global_unresolved_both_wrong": global_unresolved,
            "global_decisive_fixes_plus_regressions": global_decisive,
            "global_posterior_conditional_fix_rate": override_p,
            "global_override_logit_weight": global_weight,
            "leave_one_owner_stage_out": folds,
            "precision_gate_passed": gate,
            "base_weight_after_gate": base_weight,
        }
    grouped: dict[str, list[str]] = defaultdict(list)
    for role in ROLE_ORDER:
        grouped[ROLE_GROUP[role]].append(role)
    effective: dict[str, float] = {}
    group_report: dict[str, Any] = {}
    for group, members in sorted(grouped.items()):
        raw_sum = sum(base_weights[role] for role in members)
        cap = max((base_weights[role] for role in members), default=0.0)
        for role in members:
            effective[role] = base_weights[role] * cap / raw_sum if raw_sum > 0 else 0.0
        group_report[group] = {
            "members": members,
            "uncapped_sum": raw_sum,
            "group_cap": cap,
            "effective_sum": sum(effective[role] for role in members),
        }
    for role in ROLE_ORDER:
        roles[role]["effective_weight_after_group_cap"] = effective[role]
    label_projection = [
        {
            "task_id": task_id,
            "owner_stage": labels[task_id]["owner_stage"],
            "answer_sha256": labels[task_id]["answer_sha256"],
        }
        for task_id in sorted(labels)
    ]
    report = {
        "schema_version": CALIBRATION_SCHEMA,
        "status": "source_labels_only_no_benchmark_evaluation",
        "model_closure": [MODEL],
        "source_union_size": len(labels),
        "source_union_projection_sha256": SOURCE["union_sha256"],
        "label_projection_sha256": hashlib.sha256(canonical_bytes(label_projection, newline=False)).hexdigest(),
        "owner_stage_counts": dict(sorted(family_counts.items())),
        "beta_prior": {"alpha": ALPHA, "beta": BETA},
        "leave_one_owner_stage_out_gate": {
            "minimum_total_training_rows": MIN_FOLD_VALID,
            "minimum_global_decisive_fixes_plus_regressions": MIN_GLOBAL_DECISIVE,
            "minimum_posterior_conditional_fix_rate": MIN_FOLD_POSTERIOR,
        },
        "roles": roles,
        "correlation_groups": group_report,
        "effective_weights": effective,
        "access_attestation": {
            "official_source_final_answers_used": True,
            "benchmark_gold_score_correctness_or_judge_used": False,
            "task_id_used_as_quality_feature": False,
            "network_used": False,
            "gpu_used": False,
        },
    }
    return report, effective


def choose_uncovered(task_id: str, candidates: dict[str, dict[str, dict[str, Any]]], effective: dict[str, float]) -> tuple[str, dict[str, Any]]:
    anchor = candidate_answer(candidates["active_crop_v2"][task_id])
    tallies: dict[str, float] = defaultdict(float)
    groups: dict[str, set[str]] = defaultdict(set)
    supporters: dict[str, list[str]] = defaultdict(list)
    for role in ROLE_ORDER:
        answer = candidate_answer(candidates[role][task_id])
        weight = effective[role]
        if answer is None or weight <= 0:
            continue
        tallies[answer] += weight
        groups[answer].add(ROLE_GROUP[role])
        supporters[answer].append(role)
    if anchor is not None:
        tallies[anchor] += ANCHOR_BONUS
    total = sum(effective.values()) + (ANCHOR_BONUS if anchor is not None else 0.0)
    if total <= 0:
        return "active_crop_v2", {
            "reason": "no_calibrated_weight_fail_closed",
            "top_share": None,
            "normalized_margin_over_anchor": None,
            "support_groups": 0,
            "anchor_valid": anchor is not None,
        }
    if not tallies:
        return "active_crop_v2", {
            "reason": "no_valid_crossfitted_candidate_fail_closed_anchor_bytes",
            "top_share": None,
            "normalized_margin_over_anchor": None,
            "support_groups": 0,
            "anchor_valid": False,
        }
    ranked = sorted(tallies, key=lambda answer: (-tallies[answer], 0 if answer == anchor else 1, answer))
    top = ranked[0]
    top_share = tallies[top] / total
    margin = (tallies[top] - (tallies[anchor] if anchor is not None else 0.0)) / total
    support_groups = len(groups[top])
    if anchor is None:
        if top_share < MIN_TOP_SHARE:
            reason = "invalid_anchor_top_share_below_threshold_fail_closed_anchor_bytes"
        elif support_groups < MIN_SUPPORT_GROUPS:
            reason = "invalid_anchor_insufficient_crossfitted_groups_fail_closed_anchor_bytes"
        else:
            priority = {role: index for index, role in enumerate(DONOR_PRIORITY)}
            donor = sorted(
                supporters[top], key=lambda role: (-effective[role], priority[role])
            )[0]
            return donor, {
                "reason": "invalid_anchor_replaced_by_crossfitted_multigroup_consensus",
                "top_share": top_share,
                "normalized_margin_over_anchor": margin,
                "support_groups": support_groups,
                "anchor_valid": False,
            }
    elif top == anchor:
        reason = "anchor_has_highest_calibrated_score"
    elif top_share < MIN_TOP_SHARE:
        reason = "top_share_below_preregistered_threshold"
    elif margin < MIN_MARGIN:
        reason = "margin_over_anchor_below_preregistered_threshold"
    elif support_groups < MIN_SUPPORT_GROUPS:
        reason = "insufficient_independent_candidate_groups"
    else:
        priority = {role: index for index, role in enumerate(DONOR_PRIORITY)}
        donor = sorted(
            supporters[top], key=lambda role: (-effective[role], priority[role])
        )[0]
        return donor, {
            "reason": "source_calibrated_replacement",
            "top_share": top_share,
            "normalized_margin_over_anchor": margin,
            "support_groups": support_groups,
            "anchor_valid": True,
        }
    return "active_crop_v2", {
        "reason": reason,
        "top_share": top_share,
        "normalized_margin_over_anchor": margin,
        "support_groups": support_groups,
        "anchor_valid": anchor is not None,
    }


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise SelectorError(f"refusing to overwrite build artifact: {path.name}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise SelectorError(f"cannot atomically finalize {path.name}: {exc}") from exc


def write_canonical(path: Path, value: Any) -> None:
    write_bytes_atomic(path, canonical_bytes(value))


def artifact_descriptor(path: Path, experiment_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(experiment_root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build(experiment_root: Path) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    root = repo_root(experiment_root)
    verify_preregistration(experiment_root)
    outputs = [
        "test_attestation.json",
        "calibration.json",
        "decisions.jsonl",
        "candidate_solver.jsonl",
        "full_candidate_solver.jsonl",
        "candidate_manifest.json",
        "FREEZE.json",
        "FREEZE_PINS.json",
    ]
    existing = [name for name in outputs if (experiment_root / name).exists()]
    if existing:
        raise SelectorError(f"build outputs must be absent: {existing}")

    test_path = experiment_root / "test_selector.py"
    test_run = subprocess.run(
        [sys.executable, str(test_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    test_attestation = {
        "schema_version": "maxim-9b-source-calibrated-test-attestation-v1",
        "command": [sys.executable, str(test_path)],
        "returncode": test_run.returncode,
        "stdout": test_run.stdout,
        "stderr": test_run.stderr,
        "tests_sha256": sha256_file(test_path),
    }
    write_canonical(experiment_root / "test_attestation.json", test_attestation)
    if test_run.returncode != 0:
        raise SelectorError("tests failed; build aborted before candidate artifacts")

    labels, _, source_rows, source_raw, source_authority = load_source_authority(root)
    order, candidates, candidate_raw = load_candidates(root)
    if not set(labels).issubset(order) or len(set(order) - set(labels)) != 118:
        raise SelectorError("expected exact 156 covered and 118 uncovered partition")
    routes, input_authority = load_input_authority(
        root, expected_order=order, source_ids=set(labels)
    )
    calibration, effective = calibrate(labels, candidates)
    calibration_path = experiment_root / "calibration.json"
    write_canonical(calibration_path, calibration)

    decisions_path = experiment_root / "decisions.jsonl"
    solver_path = experiment_root / "candidate_solver.jsonl"
    full_solver_path = experiment_root / "full_candidate_solver.jsonl"
    decision_lines: list[bytes] = []
    solver_lines: list[bytes] = []
    full_solver_lines: list[bytes] = []
    reason_counts: Counter[str] = Counter()
    donor_counts: Counter[str] = Counter()
    full_reason_counts: Counter[str] = Counter()
    full_donor_counts: Counter[str] = Counter()
    covered = uncovered = replacements = 0
    primary_replacements = 0
    primary_changes_vs_anchor = 0
    primary_source_deterministic = 0
    image_rows_preserved = 0
    full_image_rows_changed = 0
    for task_id in order:
        if task_id in labels:
            covered += 1
            full_selected_role = "frozen_official_source"
            full_reason = "covered_by_frozen_source_union"
            full_selected_raw = source_raw[task_id]
            details = {
                "top_share": None,
                "normalized_margin_over_anchor": None,
                "support_groups": None,
                "anchor_valid": None,
            }
        else:
            uncovered += 1
            full_selected_role, details = choose_uncovered(task_id, candidates, effective)
            full_reason = details["reason"]
            full_selected_raw = candidate_raw[full_selected_role][task_id]
            replacements += int(full_selected_role != "active_crop_v2")
        route = routes[task_id]
        if route == "image_judge":
            selected_role = "active_crop_v2"
            reason = "image_route_active_crop_byte_preserve_safety_veto"
            selected_raw = candidate_raw["active_crop_v2"][task_id]
            image_rows_preserved += 1
            full_image_rows_changed += int(
                full_selected_raw != candidate_raw["active_crop_v2"][task_id]
            )
        else:
            selected_role = full_selected_role
            reason = full_reason
            selected_raw = full_selected_raw
            primary_source_deterministic += int(
                full_selected_role == "frozen_official_source"
            )
            primary_replacements += int(
                task_id not in labels and selected_role != "active_crop_v2"
            )
        if route == "image_judge" and selected_raw != candidate_raw["active_crop_v2"][task_id]:
            raise SelectorError("image-route byte-preservation veto failed")
        primary_changes_vs_anchor += int(
            selected_raw != candidate_raw["active_crop_v2"][task_id]
        )
        donor_counts[selected_role] += 1
        reason_counts[reason] += 1
        full_donor_counts[full_selected_role] += 1
        full_reason_counts[full_reason] += 1
        selected_row = json.loads(selected_raw)
        validate_solver_row(selected_row, role=f"output:{selected_role}")
        if selected_row["task_id"] != task_id:
            raise SelectorError("selected donor row join mismatch")
        full_selected_row = json.loads(full_selected_raw)
        validate_solver_row(full_selected_row, role=f"full_output:{full_selected_role}")
        if full_selected_row["task_id"] != task_id:
            raise SelectorError("full selected donor row join mismatch")
        solver_lines.append(selected_raw)
        full_solver_lines.append(full_selected_raw)
        decision_lines.append(
            canonical_bytes(
                {
                    "schema_version": DECISION_SCHEMA,
                    "task_id": task_id,
                    "scope": "covered_source" if task_id in labels else "uncovered",
                    "evaluation_route": route,
                    "selected_role": selected_role,
                    "reason": reason,
                    "full_selected_role": full_selected_role,
                    "full_reason": full_reason,
                    "image_route_safety_veto_applied": route == "image_judge",
                    "top_share": details["top_share"],
                    "normalized_margin_over_anchor": details["normalized_margin_over_anchor"],
                    "support_groups": details["support_groups"],
                    "anchor_valid": details["anchor_valid"],
                    "task_id_used_as_quality_feature": False,
                }
            )
        )
    if covered != 156 or uncovered != 118 or len(solver_lines) != 274:
        raise SelectorError("output partition/count invariant failed")
    if image_rows_preserved != 97 or len(full_solver_lines) != 274:
        raise SelectorError("route/count invariant failed")
    write_bytes_atomic(decisions_path, b"".join(decision_lines))
    write_bytes_atomic(solver_path, b"".join(solver_lines))
    write_bytes_atomic(full_solver_path, b"".join(full_solver_lines))

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "candidate_id": EXPERIMENT_ID,
        "status": "deterministic_only_primary_candidate_frozen_before_benchmark_evaluation",
        "evaluation_status": "not_evaluated",
        "primary_evaluation_scope": "deterministic_only_changes_all_image_rows_byte_preserved",
        "full_candidate_evaluation_status": "not_evaluated_requires_fresh_image_judge",
        "model_closure": [MODEL],
        "inherited_27b_outputs": False,
        "rows": 274,
        "full_source_covered_rows_byte_copied": covered,
        "selector_scored_only_source_uncovered_rows": uncovered,
        "full_uncovered_replacements_vs_active_crop": replacements,
        "primary_uncovered_replacements_vs_active_crop": primary_replacements,
        "primary_changes_vs_active_crop_row_bytes": primary_changes_vs_anchor,
        "primary_source_covered_deterministic_rows_byte_copied": primary_source_deterministic,
        "primary_image_rows_active_crop_byte_preserved": image_rows_preserved,
        "full_candidate_image_rows_changed_vs_active_crop": full_image_rows_changed,
        "selected_role_counts": dict(sorted(donor_counts.items())),
        "decision_reason_counts": dict(sorted(reason_counts.items())),
        "full_selected_role_counts": dict(sorted(full_donor_counts.items())),
        "full_decision_reason_counts": dict(sorted(full_reason_counts.items())),
        "upstream_candidates": {role: dict(UPSTREAM[role]) for role in ROLE_ORDER},
        "source_authority": source_authority,
        "outcome_free_input_authority": input_authority,
        "calibration": artifact_descriptor(calibration_path, experiment_root),
        "decisions": artifact_descriptor(decisions_path, experiment_root),
        "candidate_solver": artifact_descriptor(solver_path, experiment_root),
        "full_candidate_solver": artifact_descriptor(full_solver_path, experiment_root),
        "preregistration_sha256": sha256_file(experiment_root / "PREREGISTRATION.json"),
        "selector_code_sha256": sha256_file(Path(__file__)),
        "tests_sha256": sha256_file(test_path),
        "access_attestation": {
            "historical_aggregate_and_prior_project_outcomes_known": True,
            "source_union_labels_used": True,
            "outcome_free_route_used_only_as_safety_veto": True,
            "candidate_pool_content_not_parsed_or_used": True,
            "runtime_benchmark_gold_score_correctness_or_judge_used": False,
            "task_id_quality_rules_used": False,
            "network_used": False,
            "gpu_used": False,
        },
    }
    manifest_path = experiment_root / "candidate_manifest.json"
    write_canonical(manifest_path, manifest)

    frozen_artifact_names = [
        "PREREGISTRATION.json",
        "PRE_ROUTE_DRAFT_PROVENANCE.json",
        "README.md",
        "selector.py",
        "test_selector.py",
        "test_attestation.json",
        "calibration.json",
        "decisions.jsonl",
        "candidate_solver.jsonl",
        "full_candidate_solver.jsonl",
        "candidate_manifest.json",
    ]
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "selector_id": EXPERIMENT_ID,
        "status": "frozen_not_evaluated",
        "model_closure": [MODEL],
        "inherited_27b_outputs": False,
        "source_authority": source_authority,
        "outcome_free_input_authority": input_authority,
        "upstream_candidates": {role: dict(UPSTREAM[role]) for role in ROLE_ORDER},
        "artifacts": {
            name: artifact_descriptor(experiment_root / name, experiment_root)
            for name in frozen_artifact_names
        },
        "counts": {
            "rows": 274,
            "source_calibration_rows": 156,
            "uncovered_selection_rows": 118,
            "deterministic_route_rows": 177,
            "image_route_rows_byte_preserved": 97,
            "primary_changes_vs_active_crop_row_bytes": primary_changes_vs_anchor,
            "full_candidate_image_rows_changed_vs_active_crop": full_image_rows_changed,
        },
        "evaluation_access": {
            "benchmark_gold": False,
            "benchmark_score": False,
            "benchmark_correctness": False,
            "judge_outcomes": False,
            "candidate_pool_parsed_or_used": False,
        },
        "freeze_projection_sha256": None,
    }
    projection = dict(freeze)
    projection.pop("freeze_projection_sha256")
    freeze["freeze_projection_sha256"] = hashlib.sha256(
        canonical_bytes(projection, newline=False)
    ).hexdigest()
    freeze_path = experiment_root / "FREEZE.json"
    write_canonical(freeze_path, freeze)
    pins = {
        "schema_version": PINS_SCHEMA,
        "freeze_path": "FREEZE.json",
        "freeze_sha256": sha256_file(freeze_path),
        "freeze_projection_sha256": freeze["freeze_projection_sha256"],
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_solver_sha256": sha256_file(solver_path),
        "full_candidate_solver_sha256": sha256_file(full_solver_path),
        "selector_code_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(experiment_root / "PREREGISTRATION.json"),
        "evaluation_status": "not_evaluated",
    }
    pins_path = experiment_root / "FREEZE_PINS.json"
    write_canonical(pins_path, pins)
    return {
        "status": "candidate_solver_manifest_and_freeze_created_without_evaluation",
        "candidate_solver_sha256": pins["candidate_solver_sha256"],
        "full_candidate_solver_sha256": pins["full_candidate_solver_sha256"],
        "candidate_manifest_sha256": pins["candidate_manifest_sha256"],
        "freeze_sha256": pins["freeze_sha256"],
        "freeze_projection_sha256": pins["freeze_projection_sha256"],
        "freeze_pins_sha256": sha256_file(pins_path),
        "covered": covered,
        "uncovered": uncovered,
        "full_uncovered_replacements": replacements,
        "primary_uncovered_replacements": primary_replacements,
        "primary_image_rows_byte_preserved": image_rows_preserved,
    }


def verify_freeze(experiment_root: Path) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    verify_preregistration(experiment_root)
    pins = load_json(experiment_root / "FREEZE_PINS.json", "freeze pins")
    freeze_path = experiment_root / "FREEZE.json"
    if pins.get("schema_version") != PINS_SCHEMA or pins.get("evaluation_status") != "not_evaluated":
        raise SelectorError("freeze pins schema/status mismatch")
    if sha256_file(freeze_path) != pins.get("freeze_sha256"):
        raise SelectorError("freeze file differs from pin")
    freeze = load_json(freeze_path, "freeze")
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA
        or freeze.get("status") != "frozen_not_evaluated"
        or freeze.get("model_closure") != [MODEL]
        or freeze.get("inherited_27b_outputs") is not False
    ):
        raise SelectorError("freeze identity/model closure mismatch")
    projection = dict(freeze)
    declared_projection = projection.pop("freeze_projection_sha256", None)
    recomputed = hashlib.sha256(canonical_bytes(projection, newline=False)).hexdigest()
    if declared_projection != recomputed or recomputed != pins.get("freeze_projection_sha256"):
        raise SelectorError("freeze projection mismatch")
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise SelectorError("freeze artifacts missing")
    for name, descriptor in artifacts.items():
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256", "size_bytes"}:
            raise SelectorError("freeze artifact descriptor malformed")
        relative = Path(descriptor["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SelectorError("freeze artifact path unsafe")
        path = (experiment_root / relative).resolve()
        try:
            path.relative_to(experiment_root)
        except ValueError as exc:
            raise SelectorError("freeze artifact escapes experiment") from exc
        if (
            not path.is_file()
            or path.stat().st_size != descriptor["size_bytes"]
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise SelectorError(f"frozen artifact changed: {name}")
    root = repo_root(experiment_root)
    labels, _, _, _, source_authority = load_source_authority(root)
    order, candidates, candidate_raw = load_candidates(root)
    routes, input_authority = load_input_authority(
        root, expected_order=order, source_ids=set(labels)
    )
    if freeze.get("source_authority") != source_authority:
        raise SelectorError("freeze source authority changed")
    if freeze.get("outcome_free_input_authority") != input_authority:
        raise SelectorError("freeze input authority changed")
    if pins.get("candidate_manifest_sha256") != sha256_file(experiment_root / "candidate_manifest.json"):
        raise SelectorError("candidate manifest differs from pin")
    if pins.get("candidate_solver_sha256") != sha256_file(experiment_root / "candidate_solver.jsonl"):
        raise SelectorError("candidate solver differs from pin")
    if pins.get("full_candidate_solver_sha256") != sha256_file(experiment_root / "full_candidate_solver.jsonl"):
        raise SelectorError("full candidate solver differs from pin")
    primary_order, _, primary_raw = load_solver(
        experiment_root / "candidate_solver.jsonl",
        role="frozen_primary_candidate",
        expected_sha256=pins["candidate_solver_sha256"],
    )
    full_order, _, _ = load_solver(
        experiment_root / "full_candidate_solver.jsonl",
        role="frozen_full_candidate",
        expected_sha256=pins["full_candidate_solver_sha256"],
    )
    if primary_order != order or full_order != order:
        raise SelectorError("frozen candidate order differs from authority")
    for task_id, route in routes.items():
        if route == "image_judge" and primary_raw[task_id] != candidate_raw["active_crop_v2"][task_id]:
            raise SelectorError("frozen primary changed an image-route row")
    return {
        "status": "freeze_verified_not_evaluated",
        "freeze_sha256": pins["freeze_sha256"],
        "freeze_projection_sha256": recomputed,
        "freeze_pins_sha256": sha256_file(experiment_root / "FREEZE_PINS.json"),
        "candidate_solver_sha256": pins["candidate_solver_sha256"],
        "full_candidate_solver_sha256": pins["full_candidate_solver_sha256"],
        "candidate_manifest_sha256": pins["candidate_manifest_sha256"],
        "image_rows_byte_preserved": 97,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify-freeze", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.build == args.verify_freeze:
            raise SelectorError("choose exactly one of --build or --verify-freeze")
        report = build(args.root) if args.build else verify_freeze(args.root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (SelectorError, subprocess.TimeoutExpired) as exc:
        print(f"selector error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
