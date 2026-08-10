from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import compositor_v1_1 as legacy
import selector_v1_2 as selector


PROFILE_SCHEMA = "maxim-9b-baseline-selector-multi-arm-evaluation-profile-v1"
FREEZE_SCHEMA = "maxim-9b-baseline-selector-multi-arm-evaluation-freeze-v1"
WAVE_ID = "maxim_9b_baseline_selector_v1_1_v1_2_one_shot_wave"
MODEL = "Qwen/Qwen3.5-9B"
ROW_COUNT = 274
BENCHMARK_SHA256 = "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
SCORER_SHA256 = "bca10e6546b68f8a66eb4d68aa13316429e4689789be1bef14cd592955e4eacf"
FINAL_IMAGE_JUDGE_SHA256 = "c22075cf5f64fb08b073beb2bf33920b37047be7a964776dad8fe90b7660bc98"
PAGE_RAG_JUDGE_SHA256 = "59dcc93454b29dfc65b0a9b1243a177d472b6c0a13cbe46fb5c98079810a73f4"
INPUT_PACKAGE_SHA256 = "12d73ab5cd5955dffee921c97b84fb8bc6c99e3b0152c295e74aa287ad3666e8"
ORDER_SHA256 = "7140c7c01b48053f6a15a3b0113f68cad37bbb887744828b570a6eaa0447d62b"
ROUTE_SHA256 = "f89ef00f95b9d83610b66948fcb11667dc927f2452b000ef62e031a1a0de26f6"

EXPECTED_ARMS = [
    {
        "arm_id": "v1_1_primary",
        "experiment_relative_solver_path": "compositor_output_v1_1/primary_solver.jsonl",
        "solver_sha256": "93905be2f9e5c63bf2eec1c6c22e0037de617b48288e3fa5adca2d680ef1a704",
        "experiment_relative_composition_output_freeze_path": "compositor_output_v1_1/COMPOSITION_OUTPUT_FREEZE.json",
        "composition_output_freeze_sha256": "729012b01b4ebb083b0fa6500bcebccf503a909a9cd8b79fae3da113a0a6fb51",
        "expected_rows": ROW_COUNT,
        "experiment_relative_score_output_path": "evaluation_wave_v1/results/v1_1_primary_score.json",
    },
    {
        "arm_id": "v1_1_secondary",
        "experiment_relative_solver_path": "compositor_output_v1_1/secondary_solver.jsonl",
        "solver_sha256": "9fc8298f0a81d269f26bba65f11bf5afa5581ea8c76295a501b8605552a5cc73",
        "experiment_relative_composition_output_freeze_path": "compositor_output_v1_1/COMPOSITION_OUTPUT_FREEZE.json",
        "composition_output_freeze_sha256": "729012b01b4ebb083b0fa6500bcebccf503a909a9cd8b79fae3da113a0a6fb51",
        "expected_rows": ROW_COUNT,
        "experiment_relative_score_output_path": "evaluation_wave_v1/results/v1_1_secondary_score.json",
    },
    {
        "arm_id": "v1_2_primary",
        "experiment_relative_solver_path": "compositor_output_v1_2/primary_solver.jsonl",
        "solver_sha256": "09aa8d69e7de3a02bbc9b28b2b269b845a0dee1a40ef2d6aa55f7e966a779bef",
        "experiment_relative_composition_output_freeze_path": "compositor_output_v1_2/COMPOSITION_OUTPUT_FREEZE_v1_2.json",
        "composition_output_freeze_sha256": "da26ddb88f6e8d0773a0f42ec2775c42e15edd0d19af5eadb214b0bb354ef30b",
        "expected_rows": ROW_COUNT,
        "experiment_relative_score_output_path": "evaluation_wave_v1/results/v1_2_primary_score.json",
    },
    {
        "arm_id": "v1_2_exploratory",
        "experiment_relative_solver_path": "compositor_output_v1_2/exploratory_solver.jsonl",
        "solver_sha256": "cf18e0836b9af729fcfee337599e724de368a4fb78f4d62442d0dea71057da80",
        "experiment_relative_composition_output_freeze_path": "compositor_output_v1_2/COMPOSITION_OUTPUT_FREEZE_v1_2.json",
        "composition_output_freeze_sha256": "da26ddb88f6e8d0773a0f42ec2775c42e15edd0d19af5eadb214b0bb354ef30b",
        "expected_rows": ROW_COUNT,
        "experiment_relative_score_output_path": "evaluation_wave_v1/results/v1_2_exploratory_score.json",
    },
]


class EvaluationFreezeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    try:
        return legacy._sha256(path)
    except legacy.CompositorError as exc:
        raise EvaluationFreezeError(str(exc)) from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return legacy._read_json(path, label)
    except legacy.CompositorError as exc:
        raise EvaluationFreezeError(str(exc)) from exc


def _strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    try:
        legacy._strict_keys(value, expected, label)
    except legacy.CompositorError as exc:
        raise EvaluationFreezeError(str(exc)) from exc


def _safe_experiment_path(root: Path, relative_path: str, label: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise EvaluationFreezeError(f"{label} path is unsafe")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise EvaluationFreezeError(f"{label} path escaped experiment root") from exc
    return candidate


def _repository_file(root: Path, relative_path: str, expected_sha256: str, label: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise EvaluationFreezeError(f"{label} repository path is unsafe")
    if not relative.parts or relative.parts[0] not in {"artifacts", "reports", "scripts"}:
        raise EvaluationFreezeError(f"{label} repository path prefix is not allowlisted")
    try:
        return legacy._find_repository_file(root, relative_path, expected_sha256)
    except legacy.CompositorError as exc:
        raise EvaluationFreezeError(f"{label}: {exc}") from exc


def _validate_repository_descriptor(
    root: Path,
    value: Any,
    *,
    expected_path: str,
    expected_sha256: str,
    label: str,
) -> Path:
    if not isinstance(value, dict):
        raise EvaluationFreezeError(f"{label} descriptor must be object")
    expected_keys = {"repository_relative_path", "sha256"}
    if label == "benchmark":
        expected_keys.add("rows")
    _strict_keys(value, expected_keys, label)
    if value["repository_relative_path"] != expected_path or value["sha256"] != expected_sha256:
        raise EvaluationFreezeError(f"{label} descriptor pin mismatch")
    if label == "benchmark" and value["rows"] != ROW_COUNT:
        raise EvaluationFreezeError("benchmark row declaration mismatch")
    path = _repository_file(root, expected_path, expected_sha256, label)
    if _sha256(path) != expected_sha256:
        raise EvaluationFreezeError(f"{label} SHA mismatch")
    return path


def _load_order_and_routes(root: Path, routing: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    _strict_keys(
        routing,
        {
            "experiment_relative_input_package_path",
            "input_package_sha256",
            "benchmark_order_sha256",
            "outcome_free_route_map_sha256",
        },
        "routing authority",
    )
    expected = {
        "experiment_relative_input_package_path": selector.INPUT_PACKAGE_RELATIVE_PATH,
        "input_package_sha256": INPUT_PACKAGE_SHA256,
        "benchmark_order_sha256": ORDER_SHA256,
        "outcome_free_route_map_sha256": ROUTE_SHA256,
    }
    if routing != expected:
        raise EvaluationFreezeError("routing authority pins mismatch")
    profile = selector.load_profile(root / "profile_v1_2.json", require_ready=True)
    package_path = _safe_experiment_path(
        root, routing["experiment_relative_input_package_path"], "input package"
    )
    if _sha256(package_path) != INPUT_PACKAGE_SHA256:
        raise EvaluationFreezeError("routing input package SHA mismatch")
    try:
        _, ordered_ids, routes, _, _ = selector.load_input_package(package_path, profile)
    except (selector.SelectorV12Error, selector.base.SelectorError) as exc:
        raise EvaluationFreezeError(str(exc)) from exc
    if len(ordered_ids) != ROW_COUNT or len(set(ordered_ids)) != ROW_COUNT:
        raise EvaluationFreezeError("routing authority IDs are missing, duplicated, or extra")
    if Counter(routes) != {"deterministic": 177, "image_judge": 97}:
        raise EvaluationFreezeError("routing authority split is not 177/97")
    return ordered_ids, routes


def _validate_solver(path: Path, expected_sha256: str, ordered_ids: tuple[str, ...], label: str) -> None:
    try:
        rows = legacy._read_jsonl_raw(path, label, expected_sha256)
    except legacy.CompositorError as exc:
        raise EvaluationFreezeError(str(exc)) from exc
    if len(rows) != ROW_COUNT:
        raise EvaluationFreezeError(f"{label} row count mismatch")
    for index, raw_row in enumerate(rows):
        row = raw_row.value
        try:
            selector.base._assert_no_excluded_runtime_fields(row, f"{label}[{index}]")
        except selector.base.SelectorError as exc:
            raise EvaluationFreezeError(str(exc)) from exc
        if row.get("task_id") != ordered_ids[index]:
            raise EvaluationFreezeError(f"{label} task identity/order mismatch at row {index}")
        if row.get("model") != MODEL:
            raise EvaluationFreezeError(f"{label} model closure mismatch at row {index}")
        if not isinstance(row.get("final_answer"), str) or not row["final_answer"].strip():
            raise EvaluationFreezeError(f"{label} final_answer missing at row {index}")


def validate_profile(root: Path, *, require_score_outputs_absent: bool = True) -> dict[str, Any]:
    root = root.resolve()
    profile = _read_json(root / "multi_arm_evaluation_profile_v1.json", "multi-arm profile")
    _strict_keys(
        profile,
        {
            "schema_version",
            "wave_id",
            "status",
            "chronology",
            "benchmark",
            "scorer",
            "evaluator_split",
            "model_closure",
            "routing_authority",
            "candidate_arms",
            "execution_policy",
        },
        "multi-arm profile",
    )
    if profile["schema_version"] != PROFILE_SCHEMA or profile["wave_id"] != WAVE_ID:
        raise EvaluationFreezeError("multi-arm profile schema/identity mismatch")
    if profile["status"] != "preregistered_not_scored_not_executed":
        raise EvaluationFreezeError("multi-arm profile status mismatch")
    expected_chronology = {
        "historical_benchmark_aggregate_score_and_prior_task_outcomes_were_known_before_this_wave": True,
        "all_four_candidate_solvers_were_frozen_before_any_score_of_any_arm_in_this_wave": True,
        "evaluation_profile_and_output_paths_were_frozen_before_any_score_of_any_arm_in_this_wave": True,
        "same_wave_retuning_after_any_arm_score_is_forbidden": True,
        "freeze_construction_read_gold_reference_correctness_score_or_judge_outcomes": False,
    }
    if profile["chronology"] != expected_chronology:
        raise EvaluationFreezeError("multi-arm chronology contract mismatch")
    if profile["model_closure"] != [MODEL]:
        raise EvaluationFreezeError("multi-arm model closure mismatch")
    benchmark_path = _validate_repository_descriptor(
        root,
        profile["benchmark"],
        expected_path="artifacts/baselines/basic_page_rag_v1/validation_274.jsonl",
        expected_sha256=BENCHMARK_SHA256,
        label="benchmark",
    )
    scorer_path = _validate_repository_descriptor(
        root,
        profile["scorer"],
        expected_path="scripts/score_maxim_full274.py",
        expected_sha256=SCORER_SHA256,
        label="scorer",
    )
    split = profile["evaluator_split"]
    if not isinstance(split, dict):
        raise EvaluationFreezeError("evaluator split must be object")
    _strict_keys(
        split,
        {"deterministic_rows", "image_judge_rows", "final_image_judge", "page_rag_baseline_judge"},
        "evaluator split",
    )
    if split["deterministic_rows"] != 177 or split["image_judge_rows"] != 97:
        raise EvaluationFreezeError("evaluator split declaration mismatch")
    final_judge_path = _validate_repository_descriptor(
        root,
        split["final_image_judge"],
        expected_path="reports/maxim_9b_source_replay_v1_20260809/active_crop/final_evaluation/matched_image97_judge.jsonl",
        expected_sha256=FINAL_IMAGE_JUDGE_SHA256,
        label="final image judge",
    )
    baseline_judge_path = _validate_repository_descriptor(
        root,
        split["page_rag_baseline_judge"],
        expected_path="artifacts/baselines/basic_page_rag_v1/agent_rag_judge.jsonl",
        expected_sha256=PAGE_RAG_JUDGE_SHA256,
        label="page RAG baseline judge",
    )
    ordered_ids, routes = _load_order_and_routes(root, profile["routing_authority"])
    if profile["candidate_arms"] != EXPECTED_ARMS:
        raise EvaluationFreezeError("candidate arm contract differs from exact preregistration")
    output_paths: list[Path] = []
    solver_paths: dict[str, str] = {}
    for arm in EXPECTED_ARMS:
        arm_id = arm["arm_id"]
        solver_path = _safe_experiment_path(root, arm["experiment_relative_solver_path"], arm_id)
        if _sha256(solver_path) != arm["solver_sha256"]:
            raise EvaluationFreezeError(f"{arm_id} solver SHA mismatch")
        freeze_path = _safe_experiment_path(
            root, arm["experiment_relative_composition_output_freeze_path"], f"{arm_id} output freeze"
        )
        if _sha256(freeze_path) != arm["composition_output_freeze_sha256"]:
            raise EvaluationFreezeError(f"{arm_id} composition output freeze SHA mismatch")
        _validate_solver(solver_path, arm["solver_sha256"], ordered_ids, arm_id)
        solver_paths[arm_id] = str(solver_path)
        output_path = _safe_experiment_path(
            root, arm["experiment_relative_score_output_path"], f"{arm_id} score output"
        )
        output_paths.append(output_path)
    if len(set(output_paths)) != len(EXPECTED_ARMS):
        raise EvaluationFreezeError("score output paths are not unique")
    existing = [str(path) for path in output_paths if path.exists()]
    if require_score_outputs_absent and existing:
        raise EvaluationFreezeError(f"score output paths already exist: {existing}")
    expected_policy = {
        "one_shot_multi_arm_wave": True,
        "run_all_four_arms_under_the_same_pinned_evaluator_inputs": True,
        "do_not_inspect_any_arm_score_before_all_four_outputs_exist": True,
        "do_not_change_rules_candidates_or_evaluator_after_any_arm_score": True,
        "score_output_paths_must_be_unique_and_absent_at_freeze": True,
        "freeze_does_not_authorize_or_execute_scoring": True,
    }
    if profile["execution_policy"] != expected_policy:
        raise EvaluationFreezeError("multi-arm execution policy mismatch")
    return {
        "status": "multi_arm_profile_verified_unscored",
        "benchmark_path": str(benchmark_path),
        "scorer_path": str(scorer_path),
        "final_image_judge_path": str(final_judge_path),
        "page_rag_baseline_judge_path": str(baseline_judge_path),
        "rows": len(ordered_ids),
        "route_counts": dict(sorted(Counter(routes).items())),
        "solver_paths": solver_paths,
        "score_output_paths_absent": not existing,
        "score_output_paths": [str(path) for path in output_paths],
    }


def verify_freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    freeze_path = root / "MULTI_ARM_EVALUATION_FREEZE_v1.json"
    freeze = _read_json(freeze_path, "multi-arm evaluation freeze")
    _strict_keys(
        freeze,
        {
            "schema_version",
            "wave_id",
            "status",
            "all_arms_frozen_before_any_arm_score",
            "same_wave_retuning_forbidden",
            "score_execution_authorized",
            "score_outputs_absent_at_freeze",
            "benchmark_sha256",
            "scorer_sha256",
            "final_image_judge_sha256",
            "page_rag_baseline_judge_sha256",
            "expected_rows",
            "expected_deterministic_rows",
            "expected_image_judge_rows",
            "candidate_solver_sha256",
            "score_output_paths",
            "artifacts",
        },
        "multi-arm evaluation freeze",
    )
    if freeze["schema_version"] != FREEZE_SCHEMA or freeze["wave_id"] != WAVE_ID:
        raise EvaluationFreezeError("multi-arm freeze schema/identity mismatch")
    if freeze["status"] != "frozen_not_scored_not_executed":
        raise EvaluationFreezeError("multi-arm freeze status mismatch")
    if freeze["all_arms_frozen_before_any_arm_score"] is not True:
        raise EvaluationFreezeError("multi-arm freeze chronology is false")
    if freeze["same_wave_retuning_forbidden"] is not True:
        raise EvaluationFreezeError("same-wave retuning prohibition missing")
    if freeze["score_execution_authorized"] is not False:
        raise EvaluationFreezeError("freeze unexpectedly authorizes scoring")
    if freeze["score_outputs_absent_at_freeze"] is not True:
        raise EvaluationFreezeError("score output absence attestation missing")
    expected_scalars = {
        "benchmark_sha256": BENCHMARK_SHA256,
        "scorer_sha256": SCORER_SHA256,
        "final_image_judge_sha256": FINAL_IMAGE_JUDGE_SHA256,
        "page_rag_baseline_judge_sha256": PAGE_RAG_JUDGE_SHA256,
        "expected_rows": 274,
        "expected_deterministic_rows": 177,
        "expected_image_judge_rows": 97,
    }
    for key, expected in expected_scalars.items():
        if freeze[key] != expected:
            raise EvaluationFreezeError(f"multi-arm freeze {key} mismatch")
    expected_solver_map = {arm["arm_id"]: arm["solver_sha256"] for arm in EXPECTED_ARMS}
    expected_output_map = {
        arm["arm_id"]: arm["experiment_relative_score_output_path"] for arm in EXPECTED_ARMS
    }
    if freeze["candidate_solver_sha256"] != expected_solver_map:
        raise EvaluationFreezeError("multi-arm freeze candidate solver map mismatch")
    if freeze["score_output_paths"] != expected_output_map:
        raise EvaluationFreezeError("multi-arm freeze score output map mismatch")
    artifacts = freeze["artifacts"]
    if not isinstance(artifacts, dict):
        raise EvaluationFreezeError("multi-arm freeze artifacts must be object")
    expected_artifacts = {
        "profile": "multi_arm_evaluation_profile_v1.json",
        "verifier": "verify_multi_arm_evaluation_freeze_v1.py",
        "tests": "test_multi_arm_evaluation_freeze_v1.py",
    }
    _strict_keys(artifacts, set(expected_artifacts), "multi-arm freeze artifacts")
    verified: dict[str, dict[str, str]] = {}
    for role, expected_name in expected_artifacts.items():
        descriptor = artifacts[role]
        if not isinstance(descriptor, dict):
            raise EvaluationFreezeError(f"multi-arm freeze {role} descriptor must be object")
        _strict_keys(descriptor, {"path", "sha256"}, f"multi-arm freeze {role}")
        if descriptor["path"] != expected_name:
            raise EvaluationFreezeError(f"multi-arm freeze {role} path mismatch")
        path = (root / expected_name).resolve()
        if _sha256(path) != descriptor["sha256"]:
            raise EvaluationFreezeError(f"multi-arm freeze {role} SHA mismatch")
        verified[role] = {"path": str(path), "sha256": descriptor["sha256"]}
    profile_report = validate_profile(root, require_score_outputs_absent=True)
    return {
        "status": "multi_arm_evaluation_freeze_verified_not_scored",
        "freeze_path": str(freeze_path),
        "freeze_sha256": _sha256(freeze_path),
        "artifacts": verified,
        "profile_report": profile_report,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify four-arm one-shot evaluation freeze")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = validate_profile(args.root) if args.profile_only else verify_freeze(args.root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (EvaluationFreezeError, selector.SelectorV12Error, selector.base.SelectorError) as exc:
        print(f"multi-arm evaluation freeze error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
