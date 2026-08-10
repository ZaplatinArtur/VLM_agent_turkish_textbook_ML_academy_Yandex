from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import compositor_v1_1 as legacy
import verify_multi_arm_evaluation_freeze_v1 as base_freeze


EXECUTION_FREEZE_SCHEMA = "maxim-9b-baseline-selector-multi-arm-execution-freeze-v1"
ATTEMPT_SCHEMA = "maxim-9b-baseline-selector-multi-arm-attempt-v1"
COMPLETION_SCHEMA = "maxim-9b-baseline-selector-multi-arm-completion-v1"
BASE_FREEZE_SHA256 = "e9fe41db583a02044d9f693c013cc19a6c9b40c96eca03bb559b356f731db7f1"
EXECUTION_FREEZE_NAME = "MULTI_ARM_EVALUATION_EXECUTION_FREEZE_v1.json"
ATTEMPT_MARKER_RELATIVE_PATH = "evaluation_wave_v1/ATTEMPT_STARTED.json"
COMPLETION_MANIFEST_RELATIVE_PATH = "evaluation_wave_v1/WAVE_COMPLETION_MANIFEST.json"
COMPLETION_SHA_RELATIVE_PATH = "evaluation_wave_v1/WAVE_COMPLETION_MANIFEST.sha256"

OUTPUT_BUNDLES = {
    "v1_1_primary": {
        "json": "evaluation_wave_v1/results/v1_1_primary_score.json",
        "md": "evaluation_wave_v1/results/v1_1_primary_score.md",
        "sha256": "evaluation_wave_v1/results/v1_1_primary_score.sha256",
    },
    "v1_1_secondary": {
        "json": "evaluation_wave_v1/results/v1_1_secondary_score.json",
        "md": "evaluation_wave_v1/results/v1_1_secondary_score.md",
        "sha256": "evaluation_wave_v1/results/v1_1_secondary_score.sha256",
    },
    "v1_2_primary": {
        "json": "evaluation_wave_v1/results/v1_2_primary_score.json",
        "md": "evaluation_wave_v1/results/v1_2_primary_score.md",
        "sha256": "evaluation_wave_v1/results/v1_2_primary_score.sha256",
    },
    "v1_2_exploratory": {
        "json": "evaluation_wave_v1/results/v1_2_exploratory_score.json",
        "md": "evaluation_wave_v1/results/v1_2_exploratory_score.md",
        "sha256": "evaluation_wave_v1/results/v1_2_exploratory_score.sha256",
    },
}


class WaveExecutionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    try:
        return legacy._sha256(path)
    except legacy.CompositorError as exc:
        raise WaveExecutionError(str(exc)) from exc


def _canonical_json(value: Any) -> bytes:
    return legacy._canonical_json(value)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return legacy._read_json(path, label)
    except legacy.CompositorError as exc:
        raise WaveExecutionError(str(exc)) from exc


def _strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    try:
        legacy._strict_keys(value, expected, label)
    except legacy.CompositorError as exc:
        raise WaveExecutionError(str(exc)) from exc


def _safe_path(root: Path, relative_path: str, label: str) -> Path:
    try:
        return base_freeze._safe_experiment_path(root, relative_path, label)
    except base_freeze.EvaluationFreezeError as exc:
        raise WaveExecutionError(str(exc)) from exc


def _all_declared_paths(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "attempt_marker": _safe_path(root, ATTEMPT_MARKER_RELATIVE_PATH, "attempt marker"),
        "completion_manifest": _safe_path(
            root, COMPLETION_MANIFEST_RELATIVE_PATH, "completion manifest"
        ),
        "completion_sha256": _safe_path(root, COMPLETION_SHA_RELATIVE_PATH, "completion SHA"),
    }
    for arm_id, bundle in OUTPUT_BUNDLES.items():
        for kind, relative in bundle.items():
            paths[f"{arm_id}.{kind}"] = _safe_path(root, relative, f"{arm_id} {kind} output")
    if len(paths) != len(set(paths.values())):
        raise WaveExecutionError("execution wave paths are not globally unique")
    return paths


def verify_execution_freeze(root: Path, *, require_all_outputs_absent: bool = True) -> dict[str, Any]:
    root = root.resolve()
    base_report = base_freeze.verify_freeze(root)
    base_path = root / "MULTI_ARM_EVALUATION_FREEZE_v1.json"
    if _sha256(base_path) != BASE_FREEZE_SHA256:
        raise WaveExecutionError("base multi-arm evaluation freeze SHA mismatch")
    path = root / EXECUTION_FREEZE_NAME
    value = _read_json(path, "multi-arm execution freeze")
    _strict_keys(
        value,
        {
            "schema_version",
            "wave_id",
            "status",
            "base_evaluation_freeze_sha256",
            "base_freeze_no_execution_superseded_only_by_explicit_execute_invocation_of_this_pinned_launcher",
            "requires_explicit_execute_flag",
            "attempt_marker_created_atomically_before_any_scorer",
            "all_four_processes_launched_and_awaited_before_status_decision",
            "subprocess_stdout_stderr_captured_and_never_forwarded",
            "individual_score_content_not_parsed_or_printed_before_completion_barrier",
            "attempt_marker_preserved_on_failure",
            "completion_manifest_written_only_after_all_four_successes_and_outputs_exist",
            "same_wave_retuning_forbidden",
            "output_bundles",
            "attempt_marker_path",
            "completion_manifest_path",
            "completion_sha256_path",
            "artifacts",
        },
        "multi-arm execution freeze",
    )
    expected_flags = {
        "schema_version": EXECUTION_FREEZE_SCHEMA,
        "wave_id": base_freeze.WAVE_ID,
        "status": "launcher_frozen_not_executed",
        "base_evaluation_freeze_sha256": BASE_FREEZE_SHA256,
        "base_freeze_no_execution_superseded_only_by_explicit_execute_invocation_of_this_pinned_launcher": True,
        "requires_explicit_execute_flag": True,
        "attempt_marker_created_atomically_before_any_scorer": True,
        "all_four_processes_launched_and_awaited_before_status_decision": True,
        "subprocess_stdout_stderr_captured_and_never_forwarded": True,
        "individual_score_content_not_parsed_or_printed_before_completion_barrier": True,
        "attempt_marker_preserved_on_failure": True,
        "completion_manifest_written_only_after_all_four_successes_and_outputs_exist": True,
        "same_wave_retuning_forbidden": True,
        "output_bundles": OUTPUT_BUNDLES,
        "attempt_marker_path": ATTEMPT_MARKER_RELATIVE_PATH,
        "completion_manifest_path": COMPLETION_MANIFEST_RELATIVE_PATH,
        "completion_sha256_path": COMPLETION_SHA_RELATIVE_PATH,
    }
    for key, expected in expected_flags.items():
        if value[key] != expected:
            raise WaveExecutionError(f"execution freeze {key} mismatch")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, dict):
        raise WaveExecutionError("execution freeze artifacts must be object")
    expected_artifacts = {
        "launcher": "execute_multi_arm_evaluation_wave_v1.py",
        "tests": "test_execute_multi_arm_evaluation_wave_v1.py",
    }
    _strict_keys(artifacts, set(expected_artifacts), "execution freeze artifacts")
    verified_artifacts: dict[str, dict[str, str]] = {}
    for role, filename in expected_artifacts.items():
        descriptor = artifacts[role]
        if not isinstance(descriptor, dict):
            raise WaveExecutionError(f"execution freeze {role} descriptor must be object")
        _strict_keys(descriptor, {"path", "sha256"}, f"execution freeze {role}")
        if descriptor["path"] != filename:
            raise WaveExecutionError(f"execution freeze {role} path mismatch")
        artifact_path = root / filename
        if _sha256(artifact_path) != descriptor["sha256"]:
            raise WaveExecutionError(f"execution freeze {role} SHA mismatch")
        verified_artifacts[role] = {"path": str(artifact_path), "sha256": descriptor["sha256"]}
    declared = _all_declared_paths(root)
    existing = {label: str(item) for label, item in declared.items() if item.exists()}
    if require_all_outputs_absent and existing:
        raise WaveExecutionError(f"execution wave is already attempted or has outputs: {sorted(existing)}")
    return {
        "status": "execution_launcher_freeze_verified_not_executed",
        "execution_freeze_path": str(path),
        "execution_freeze_sha256": _sha256(path),
        "base_freeze_sha256": BASE_FREEZE_SHA256,
        "all_declared_paths_absent": not existing,
        "declared_paths": {label: str(item) for label, item in declared.items()},
        "artifacts": verified_artifacts,
        "base_report": base_report,
    }


def _atomic_create_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise WaveExecutionError(f"refusing to overwrite existing artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The exclusive path remains as evidence of an attempted write.
        raise


def _build_command_plans(root: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    root = root.resolve()
    profile_report = report["base_report"]["profile_report"]
    plans: list[dict[str, Any]] = []
    for arm in base_freeze.EXPECTED_ARMS:
        arm_id = arm["arm_id"]
        bundle = OUTPUT_BUNDLES[arm_id]
        command = [
            sys.executable,
            profile_report["scorer_path"],
            "--benchmark",
            profile_report["benchmark_path"],
            "--solver-results",
            profile_report["solver_paths"][arm_id],
            "--image-judge",
            profile_report["final_image_judge_path"],
            "--baseline-judge",
            profile_report["page_rag_baseline_judge_path"],
            "--out-json",
            str(_safe_path(root, bundle["json"], f"{arm_id} json")),
            "--out-md",
            str(_safe_path(root, bundle["md"], f"{arm_id} md")),
            "--out-sha256",
            str(_safe_path(root, bundle["sha256"], f"{arm_id} sha256")),
            "--label",
            arm_id,
            "--expected-rows",
            "274",
            "--expected-deterministic",
            "177",
            "--expected-image-judge",
            "97",
            "--expected-benchmark-sha256",
            base_freeze.BENCHMARK_SHA256,
            "--expected-baseline-judge-sha256",
            base_freeze.PAGE_RAG_JUDGE_SHA256,
        ]
        plans.append({"arm_id": arm_id, "command": command, "outputs": bundle})
    if [plan["arm_id"] for plan in plans] != list(OUTPUT_BUNDLES):
        raise WaveExecutionError("command plan arm order mismatch")
    return plans


def _launch_all(
    plans: list[dict[str, Any]],
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> list[dict[str, Any]]:
    if len(plans) != 4 or len({plan["arm_id"] for plan in plans}) != 4:
        raise WaveExecutionError("exactly four unique command plans are required")

    def invoke(plan: dict[str, Any]) -> dict[str, Any]:
        try:
            completed = runner(
                plan["command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            return {
                "arm_id": plan["arm_id"],
                "returncode": completed.returncode,
                # Captured bytes are intentionally retained only in memory and never returned.
                "captured_stdout_bytes": len(completed.stdout or b""),
                "captured_stderr_bytes": len(completed.stderr or b""),
                "launch_error": None,
            }
        except OSError as exc:
            return {
                "arm_id": plan["arm_id"],
                "returncode": None,
                "captured_stdout_bytes": 0,
                "captured_stderr_bytes": 0,
                "launch_error": type(exc).__name__,
            }

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(plan["arm_id"], pool.submit(invoke, plan)) for plan in plans]
        for arm_id, future in futures:
            try:
                result = future.result()
            except BaseException as exc:
                result = {
                    "arm_id": arm_id,
                    "returncode": None,
                    "captured_stdout_bytes": 0,
                    "captured_stderr_bytes": 0,
                    "launch_error": type(exc).__name__,
                }
            results.append(result)
    # The executor context is the completion barrier: all four futures are done here.
    return results


def _completion_payload(
    root: Path,
    *,
    execution_freeze_sha256: str,
    process_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(process_results) != 4:
        raise WaveExecutionError("completion requires four process results")
    failures = [
        row
        for row in process_results
        if row["returncode"] != 0 or row["launch_error"] is not None
    ]
    if failures:
        raise WaveExecutionError(
            "one or more scorers failed after the four-process barrier; attempt marker preserved"
        )
    artifacts: dict[str, dict[str, dict[str, str]]] = {}
    for arm_id, bundle in OUTPUT_BUNDLES.items():
        arm_artifacts: dict[str, dict[str, str]] = {}
        for kind, relative in bundle.items():
            path = _safe_path(root, relative, f"{arm_id} {kind}")
            if not path.is_file():
                raise WaveExecutionError(
                    "a scorer reported success but its complete output bundle is missing; "
                    "attempt marker preserved"
                )
            arm_artifacts[kind] = {"path": relative, "sha256": _sha256(path)}
        artifacts[arm_id] = arm_artifacts
    return {
        "schema_version": COMPLETION_SCHEMA,
        "wave_id": base_freeze.WAVE_ID,
        "status": "all_four_scores_completed_outputs_hash_frozen",
        "base_evaluation_freeze_sha256": BASE_FREEZE_SHA256,
        "execution_freeze_sha256": execution_freeze_sha256,
        "all_four_processes_completed_before_manifest": True,
        "all_four_returncodes_zero": True,
        "individual_score_content_was_not_parsed_or_printed_by_launcher": True,
        "same_wave_retuning_forbidden": True,
        "artifacts": artifacts,
    }


def execute_wave(root: Path) -> dict[str, Any]:
    root = root.resolve()
    report = verify_execution_freeze(root, require_all_outputs_absent=True)
    declared = _all_declared_paths(root)
    for path in declared.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    attempt_payload = {
        "schema_version": ATTEMPT_SCHEMA,
        "wave_id": base_freeze.WAVE_ID,
        "status": "attempt_started_before_any_scorer_output",
        "base_evaluation_freeze_sha256": BASE_FREEZE_SHA256,
        "execution_freeze_sha256": report["execution_freeze_sha256"],
        "candidate_solver_sha256": {
            arm["arm_id"]: arm["solver_sha256"] for arm in base_freeze.EXPECTED_ARMS
        },
        "output_bundles": OUTPUT_BUNDLES,
        "individual_score_content_access_before_completion": False,
    }
    attempt_path = declared["attempt_marker"]
    _atomic_create_new(attempt_path, _canonical_json(attempt_payload))
    # From this point onward the attempt marker is intentionally never removed.
    plans = _build_command_plans(root, report)
    process_results = _launch_all(plans)
    completion = _completion_payload(
        root,
        execution_freeze_sha256=report["execution_freeze_sha256"],
        process_results=process_results,
    )
    completion_path = declared["completion_manifest"]
    _atomic_create_new(completion_path, _canonical_json(completion))
    completion_sha = _sha256(completion_path)
    _atomic_create_new(
        declared["completion_sha256"],
        f"{completion_sha}  {completion_path.name}\n".encode("ascii"),
    )
    return {
        "status": completion["status"],
        "completion_manifest_path": str(completion_path),
        "completion_manifest_sha256": completion_sha,
        "completion_sha256_path": str(declared["completion_sha256"]),
        "attempt_marker_path": str(attempt_path),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Atomic-ish four-arm one-shot evaluation launcher")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-freeze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.verify_freeze == args.execute:
        print("launcher error: choose exactly one of --verify-freeze or --execute")
        return 2
    try:
        report = execute_wave(args.root) if args.execute else verify_execution_freeze(args.root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (
        WaveExecutionError,
        base_freeze.EvaluationFreezeError,
        base_freeze.selector.SelectorV12Error,
        base_freeze.selector.base.SelectorError,
    ) as exc:
        print(f"launcher error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
