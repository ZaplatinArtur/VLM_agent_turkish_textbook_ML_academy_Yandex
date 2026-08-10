from __future__ import annotations

import inspect
import subprocess
import threading
from pathlib import Path

import pytest

import execute_multi_arm_evaluation_wave_v1 as launcher


ROOT = Path(__file__).resolve().parent


def test_all_declared_attempt_completion_and_score_bundle_paths_are_unique_and_absent() -> None:
    paths = launcher._all_declared_paths(ROOT)
    assert len(paths) == 15
    assert len(set(paths.values())) == 15
    assert all(not path.exists() for path in paths.values())
    assert set(launcher.OUTPUT_BUNDLES) == {
        "v1_1_primary",
        "v1_1_secondary",
        "v1_2_primary",
        "v1_2_exploratory",
    }
    for bundle in launcher.OUTPUT_BUNDLES.values():
        assert set(bundle) == {"json", "md", "sha256"}


def test_atomic_attempt_marker_is_exclusive_and_preserves_first_bytes(tmp_path: Path) -> None:
    path = tmp_path / "wave" / "ATTEMPT_STARTED.json"
    launcher._atomic_create_new(path, b"first\n")
    with pytest.raises(launcher.WaveExecutionError, match="refusing to overwrite"):
        launcher._atomic_create_new(path, b"second\n")
    assert path.read_bytes() == b"first\n"


def test_command_plan_is_fixed_four_arm_same_evaluator_with_unique_json_md_sha_outputs() -> None:
    base_report = launcher.base_freeze.verify_freeze(ROOT)
    plans = launcher._build_command_plans(ROOT, {"base_report": base_report})
    assert [plan["arm_id"] for plan in plans] == list(launcher.OUTPUT_BUNDLES)
    assert len(plans) == 4
    scorer_paths = {plan["command"][1] for plan in plans}
    benchmark_paths = {
        plan["command"][plan["command"].index("--benchmark") + 1] for plan in plans
    }
    judge_paths = {
        plan["command"][plan["command"].index("--image-judge") + 1] for plan in plans
    }
    baseline_paths = {
        plan["command"][plan["command"].index("--baseline-judge") + 1] for plan in plans
    }
    assert len(scorer_paths) == len(benchmark_paths) == len(judge_paths) == len(baseline_paths) == 1
    all_outputs: list[str] = []
    for plan in plans:
        command = plan["command"]
        assert command[command.index("--expected-rows") + 1] == "274"
        assert command[command.index("--expected-deterministic") + 1] == "177"
        assert command[command.index("--expected-image-judge") + 1] == "97"
        for flag in ("--out-json", "--out-md", "--out-sha256"):
            all_outputs.append(command[command.index(flag) + 1])
    assert len(all_outputs) == len(set(all_outputs)) == 12


def test_launch_barrier_starts_and_awaits_all_four_and_never_prints_captured_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    barrier = threading.Barrier(4)
    finished: list[str] = []
    lock = threading.Lock()

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert kwargs == {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "check": False,
        }
        arm_id = command[0]
        barrier.wait(timeout=2)
        with lock:
            finished.append(arm_id)
        return subprocess.CompletedProcess(
            command,
            7 if arm_id == "arm2" else 0,
            stdout=f"SECRET_SCORE_{arm_id}".encode(),
            stderr=f"SECRET_ERROR_{arm_id}".encode(),
        )

    plans = [
        {"arm_id": f"arm{index}", "command": [f"arm{index}"], "outputs": {}}
        for index in range(4)
    ]
    results = launcher._launch_all(plans, runner=fake_runner)
    assert set(finished) == {"arm0", "arm1", "arm2", "arm3"}
    assert [row["arm_id"] for row in results] == [f"arm{index}" for index in range(4)]
    assert next(row for row in results if row["arm_id"] == "arm2")["returncode"] == 7
    captured = capsys.readouterr()
    assert "SECRET_SCORE" not in captured.out + captured.err
    assert "SECRET_ERROR" not in captured.out + captured.err


def test_completion_rejects_any_failure_after_barrier_without_creating_manifest(tmp_path: Path) -> None:
    results = [
        {
            "arm_id": arm_id,
            "returncode": 1 if index == 0 else 0,
            "captured_stdout_bytes": 0,
            "captured_stderr_bytes": 0,
            "launch_error": None,
        }
        for index, arm_id in enumerate(launcher.OUTPUT_BUNDLES)
    ]
    with pytest.raises(launcher.WaveExecutionError, match="four-process barrier"):
        launcher._completion_payload(
            tmp_path,
            execution_freeze_sha256="0" * 64,
            process_results=results,
        )
    assert not (tmp_path / launcher.COMPLETION_MANIFEST_RELATIVE_PATH).exists()


def test_completion_hashes_all_twelve_outputs_without_parsing_score_content(tmp_path: Path) -> None:
    for arm_id, bundle in launcher.OUTPUT_BUNDLES.items():
        for kind, relative in bundle.items():
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Deliberately not valid JSON/Markdown/checksum: completion only establishes byte hashes.
            path.write_bytes(f"opaque-{arm_id}-{kind}\n".encode())
    results = [
        {
            "arm_id": arm_id,
            "returncode": 0,
            "captured_stdout_bytes": 123,
            "captured_stderr_bytes": 0,
            "launch_error": None,
        }
        for arm_id in launcher.OUTPUT_BUNDLES
    ]
    payload = launcher._completion_payload(
        tmp_path,
        execution_freeze_sha256="1" * 64,
        process_results=results,
    )
    assert payload["status"] == "all_four_scores_completed_outputs_hash_frozen"
    assert set(payload["artifacts"]) == set(launcher.OUTPUT_BUNDLES)
    assert sum(len(bundle) for bundle in payload["artifacts"].values()) == 12
    assert payload["individual_score_content_was_not_parsed_or_printed_by_launcher"] is True


def test_execute_wave_source_places_exclusive_attempt_before_four_process_launch() -> None:
    source = inspect.getsource(launcher.execute_wave)
    marker = source.index("_atomic_create_new(attempt_path")
    launch = source.index("_launch_all(plans)")
    completion = source.index("_completion_payload(")
    manifest = source.index("_atomic_create_new(completion_path")
    assert marker < launch < completion < manifest
    assert "unlink" not in source


@pytest.mark.skipif(
    not (ROOT / launcher.EXECUTION_FREEZE_NAME).is_file(),
    reason="execution freeze is created only after launcher and tests are pinned",
)
def test_final_execution_freeze_verifies_without_creating_attempt_or_scores() -> None:
    report = launcher.verify_execution_freeze(ROOT)
    assert report["status"] == "execution_launcher_freeze_verified_not_executed"
    assert report["all_declared_paths_absent"] is True
