from __future__ import annotations

import json
from pathlib import Path

import pytest

import verify_multi_arm_evaluation_freeze_v1 as freeze


ROOT = Path(__file__).resolve().parent


def test_profile_verifies_all_four_frozen_solvers_without_scoring() -> None:
    report = freeze.validate_profile(ROOT)
    assert report["status"] == "multi_arm_profile_verified_unscored"
    assert report["rows"] == 274
    assert report["route_counts"] == {"deterministic": 177, "image_judge": 97}
    assert set(report["solver_paths"]) == {
        "v1_1_primary",
        "v1_1_secondary",
        "v1_2_primary",
        "v1_2_exploratory",
    }
    assert report["score_output_paths_absent"] is True
    assert len(report["score_output_paths"]) == len(set(report["score_output_paths"])) == 4


def test_profile_pins_exact_benchmark_scorer_and_judges() -> None:
    profile = json.loads((ROOT / "multi_arm_evaluation_profile_v1.json").read_text(encoding="utf-8"))
    assert profile["benchmark"]["sha256"] == freeze.BENCHMARK_SHA256
    assert profile["scorer"]["sha256"] == freeze.SCORER_SHA256
    assert profile["evaluator_split"]["final_image_judge"]["sha256"] == (
        freeze.FINAL_IMAGE_JUDGE_SHA256
    )
    assert profile["evaluator_split"]["page_rag_baseline_judge"]["sha256"] == (
        freeze.PAGE_RAG_JUDGE_SHA256
    )
    assert profile["evaluator_split"]["deterministic_rows"] == 177
    assert profile["evaluator_split"]["image_judge_rows"] == 97


def test_profile_states_one_shot_no_same_wave_retuning_and_no_score_authorization() -> None:
    profile = json.loads((ROOT / "multi_arm_evaluation_profile_v1.json").read_text(encoding="utf-8"))
    chronology = profile["chronology"]
    policy = profile["execution_policy"]
    assert chronology["all_four_candidate_solvers_were_frozen_before_any_score_of_any_arm_in_this_wave"]
    assert chronology["same_wave_retuning_after_any_arm_score_is_forbidden"]
    assert chronology["freeze_construction_read_gold_reference_correctness_score_or_judge_outcomes"] is False
    assert policy["one_shot_multi_arm_wave"]
    assert policy["do_not_inspect_any_arm_score_before_all_four_outputs_exist"]
    assert policy["freeze_does_not_authorize_or_execute_scoring"]


def test_candidate_contract_is_exact_and_every_output_path_is_currently_absent() -> None:
    profile = json.loads((ROOT / "multi_arm_evaluation_profile_v1.json").read_text(encoding="utf-8"))
    assert profile["candidate_arms"] == freeze.EXPECTED_ARMS
    paths = [
        freeze._safe_experiment_path(ROOT, arm["experiment_relative_score_output_path"], arm["arm_id"])
        for arm in freeze.EXPECTED_ARMS
    ]
    assert len(paths) == len(set(paths)) == 4
    assert all(not path.exists() for path in paths)


def test_path_guards_reject_absolute_and_parent_traversal() -> None:
    with pytest.raises(freeze.EvaluationFreezeError, match="unsafe"):
        freeze._safe_experiment_path(ROOT, "../score.json", "tampered output")
    with pytest.raises(freeze.EvaluationFreezeError, match="unsafe"):
        freeze._safe_experiment_path(ROOT, str((ROOT / "score.json").resolve()), "absolute output")
    with pytest.raises(freeze.EvaluationFreezeError, match="prefix"):
        freeze._repository_file(ROOT, "unknown/score.py", "0" * 64, "tampered scorer")


def test_solver_validator_rejects_identity_swap_even_when_tampered_file_hash_is_recomputed(
    tmp_path: Path,
) -> None:
    source = ROOT / freeze.EXPECTED_ARMS[0]["experiment_relative_solver_path"]
    lines = source.read_bytes().splitlines(keepends=True)
    first = json.loads(lines[0].decode("utf-8"))
    first["task_id"] = "swapped_task"
    lines[0] = (
        json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    tampered = tmp_path / "solver.jsonl"
    tampered.write_bytes(b"".join(lines))
    profile = json.loads((ROOT / "profile_v1_2.json").read_text(encoding="utf-8"))
    rows, ordered_ids, _, _, _ = freeze.selector.load_input_package(
        ROOT / freeze.selector.INPUT_PACKAGE_RELATIVE_PATH, profile
    )
    assert len(rows) == 274
    with pytest.raises(freeze.EvaluationFreezeError, match="identity/order"):
        freeze._validate_solver(tampered, freeze._sha256(tampered), ordered_ids, "tampered")


@pytest.mark.skipif(
    not (ROOT / "MULTI_ARM_EVALUATION_FREEZE_v1.json").is_file(),
    reason="freeze is created only after profile, verifier, and tests are pinned",
)
def test_final_multi_arm_freeze_verifies_and_remains_unscored() -> None:
    report = freeze.verify_freeze(ROOT)
    assert report["status"] == "multi_arm_evaluation_freeze_verified_not_scored"
    assert report["profile_report"]["score_output_paths_absent"] is True


@pytest.mark.skipif(
    not (ROOT / "MULTI_ARM_EVALUATION_FREEZE_v1.json").is_file(),
    reason="freeze is created only after profile, verifier, and tests are pinned",
)
def test_final_freeze_rejects_profile_tamper(tmp_path: Path) -> None:
    for name in (
        "multi_arm_evaluation_profile_v1.json",
        "verify_multi_arm_evaluation_freeze_v1.py",
        "test_multi_arm_evaluation_freeze_v1.py",
        "MULTI_ARM_EVALUATION_FREEZE_v1.json",
    ):
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    (tmp_path / "multi_arm_evaluation_profile_v1.json").write_bytes(
        (ROOT / "multi_arm_evaluation_profile_v1.json").read_bytes() + b"\n"
    )
    with pytest.raises(freeze.EvaluationFreezeError, match="profile SHA mismatch"):
        freeze.verify_freeze(tmp_path)
