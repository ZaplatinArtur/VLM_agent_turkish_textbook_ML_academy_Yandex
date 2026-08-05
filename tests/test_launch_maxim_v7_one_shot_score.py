from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.launch_maxim_v7_one_shot_score import audit


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE = (
    REPO_ROOT
    / "reports"
    / "maxim_official_exact_source_v2_20260805"
    / "V7_ONE_SHOT_EVALUATION_FREEZE.json"
)


def _freeze() -> dict:
    return json.loads(FREEZE.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def test_frozen_launch_is_irreversibly_consumed_after_first_attempt() -> None:
    with pytest.raises(ValueError, match="score output already exists"):
        audit(FREEZE, REPO_ROOT, check_remote=False)

    evaluation = (
        REPO_ROOT
        / "reports"
        / "maxim_official_exact_source_v2_20260805"
        / "fill_blank_page_activity_history_v7_evaluation"
    )
    marker = json.loads((evaluation / "score_attempt.json").read_text(encoding="utf-8"))
    assert marker["attempt"] == 1
    assert (evaluation / "score.json").is_file()
    assert (evaluation / "score.md").is_file()
    assert (evaluation / "score.sha256").is_file()


def test_rejects_final_judge_hash_change(tmp_path: Path) -> None:
    freeze = _freeze()
    freeze["final_image_judge"]["sha256"] = "0" * 64
    mutated = _write(tmp_path / "freeze.json", freeze)
    with pytest.raises(ValueError, match="final image judge"):
        audit(mutated, REPO_ROOT, check_remote=False)


def test_rejects_launch_argument_change(tmp_path: Path) -> None:
    freeze = _freeze()
    freeze["launch"]["arguments"][-1] = "0" * 64
    mutated = _write(tmp_path / "freeze.json", freeze)
    with pytest.raises(ValueError, match="launch arguments"):
        audit(mutated, REPO_ROOT, check_remote=False)


def test_rejects_existing_attempt_marker(tmp_path: Path) -> None:
    freeze = _freeze()
    freeze["launch"]["outputs"]["attempt_marker"] = freeze["scorer"]["path"]
    mutated = _write(tmp_path / "freeze.json", freeze)
    with pytest.raises(ValueError, match="score output already exists"):
        audit(mutated, REPO_ROOT, check_remote=False)


def test_rejects_same_wave_retuning(tmp_path: Path) -> None:
    freeze = _freeze()
    freeze["policy"]["same_wave_retuning_allowed"] = True
    mutated = _write(tmp_path / "freeze.json", freeze)
    with pytest.raises(ValueError, match="same-wave retuning"):
        audit(mutated, REPO_ROOT, check_remote=False)
