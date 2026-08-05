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


def test_frozen_launch_is_ready_before_first_attempt() -> None:
    result = audit(FREEZE, REPO_ROOT, check_remote=False)
    assert result["status"] == "ready"
    assert result["score_attempts"] == 0
    assert result["score_outputs_absent"] is True


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
