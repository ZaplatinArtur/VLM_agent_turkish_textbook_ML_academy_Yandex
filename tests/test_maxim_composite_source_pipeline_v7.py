from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.audit_maxim_composite_source_pipeline_v7 import audit


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = REPO_ROOT / "configs" / "maxim_composite_source_pipeline_v7.json"


def _profile() -> dict:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def _write_profile(path: Path, value: dict) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_frozen_composite_chain_passes_without_judge_outputs(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    result = audit(PROFILE, output, REPO_ROOT)

    assert result["status"] == "pass"
    assert result["candidate_task_ids"] == [
        "val_0042",
        "val_0043",
        "val_0044",
        "val_0046",
        "val_0149",
        "val_0150",
        "val_0178",
        "val_0196",
    ]
    assert result["main"]["overrides"] == 1
    assert result["history"]["overrides"] == 1
    assert result["fixed_outputs_absent"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_rejects_outcome_guided_policy(tmp_path: Path) -> None:
    profile = _profile()
    profile["policy"]["outcome_guided_candidate_removal_allowed"] = True
    mutated = _write_profile(tmp_path / "profile.json", profile)

    with pytest.raises(ValueError, match="outcome_guided_candidate_removal_allowed"):
        audit(mutated, tmp_path / "audit.json", REPO_ROOT)


def test_rejects_candidate_removal(tmp_path: Path) -> None:
    profile = _profile()
    profile["preregistration"]["candidate_task_ids"].pop()
    mutated = _write_profile(tmp_path / "profile.json", profile)

    with pytest.raises(ValueError, match="candidate set or order"):
        audit(mutated, tmp_path / "audit.json", REPO_ROOT)


def test_rejects_pinned_hash_mismatch(tmp_path: Path) -> None:
    profile = _profile()
    profile["main_stage"]["solver"]["sha256"] = "0" * 64
    mutated = _write_profile(tmp_path / "profile.json", profile)

    with pytest.raises(ValueError, match="SHA-256 mismatch for main solver"):
        audit(mutated, tmp_path / "audit.json", REPO_ROOT)


def test_rejects_judge_output_collision_with_input(tmp_path: Path) -> None:
    profile = _profile()
    profile["main_stage"]["image_judge_output"] = copy.deepcopy(
        profile["main_stage"]["solver"]["path"]
    )
    mutated = _write_profile(tmp_path / "profile.json", profile)

    with pytest.raises(ValueError, match="overwrites an input"):
        audit(mutated, tmp_path / "audit.json", REPO_ROOT)


def test_rejects_audit_overwrite_of_pinned_profile() -> None:
    with pytest.raises(ValueError, match="overwrite the profile"):
        audit(PROFILE, PROFILE, REPO_ROOT)
