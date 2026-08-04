from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compose_maxim_consensus_repair_sweep_v1 as composer


def _row(task_id: str, answer: str, *, forced: bool = False) -> dict:
    return {
        "task_id": task_id,
        "condition": "source",
        "final_answer": answer,
        "forced_answer": forced,
        "error": None,
        "reasoning": "r",
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _profile(tmp_path: Path, rows_by_source: dict[str, list[dict]]) -> Path:
    sources = {}
    for name in ("default_v21", "v31", "active", "no_tools"):
        path = tmp_path / f"{name}.jsonl"
        _write(path, rows_by_source[name])
        sources[name] = {"path": str(path), "sha256": composer.sha256_file(path), "rows": 2}
    profile = {
        "schema_version": composer.PROFILE_SCHEMA_VERSION,
        "status": "frozen_before_source_row_values_and_target_scoring",
        "rows": 2,
        "policies": list(composer.POLICIES),
        "source_outcomes_used_for_policy_design": False,
        "composer_sha256": composer.sha256_file(Path(composer.__file__).resolve()),
        "conditions": {policy: f"maxim_consensus_repair_{policy}_v1" for policy in composer.POLICIES},
        "sources": sources,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> Path:
    return _profile(
        tmp_path,
        {
            "default_v21": [_row("a", "old"), _row("b", "same")],
            "v31": [_row("a", "  Answer: NEW! "), _row("b", "x")],
            "active": [_row("a", "new"), _row("b", "y")],
            "no_tools": [_row("a", "other"), _row("b", "same")],
        },
    )


def test_pair_agreement_overrides_exact_source_row(tmp_path: Path) -> None:
    profile = _fixture(tmp_path)
    output = tmp_path / "out"
    result = composer.compose(profile_path=profile, policy="v31_active_pair", output_dir=output)
    rows = [json.loads(line) for line in (output / "solver.jsonl").read_text().splitlines()]
    assert result["override_rows"] == 1
    assert result["source_counts"]["v31"] == 1
    assert rows[0]["final_answer"] == "  Answer: NEW! "
    assert rows[1]["final_answer"] == "same"


def test_two_of_three_majority_uses_frozen_priority(tmp_path: Path) -> None:
    profile = _fixture(tmp_path)
    output = tmp_path / "out"
    result = composer.compose(
        profile_path=profile, policy="two_of_three_majority", output_dir=output
    )
    assert result["source_counts"] == {"default": 1, "v31": 1, "active": 0, "no_tools": 0}


def test_forced_vote_is_ineligible(tmp_path: Path) -> None:
    profile = _profile(
        tmp_path,
        {
            "default_v21": [_row("a", "old"), _row("b", "same")],
            "v31": [_row("a", "new", forced=True), _row("b", "x")],
            "active": [_row("a", "new"), _row("b", "y")],
            "no_tools": [_row("a", "other"), _row("b", "z")],
        },
    )
    result = composer.compose(
        profile_path=profile, policy="v31_active_pair", output_dir=tmp_path / "out"
    )
    assert result["override_rows"] == 0


def test_source_change_after_freeze_is_rejected(tmp_path: Path) -> None:
    profile = _fixture(tmp_path)
    value = json.loads(profile.read_text())
    source = Path(value["sources"]["v31"]["path"])
    source.write_text(source.read_text() + "\n", encoding="utf-8")
    with pytest.raises(composer.ConsensusRepairError, match="SHA256 mismatch"):
        composer.compose(profile_path=profile, policy="v31_active_pair", output_dir=tmp_path / "out")


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    profile = _fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "owned.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(composer.ConsensusRepairError, match="already exists"):
        composer.compose(profile_path=profile, policy="v31_active_pair", output_dir=output)
    assert sentinel.read_text() == "keep"
