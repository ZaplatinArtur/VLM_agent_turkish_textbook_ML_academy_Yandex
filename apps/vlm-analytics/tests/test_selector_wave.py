from __future__ import annotations

from pathlib import Path

import pytest

from vlm_trace_viewer.adapter import ArtifactError, discover_artifact_root
from vlm_trace_viewer.nine_b_adapter import NineBV7ArtifactAdapter
from vlm_trace_viewer.replay_aggregate import load_frozen_9b_comparison
from vlm_trace_viewer.selector_wave import (
    COMPARISON,
    SelectorWaveAdapter,
    _declared_path_matches,
    _safe_manifest_path,
    build_active_selector_dataset,
)


def _real_root_or_skip() -> Path:
    try:
        return discover_artifact_root()
    except ArtifactError as exc:
        pytest.skip(f"frozen all-9B artifacts are not installed: {exc}")


def test_real_selector_wave_is_bound_to_240_and_seven_canonical_milestones() -> None:
    summary = SelectorWaveAdapter(_real_root_or_skip()).load()

    assert [item.correct for item in summary.milestones] == [141, 193, 194, 218, 227, 235, 238]
    assert summary.correct == 240
    assert summary.rows == 274
    assert summary.math_correct == 109
    assert summary.history_correct == 10
    assert (summary.deterministic_correct, summary.deterministic_rows) == (158, 177)
    assert (summary.image_correct, summary.image_rows) == (82, 97)
    assert summary.fixes == 2
    assert summary.regressions == 0
    assert summary.passthrough_rows == 272
    assert [item.task_id for item in summary.tasks] == ["val_0089", "val_0251"]
    assert summary.repair_task_id == "val_0223"
    assert summary.repair_score_sha256.startswith("453970038673")


def test_selector_adapter_fails_closed_on_a_frozen_hash_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _real_root_or_skip()
    from vlm_trace_viewer import selector_wave

    real_sha256 = selector_wave._sha256

    def corrupt_comparison(path: Path) -> str:
        if path == (root / COMPARISON).resolve():
            return "0" * 64
        return real_sha256(path)

    monkeypatch.setattr(selector_wave, "_sha256", corrupt_comparison)
    with pytest.raises(ArtifactError, match="frozen hash mismatch"):
        SelectorWaveAdapter(root).load()


def test_active_analytics_projects_exactly_two_selector_fixes() -> None:
    root = _real_root_or_skip()
    selector = SelectorWaveAdapter(root).load()
    source_v7 = NineBV7ArtifactAdapter(
        load_frozen_9b_comparison(root / COMPARISON),
        display_asset_root=root,
    ).load()

    active = build_active_selector_dataset(source_v7, selector)

    assert (active.summary.correct, active.summary.rows) == (240, 274)
    assert active.summary.accuracy == pytest.approx(0.8759124087591241)
    assert (active.summary.math_correct, active.summary.math_rows) == (109, 139)
    assert active.summary.by_subject["History"] == {
        "n": 10,
        "new_correct": 10,
        "new_accuracy": 1.0,
    }
    changed = {
        task.task_id: task
        for task in active.tasks
        if "selector_v1_2" in task.raw
    }
    assert {task_id: task.final_answer for task_id, task in changed.items()} == {
        "val_0089": "D",
        "val_0251": "B",
    }
    assert all(task.correct for task in changed.values())
    assert sum(task.correct for task in active.tasks) == 240


@pytest.mark.parametrize(
    "value",
    ("../score.json", "a/../../score.json", "/absolute/score.json", "C:/score.json"),
)
def test_completion_manifest_paths_cannot_escape_experiment_root(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ArtifactError, match="unsafe path"):
        _safe_manifest_path(tmp_path, value)


def test_machine_specific_declared_path_is_only_used_as_a_suffix_binding() -> None:
    relative = Path("reports/example/aggregate.json")
    assert _declared_path_matches(
        r"C:\Users\someone\project\reports\example\aggregate.json", relative
    )
    assert not _declared_path_matches(
        r"C:\Users\someone\project\reports\other\aggregate.json", relative
    )
