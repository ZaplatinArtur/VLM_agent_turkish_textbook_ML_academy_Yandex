from __future__ import annotations

from pathlib import Path

import pytest

from vlm_trace_viewer.adapter import ArtifactError, discover_artifact_root
from vlm_trace_viewer.nine_b_adapter import NineBV7ArtifactAdapter
from vlm_trace_viewer.replay_aggregate import load_frozen_9b_comparison
from vlm_trace_viewer.selector_wave import COMPARISON, SelectorWaveAdapter, build_active_selector_dataset
from vlm_trace_viewer.source_wave import (
    COMPLETION,
    OFFICIAL_FIX_TASK_IDS,
    SourceExpansionWaveAdapter,
    _safe_relative,
    build_active_source_wave_dataset,
)


def _real_root_or_skip() -> Path:
    try:
        return discover_artifact_root()
    except ArtifactError as exc:
        pytest.skip(f"frozen all-9B artifacts are not installed: {exc}")


def test_real_source_wave_is_bound_to_official_249_and_separate_research() -> None:
    summary = SourceExpansionWaveAdapter(_real_root_or_skip()).load()

    assert (summary.correct, summary.rows) == (249, 274)
    assert summary.accuracy == pytest.approx(249 / 274)
    assert (summary.math_correct, summary.math_rows) == (117, 139)
    assert (summary.english_correct, summary.english_rows) == (9, 9)
    assert (summary.deterministic_correct, summary.deterministic_rows) == (158, 177)
    assert (summary.image_correct, summary.image_rows) == (91, 97)
    assert (summary.fixes, summary.regressions) == (9, 0)
    assert summary.fix_task_ids == OFFICIAL_FIX_TASK_IDS
    assert len(summary.tasks) == 274
    assert len(summary.answer_changed_task_ids) == 14
    assert summary.research_all36.correct == 251
    assert summary.research_all36.eligible_for_official_headline is False
    assert summary.research_all36.license_status == "unverified"


def test_source_wave_fails_closed_on_completion_hash_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _real_root_or_skip()
    from vlm_trace_viewer import source_wave

    real_sha256 = source_wave._sha256

    def corrupt_completion(path: Path) -> str:
        if path == (root / COMPLETION).resolve():
            return "0" * 64
        return real_sha256(path)

    monkeypatch.setattr(source_wave, "_sha256", corrupt_completion)
    with pytest.raises(ArtifactError, match="frozen hash mismatch"):
        SourceExpansionWaveAdapter(root).load()


def test_active_projection_recomputes_all_274_outcomes_and_exact_nine_fixes() -> None:
    root = _real_root_or_skip()
    selector = SelectorWaveAdapter(root).load()
    source_v7 = NineBV7ArtifactAdapter(
        load_frozen_9b_comparison(root / COMPARISON),
        display_asset_root=root,
    ).load()
    active_240 = build_active_selector_dataset(source_v7, selector)
    wave = SourceExpansionWaveAdapter(root).load()

    active_249 = build_active_source_wave_dataset(active_240, selector, wave)

    assert (active_249.summary.correct, active_249.summary.rows) == (249, 274)
    assert sum(task.correct for task in active_249.tasks) == 249
    assert (active_249.summary.math_correct, active_249.summary.math_rows) == (117, 139)
    fixed = {
        task.task_id
        for task in active_249.tasks
        if (task.raw.get("source_wave_v1_1") or {}).get("correctness_fix_vs_240")
    }
    assert fixed == set(OFFICIAL_FIX_TASK_IDS)
    assert all(
        task.decision_action in {"source_wave_replace", "source_wave_confirm"}
        for task in active_249.tasks
        if task.task_id in wave.target_task_ids
    )


@pytest.mark.parametrize(
    "value",
    ("../metrics.json", "a/../../metrics.json", "/metrics.json", "C:/metrics.json"),
)
def test_source_wave_completion_paths_cannot_escape_wave(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(ArtifactError, match="unsafe"):
        _safe_relative(tmp_path, value, "fixture")
