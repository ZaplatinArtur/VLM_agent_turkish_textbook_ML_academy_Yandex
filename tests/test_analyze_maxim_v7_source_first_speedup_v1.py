from __future__ import annotations

from pathlib import Path

from scripts.analyze_maxim_v7_source_first_speedup_v1 import analyze


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_v7_source_first_replay_is_answer_equivalent(tmp_path: Path) -> None:
    result = analyze(
        ROOT / "configs/maxim_composite_source_pipeline_v7.json",
        tmp_path / "analysis.json",
        tmp_path / "REPORT.md",
        ROOT,
    )

    assert result["rows"] == 274
    assert result["source_shortcuts"] == 131
    assert result["anchor_fallbacks"] == 143
    assert result["answer_equivalent_rows"] == 131
    assert result["conflicting_source_rows"] == []
    assert result["recorded_anchor_usage"]["avoidable_latency_fraction"] > 0.44
    assert result["claims"] == {
        "artifact_answer_equivalence_measured": True,
        "online_wall_clock_speedup_measured": False,
        "source_lookup_cost_included": False,
        "accuracy_or_gold_read": False,
        "task_id_policy_feature": False,
        "task_id_alignment_only": True,
    }
    assert (tmp_path / "analysis.json").is_file()
    assert "не новый score" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")
