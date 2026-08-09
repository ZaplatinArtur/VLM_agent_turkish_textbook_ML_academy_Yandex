from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_trace_viewer.holdout80 import (
    EXPECTED_SUMMARY_PROJECTION_SHA256,
    SUMMARY_FILE,
    HoldoutIntegrityError,
    load_holdout80_summary,
)


def test_frozen_public_summary_has_expected_scopes_and_scores() -> None:
    summary = load_holdout80_summary()

    assert (summary.raw.correct, summary.raw.total, summary.raw.accuracy) == (
        71,
        80,
        0.8875,
    )
    assert (
        summary.erratum_inclusive.correct,
        summary.erratum_inclusive.total,
        summary.erratum_inclusive.accuracy,
    ) == (79, 80, 0.9875)
    assert (summary.valid.correct, summary.valid.total, summary.valid.accuracy) == (
        79,
        79,
        1.0,
    )
    assert summary.projection_sha256 == EXPECTED_SUMMARY_PROJECTION_SHA256
    assert summary.scope["metric_kind"] == "source lookup and source binding"
    assert summary.scope["book_disjoint"] is False
    assert summary.scope["private_rows_embedded"] is False


def test_subjects_and_errata_reconcile_without_overwriting_raw() -> None:
    summary = load_holdout80_summary()
    subjects = {row.subject: row for row in summary.subjects}

    assert (subjects["Math 12"].raw_correct, subjects["Math 12"].raw_total) == (
        20,
        20,
    )
    assert (subjects["Biology 9"].raw_correct, subjects["Biology 9"].raw_total) == (
        30,
        30,
    )
    assert (
        subjects["Physics 12"].raw_correct,
        subjects["Physics 12"].raw_total,
        subjects["Physics 12"].valid_correct,
        subjects["Physics 12"].valid_total,
    ) == (21, 30, 29, 29)
    assert {row.kind: row.affected_rows for row in summary.errata} == {
        "swapped_gold_sections": 8,
        "invalid_task_type": 1,
    }
    assert summary.mcq["raw_correct"] == 51
    assert summary.mcq["valid_correct"] == summary.mcq["valid_total"] == 59


def test_public_summary_contains_no_private_rows_or_task_ids() -> None:
    text = SUMMARY_FILE.read_text(encoding="utf-8")

    assert '"task_id"' not in text
    assert "h80-" not in text
    assert '"prediction"' not in text
    assert '"gold_answer"' not in text


def test_semantic_tampering_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    payload["scores"]["raw_protocol"]["correct"] = 72
    tampered = tmp_path / "holdout80.json"
    tampered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(HoldoutIntegrityError, match="projection mismatch"):
        load_holdout80_summary(tampered)
