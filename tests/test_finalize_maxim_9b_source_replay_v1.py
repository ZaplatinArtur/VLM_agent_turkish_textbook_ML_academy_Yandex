from __future__ import annotations

import pytest

from scripts.finalize_maxim_9b_source_replay_v1 import (
    FinalizeError,
    KNOWN_27B_SOLVER_SHA256,
    guard_no_known_27b,
    referenced,
    stable_projection,
    validate_solver_rows,
)


def test_validate_solver_rows_rejects_top_level_judge_outcome() -> None:
    rows = {
        "val_0001": {
            "task_id": "val_0001",
            "model": "Qwen/Qwen3.5-9B",
            "final_answer": "A",
            "strict_correct": True,
        }
    }
    with pytest.raises(FinalizeError, match="forbidden outcome fields"):
        validate_solver_rows(rows, "tampered")


def test_validate_solver_rows_rejects_generation_gold_access() -> None:
    rows = {
        "val_0001": {
            "task_id": "val_0001",
            "model": "Qwen/Qwen3.5-9B",
            "final_answer": "A",
            "generation": {"gold_access": True},
        }
    }
    with pytest.raises(FinalizeError, match="gold_access"):
        validate_solver_rows(rows, "tampered")


def test_recursive_guard_rejects_renamed_known_27b_sha() -> None:
    digest = sorted(KNOWN_27B_SOLVER_SHA256)[0]
    with pytest.raises(FinalizeError, match="known 27B solver SHA"):
        guard_no_known_27b({"nested": [{"opaque": digest}]}, "tampered")


def test_stable_projection_ignores_only_timestamp_and_path() -> None:
    left = {
        "created_at_utc": "2026-08-09T00:00:00+00:00",
        "artifact": {"path": "C:/one/solver.jsonl", "sha256": "a" * 64},
        "score": 238,
    }
    right = {
        "created_at_utc": "2026-08-10T00:00:00+00:00",
        "artifact": {"path": "D:/two/solver.jsonl", "sha256": "a" * 64},
        "score": 238,
    }
    assert stable_projection(left) == stable_projection(right)
    right["score"] = 237
    assert stable_projection(left) != stable_projection(right)


def test_referenced_rejects_stale_manifest_hash(tmp_path) -> None:
    manifest = tmp_path / "image_judge_manifest.json"
    manifest.write_text('{"version":2}\n', encoding="utf-8", newline="\n")
    stale = {"path": str(manifest), "sha256": "0" * 64}
    with pytest.raises(FinalizeError, match="hash mismatch"):
        referenced(stale, "stale closure")
