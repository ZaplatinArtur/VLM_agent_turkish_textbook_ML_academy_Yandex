from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_maxim_9b_comparison_v2 import (
    BuildError,
    comparison,
    normalized_solver,
)


def test_comparison_counts_fix_regression_and_unchanged() -> None:
    before = {"a": False, "b": True, "c": True, "d": False}
    after = {"a": True, "b": False, "c": True, "d": False}
    result = comparison("page_rag_9b", "a" * 64, before, after)
    assert (result["fixes"], result["regressions"], result["unchanged"]) == (1, 1, 2)


def test_normalized_solver_adds_only_explicit_model_origin(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "normalized.jsonl"
    row = {"task_id": "val_0001", "model": "Qwen/Qwen3.5-9B", "final_answer": "A"}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
    _, indexed = normalized_solver(source, output, {"val_0001"})
    assert indexed["val_0001"]["final_origin"] == "model_anchor"
    assert indexed["val_0001"]["final_answer"] == "A"
    assert source.read_text(encoding="utf-8") == json.dumps(row) + "\n"


def test_normalized_solver_rejects_foreign_model(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    row = {"task_id": "val_0001", "model": "Qwen/Qwen3.5-27B", "final_answer": "A"}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(BuildError, match="not an exact"):
        normalized_solver(source, tmp_path / "normalized.jsonl", {"val_0001"})
