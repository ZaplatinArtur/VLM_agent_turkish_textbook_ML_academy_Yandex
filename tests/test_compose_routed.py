from __future__ import annotations

import json
from pathlib import Path

import pytest

from mla_baseline.compose_routed import compose_routed_results


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _result(task_id: str, condition: str, answer: str) -> dict:
    return {
        "task_id": task_id,
        "condition": condition,
        "model": "qwen/qwen3.5-9b",
        "prompt_version": "v2_cot",
        "final_answer": answer,
        "generation": {"experiment_id": condition},
        "tool_calls": ([{"tool": "search_textbooks"}] if condition == "agent_rag" else []),
        "error": None,
    }


def test_composes_exact_baseline_and_rag_rows_by_subject(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    no_tools = tmp_path / "b0.jsonl"
    rag = tmp_path / "e3.jsonl"
    output = tmp_path / "e4.jsonl"
    _write_jsonl(
        tasks,
        [
            {
                "task_id": "math",
                "subject": "Math",
                "grade": None,
                "question": "image",
                "question_images": [
                    {
                        "image_id": "i1",
                        "format": "file_path",
                        "data": "images/q.png",
                        "mime_type": "image/png",
                    }
                ],
                "reference_answer": "A",
                "answer_type": "choice",
            },
            {
                "task_id": "history",
                "subject": "History",
                "grade": None,
                "question": "image",
                "reference_answer": "B",
                "answer_type": "choice",
            },
        ],
    )
    _write_jsonl(
        no_tools,
        [_result("math", "b0_no_tools", "A"), _result("history", "b0_no_tools", "C")],
    )
    _write_jsonl(
        rag,
        [_result("math", "agent_rag", "D"), _result("history", "agent_rag", "B")],
    )

    report = compose_routed_results(
        tasks_path=tasks,
        no_tools_path=no_tools,
        rag_path=rag,
        output_path=output,
        no_retrieval_subjects="Math",
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert report["router_skips"] == 1
    assert report["router_allows"] == 1
    assert [row["final_answer"] for row in rows] == ["A", "B"]
    assert rows[0]["tool_calls"] == []
    assert rows[0]["retrieval_relevance"] == "not_attempted"
    assert rows[0]["answer_source"] == "image_only_no_retrieval"
    assert rows[0]["generation"]["composed_from_condition"] == "b0_no_tools"
    assert rows[1]["tool_calls"] == [{"tool": "search_textbooks"}]
    assert rows[1]["generation"]["composed_from_condition"] == "agent_rag"


def test_rejects_different_task_sets(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    no_tools = tmp_path / "b0.jsonl"
    rag = tmp_path / "e3.jsonl"
    _write_jsonl(
        tasks,
        [
            {
                "task_id": "q1",
                "subject": "Math",
                "grade": None,
                "question": "q",
                "reference_answer": "A",
                "answer_type": "choice",
            }
        ],
    )
    _write_jsonl(no_tools, [_result("q1", "b0_no_tools", "A")])
    _write_jsonl(rag, [])

    with pytest.raises(ValueError, match="task sets differ"):
        compose_routed_results(
            tasks_path=tasks,
            no_tools_path=no_tools,
            rag_path=rag,
            output_path=tmp_path / "out.jsonl",
            no_retrieval_subjects="Math",
        )
