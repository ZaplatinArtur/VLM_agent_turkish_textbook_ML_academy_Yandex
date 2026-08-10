from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from vlm_analytics.analytics import AnalyticsService
from vlm_analytics.database import SCHEMA, Database
from vlm_analytics.importer import import_run


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _raw(task_id: str, *, condition: str, with_trace: bool = False) -> dict:
    row = {
        "task_id": task_id,
        "condition": condition,
        "model": "qwen/qwen3.5-9b",
        "prompt_version": "v1",
        "final_answer": "A",
        "solution_steps": "solution",
        "generation": {"experiment_id": condition},
        "usage": {"input_tokens": 10, "output_tokens": 5, "latency_s": 1.0},
        "tool_calls": [],
    }
    if with_trace:
        row["generation"].update(
            {
                "retrieval_route": "allow",
                "retrieval_route_reason": "subject_allowed",
            }
        )
        row.update(
            {
                "exit_reason": "answered_with_retrieval",
                "image_evidence": ["question fact"],
                "retrieval_relevance": "confident",
                "retrieval_conflict": False,
                "answer_source": "image_with_retrieval_support",
                "tool_calls": [
                    {
                        "tool": "search_textbooks",
                        "args": {"query": "topic"},
                        "returned_chunk_ids": ["chunk-1"],
                        "relevance": {"label": "confident", "top_score": 0.9},
                    }
                ],
            }
        )
    return row


def _judge(task_id: str, correct: bool) -> dict:
    return {
        "task_id": task_id,
        "verdict": {
            "strict_correct": correct,
            "final_answer_correct": correct,
            "label": "fully_correct" if correct else "incorrect",
            "score": 4 if correct else 0,
            "confidence": 0.9,
            "error_types": [],
        },
    }


def test_imports_rag_trace_and_computes_paired_flips(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"task_id": "q1", "subject": "Math", "answer_type": "choice"},
            {"task_id": "q2", "subject": "Geography", "answer_type": "choice"},
        ],
    )
    database = Database(tmp_path / "analytics.db")

    baseline_raw = tmp_path / "baseline_raw.jsonl"
    baseline_judge = tmp_path / "baseline_judge.jsonl"
    _write_jsonl(
        baseline_raw,
        [_raw("q1", condition="b0_no_tools"), _raw("q2", condition="b0_no_tools")],
    )
    _write_jsonl(baseline_judge, [_judge("q1", True), _judge("q2", False)])
    import_run(
        database,
        run_key="b0_no_tools",
        display_name="Без тулов",
        raw_path=baseline_raw,
        judge_path=baseline_judge,
        manifest_path=manifest,
    )

    candidate_raw = tmp_path / "candidate_raw.jsonl"
    candidate_judge = tmp_path / "candidate_judge.jsonl"
    _write_jsonl(
        candidate_raw,
        [
            _raw("q1", condition="agent_rag_routed", with_trace=True),
            _raw("q2", condition="agent_rag_routed", with_trace=True),
        ],
    )
    _write_jsonl(candidate_judge, [_judge("q1", False), _judge("q2", True)])
    import_run(
        database,
        run_key="agent_rag_routed",
        display_name="Routed image-first RAG",
        raw_path=candidate_raw,
        judge_path=candidate_judge,
        manifest_path=manifest,
    )

    trace = database.rows(
        """
        SELECT exit_reason, retrieval_relevance, retrieval_conflict, answer_source,
               image_evidence_json, experiment_id, retrieval_route,
               retrieval_route_reason
        FROM task_results
        WHERE run_id = (SELECT MAX(id) FROM runs WHERE run_key = 'agent_rag_routed')
        ORDER BY task_id
        """
    )[0]
    assert trace["exit_reason"] == "answered_with_retrieval"
    assert trace["retrieval_relevance"] == "confident"
    assert trace["retrieval_conflict"] == 0
    assert trace["answer_source"] == "image_with_retrieval_support"
    assert json.loads(trace["image_evidence_json"]) == ["question fact"]
    assert trace["experiment_id"] == "agent_rag_routed"
    assert trace["retrieval_route"] == "allow"
    assert trace["retrieval_route_reason"] == "subject_allowed"

    tool = database.rows(
        "SELECT relevance_label, relevance_json FROM tool_calls ORDER BY id LIMIT 1"
    )[0]
    assert tool["relevance_label"] == "confident"
    assert json.loads(tool["relevance_json"])["top_score"] == 0.9

    comparisons = AnalyticsService(database).paired_comparisons()
    assert comparisons == [
        {
            "baseline_run_key": "b0_no_tools",
            "baseline_display_name": "Без тулов",
            "dataset_version": "validation_v2_274_hybrid",
            "candidate_run_key": "agent_rag_routed",
            "candidate_display_name": "Routed image-first RAG",
            "paired_tasks": 2,
            "baseline_correct": 1,
            "candidate_correct": 1,
            "baseline_accuracy": 50.0,
            "candidate_accuracy": 50.0,
            "delta_pp": 0.0,
            "fixed": 1,
            "regressed": 1,
            "net_fixes": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "oracle_accuracy": 100.0,
            "mcnemar_exact_p": 1.0,
        }
    ]
    routed_stats = next(
        item
        for item in AnalyticsService(database).tool_stats()
        if item["run_key"] == "agent_rag_routed"
    )
    assert routed_stats["confident_calls"] == 2
    assert routed_stats["weak_calls"] == 0
    assert routed_stats["conflicts"] == 0

    import_run(
        database,
        run_key="other_dataset",
        display_name="Other dataset",
        raw_path=candidate_raw,
        judge_path=candidate_judge,
        manifest_path=manifest,
        dataset_version="different_dataset",
    )
    assert len(AnalyticsService(database).paired_comparisons()) == 1


def test_existing_database_is_migrated_for_agent_trace(tmp_path: Path) -> None:
    database_path = tmp_path / "old.db"
    old_schema = (
        SCHEMA.replace("    exit_reason TEXT,\n", "")
        .replace("    image_evidence_json TEXT NOT NULL DEFAULT '[]',\n", "")
        .replace("    retrieval_relevance TEXT,\n", "")
        .replace("    retrieval_conflict INTEGER,\n", "")
        .replace("    answer_source TEXT,\n", "")
        .replace("    experiment_id TEXT,\n", "")
        .replace("    retrieval_route TEXT,\n", "")
        .replace("    retrieval_route_reason TEXT,\n", "")
        .replace("    relevance_json TEXT NOT NULL DEFAULT '{}',\n", "")
        .replace("    relevance_label TEXT,\n", "")
    )
    with sqlite3.connect(database_path) as connection:
        connection.executescript(old_schema)

    database = Database(database_path)

    result_columns = {
        row["name"] for row in database.rows("PRAGMA table_info(task_results)")
    }
    tool_columns = {
        row["name"] for row in database.rows("PRAGMA table_info(tool_calls)")
    }
    assert {
        "exit_reason",
        "image_evidence_json",
        "retrieval_relevance",
        "retrieval_conflict",
        "answer_source",
        "experiment_id",
        "retrieval_route",
        "retrieval_route_reason",
    } <= result_columns
    assert {"relevance_json", "relevance_label"} <= tool_columns
    assert database.scalar("SELECT value FROM schema_info WHERE key = 'version'") == "6"
