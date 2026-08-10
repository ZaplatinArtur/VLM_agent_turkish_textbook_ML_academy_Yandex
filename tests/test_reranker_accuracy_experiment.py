import json
import zipfile

import pytest
from langchain_core.messages import AIMessage

from mla_baseline.config import Settings
from mla_baseline.contracts import Task
from mla_baseline.reranker_accuracy_experiment import (
    DEFAULT_ARMS,
    FrozenRerankerSolver,
    build_context_records,
    load_rankings_archive,
    summarize_judges,
    validate_ranking_arms,
)


class FakeLlm:
    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        return AIMessage(
            content='{"solution_steps":"used frozen ranking","final_answer":"A"}',
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )


def _tasks():
    return [
        Task(
            task_id="task-1",
            subject="math",
            grade=5,
            question="question one",
            reference_answer="A",
            answer_type="choice",
        ),
        Task(
            task_id="task-2",
            subject="science",
            grade=6,
            question="question two",
            reference_answer="B",
            answer_type="choice",
        ),
    ]


def _hit(chunk_id, rank, score):
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "page_id": chunk_id,
        "score": score,
        "subject": "math",
        "grade": 5,
        "textbook": "book",
        "page": rank,
        "text": f"long textbook text for {chunk_id}",
    }


def _rows(order):
    scores = {"c1": 0.9, "c2": 0.8, "c3": 0.7}
    return [
        {
            "task_id": "task-1",
            "query": "rectangle area",
            "subject": "math",
            "grade": 5,
            "rankings": [
                _hit(chunk_id, rank, scores[chunk_id])
                for rank, chunk_id in enumerate(order, 1)
            ],
        },
        {
            "task_id": "task-2",
            "query": "plants",
            "subject": "science",
            "grade": 6,
            "rankings": [],
        },
    ]


def _write_archive(path):
    rows = {
        "dense": _rows(["c1", "c2", "c3"]),
        "gte_multilingual": _rows(["c2", "c1", "c3"]),
        "bge_v2_m3": _rows(["c3", "c2", "c1"]),
    }
    filenames = {
        "dense": "rankings_dense.jsonl",
        "gte_multilingual": "rankings_gte_multilingual.jsonl",
        "bge_v2_m3": "rankings_bge_v2_m3.jsonl",
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "experiment_manifest.json",
            json.dumps({"tasks": 2, "rerankable_tasks": 1}),
        )
        for arm, arm_rows in rows.items():
            archive.writestr(
                filenames[arm],
                "".join(json.dumps(row) + "\n" for row in arm_rows),
            )


def _write_judge(path, values):
    path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "verdict": {"strict_correct": strict_correct},
                }
            )
            + "\n"
            for task_id, strict_correct in values.items()
        ),
        encoding="utf-8",
    )


def test_archive_builds_same_task_set_and_keeps_empty_context(tmp_path):
    archive_path = tmp_path / "rankings.zip"
    _write_archive(archive_path)

    _, rows_by_arm = load_rankings_archive(archive_path)
    indexed, rerankable = validate_ranking_arms(
        tasks=_tasks(),
        rows_by_arm=rows_by_arm,
    )
    records = build_context_records(
        tasks=_tasks(),
        rankings_by_task=indexed["gte_multilingual"],
        arm="gte_multilingual",
        top_k=2,
        max_text_chars=100,
    )

    assert set(indexed) == set(DEFAULT_ARMS)
    assert rerankable == {"task-1"}
    assert [record["task_id"] for record in records] == ["task-1", "task-2"]
    assert [
        hit["chunk_id"] for hit in records[0]["payload"]["hits"]
    ] == ["c2", "c1"]
    assert sum(len(hit["text"]) for hit in records[0]["payload"]["hits"]) <= 100
    assert records[1]["payload"]["hits"] == []


def test_validation_rejects_a_reranker_with_a_different_candidate_pool(tmp_path):
    archive_path = tmp_path / "rankings.zip"
    _write_archive(archive_path)
    _, rows_by_arm = load_rankings_archive(archive_path)
    rows_by_arm["gte_multilingual"][0]["rankings"][0]["chunk_id"] = "other"

    with pytest.raises(ValueError, match="candidate pool differs from dense"):
        validate_ranking_arms(tasks=_tasks(), rows_by_arm=rows_by_arm)


def test_frozen_solver_records_the_reranker_arm(tmp_path):
    archive_path = tmp_path / "rankings.zip"
    _write_archive(archive_path)
    _, rows_by_arm = load_rankings_archive(archive_path)
    indexed, _ = validate_ranking_arms(tasks=_tasks(), rows_by_arm=rows_by_arm)
    records = build_context_records(
        tasks=_tasks(),
        rankings_by_task=indexed["gte_multilingual"],
        arm="gte_multilingual",
        top_k=2,
        max_text_chars=100,
    )
    settings = Settings(
        _env_file=None,
        structured_mode="none",
        prompt_version="v1",
    )

    result = FrozenRerankerSolver(
        settings,
        records=records,
        arm="gte_multilingual",
        llm=FakeLlm(),
    ).solve(_tasks()[0])

    assert result.error is None
    assert result.condition == "agent_rag_frozen_gte_multilingual"
    assert result.generation["reranker_arm"] == "gte_multilingual"
    assert result.tool_calls[0].returned_chunk_ids == ["c2", "c1"]


def test_summary_reports_accuracy_and_paired_changes_on_both_slices(tmp_path):
    dense = tmp_path / "dense.jsonl"
    gte = tmp_path / "gte.jsonl"
    bge = tmp_path / "bge.jsonl"
    _write_judge(dense, {"task-1": False, "task-2": True})
    _write_judge(gte, {"task-1": True, "task-2": True})
    _write_judge(bge, {"task-1": False, "task-2": False})

    summary = summarize_judges(
        judge_paths={
            "dense": dense,
            "gte_multilingual": gte,
            "bge_v2_m3": bge,
        },
        rerankable_task_ids={"task-1"},
    )

    assert summary["accuracy"]["dense"]["all_evaluated"]["accuracy"] == 0.5
    assert summary["accuracy"]["gte_multilingual"]["all_evaluated"][
        "accuracy"
    ] == 1.0
    assert summary["accuracy"]["bge_v2_m3"]["rerankable_only"][
        "accuracy"
    ] == 0.0
    assert summary["accuracy"]["gte_multilingual"]["unchanged_input_control"][
        "accuracy"
    ] == 1.0
    assert summary["comparisons_vs_dense"]["gte_multilingual"][
        "rerankable_only"
    ]["fixed"] == 1
    assert summary["comparisons_vs_dense"]["bge_v2_m3"]["all_evaluated"][
        "regressed"
    ] == 1
    assert summary["comparisons_vs_dense"]["gte_multilingual"][
        "unchanged_input_control"
    ]["fixed"] == 0
