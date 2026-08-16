import json
from pathlib import Path

import pytest

from retrieve.evaluation.scoring import (
    evaluate_dense_mmr,
    prepare_qrels_from_agent_run,
    score_ranking,
)
from schemas.retrieve import RetrievedChunk


def _chunk(chunk_id: str, subject: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        images=[],
        score=0.0,
        metadata={
            "textbook": chunk_id.split(":", 1)[0],
            "page": int(chunk_id.rsplit(":", 1)[-1]),
            "subject": subject,
            "grade": 5,
        },
    )


class _DenseStub:
    def __init__(self, chunks: list[RetrievedChunk], orders: dict[str, list[str]]) -> None:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.orders = orders

    def rank(self, query: str, chunks=None, subject=None):
        del chunks, subject
        return [
            self.chunks[chunk_id].model_copy(update={"score": 1.0 / rank})
            for rank, chunk_id in enumerate(self.orders[query], start=1)
        ]


class _MmrStub:
    def __init__(self, orders: dict[str, list[str]]) -> None:
        self.orders = orders

    def rank(self, query: str, chunks=None, subject=None):
        del subject
        by_id = {chunk.chunk_id: chunk for chunk in chunks or []}
        return [by_id[chunk_id] for chunk_id in self.orders[query]]


class _PipelineStub:
    def __init__(self, dense, mmr) -> None:
        self.rankers = [dense, mmr]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_score_ranking_computes_recall_map_hit_and_mrr() -> None:
    scores = score_ranking(
        ["a", "b", "c", "c"],
        {"a", "c"},
        cutoffs=(1, 3, 5),
    )

    assert scores["1"] == {
        "hit": 1.0,
        "recall": 0.5,
        "average_precision": 1.0,
        "reciprocal_rank": 1.0,
    }
    assert scores["3"]["recall"] == 1.0
    assert scores["3"]["average_precision"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert scores["5"]["average_precision"] == scores["3"]["average_precision"]


def test_dense_mmr_report_uses_only_corpus_covered_qrels(tmp_path: Path) -> None:
    chunks = [
        _chunk("math-book:0001", "math"),
        _chunk("math-book:0002", "math"),
        _chunk("science-book:0003", "science"),
    ]
    dense_orders = {
        "q1": ["math-book:0001", "math-book:0002", "science-book:0003"],
        "q2": ["math-book:0002", "math-book:0001", "science-book:0003"],
        "q3": ["math-book:0001", "math-book:0002", "science-book:0003"],
        "q4": ["math-book:0001", "math-book:0002", "science-book:0003"],
        "q5": ["math-book:0002", "math-book:0001", "science-book:0003"],
        "q7": ["math-book:0001", "math-book:0002", "science-book:0003"],
    }
    mmr_orders = {
        "q1": ["math-book:0002", "math-book:0001", "science-book:0003"],
        "q2": dense_orders["q2"],
        "q3": dense_orders["q3"],
        "q4": ["science-book:0003", "math-book:0001", "math-book:0002"],
        "q5": ["math-book:0001", "math-book:0002", "science-book:0003"],
        "q7": ["math-book:0002", "math-book:0001", "science-book:0003"],
    }
    qrels = tmp_path / "qrels.jsonl"
    _write_jsonl(
        qrels,
        [
            {
                "query_id": "1",
                "query": "q1",
                "subject": "Mathematics",
                "relevant_chunk_ids": ["math-book:0001", "science-book:0003"],
            },
            {
                "query_id": "2",
                "query": "q2",
                "subject": "Math",
                "relevant_chunk_ids": ["missing:0001"],
            },
            {
                "query_id": "3",
                "query": "q3",
                "subject": "History",
                "relevant_chunk_ids": [],
            },
            {
                "query_id": "4",
                "query": "q4",
                "subject": "Science",
                "relevant_chunk_ids": ["science-book:0003"],
            },
            {
                "query_id": "5",
                "query": "q5",
                "subject": "Math",
                "relevant_chunk_ids": ["math-book:0002", "missing:0002"],
            },
            {
                "query_id": "6",
                "query": "",
                "subject": "Math",
                "relevant_chunk_ids": ["math-book:0001"],
            },
            {
                "query_id": "7",
                "query": "q7",
                "subject": "Math",
                "relevant_chunk_ids": [],
                "annotation_status": "complete",
            },
        ],
    )
    pipeline = _PipelineStub(
        _DenseStub(chunks, dense_orders),
        _MmrStub(mmr_orders),
    )

    report = evaluate_dense_mmr(
        qrels,
        cutoffs=(1, 2, 3),
        fetch_k=3,
        chunks=chunks,
        pipeline=pipeline,
        candidate_text_chars=20,
    )

    assert report["coverage"] == {
        "total_qrels": 7,
        "scored_qrels": 3,
        "missing_query_qrels": 1,
        "unannotated_qrels": 1,
        "uncovered_qrels": 2,
        "partially_covered_qrels": 1,
        "fully_covered_qrels": 2,
    }
    assert report["variants"]["dense"]["metrics"]["recall_at_3"] == 1.0
    assert report["variants"]["dense_mmr"]["metrics"]["recall_at_3"] == 1.0
    assert report["variants"]["dense"]["by_subject"]["math"]["queries"] == 2
    assert report["variants"]["dense"]["by_subject"]["science"]["queries"] == 1
    assert report["per_query"][1]["coverage_status"] == "uncovered"
    assert report["per_query"][4]["evaluated_relevant_ids"] == ["math-book:0002"]
    for row in report["per_query"]:
        assert {
            candidate["chunk_id"] for candidate in row["dense"]["candidates"]
        } == {
            candidate["chunk_id"] for candidate in row["dense_mmr"]["candidates"]
        }
    assert report["per_query"][5]["coverage_status"] == "missing_query"
    assert report["per_query"][6]["coverage_status"] == "uncovered"


def test_fetch_k_must_cover_largest_metric_cutoff(tmp_path: Path) -> None:
    qrels = tmp_path / "qrels.jsonl"
    _write_jsonl(qrels, [{"query": "q", "relevant_chunk_ids": ["book:0001"]}])

    with pytest.raises(ValueError, match="largest cutoff"):
        evaluate_dense_mmr(
            qrels,
            cutoffs=(1, 5),
            fetch_k=4,
            chunks=[_chunk("book:0001", "math")],
            pipeline=_PipelineStub(None, None),
        )


def test_prepare_qrels_freezes_first_agent_search_query(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    run = tmp_path / "run.jsonl"
    output = tmp_path / "qrels.jsonl"
    _write_jsonl(
        tasks,
        [
            {"task_id": "t1", "subject": "History", "grade": 8},
            {"task_id": "t2", "subject": "Math", "grade": 5},
        ],
    )
    _write_jsonl(
        run,
        [
            {
                "task_id": "t1",
                "tool_calls": [
                    {
                        "tool": "search_textbooks",
                        "args": {"query": "first query", "subject": "History", "grade": 8},
                    },
                    {"tool": "search_textbooks", "args": {"query": "rewrite"}},
                ],
            },
            {"task_id": "t2", "tool_calls": []},
        ],
    )

    summary = prepare_qrels_from_agent_run(tasks, run, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["with_agent_query"] == 1
    assert summary["needs_manual_query"] == 1
    assert rows[0]["query"] == "first query"
    assert rows[0]["subject"] == "History"
    assert rows[0]["retrieval_subject"] == "History"
    assert rows[0]["retrieval_grade"] == 8
    assert rows[0]["relevant_chunk_ids"] == []
    assert rows[1]["needs_manual_query"] is True
