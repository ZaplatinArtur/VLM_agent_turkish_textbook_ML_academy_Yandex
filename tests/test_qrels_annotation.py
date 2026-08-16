import json
from pathlib import Path

import pytest

from retrieve.evaluation.qrels_annotation import (
    annotate_candidate_pool,
    build_candidate_pool,
    parse_annotation,
)
from schemas.retrieve import RetrievedChunk
from vlm_judge.backends import ReplayBackend


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        images=[],
        score=0.0,
        metadata={"textbook": "book", "page": int(chunk_id[-1]), "subject": "math", "grade": 5},
    )


class _DenseStub:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def rank(self, query: str, chunks=None, subject=None):
        del query, chunks, subject
        return [
            chunk.model_copy(update={"score": 1.0 / rank})
            for rank, chunk in enumerate(self.chunks, start=1)
        ]


def test_parse_annotation_validates_candidate_ids() -> None:
    parsed = parse_annotation(
        json.dumps(
            {
                "relevant_chunk_ids": ["c1"],
                "uncertain_chunk_ids": ["c2", "c1"],
                "confidence": 0.8,
                "rationale": "c1 states the required formula",
            }
        ),
        {"c1", "c2"},
    )
    assert parsed["relevant_chunk_ids"] == ["c1"]
    assert parsed["uncertain_chunk_ids"] == ["c2"]

    with pytest.raises(ValueError, match="unknown IDs"):
        parse_annotation(
            '{"relevant_chunk_ids":["missing"],"uncertain_chunk_ids":[],"confidence":1,"rationale":"x"}',
            {"c1"},
        )


def test_build_candidate_pool_combines_dense_and_bm25(tmp_path: Path) -> None:
    chunks = [
        _chunk("book:1", "triangle area base height"),
        _chunk("book:2", "fraction addition equal denominators"),
    ]
    qrels = tmp_path / "qrels.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    run = tmp_path / "run.jsonl"
    output = tmp_path / "pool.jsonl"
    _write_jsonl(
        qrels,
        [
            {"task_id": "t1", "query": "fraction denominators", "retrieval_subject": None},
            {"task_id": "t2", "query": "", "retrieval_subject": None},
        ],
    )
    _write_jsonl(
        tasks,
        [
            {"task_id": "t1", "subject": "Math", "grade": 5, "reference_answer": "1/2"},
            {"task_id": "t2", "subject": "Math", "grade": 5, "reference_answer": "6"},
        ],
    )
    _write_jsonl(
        run,
        [
            {"task_id": "t1", "image_evidence": ["add fractions"]},
            {"task_id": "t2", "image_evidence": ["triangle area base height"]},
        ],
    )

    summary = build_candidate_pool(
        qrels,
        tasks,
        run,
        output,
        bm25_source_path=tmp_path / "bm25.jsonl",
        bm25_index_path=tmp_path / "bm25.sqlite",
        dense_k=2,
        bm25_k=2,
        chunks=chunks,
        dense_ranker=_DenseStub(chunks),
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["tasks"] == 2
    assert summary["fallback_queries"] == 1
    assert rows[0]["candidates"][0]["dense_rank"] == 1
    assert any(candidate["bm25_rank"] == 1 for candidate in rows[0]["candidates"])
    assert rows[1]["query_source"] == "image_evidence_fallback"


def test_annotation_is_resumable_and_writes_completed_qrels(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    output = tmp_path / "annotated.jsonl"
    _write_jsonl(
        pool,
        [
            {
                "task_id": "t1",
                "query": "triangle area",
                "subject": "Math",
                "grade": 5,
                "retrieval_subject": None,
                "retrieval_grade": None,
                "reference_answer": "6",
                "image_evidence": ["base 3", "height 4"],
                "pool": {"dense_k": 200, "bm25_k": 100},
                "candidates": [{"chunk_id": "c1", "text": "area = base * height / 2"}],
            }
        ],
    )
    response = json.dumps(
        {
            "relevant_chunk_ids": ["c1"],
            "uncertain_chunk_ids": [],
            "confidence": 0.9,
            "rationale": "The chunk states the necessary formula.",
        }
    )
    first = annotate_candidate_pool(pool, output, ReplayBackend([response]), limit=1)
    second = annotate_candidate_pool(pool, output, ReplayBackend([]), limit=1)
    row = json.loads(output.read_text(encoding="utf-8"))

    assert first["completed"] == 1
    assert second["processed"] == 0
    assert row["annotation_status"] == "complete"
    assert row["relevant_chunk_ids"] == ["c1"]
