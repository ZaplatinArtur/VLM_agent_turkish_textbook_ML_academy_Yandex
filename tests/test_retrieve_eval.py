import math

import pytest

from retrieve.evaluation.evaluate import evaluate_pipeline
from schemas.retrieve import RetrievedChunk


class StubPipeline:
    """Отдаёт фиксированный порядок chunk_id независимо от запроса."""

    def __init__(self, order: list[str], metadata: dict | None = None) -> None:
        self.order = order
        self.metadata = metadata or {}

    def run(self, query: str, k: int, subject=None) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(chunk_id=cid, text=cid, score=1.0, metadata=self.metadata)
            for cid in self.order[:k]
        ]


def test_metrics_match_hand_computed_values():
    # Релевантен только "b"; выдача a,b,c → b на 2-й позиции.
    qrels = [{"query_id": "1", "query": "q", "relevant_chunk_ids": ["b"]}]
    report = evaluate_pipeline(StubPipeline(["a", "b", "c"]), qrels, ks=[1, 3])
    metrics = report["metrics"]

    # @1: b ещё не найден.
    assert metrics["hit_rate_at_1"] == 0.0
    assert metrics["recall_at_1"] == 0.0
    assert metrics["mrr_at_1"] == 0.0
    # @3: найден на позиции 2.
    assert metrics["hit_rate_at_3"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr_at_3"] == pytest.approx(0.5)
    assert metrics["ndcg_at_3"] == pytest.approx(1.0 / math.log2(3))


def test_perfect_ranking_scores_one():
    qrels = [{"query_id": "1", "query": "q", "relevant_chunk_ids": ["a"]}]
    report = evaluate_pipeline(StubPipeline(["a", "b"]), qrels, ks=[1])
    assert report["metrics"]["ndcg_at_1"] == pytest.approx(1.0)
    assert report["metrics"]["mrr_at_1"] == pytest.approx(1.0)


def test_page_id_target_used_when_no_chunk_ids():
    qrels = [{"query_id": "1", "query": "q", "relevant_page_ids": ["p7"]}]
    pipeline = StubPipeline(["a"], metadata={"page_id": "p7"})
    report = evaluate_pipeline(pipeline, qrels, ks=[1])
    assert report["metrics"]["hit_rate_at_1"] == 1.0


def test_missing_relevant_ids_raise():
    with pytest.raises(ValueError):
        evaluate_pipeline(StubPipeline(["a"]), [{"query_id": "1", "query": "q"}], ks=[1])
