from __future__ import annotations

import pytest

from retrieve.rankers.mmr import MaximalMarginalRelevanceRanker
from schemas.retrieve import RetrievedChunk


class FixedEmbedder:
    def __init__(self) -> None:
        self.vectors = {
            "query": [1.0, 0.0],
            "best": [1.0, 0.0],
            "duplicate": [0.99, 0.01],
            "diverse": [0.6, 0.8],
        }

    def embed_query(self, query: str) -> list[float]:
        return self.vectors[query]

    def embed_chunks(self, chunks: list[RetrievedChunk]) -> list[list[float]]:
        return [self.vectors[chunk.text] for chunk in chunks]


class DictCache:
    def __init__(self) -> None:
        self.store: dict[str, list[float]] = {}

    def get_embedding(self, chunk_id: str) -> list[float] | None:
        return self.store.get(chunk_id)

    def set_embedding(self, chunk_id: str, vector: list[float]) -> None:
        self.store[chunk_id] = vector


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=chunk_id,
        score=score,
        metadata={"page": 1},
    )


@pytest.fixture
def candidates() -> list[RetrievedChunk]:
    return [
        _chunk("best", 0.99),
        _chunk("duplicate", 0.98),
        _chunk("diverse", 0.70),
    ]


def test_mmr_promotes_diverse_candidate_after_best(candidates) -> None:
    ranker = MaximalMarginalRelevanceRanker(
        FixedEmbedder(),
        embedding_cache=DictCache(),
        lambda_mult=0.3,
    )

    results = ranker.rank("query", candidates)

    assert [chunk.chunk_id for chunk in results] == [
        "best",
        "diverse",
        "duplicate",
    ]


def test_mmr_lambda_one_reduces_to_query_relevance(candidates) -> None:
    ranker = MaximalMarginalRelevanceRanker(
        FixedEmbedder(),
        embedding_cache=DictCache(),
        lambda_mult=1.0,
    )

    assert [chunk.chunk_id for chunk in ranker.rank("query", candidates)] == [
        "best",
        "duplicate",
        "diverse",
    ]


def test_mmr_preserves_dense_scores_for_relevance_gate(candidates) -> None:
    ranker = MaximalMarginalRelevanceRanker(
        FixedEmbedder(),
        embedding_cache=DictCache(),
        lambda_mult=0.3,
    )

    results = ranker.rank("query", candidates)

    assert {chunk.chunk_id: chunk.score for chunk in results} == {
        "best": 0.99,
        "duplicate": 0.98,
        "diverse": 0.70,
    }
    assert results[0].chunk_id == "best"


def test_mmr_rejects_invalid_lambda() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        MaximalMarginalRelevanceRanker(FixedEmbedder(), lambda_mult=1.1)


def test_mmr_without_candidates_returns_empty() -> None:
    ranker = MaximalMarginalRelevanceRanker(FixedEmbedder(), lambda_mult=0.5)

    assert ranker.rank("query", []) == []
