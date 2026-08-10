from __future__ import annotations

import math

from schemas.retrieve import RetrievedChunk

from ..cache import EmbeddingCache
from ..embedders import Embedder
from .base import Ranker


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def maximal_marginal_relevance_order(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    *,
    lambda_mult: float,
) -> list[int]:
    """Return a deterministic MMR permutation of all candidate indices."""
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("lambda_mult must be between 0 and 1")
    if not candidate_embeddings:
        return []

    query_scores = [
        _cosine(query_embedding, embedding)
        for embedding in candidate_embeddings
    ]
    pair_scores = [
        [_cosine(left, right) for right in candidate_embeddings]
        for left in candidate_embeddings
    ]
    remaining = list(range(len(candidate_embeddings)))
    selected: list[int] = []
    while remaining:
        if not selected:
            best = max(remaining, key=lambda index: query_scores[index])
        else:
            best = max(
                remaining,
                key=lambda index: (
                    lambda_mult * query_scores[index]
                    - (1.0 - lambda_mult)
                    * max(pair_scores[index][chosen] for chosen in selected)
                ),
            )
        selected.append(best)
        remaining.remove(best)
    return selected


class MaximalMarginalRelevanceRanker(Ranker):
    """Diversify dense candidates while preserving their original scores."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        lambda_mult: float = 0.5,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be between 0 and 1")
        self.embedder = embedder
        self.lambda_mult = lambda_mult
        self.embedding_cache = embedding_cache or EmbeddingCache()

    def _candidate_embeddings(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[list[float]]:
        cached: dict[str, list[float]] = {}
        missing: list[RetrievedChunk] = []
        for chunk in chunks:
            vector = self.embedding_cache.get_embedding(chunk.chunk_id)
            if vector is None:
                missing.append(chunk)
            else:
                cached[chunk.chunk_id] = vector
        if missing:
            for chunk, vector in zip(missing, self.embedder.embed_chunks(missing)):
                self.embedding_cache.set_embedding(chunk.chunk_id, vector)
                cached[chunk.chunk_id] = vector
        return [cached[chunk.chunk_id] for chunk in chunks]

    def rank(
        self,
        query: str,
        chunks: list[RetrievedChunk] | None = None,
        subject: str | None = None,
        grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        del subject, grade
        candidates = list(chunks or [])
        if not candidates:
            return []
        order = maximal_marginal_relevance_order(
            self.embedder.embed_query(query),
            self._candidate_embeddings(candidates),
            lambda_mult=self.lambda_mult,
        )
        return [candidates[index] for index in order]
