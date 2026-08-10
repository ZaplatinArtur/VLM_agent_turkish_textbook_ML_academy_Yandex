import math
from collections.abc import Sequence

from schemas.retrieve import RetrievedChunk

from .base import Ranker

DEFAULT_RRF_K = 60
DEFAULT_PRIMARY_CANDIDATE_WEIGHT = 1.0
DEFAULT_SEMANTIC_CANDIDATE_WEIGHT = 0.85
_SEMANTIC_CANDIDATE_MODES = frozenset({"union", "fallback"})


class PrimaryCandidateUnion(Ranker):
    """Add semantic candidates without displacing the primary lexical head."""

    def __init__(
        self,
        primary: Ranker,
        semantic: Ranker,
        *,
        primary_k: int = 32,
        semantic_k: int = 32,
        mode: str = "union",
        fallback_min_candidates: int = 5,
        rank_k: int = DEFAULT_RRF_K,
        primary_weight: float = DEFAULT_PRIMARY_CANDIDATE_WEIGHT,
        semantic_weight: float = DEFAULT_SEMANTIC_CANDIDATE_WEIGHT,
    ) -> None:
        if primary_k <= 0 or semantic_k <= 0:
            raise ValueError("candidate limits must be positive")
        if fallback_min_candidates <= 0:
            raise ValueError("fallback_min_candidates must be positive")
        if mode not in _SEMANTIC_CANDIDATE_MODES:
            raise ValueError("mode must be 'union' or 'fallback'")
        if rank_k <= 0:
            raise ValueError("rank_k must be positive")
        if (
            not math.isfinite(primary_weight)
            or not math.isfinite(semantic_weight)
            or primary_weight <= 0
            or semantic_weight <= 0
            or semantic_weight > primary_weight
        ):
            raise ValueError(
                "candidate weights must be finite, positive, and lexical-first"
            )
        self.primary = primary
        self.semantic = semantic
        self.primary_k = primary_k
        self.semantic_k = semantic_k
        self.mode = mode
        self.fallback_min_candidates = fallback_min_candidates
        self.rank_k = rank_k
        self.primary_weight = float(primary_weight)
        self.semantic_weight = float(semantic_weight)

    def build(self) -> None:
        for ranker in (self.primary, self.semantic):
            build = getattr(ranker, "build", None)
            if callable(build):
                build()

    def persist(self) -> None:
        for ranker in (self.primary, self.semantic):
            persist = getattr(ranker, "persist", None)
            if callable(persist):
                persist()

    @staticmethod
    def _unique(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        unique: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk in chunks:
            if not chunk.chunk_id or chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            unique.append(chunk)
        return unique

    def rank(
        self,
        query: str,
        chunks: list[RetrievedChunk] | None = None,
        subject: str | None = None,
        grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        primary = self._unique(
            self.primary.rank(
                query,
                chunks,
                subject=subject,
                grade=grade,
            )
        )
        if self.mode == "fallback" and len(primary) >= self.fallback_min_candidates:
            return primary

        semantic = self._unique(
            self.semantic.rank(
                query,
                chunks,
                subject=subject,
                grade=grade,
            )
        )
        if any(not math.isfinite(float(chunk.score)) for chunk in semantic):
            raise ValueError("semantic candidate generator returned a non-finite score")
        semantic.sort(key=lambda chunk: (-float(chunk.score), chunk.chunk_id))

        # Primary candidates own their positions even when semantic retrieval finds
        # the same IDs. New semantic candidates are inserted after the protected
        # lexical head so the downstream reranker can inspect both arms.
        primary_ids = {chunk.chunk_id for chunk in primary}
        semantic_only = [
            (position, chunk)
            for position, chunk in enumerate(semantic, 1)
            if chunk.chunk_id not in primary_ids
        ][: self.semantic_k]
        scored_primary = [
            chunk.model_copy(
                update={
                    "score": self.primary_weight / (self.rank_k + position)
                }
            )
            for position, chunk in enumerate(primary, 1)
        ]
        scored_semantic = [
            chunk.model_copy(
                update={
                    "score": self.semantic_weight / (self.rank_k + position)
                }
            )
            for position, chunk in semantic_only
        ]
        primary_head = scored_primary[: self.primary_k]
        primary_tail = scored_primary[self.primary_k :]
        return primary_head + scored_semantic + primary_tail


class ReciprocalRankFusion(Ranker):
    def __init__(
            self,
            rankers: Sequence[Ranker],
            rrf_k: int = DEFAULT_RRF_K,
            weights: Sequence[float] | None = None,
    ) -> None:
        if not rankers:
            raise ValueError("ReciprocalRankFusion requires at least one ranker")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if weights is not None and len(weights) != len(rankers):
            raise ValueError("weights must match the number of rankers")
        self.rankers = list(rankers)
        self.rrf_k = rrf_k
        self.weights = list(weights) if weights is not None else [1.0] * len(rankers)

    def build(self) -> None:
        for ranker in self.rankers:
            build = getattr(ranker, "build", None)
            if callable(build):
                build()

    def persist(self) -> None:
        for ranker in self.rankers:
            persist = getattr(ranker, "persist", None)
            if callable(persist):
                persist()

    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
            grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        scores = {}
        seen = {}
        for ranker, weight in zip(self.rankers, self.weights):
            for position, chunk in enumerate(
                    ranker.rank(query, chunks, subject=subject, grade=grade), start=1
            ):
                scores[chunk.chunk_id] = (
                        scores.get(chunk.chunk_id, 0.0) + weight / (self.rrf_k + position)
                )
                seen.setdefault(chunk.chunk_id, chunk)
        # sorted == ничьи разрешаются порядком первого вхождения
        ordered = sorted(seen.values(), key=lambda chunk: -scores[chunk.chunk_id])
        return [
            chunk.model_copy(update={"score": scores[chunk.chunk_id]})
            for chunk in ordered
        ]
