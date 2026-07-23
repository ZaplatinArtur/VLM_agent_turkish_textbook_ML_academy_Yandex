from collections.abc import Sequence

from ...schemas.retrieve import RetrievedChunk

from .base import Ranker

DEFAULT_RRF_K = 60


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

    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
    ) -> list[RetrievedChunk]:
        scores = {}
        seen = {}
        for ranker, weight in zip(self.rankers, self.weights):
            for position, chunk in enumerate(
                    ranker.rank(query, chunks, subject=subject), start=1
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
