from typing import Protocol

from schemas.retrieve import RetrievedChunk


class Ranker(Protocol):
    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
            grade: int | str | None = None,
    ) -> list[RetrievedChunk]: ...


def rescored(
        head: list[RetrievedChunk],
        tail: list[RetrievedChunk],
        scores: list[float],
) -> list[RetrievedChunk]:
    """Return a stably score-sorted head followed by the untouched tail."""
    if len(head) != len(scores):
        raise ValueError("reranker returned a different number of scores")
    scored = [
        chunk.model_copy(update={"score": float(score)})
        for chunk, score in zip(head, scores)
    ]
    # Python's sort is stable, so equal scores preserve the candidate order.
    scored.sort(key=lambda chunk: -chunk.score)
    return scored + tail
