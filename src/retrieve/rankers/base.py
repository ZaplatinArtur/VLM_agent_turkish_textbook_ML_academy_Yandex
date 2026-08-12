from typing import Protocol

from schemas.retrieve import RetrievedChunk


class Ranker(Protocol):
    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
    ) -> list[RetrievedChunk]: ...


def rescored(
        head: list[RetrievedChunk],
        tail: list[RetrievedChunk],
        scores: list[float],
) -> list[RetrievedChunk]:
    """Переставляет head по новым скорам; хвост за пределами top_n не трогаем."""
    scored = [
        chunk.model_copy(update={"score": score})
        for chunk, score in zip(head, scores)
    ]
    scored.sort(key=lambda chunk: -chunk.score)
    return scored + tail
