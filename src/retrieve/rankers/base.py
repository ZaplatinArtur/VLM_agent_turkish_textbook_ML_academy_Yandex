from typing import Protocol

from schemas.retrieve import RetrievedChunk


class Ranker(Protocol):
    k: int

    def rank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        subject: str | None = None,
    ) -> list[RetrievedChunk]: ...
