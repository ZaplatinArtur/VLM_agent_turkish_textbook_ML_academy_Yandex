from typing import Protocol

from ...schemas.retrieve import RetrievedChunk


class Ranker(Protocol):
    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
    ) -> list[RetrievedChunk]: ...
