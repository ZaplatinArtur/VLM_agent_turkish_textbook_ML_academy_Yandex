from typing import Protocol

from ...schemas.retrieve import RetrievedChunk


class Embedder(Protocol):
    def embed_chunks(self, chunks: list[RetrievedChunk]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...
