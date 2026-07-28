from abc import ABC, abstractmethod
from typing import Protocol

from schemas.retrieve import RetrievedChunk


class Embedder(Protocol):
    def embed_chunks(self, chunks: list[RetrievedChunk]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class SymmetricTextEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        pass

    def embed_chunks(self, chunks: list[RetrievedChunk]) -> list[list[float]]:
        if not chunks:
            return []
        return self.encode(
            [
                str(chunk.metadata.get("retrieval_text") or chunk.text)
                for chunk in chunks
            ]
        )

    def embed_query(self, query: str) -> list[float]:
        return self.encode([query])[0]
