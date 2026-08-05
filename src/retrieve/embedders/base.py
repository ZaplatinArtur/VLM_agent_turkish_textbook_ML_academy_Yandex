from abc import ABC, abstractmethod
from typing import Protocol

from schemas.retrieve import RetrievedChunk


class Embedder(Protocol):
    def embed_chunks(self, chunks: list[RetrievedChunk]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class TextEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        pass

class SymmetricTextEmbedder(TextEmbedder):
    def embed_chunks(self, chunks: list[RetrievedChunk]) -> list[list[float]]:
        if not chunks:
            return []
        return self.encode([chunk.text for chunk in chunks])

    def embed_query(self, query: str) -> list[float]:
        return self.encode([query])[0]

class AsymmetricTextEmbedder(TextEmbedder):
    query_prefix: str = ""
    passage_prefix: str = ""

    def embed_chunks(self, chunks: list[RetrievedChunk]) -> list[list[float]]:
        if not chunks:
            return []
        return self.encode([self.passage_prefix + chunk.text for chunk in chunks])

    def embed_query(self, query: str) -> list[float]:
        return self.encode([self.query_prefix + query])[0]