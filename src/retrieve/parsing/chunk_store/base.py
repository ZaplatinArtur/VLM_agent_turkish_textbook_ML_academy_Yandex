from abc import ABC, abstractmethod

from schemas.retrieve import RetrievedChunk


class ChunkStore(ABC):
    @abstractmethod
    def save_book(self, book_slug: str, chunks: list[RetrievedChunk]) -> None: ...

    @abstractmethod
    def load_book(self, book_slug: str) -> list[RetrievedChunk]: ...

    @abstractmethod
    def load(self) -> list[RetrievedChunk]: ...
