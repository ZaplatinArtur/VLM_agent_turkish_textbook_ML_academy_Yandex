from ...schemas.retrieve import RetrievedChunk
from .base import CACHE_DIR, Cache


class ParsingCache(Cache):
    def __init__(self) -> None:
        super().__init__(CACHE_DIR / "parsing")

    def get_chunk(self, chunk_id: str) -> RetrievedChunk | None:
        return self.get(chunk_id)

    def set_chunk(self, chunk: RetrievedChunk) -> None:
        self.set(chunk.chunk_id, chunk)

    def get_chunks(self) -> list[RetrievedChunk]:
        return list(self.values())

    def set_chunks(self, chunks: list[RetrievedChunk]) -> None:
        for chunk in chunks:
            self.set_chunk(chunk)
