from ..schemas.retrieve import RetrievedChunk
from .cache import ParsingCache


class Index:
    def __init__(self, cache: ParsingCache | None = None) -> None:
        self.cache = cache or ParsingCache()

    def add(self, chunks: list[RetrievedChunk]) -> None:
        self.cache.set_chunks(chunks)

    def get(self, subject: str | None = None) -> list[RetrievedChunk]:
        chunks = self.cache.get_chunks()
        if subject is None:
            return chunks
        return [c for c in chunks if c.metadata.get("subject") == subject]

    def get_by_id(self, chunk_id: str) -> RetrievedChunk | None:
        return self.cache.get_chunk(chunk_id)
