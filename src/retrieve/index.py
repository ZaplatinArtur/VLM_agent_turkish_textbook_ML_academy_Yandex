from ..schemas.retrieve import RetrievedChunk


class Index:
    """In-memory view of chunks loaded from the parser's JSONL store."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks_by_id: dict[str, RetrievedChunk] = {}
        if chunks:
            self.add(chunks)

    def add(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks_by_id.update({chunk.chunk_id: chunk for chunk in chunks})

    def get(self, subject: str | None = None) -> list[RetrievedChunk]:
        chunks = list(self._chunks_by_id.values())
        if subject is None:
            return chunks
        return [
            chunk
            for chunk in chunks
            if chunk.metadata.get("subject") == subject
        ]

    def get_by_id(self, chunk_id: str) -> RetrievedChunk | None:
        return self._chunks_by_id.get(chunk_id)
