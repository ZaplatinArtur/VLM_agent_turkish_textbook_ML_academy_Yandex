from ...schemas.retrieve import RetrievedChunk

from .factory import get_chunk_store

_cache: list[RetrievedChunk] | None = None  # TODO: do smth better than global variable


def get_retrieved_chunks() -> list[RetrievedChunk]:
    global _cache
    if _cache is None:
        _cache = get_chunk_store().load()
    return _cache
