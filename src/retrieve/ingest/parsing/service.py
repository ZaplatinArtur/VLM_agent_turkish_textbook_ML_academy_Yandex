from functools import cache

from schemas.retrieve import RetrievedChunk

from .factory import get_chunk_store


@cache
def get_retrieved_chunks() -> tuple[RetrievedChunk, ...]:
    """Корпус целиком, читается с диска один раз за процесс."""
    return tuple(get_chunk_store().load())
