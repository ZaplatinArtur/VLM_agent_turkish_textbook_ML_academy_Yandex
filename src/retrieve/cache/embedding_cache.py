from .base import CACHE_DIR, Cache


class EmbeddingCache(Cache):
    def __init__(self) -> None:
        super().__init__(CACHE_DIR / "embeddings")

    def get_embedding(self, chunk_id: str) -> list[float] | None:
        return self.get(chunk_id)

    def set_embedding(self, chunk_id: str, vector: list[float]) -> None:
        self.set(chunk_id, vector)
