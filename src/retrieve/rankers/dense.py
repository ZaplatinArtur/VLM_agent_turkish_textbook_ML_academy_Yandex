from collections.abc import Callable

from ...schemas.retrieve import RetrievedChunk

from ..cache import EmbeddingCache
from ..embedders import Embedder
from ..index import Index
from ..vector_store import FaissVectorStore, VectorStore
from .base import Ranker


class DenseRanker(Ranker):
    def __init__(
        self,
        embedder: Embedder,
        index: Index | None = None,
        embedding_cache: EmbeddingCache | None = None,
        fetch_k: int = 200,
        store_factory: Callable[
            [list[str], list[list[float]]], VectorStore
        ] = FaissVectorStore.from_vectors,
    ) -> None:
        self.embedder = embedder
        self.index = index or Index()
        self.embedding_cache = embedding_cache or EmbeddingCache()
        self.fetch_k = fetch_k
        self._store_factory = store_factory
        self._store: VectorStore | None = None
        self._chunks_by_id = {}
        self._built = False

    def build(self) -> None:
        chunks = self.index.get()
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if chunks:
            vectors = self._embed_chunks(chunks)
            self._store = self._store_factory([chunk.chunk_id for chunk in chunks], vectors)
        else:
            self._store = None
        self._built = True

    def invalidate(self) -> None:
        self._built = False

    def _ensure_built(self) -> None:
        if not self._built:
            self.build()

    def _embed_chunks(self, chunks: list[RetrievedChunk]) -> list[list[float]]:
        cached = {}
        missing = []
        for chunk in chunks:
            vector = self.embedding_cache.get_embedding(chunk.chunk_id)
            if vector is None:
                missing.append(chunk)
            else:
                cached[chunk.chunk_id] = vector
        if missing:
            for chunk, vector in zip(missing, self.embedder.embed_chunks(missing)):
                self.embedding_cache.set_embedding(chunk.chunk_id, vector)
                cached[chunk.chunk_id] = vector
        return [cached[chunk.chunk_id] for chunk in chunks]

    def rank(
        self,
        query: str,
        chunks: list[RetrievedChunk] | None = None,
        subject: str | None = None,
    ) -> list[RetrievedChunk]:
        self._ensure_built()
        if self._store is None:
            return []
        allowed_ids = None  # None means that the whole index is searchable.
        if subject is not None:
            allowed_ids = {chunk.chunk_id for chunk in self.index.get(subject)}
        if chunks:
            subset_ids = {chunk.chunk_id for chunk in chunks}
            allowed_ids = subset_ids if allowed_ids is None else allowed_ids & subset_ids
        limit = None if chunks else self.fetch_k
        pool = len(self._chunks_by_id) if allowed_ids is not None else self.fetch_k
        query_vector = self.embedder.embed_query(query)
        results = []
        for chunk_id, score in self._store.search(query_vector, pool):
            if allowed_ids is not None and chunk_id not in allowed_ids:
                continue
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(chunk.model_copy(update={"score": score}))
            if limit is not None and len(results) >= limit:
                break
        return results
