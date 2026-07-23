import threading
from pathlib import Path

from schemas.retrieve import RetrievedChunk

from ..cache import EmbeddingCache
from ..embedders import Embedder
from ..index import Index
from ..persistence import load_index, save_index
from ..vector_store import FaissVectorStore
from .base import Ranker


class DenseRanker(Ranker):
    def __init__(
            self,
            embedder: Embedder,
            index: Index | None = None,
            embedding_cache: EmbeddingCache | None = None,
            fetch_k: int = 200,
            index_dir: Path | str | None = None,
    ) -> None:
        self.embedder = embedder
        self.index = index or Index()
        self.embedding_cache = embedding_cache or EmbeddingCache()
        self.fetch_k = fetch_k
        self.index_dir = index_dir
        self._store: FaissVectorStore | None = None
        self._chunks_by_id = {}
        self._built = False
        self._build_lock = threading.Lock()

    @property
    def _embedder_name(self) -> str:
        return getattr(self.embedder, "model_name", type(self.embedder).__name__)

    def build(self) -> None:
        if self._built:
            return
        with self._build_lock:
            if self._built:
                return
            chunks = self.index.get()
            self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
            if not chunks:
                self._store = None
                self._built = True
                return
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            if self.index_dir is not None:
                store = load_index(self.index_dir, chunk_ids, self._embedder_name)
                if store is not None:
                    self._store = store
                    self._built = True
                    return
            vectors = self._embed_chunks(chunks)
            self._store = FaissVectorStore.from_vectors(chunk_ids, vectors)
            self._built = True
            self.persist()

    def persist(self) -> None:
        if self.index_dir is None or not isinstance(self._store, FaissVectorStore):
            return
        save_index(self.index_dir, self._store, self._embedder_name)

    def invalidate(self) -> None:
        self._built = False

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
        if not self._built:
            self.build()
        if self._store is None:
            return []
        allowed_ids = None  # None == поиск везде
        if subject is not None:
            allowed_ids = {chunk.chunk_id for chunk in self.index.get(subject)}
        if chunks:
            subset_ids = {chunk.chunk_id for chunk in chunks}
            allowed_ids = subset_ids if allowed_ids is None else allowed_ids & subset_ids
        if allowed_ids is not None and not allowed_ids:
            return []
        pool = len(allowed_ids) if chunks else self.fetch_k
        query_vector = self.embedder.embed_query(query)
        results = []
        for chunk_id, score in self._store.search(query_vector, pool, allowed_ids):
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(chunk.model_copy(update={"score": score}))
        return results
