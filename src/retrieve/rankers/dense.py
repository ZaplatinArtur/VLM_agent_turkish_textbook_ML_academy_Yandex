import math
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from schemas.retrieve import RetrievedChunk

from ..cache import EmbeddingCache
from ..embedders import Embedder
from ..index import Index
from ..storage.persistence import (
    IndexValidationError,
    StrictBuildLock,
    acquire_strict_build_lock,
    load_index,
    provenance_fingerprint,
    retrieval_chunk_projection_sha256,
    retrieval_corpus_projection_sha256,
    save_index,
)
from ..storage.vector_store import (
    FaissVectorStore,
    IndexKind,
    resolve_index_kind,
)
from .base import Ranker


class DenseRanker(Ranker):
    def __init__(
            self,
            embedder: Embedder,
            index: Index | None = None,
            embedding_cache: EmbeddingCache | None = None,
            fetch_k: int = 200,
            index_dir: Path | str | None = None,
            strict_provenance: bool = False,
            expected_embedder_provenance: Mapping[str, object] | None = None,
            expected_embedding_dimension: int | None = None,
            index_kind: IndexKind | str | None = None,
    ) -> None:
        if type(strict_provenance) is not bool:
            raise ValueError("strict_provenance must be a boolean")
        if type(fetch_k) is not int or fetch_k <= 0:
            raise ValueError("fetch_k must be positive")
        if expected_embedder_provenance is not None and not strict_provenance:
            raise ValueError(
                "expected_embedder_provenance requires strict_provenance=True"
            )
        if expected_embedding_dimension is not None and (
            type(expected_embedding_dimension) is not int
            or expected_embedding_dimension <= 0
        ):
            raise ValueError("expected_embedding_dimension must be positive")
        if strict_provenance and not expected_embedder_provenance:
            raise ValueError(
                "strict dense retrieval requires expected embedder provenance"
            )
        if strict_provenance and expected_embedding_dimension is None:
            raise ValueError(
                "strict dense retrieval requires expected_embedding_dimension"
            )
        if strict_provenance and index_dir is None:
            raise ValueError("strict dense retrieval requires index_dir")
        expected_kind_value = (
            expected_embedder_provenance.get("faiss_index_kind")
            if expected_embedder_provenance is not None
            else None
        )
        requested_kind_value = (
            index_kind if index_kind is not None else expected_kind_value
        )
        try:
            requested_index_kind = (
                IndexKind(requested_kind_value)
                if requested_kind_value is not None
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("index_kind must be 'auto', 'flat', or 'hnsw'") from exc
        if strict_provenance and requested_index_kind is None:
            raise ValueError("strict dense retrieval requires index_kind")
        if (
            strict_provenance
            and expected_kind_value != requested_index_kind.value
        ):
            raise ValueError(
                "strict index_kind does not match expected embedder provenance"
            )
        self.embedder = embedder
        self.index = index or Index()
        self.embedding_cache = embedding_cache or EmbeddingCache()
        self.fetch_k = fetch_k
        self.index_dir = index_dir
        self.strict_provenance = bool(strict_provenance)
        self.expected_embedder_provenance = (
            dict(expected_embedder_provenance)
            if expected_embedder_provenance is not None
            else None
        )
        self.expected_embedding_dimension = expected_embedding_dimension
        self.index_kind = requested_index_kind
        self._store: FaissVectorStore | None = None
        self._chunks_by_id = {}
        self._corpus_projection_sha256: str | None = None
        self._strict_build_lock: StrictBuildLock | None = None
        self._persisted = False
        self._built = False
        self._build_lock = threading.Lock()

    @property
    def _embedder_name(self) -> str:
        return getattr(self.embedder, "model_name", type(self.embedder).__name__)

    def _embedder_provenance(self) -> dict[str, object] | None:
        if not self.strict_provenance:
            return None
        raw = getattr(self.embedder, "provenance", None)
        if callable(raw):
            raw = raw()
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(
                "strict dense retrieval requires non-empty embedder provenance"
            )
        provenance = dict(raw)
        if provenance.get("embedding_dimension") != self.expected_embedding_dimension:
            raise ValueError(
                "embedder provenance has the wrong embedding dimension"
            )
        if self.strict_provenance and provenance.get("faiss_index_kind") != (
            self.index_kind.value
        ):
            raise ValueError("embedder provenance has the wrong FAISS index kind")
        actual_fingerprint = provenance_fingerprint(provenance)
        if self.expected_embedder_provenance is not None:
            expected_fingerprint = provenance_fingerprint(
                self.expected_embedder_provenance
            )
            if actual_fingerprint != expected_fingerprint:
                raise ValueError(
                    "embedder provenance does not match the configured immutable pin"
                )
        return provenance

    def _embedding_cache_key(self, chunk: RetrievedChunk) -> str:
        provenance = self._embedder_provenance()
        if provenance is None:
            # Migration guardrail: the default/legacy arm keeps its old cache keys.
            return chunk.chunk_id
        return (
            f"v2:{provenance_fingerprint(provenance)}:"
            f"{retrieval_chunk_projection_sha256(chunk)}:{chunk.chunk_id}"
        )

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
            provenance = self._embedder_provenance()
            self._corpus_projection_sha256 = (
                retrieval_corpus_projection_sha256(chunks)
                if self.strict_provenance
                else None
            )
            if self.index_dir is not None:
                store = load_index(
                    self.index_dir,
                    chunk_ids,
                    self._embedder_name,
                    embedder_provenance=provenance,
                    corpus_projection_sha256=self._corpus_projection_sha256,
                    require_strict_manifest=self.strict_provenance,
                )
                if store is not None:
                    self._store = store
                    self._persisted = True
                    self._built = True
                    return
                if self.strict_provenance:
                    self._strict_build_lock = acquire_strict_build_lock(
                        self.index_dir
                    )
            vectors = self._embed_chunks(chunks)
            if self.strict_provenance:
                resolved_kind = resolve_index_kind(self.index_kind, len(chunk_ids))
                self._store = FaissVectorStore.from_vectors(
                    chunk_ids,
                    vectors,
                    kind=resolved_kind,
                )
            elif self.index_kind is None:
                # Preserve legacy process-global MLA_FAISS_INDEX_KIND behavior.
                self._store = FaissVectorStore.from_vectors(chunk_ids, vectors)
            else:
                self._store = FaissVectorStore.from_vectors(
                    chunk_ids,
                    vectors,
                    kind=self.index_kind,
                )
            if self.strict_provenance:
                # A strict ranker is not usable until its provenance-bound
                # snapshot has been written successfully.
                self.persist()
                if self.index_dir is not None:
                    validated = load_index(
                        self.index_dir,
                        chunk_ids,
                        self._embedder_name,
                        embedder_provenance=provenance,
                        corpus_projection_sha256=self._corpus_projection_sha256,
                        require_strict_manifest=True,
                        active_build_lock=self._strict_build_lock,
                    )
                    if validated is None:
                        raise IndexValidationError(
                            "strict index could not be revalidated after build"
                        )
                    self._store = validated
                    assert self._strict_build_lock is not None
                    self._strict_build_lock.release()
                    self._strict_build_lock = None
                    self._persisted = True
                self._built = True
            else:
                # Preserve the legacy state transition while the arm is off.
                self._built = True
                self.persist()

    def persist(self) -> None:
        if self.index_dir is None or not isinstance(self._store, FaissVectorStore):
            return
        if self.strict_provenance and self._persisted:
            return
        book_ids = [
            str(
                chunk.metadata.get("textbook")
                or chunk.chunk_id.split(":", 1)[0]
            )
            for chunk in self.index.get()
        ]
        save_index(
            self.index_dir,
            self._store,
            self._embedder_name,
            book_ids=book_ids,
            embedder_provenance=self._embedder_provenance(),
            corpus_projection_sha256=self._corpus_projection_sha256,
            strict_build_lock=self._strict_build_lock,
        )

    def invalidate(self) -> None:
        self._built = False

    def _validated_vector(self, value: Any, *, label: str) -> list[float]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{label} is not a vector")
        try:
            vector = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains non-numeric values") from exc
        if not vector or any(not math.isfinite(item) for item in vector):
            raise ValueError(f"{label} is empty or contains non-finite values")
        if (
            self.expected_embedding_dimension is not None
            and len(vector) != self.expected_embedding_dimension
        ):
            raise ValueError(
                f"{label} has dimension {len(vector)}; expected "
                f"{self.expected_embedding_dimension}"
            )
        if self.strict_provenance:
            norm = math.hypot(*vector)
            if not math.isfinite(norm) or norm <= 1e-12:
                raise ValueError(
                    f"{label} has a non-finite or near-zero L2 norm"
                )
            # FAISS consumes float32. Normalize while the values are still
            # float64 so large-but-finite inputs cannot overflow inside
            # np.linalg.norm(float32), then validate the exact float32 vector
            # that will be handed to the vector store.
            float32_vector = np.asarray(
                [item / norm for item in vector],
                dtype=np.float32,
            )
            if not bool(np.isfinite(float32_vector).all()):
                raise ValueError(
                    f"{label} becomes non-finite after float32 normalization"
                )
            vector = [float(item) for item in float32_vector]
            float32_norm = math.hypot(*vector)
            if (
                not math.isfinite(float32_norm)
                or float32_norm <= 1e-12
                or not math.isclose(
                    float32_norm,
                    1.0,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError(
                    f"{label} is not a finite unit vector after float32 normalization"
                )
        return vector

    def _embed_chunks(self, chunks: list[RetrievedChunk]) -> list[list[float]]:
        cached = {}
        missing = []
        for chunk in chunks:
            cache_key = self._embedding_cache_key(chunk)
            vector = self.embedding_cache.get_embedding(cache_key)
            if vector is None:
                missing.append(chunk)
            else:
                cached[chunk.chunk_id] = self._validated_vector(
                    vector,
                    label=f"cached embedding for {chunk.chunk_id!r}",
                )
        if missing:
            generated = self.embedder.embed_chunks(missing)
            if len(generated) != len(missing):
                raise ValueError(
                    "embedder returned a different number of vectors than chunks"
                )
            for chunk, raw_vector in zip(missing, generated):
                vector = self._validated_vector(
                    raw_vector,
                    label=f"embedding for {chunk.chunk_id!r}",
                )
                cache_key = self._embedding_cache_key(chunk)
                self.embedding_cache.set_embedding(cache_key, vector)
                cached[chunk.chunk_id] = vector
        vectors = [cached[chunk.chunk_id] for chunk in chunks]
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("dense corpus embeddings have inconsistent dimensions")
        return vectors

    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
            grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        if not self._built:
            self.build()
        if self._store is None:
            return []
        allowed_ids = None  # None == поиск везде
        if subject is not None or grade is not None:
            filtered = (
                self.index.get(subject)
                if grade is None
                else self.index.get(subject=subject, grade=grade)
            )
            allowed_ids = {chunk.chunk_id for chunk in filtered}
        if chunks:
            subset_ids = {chunk.chunk_id for chunk in chunks}
            allowed_ids = subset_ids if allowed_ids is None else allowed_ids & subset_ids
        if allowed_ids is not None and not allowed_ids:
            return []
        pool = len(allowed_ids) if chunks else self.fetch_k
        query_vector = self._validated_vector(
            self.embedder.embed_query(query),
            label="query embedding",
        )
        hits = self._store.search(query_vector, pool, allowed_ids)
        if self.strict_provenance:
            hits.sort(key=lambda item: (-float(item[1]), item[0]))
        results = []
        for chunk_id, score in hits:
            if not math.isfinite(float(score)):
                raise ValueError("dense vector store returned a non-finite score")
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(chunk.model_copy(update={"score": score}))
        return results
