from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from schemas.retrieve import RetrievedChunk

from .base import AsymmetricTextEmbedder, Embedder, SymmetricTextEmbedder

MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL = MINILM_MODEL
E5_SMALL_MODEL = "intfloat/multilingual-e5-small"
E5_BASE_MODEL = "intfloat/multilingual-e5-base"
E5_MODEL = E5_BASE_MODEL
M3_MODEL = "BAAI/bge-m3"
QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
QWEN3_TASK = (
    "Given a query from a Turkish school task, retrieve the textbook page "
    "that answers it"
)
QWEN3_QUERY_PREFIX = f"Instruct: {QWEN3_TASK}\nQuery:"

BGE_M3_SEMANTIC_SMOKE_CONTRACT = "official_model_card_dense_score_v1"
BGE_M3_SEMANTIC_SMOKE_EXPECTED = (
    (0.6260, 0.3474),
    (0.3499, 0.6782),
)
BGE_M3_SEMANTIC_SMOKE_TOLERANCE = 3e-4
BGE_M3_SEMANTIC_SMOKE_MAX_LENGTH = 8192
_BGE_M3_SEMANTIC_SMOKE_LEFT = ("What is BGE M3?", "Defination of BM25")
_BGE_M3_SEMANTIC_SMOKE_RIGHT = (
    "BGE M3 is an embedding model supporting dense retrieval, lexical "
    "matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set of "
    "documents based on the query terms appearing in each document",
)


def _sentence_transformer_class() -> type[Any]:
    # Keep the heavyweight optional dependency out of import-time code paths.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


class SentenceTransformerEmbedder(SymmetricTextEmbedder, Embedder):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        *,
        revision: str | None = None,
        license_id: str | None = None,
        max_length: int | None = None,
        expected_dimension: int | None = None,
        normalize_embeddings: bool = False,
        task_contract: str = "symmetric_text_v1",
        local_files_only: bool = False,
        cache_dir: Path | str | None = None,
        device: str | None = None,
        trust_remote_code: bool = False,
        runtime_versions: Mapping[str, str] | None = None,
        validate_bge_m3_semantics: bool = False,
        faiss_index_kind: str = "auto",
        normalize: bool | None = None,
    ) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_name must not be empty")
        if revision is not None and not revision.strip():
            raise ValueError("revision must not be blank")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive")
        if expected_dimension is not None and expected_dimension <= 0:
            raise ValueError("expected_dimension must be positive")
        if not task_contract.strip():
            raise ValueError("task_contract must not be empty")
        if trust_remote_code:
            raise ValueError("remote model code is not permitted for retrieval embedders")
        if type(validate_bge_m3_semantics) is not bool:
            raise ValueError("validate_bge_m3_semantics must be a boolean")
        if faiss_index_kind not in {"auto", "flat", "hnsw"}:
            raise ValueError("faiss_index_kind must be 'auto', 'flat', or 'hnsw'")
        if normalize is not None:
            normalize_embeddings = normalize
        if runtime_versions is not None and (
            not runtime_versions
            or any(
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(version, str)
                or not version.strip()
                for name, version in runtime_versions.items()
            )
        ):
            raise ValueError("runtime_versions must contain non-empty names and versions")

        self.model_name = model_name
        self.batch_size = batch_size
        self.revision = revision.strip() if revision is not None else None
        self.license_id = license_id.casefold() if license_id else None
        self.max_length = max_length
        self.expected_dimension = expected_dimension
        self.normalize_embeddings = bool(normalize_embeddings)
        self.task_contract = task_contract.strip()
        self.local_files_only = bool(local_files_only)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self.device = device.strip() if device else None
        self.trust_remote_code = False
        self.runtime_versions = (
            {
                name.strip(): version.strip()
                for name, version in sorted(runtime_versions.items())
            }
            if runtime_versions is not None
            else None
        )
        self.validate_bge_m3_semantics = validate_bge_m3_semantics
        self.faiss_index_kind = faiss_index_kind
        self.runtime_validation: dict[str, object] | None = None
        self._model: Any | None = None
        self._model_lock = threading.Lock()

    @property
    def provenance(self) -> dict[str, object]:
        """Canonical fields that affect the meaning of persisted embeddings."""
        provenance: dict[str, object] = {
            "backend": "sentence-transformers",
            "batch_size": self.batch_size,
            "device": self.device,
            "embedding_dimension": self.expected_dimension,
            "encode_normalize_embeddings": self.normalize_embeddings,
            "faiss_index_kind": self.faiss_index_kind,
            "license": self.license_id,
            "max_length": self.max_length,
            "model_id": self.model_name,
            "revision": self.revision,
            "task_contract": self.task_contract,
            "trust_remote_code": self.trust_remote_code,
            "vector_store_normalization": "l2",
        }
        if self.runtime_versions is not None:
            provenance["runtime_packages"] = dict(self.runtime_versions)
        if self.validate_bge_m3_semantics:
            provenance["semantic_smoke_contract"] = BGE_M3_SEMANTIC_SMOKE_CONTRACT
        return provenance

    @property
    def model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    kwargs: dict[str, object] = {
                        "local_files_only": self.local_files_only,
                        "trust_remote_code": self.trust_remote_code,
                    }
                    if self.revision is not None:
                        kwargs["revision"] = self.revision
                    if self.cache_dir is not None:
                        kwargs["cache_folder"] = str(self.cache_dir)
                    if self.device is not None:
                        kwargs["device"] = self.device
                    model = _sentence_transformer_class()(self.model_name, **kwargs)
                    if self.max_length is not None:
                        model.max_seq_length = self.max_length
                    if self.validate_bge_m3_semantics:
                        self._validate_bge_m3_semantic_smoke(model)
                    self._model = model
        return self._model

    def _encode_with_model(
        self,
        model: Any,
        texts: list[str],
    ) -> list[list[float]]:
        encoded = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        vectors = encoded.tolist()
        if len(vectors) != len(texts):
            raise ValueError(
                "sentence-transformer returned a different number of vectors than texts"
            )
        dimensions = {len(vector) for vector in vectors if isinstance(vector, list)}
        if (
            len(dimensions) != 1
            or 0 in dimensions
            or any(not isinstance(vector, list) for vector in vectors)
            or any(
                not math.isfinite(float(value))
                for vector in vectors
                for value in vector
            )
        ):
            raise ValueError("sentence-transformer returned malformed embeddings")
        if (
            self.expected_dimension is not None
            and dimensions != {self.expected_dimension}
        ):
            raise ValueError(
                "sentence-transformer embedding dimension does not match the "
                f"configured pin ({self.expected_dimension})"
            )
        return vectors

    def _validate_bge_m3_semantic_smoke(self, model: Any) -> None:
        texts = list(_BGE_M3_SEMANTIC_SMOKE_LEFT + _BGE_M3_SEMANTIC_SMOKE_RIGHT)
        configured_max_length = getattr(model, "max_seq_length", None)
        model.max_seq_length = BGE_M3_SEMANTIC_SMOKE_MAX_LENGTH
        try:
            vectors = self._encode_with_model(model, texts)
        finally:
            model.max_seq_length = configured_max_length
        left = vectors[:2]
        right = vectors[2:]
        actual = [
            [
                sum(a * b for a, b in zip(left_vector, right_vector))
                for right_vector in right
            ]
            for left_vector in left
        ]
        if any(
            abs(actual[row][column] - BGE_M3_SEMANTIC_SMOKE_EXPECTED[row][column])
            > BGE_M3_SEMANTIC_SMOKE_TOLERANCE
            for row in range(2)
            for column in range(2)
        ):
            raise RuntimeError(
                "BGE-M3 official semantic smoke test failed: "
                f"actual={actual}, expected={BGE_M3_SEMANTIC_SMOKE_EXPECTED}, "
                f"atol={BGE_M3_SEMANTIC_SMOKE_TOLERANCE}"
            )
        self.runtime_validation = {
            "absolute_tolerance": BGE_M3_SEMANTIC_SMOKE_TOLERANCE,
            "actual": actual,
            "expected": [list(row) for row in BGE_M3_SEMANTIC_SMOKE_EXPECTED],
            "name": BGE_M3_SEMANTIC_SMOKE_CONTRACT,
            "passed": True,
            "smoke_max_length": BGE_M3_SEMANTIC_SMOKE_MAX_LENGTH,
        }

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode_with_model(self.model, texts)


# Backward-compatible public name used by the profile-based retrieval branch.
SentenceTransformerBackend = SentenceTransformerEmbedder


class PlainEmbedder(SentenceTransformerEmbedder):
    """Compatibility name for symmetric MiniLM and BGE-M3 profiles."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)


class M3Embedder(SentenceTransformerEmbedder):
    def __init__(self, model_name: str = M3_MODEL, **kwargs) -> None:
        super().__init__(model_name=model_name, **kwargs)


class E5Embedder(SentenceTransformerEmbedder, AsymmetricTextEmbedder):
    query_prefix = E5_QUERY_PREFIX
    passage_prefix = E5_PASSAGE_PREFIX

    def __init__(self, model_name: str = E5_MODEL, **kwargs) -> None:
        super().__init__(model_name=model_name, **kwargs)

    def embed_chunks(self, chunks: list[RetrievedChunk]) -> list[list[float]]:
        if not chunks:
            return []
        return self.encode(
            [
                self.passage_prefix
                + str(chunk.metadata.get("retrieval_text") or chunk.text)
                for chunk in chunks
            ]
        )

    def embed_query(self, query: str) -> list[float]:
        return self.encode([self.query_prefix + query])[0]


class Qwen3Embedder(SentenceTransformerEmbedder, AsymmetricTextEmbedder):
    """Qwen3 embedding profile with an instruction on the query only."""

    query_prefix = QWEN3_QUERY_PREFIX

    def __init__(
        self,
        model_name: str = QWEN3_EMBEDDING_MODEL,
        batch_size: int = 32,
        normalize: bool = False,
        max_seq_length: int | None = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            normalize=normalize,
            max_length=max_seq_length,
            **kwargs,
        )

    def embed_chunks(self, chunks: list[RetrievedChunk]) -> list[list[float]]:
        if not chunks:
            return []
        return self.encode(
            [str(chunk.metadata.get("retrieval_text") or chunk.text) for chunk in chunks]
        )

    def embed_query(self, query: str) -> list[float]:
        return self.encode([self.query_prefix + query])[0]
