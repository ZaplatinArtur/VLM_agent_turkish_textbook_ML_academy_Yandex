import json
import hashlib
import os
from collections import OrderedDict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

import faiss
import numpy as np

INDEX_FILE = "index.faiss"
CHUNK_IDS_FILE = "chunk_ids.json"

FLAT_MAX_VECTORS = 10_000

# HNSW: M — рёбер на узел (больше — точнее и прожорливее по памяти),
# efConstruction — ширина обхода при сборке, efSearch — при поиске.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

# Жёсткий фильтр (одна книга/предмет) на графе деградирует до пустой выдачи,
# поэтому подмножества такого размера обсчитываем точным перебором.
EXACT_SUBSET_MAX = 50_000
OVERFETCH_FACTOR = 4
SUBSET_CACHE_MAX_VECTORS = 100_000


class IndexKind(str, Enum):
    """Тип FAISS-индекса, который просим построить."""
    FLAT = "flat"
    HNSW = "hnsw"
    AUTO = "auto"


def _to_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def make_index(dim: int, kind: IndexKind = IndexKind.AUTO, size_hint: int = 0):
    """Создаёт пустой FAISS-индекс заданного типа."""
    if kind is IndexKind.AUTO:
        kind = IndexKind.FLAT if size_hint <= FLAT_MAX_VECTORS else IndexKind.HNSW
    if kind is IndexKind.FLAT:
        return faiss.IndexFlatIP(dim)
    if kind is IndexKind.HNSW:
        index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = HNSW_EF_SEARCH
        return index
    raise ValueError(f"unsupported index kind: {kind}")


def migrate_index(source, target_kind: IndexKind):
    """Пересобирает индекс в другой тип, перенося уже добавленные векторы."""
    count = source.ntotal
    vectors = (
        source.reconstruct_n(0, count)
        if count
        else np.empty((0, source.d), dtype="float32")
    )
    target = make_index(source.d, target_kind, count)
    if count:
        target.add(vectors)
    return target


class FaissVectorStore:
    def __init__(self, chunk_ids: list[str], index) -> None:
        self.chunk_ids = list(chunk_ids)
        self._positions = {chunk_id: i for i, chunk_id in enumerate(self.chunk_ids)}
        self._index = index
        self._subset_cache: OrderedDict[bytes, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self._subset_cache_vectors = 0

    @classmethod
    def from_vectors(
            cls,
            chunk_ids: list[str],
            vectors: list[list[float]],
            kind: IndexKind = IndexKind.AUTO,
    ) -> "FaissVectorStore":
        configured_kind = os.environ.get("MLA_FAISS_INDEX_KIND", "").strip()
        if kind is IndexKind.AUTO and configured_kind:
            kind = IndexKind(configured_kind.casefold())
        matrix = _normalize(_to_matrix(vectors))
        index = make_index(matrix.shape[1], kind, matrix.shape[0])
        index.add(matrix)
        return cls(chunk_ids, index)

    @property
    def is_exact(self) -> bool:
        """True, если индекс Flat."""
        return isinstance(self._index, faiss.IndexFlat)

    @property
    def index_type(self) -> str:
        return type(self._index).__name__

    @property
    def size(self) -> int:
        return self._index.ntotal

    def save(self, directory: Path | str) -> None:
        """Сохраняет снимок индекса на диск: сам индекс FAISS + порядок chunk_id.

        Пишем обычным Python-IO, а не через faiss.write_index (ломается на Unicode)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / INDEX_FILE).write_bytes(
            faiss.serialize_index(self._index).tobytes()
        )
        (directory / CHUNK_IDS_FILE).write_text(
            json.dumps(self.chunk_ids, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path | str) -> "FaissVectorStore | None":
        """Загружает снимок с диска. Возвращает None, если файлов нет."""
        directory = Path(directory)
        index_path = directory / INDEX_FILE
        ids_path = directory / CHUNK_IDS_FILE
        if not index_path.exists() or not ids_path.exists():
            return None
        buffer = np.frombuffer(index_path.read_bytes(), dtype="uint8")
        index = faiss.deserialize_index(buffer)
        chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        return cls(chunk_ids, index)

    def migrate_to(self, kind: IndexKind) -> None:
        """Меняет тип индекса на месте, сохраняя все векторы (см. migrate_index)."""
        self._index = migrate_index(self._index, kind)

    def search(
            self,
            vector: list[float],
            k: int,
            allowed_ids: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Возвращает до k пар (chunk_id, score), отсортированных по убыванию.

        allowed_ids ограничивает поиск подмножеством (предмет, конкретная книга,
        кандидаты предыдущей стадии). Фильтр применяется до поиска, а не после,
        поэтому k результатов набирается из разрешённых чанков, а не из общего
        топа, где разрешённых могло не оказаться вовсе.
        """
        if k <= 0 or self._index.ntotal == 0:
            return []
        query = _normalize(_to_matrix([vector]))
        if allowed_ids is None:
            return self._search_index(query, k)

        allowed = set(allowed_ids)
        positions = np.fromiter(
            (self._positions[cid] for cid in allowed if cid in self._positions),
            dtype="int64",
        )
        positions.sort()
        if positions.size == 0:
            return []
        if self.is_exact:
            return self._search_index(query, k, positions)
        if positions.size <= EXACT_SUBSET_MAX:
            return self._search_subset(query, k, positions)
        hits = self._search_index(query, min(k * OVERFETCH_FACTOR, self._index.ntotal))
        return [hit for hit in hits if hit[0] in allowed][:k]

    def _search_index(
            self,
            query: np.ndarray,
            k: int,
            positions: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        k = min(k, self._index.ntotal)
        hnsw = getattr(self._index, "hnsw", None)
        if hnsw is not None:
            params = faiss.SearchParametersHNSW()
            # HNSW не вернёт больше кандидатов, чем ширина обхода.
            params.efSearch = max(hnsw.efSearch, k)
        else:
            params = faiss.SearchParameters()
        # selector должен пережить вызов search: держим ссылку в локальной переменной.
        selector = None
        if positions is not None:
            selector = faiss.IDSelectorBatch(positions)
            params.sel = selector
        scores, indices = self._index.search(query, k, params=params)
        return [
            (self.chunk_ids[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    def _search_subset(
            self,
            query: np.ndarray,
            k: int,
            positions: np.ndarray,
    ) -> list[tuple[str, float]]:
        cache_key = hashlib.sha1(positions.tobytes()).digest()
        cached = self._subset_cache.get(cache_key)
        if cached is None or not np.array_equal(cached[0], positions):
            vectors = self._index.reconstruct_batch(positions)
            cached = (positions.copy(), vectors)
            self._subset_cache[cache_key] = cached
            self._subset_cache_vectors += positions.size
            while (
                self._subset_cache_vectors > SUBSET_CACHE_MAX_VECTORS
                and len(self._subset_cache) > 1
            ):
                _, (removed_positions, _) = self._subset_cache.popitem(last=False)
                self._subset_cache_vectors -= removed_positions.size
        else:
            self._subset_cache.move_to_end(cache_key)
            vectors = cached[1]
        scores = vectors @ query[0]
        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunk_ids[positions[i]], float(scores[i])) for i in top]
