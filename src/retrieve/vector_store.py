from typing import Protocol

import numpy as np
import faiss

def _to_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore(Protocol):
    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]: ...


class FaissVectorStore:
    def __init__(self, chunk_ids: list[str], index) -> None:
        self.chunk_ids = chunk_ids
        self._index = index

    @classmethod
    def from_vectors(
        cls,
        chunk_ids: list[str],
        vectors: list[list[float]],
    ) -> "FaissVectorStore":
        matrix = _normalize(_to_matrix(vectors))
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(chunk_ids, index)

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        query = _normalize(_to_matrix([vector]))
        scores, indices = self._index.search(query, k)
        return [
            (self.chunk_ids[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]
