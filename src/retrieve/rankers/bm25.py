"""Лексический BM25 поверх bm25s (разреженные матрицы) со стеммингом Snowball.

Регистр приводится fold_case() по правилам языка, а не casefold().
"""

from __future__ import annotations

import threading

import numpy as np

from schemas.retrieve import RetrievedChunk

from ..index import Index
from .base import Ranker

DEFAULT_LANGUAGE = "turkish"
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def fold_case(text: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Приводит регистр по правилам языка. Порядок замен важен: İ → i должно
    произойти раньше lower(), иначе останется комбинирующая точка U+0307."""
    if language == "turkish":
        text = text.replace("İ", "i").replace("I", "ı")
    return text.lower()


class BM25Ranker(Ranker):
    def __init__(
            self,
            index: Index | None = None,
            fetch_k: int = 200,
            k1: float = DEFAULT_K1,
            b: float = DEFAULT_B,
            language: str = DEFAULT_LANGUAGE,
    ) -> None:
        self.index = index or Index()
        self.fetch_k = fetch_k
        self.k1 = k1
        self.b = b
        self.language = language
        self._retriever = None
        self._stemmer = None
        self._chunk_ids: list[str] = []
        self._chunks_by_id: dict[str, RetrievedChunk] = {}
        self._built = False
        self._build_lock = threading.Lock()

    def build(self) -> None:
        if self._built:
            return
        with self._build_lock:
            if self._built:
                return
            import bm25s
            import Stemmer

            chunks = self.index.get()
            self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
            self._chunk_ids = [chunk.chunk_id for chunk in chunks]
            if not chunks:
                self._retriever = None
                self._built = True
                return
            self._stemmer = Stemmer.Stemmer(self.language)
            tokens = bm25s.tokenize(
                [fold_case(chunk.text, self.language) for chunk in chunks],
                lower=False,  # регистр уже приведён по правилам языка
                stopwords=self.language,
                stemmer=self._stemmer,
                show_progress=False,
            )
            retriever = bm25s.BM25(k1=self.k1, b=self.b)
            retriever.index(tokens, show_progress=False)
            self._retriever = retriever
            self._built = True

    def invalidate(self) -> None:
        self._built = False

    def _query_tokens(self, query: str) -> list[str]:
        import bm25s

        tokenized = bm25s.tokenize(
            fold_case(query, self.language),
            lower=False,
            stopwords=self.language,
            stemmer=self._stemmer,
            show_progress=False,
            return_ids=False,
        )
        return list(tokenized[0]) if tokenized else []

    def _allowed_positions(
            self,
            chunks: list[RetrievedChunk] | None,
            subject: str | None,
    ) -> set[int] | None:
        allowed_ids: set[str] | None = None
        if subject is not None:
            allowed_ids = {chunk.chunk_id for chunk in self.index.get(subject)}
        if chunks:
            subset = {chunk.chunk_id for chunk in chunks}
            allowed_ids = subset if allowed_ids is None else allowed_ids & subset
        if allowed_ids is None:
            return None
        return {
            position
            for position, chunk_id in enumerate(self._chunk_ids)
            if chunk_id in allowed_ids
        }

    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
    ) -> list[RetrievedChunk]:
        if not self._built:
            self.build()
        if self._retriever is None:
            return []
        tokens = self._query_tokens(query)
        if not tokens:
            return []
        allowed = self._allowed_positions(chunks, subject)
        if allowed is not None and not allowed:
            return []

        scores = self._retriever.get_scores(tokens)
        if allowed is not None:
            mask = np.zeros(len(scores), dtype=bool)
            mask[list(allowed)] = True
            scores = np.where(mask, scores, 0.0)
        matched = np.flatnonzero(scores > 0)
        if matched.size == 0:
            return []
        limit = min(len(allowed) if chunks else self.fetch_k, matched.size)
        top = matched[np.argsort(-scores[matched], kind="stable")[:limit]]
        return [
            self._chunks_by_id[self._chunk_ids[position]].model_copy(
                update={"score": float(scores[position])}
            )
            for position in top
        ]
