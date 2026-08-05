from __future__ import annotations

import math
import re
import threading
from collections import Counter

from schemas.retrieve import RetrievedChunk

from ..index import Index
from .base import Ranker

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(text.casefold()):
        if len(token) < 2 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


class BM25Ranker(Ranker):
    def __init__(
            self,
            index: Index | None = None,
            fetch_k: int = 200,
            k1: float = DEFAULT_K1,
            b: float = DEFAULT_B,
    ) -> None:
        self.index = index or Index()
        self.fetch_k = fetch_k
        self.k1 = k1
        self.b = b
        self._chunk_ids: list[str] = []
        self._chunks_by_id: dict[str, RetrievedChunk] = {}
        self._doc_len: list[int] = []
        self._doc_terms: list[Counter] = []
        self._postings: dict[str, list[int]] = {}
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0
        self._built = False
        self._build_lock = threading.Lock()

    def build(self) -> None:
        if self._built:
            return
        with self._build_lock:
            if self._built:
                return
            chunks = self.index.get()
            self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
            self._chunk_ids = [chunk.chunk_id for chunk in chunks]
            self._doc_terms = []
            self._doc_len = []
            self._postings = {}
            document_freq: Counter = Counter()
            for position, chunk in enumerate(chunks):
                terms = Counter(tokenize(chunk.text))
                self._doc_terms.append(terms)
                self._doc_len.append(sum(terms.values()))
                for term in terms:
                    self._postings.setdefault(term, []).append(position)
                    document_freq[term] += 1
            total = len(chunks)
            self._avgdl = (sum(self._doc_len) / total) if total else 0.0
            self._idf = {
                term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
                for term, freq in document_freq.items()
            }
            self._built = True

    def invalidate(self) -> None:
        self._built = False

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
        positions = {
            position
            for position, chunk_id in enumerate(self._chunk_ids)
            if chunk_id in allowed_ids
        }
        return positions

    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
    ) -> list[RetrievedChunk]:
        if not self._built:
            self.build()
        if not self._chunk_ids:
            return []
        query_terms = [term for term in tokenize(query) if term in self._postings]
        if not query_terms:
            return []
        allowed = self._allowed_positions(chunks, subject)
        if allowed is not None and not allowed:
            return []

        scores: dict[int, float] = {}
        for term in query_terms:
            idf = self._idf[term]
            for position in self._postings[term]:
                if allowed is not None and position not in allowed:
                    continue
                freq = self._doc_terms[position][term]
                length = self._doc_len[position]
                denom = freq + self.k1 * (
                    1 - self.b + self.b * length / (self._avgdl or 1.0)
                )
                contribution = idf * freq * (self.k1 + 1) / denom
                scores[position] = scores.get(position, 0.0) + contribution

        limit = len(allowed) if chunks else self.fetch_k
        top = sorted(scores.items(), key=lambda item: -item[1])[:limit]
        results = []
        for position, score in top:
            chunk = self._chunks_by_id.get(self._chunk_ids[position])
            if chunk is not None:
                results.append(chunk.model_copy(update={"score": float(score)}))
        return results