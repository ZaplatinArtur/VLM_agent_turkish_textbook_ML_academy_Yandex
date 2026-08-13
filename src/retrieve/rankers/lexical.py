from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections import Counter, defaultdict

from schemas.retrieve import RetrievedChunk

from ..index import Index
from .bm25 import fold_case


# ı (U+0131) не раскладывается NFKD и не входит в [a-z0-9], поэтому её нужно
# перевести явно — иначе фильтр вырезает её и рвёт слово (ışık, ısı, sıfır → []).
_TURKISH_ASCII = str.maketrans(
    {"ı": "i", "İ": "i", "ş": "s", "ğ": "g", "ç": "c", "ö": "o", "ü": "u"}
)


def tokenize(value: str) -> list[str]:
    folded = fold_case(value).translate(_TURKISH_ASCII)
    folded = unicodedata.normalize("NFKD", folded)
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    return [
        token
        for token in re.findall(r"[a-z0-9]+", plain)
        if len(token) > 1 or token.isdigit()
    ]


class BM25Ranker:
    """Small dependency-free BM25 candidate generator."""

    def __init__(
        self,
        index: Index,
        *,
        fetch_k: int = 80,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.index = index
        self.fetch_k = fetch_k
        self.k1 = k1
        self.b = b
        self._built = False
        self._chunks: dict[str, RetrievedChunk] = {}
        self._term_frequency: dict[str, Counter[str]] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._lengths: dict[str, int] = {}
        self._average_length = 1.0
        self._build_lock = threading.Lock()

    def build(self) -> None:
        if self._built:
            return
        with self._build_lock:
            if self._built:
                return
            chunks = self.index.get()
            self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
            for chunk in chunks:
                text = str(chunk.metadata.get("retrieval_text") or chunk.text)
                frequency = Counter(tokenize(text))
                self._term_frequency[chunk.chunk_id] = frequency
                length = sum(frequency.values())
                self._lengths[chunk.chunk_id] = length
                for token in frequency:
                    self._postings[token].add(chunk.chunk_id)
            if self._lengths:
                self._average_length = (
                    sum(self._lengths.values()) / len(self._lengths)
                )
            self._built = True

    def rank(
        self,
        query: str,
        chunks: list[RetrievedChunk] | None = None,
        subject: str | None = None,
        grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        self.build()
        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []
        allowed = {
            chunk.chunk_id
            for chunk in self.index.get(subject=subject, grade=grade)
        }
        if chunks:
            allowed &= {chunk.chunk_id for chunk in chunks}
        candidate_ids: set[str] = set()
        for term in query_terms:
            candidate_ids.update(self._postings.get(term, set()))
        candidate_ids &= allowed
        document_count = max(1, len(self._chunks))
        scores: dict[str, float] = defaultdict(float)
        for term, query_frequency in query_terms.items():
            posting = self._postings.get(term, set())
            document_frequency = len(posting)
            if not document_frequency:
                continue
            inverse_document_frequency = math.log(
                1.0 + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for chunk_id in posting & candidate_ids:
                frequency = self._term_frequency[chunk_id][term]
                length = self._lengths[chunk_id]
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / self._average_length
                )
                scores[chunk_id] += (
                    query_frequency
                    * inverse_document_frequency
                    * frequency
                    * (self.k1 + 1.0)
                    / denominator
                )
        ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        return [
            self._chunks[chunk_id].model_copy(update={"score": scores[chunk_id]})
            for chunk_id in ordered[: self.fetch_k]
        ]
