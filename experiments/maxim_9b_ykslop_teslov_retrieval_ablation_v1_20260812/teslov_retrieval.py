"""Dependency-light port of Teslov's BM25/RRF/cross-encoder retrieval shape.

This module is deliberately limited to retrieval.  It neither reads benchmark
labels nor calls a model/API: a cross-encoder scorer must be injected.  The
real tuned scorer is enabled only after its local LoRA directory is present and
separately attested.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata
from pathlib import Path
from typing import Callable, Iterator, Protocol, Sequence


TESLOV_FINAL_COMMIT = "2116cb1288a8f773e1ccd929dac05382d319327e"
TESLOV_HYBRID_COMMIT = "6aa0683c85c2e69f47c04f8f5da31bc18509503f"
BASE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BASE_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
TUNED_ADAPTER_RELATIVE_PATH = Path("data/models/bge-reranker-v2/chosen")

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class TheoryChunk:
    chunk_id: str
    text: str
    grade: int | str | None = None
    subject: str | None = None


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk: TheoryChunk
    score: float


class CrossEncoderScorer(Protocol):
    """Small seam implemented by a local model or a deterministic test stub."""

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


DenseRanker = Callable[[str, Sequence[TheoryChunk]], Sequence[RankedChunk]]


_YKS_SUBJECTS = {
    "Biyoloji": "biyoloji",
    "Coğrafya": "cografya",
    "Din Kültürü ve Ahlak Bilgisi": "din_kulturu",
    "Felsefe": "felsefe",
    "Fizik": "fizik",
    "Kimya": "kimya",
    "Matematik": "matematik",
    "T.C. İnkılap Tarihi ve Atatürkçülük": "inkilap",
    "Tarih": "tarih",
    "Türk Dili ve Edebiyatı": "turk_dili_ve_edebiyati",
}


def _mojibake_candidates(text: str) -> Iterator[str]:
    """Yield the exact frozen YKS normalization candidates."""

    yield text
    for encoding in ("cp1251", "cp1252", "latin1"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired != text:
            yield repaired
    suspicious_symbols = frozenset(
        chr(value)
        for value in (
            0x00B6, 0x00B1, 0x2021, 0x00A7, 0x20AC, 0x045C, 0x045A,
            0x045F, 0x0459, 0x045B, 0x040E, 0x045E, 0x0408, 0x00A4,
            0x00A6, 0x0401, 0x0451, 0x0407, 0x0457, 0x0406, 0x0456,
            0x0405, 0x0455,
        )
    )
    pieces: list[str] = []
    index = 0
    changed = False
    while index < len(text):
        char = text[index]
        suspicious = "\u0400" <= char <= "\u04ff" or char in suspicious_symbols
        if not suspicious:
            pieces.append(char)
            index += 1
            continue
        end = index
        raw = bytearray()
        while end < len(text):
            item = text[end]
            if not ("\u0400" <= item <= "\u04ff" or item in suspicious_symbols):
                break
            try:
                encoded = (
                    item.encode("cp1251")
                    if "\u0400" <= item <= "\u04ff"
                    else item.encode("latin1")
                )
            except UnicodeEncodeError:
                break
            if len(encoded) != 1:
                break
            raw.extend(encoded)
            end += 1
        if end == index:
            pieces.append(char)
            index += 1
            continue
        try:
            decoded = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            pieces.append(text[index:end] or char)
        else:
            pieces.append(decoded)
            changed = True
        index = end if end > index else index + 1
    if changed:
        yield "".join(pieces)


def normalize_text(text: str) -> str:
    """Exact frozen YKS NFKC/mojibake/Turkish-case normalization."""

    if type(text) is not str:
        return ""
    variants = list(_mojibake_candidates(text))
    bad_markers = "ГРДМ‡С€СџВÂÃ�"
    selected = min(
        variants,
        key=lambda item: sum(item.count(marker) for marker in bad_markers),
    )
    selected = unicodedata.normalize("NFKC", selected)
    selected = selected.translate(str.maketrans({"I": "ı", "İ": "i"})).casefold()
    selected = " ".join(_TOKEN.findall(selected))
    return _WHITESPACE.sub(" ", selected).strip()


def _tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def _grade_key(value: int | str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().casefold()
    return normalized or None


def grade_filter(
    chunks: Sequence[TheoryChunk], grade: int | str | None
) -> list[TheoryChunk]:
    """Strict grade filter; unknown-grade chunks never leak into a grade arm."""

    expected = _grade_key(grade)
    if expected is None:
        return list(chunks)
    return [chunk for chunk in chunks if _grade_key(chunk.grade) == expected]


def subject_filter(
    chunks: Sequence[TheoryChunk], subject: str | None
) -> list[TheoryChunk]:
    """Fail closed to the exact YKS subject; never search an unrelated book."""

    if subject is None:
        return list(chunks)
    desired = _YKS_SUBJECTS.get(subject, subject).strip().casefold()
    if not desired:
        raise ValueError("subject must not be blank")
    return [
        chunk
        for chunk in chunks
        if (chunk.subject or "").strip().casefold() == desired
    ]


def eligible_chunks(
    chunks: Sequence[TheoryChunk],
    *,
    subject: str | None,
    grade: int | str | None,
) -> list[TheoryChunk]:
    return grade_filter(subject_filter(chunks, subject), grade)


def bm25_rank(
    query: str,
    chunks: Sequence[TheoryChunk],
    *,
    grade: int | str | None = None,
    subject: str | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[RankedChunk]:
    """Deterministic lexical arm matching the existing YKSLOP BM25 formula."""

    corpus = eligible_chunks(chunks, subject=subject, grade=grade)
    query_terms = set(_tokens(query))
    documents = [_tokens(chunk.text) for chunk in corpus]
    if not query_terms or not documents:
        return []
    average_length = sum(map(len, documents)) / len(documents)
    if average_length == 0:
        return []
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    ranked: list[RankedChunk] = []
    for chunk, document in zip(corpus, documents):
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1.0 - b + b * len(document) / average_length
            )
            score += inverse * (frequency * (k1 + 1.0)) / denominator
        if score > 0.0:
            ranked.append(RankedChunk(chunk, score))
    return sorted(ranked, key=lambda item: (-item.score, item.chunk.chunk_id))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedChunk]],
    *,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RankedChunk]:
    """Teslov-style weighted RRF with deterministic first-seen tie handling."""

    if not rankings:
        raise ValueError("at least one ranking is required")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    selected_weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(selected_weights) != len(rankings):
        raise ValueError("weights must match rankings")
    scores: dict[str, float] = {}
    seen: dict[str, tuple[int, TheoryChunk]] = {}
    ordinal = 0
    for ranking, weight in zip(rankings, selected_weights):
        for position, item in enumerate(ranking, start=1):
            scores[item.chunk.chunk_id] = scores.get(item.chunk.chunk_id, 0.0) + (
                float(weight) / (rrf_k + position)
            )
            if item.chunk.chunk_id not in seen:
                seen[item.chunk.chunk_id] = (ordinal, item.chunk)
                ordinal += 1
    ordered = sorted(seen.items(), key=lambda pair: (-scores[pair[0]], pair[1][0]))
    return [RankedChunk(value[1], scores[chunk_id]) for chunk_id, value in ordered]


class GradeAwareTeslovCrossEncoder:
    """Rerank the filtered RRF head while keeping its tail stable."""

    def __init__(self, scorer: CrossEncoderScorer, *, top_n: int = 100) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self.scorer = scorer
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        candidates: Sequence[RankedChunk],
        *,
        grade: int | str | None = None,
    ) -> list[RankedChunk]:
        allowed = {_chunk.chunk_id for _chunk in grade_filter(
            [item.chunk for item in candidates], grade
        )}
        filtered = [item for item in candidates if item.chunk.chunk_id in allowed]
        head, tail = filtered[: self.top_n], filtered[self.top_n :]
        if not head:
            return []
        scores = [float(value) for value in self.scorer.score(
            query, [item.chunk.text for item in head]
        )]
        if len(scores) != len(head) or any(not math.isfinite(value) for value in scores):
            raise ValueError("cross-encoder must return one finite score per document")
        rescored = [RankedChunk(item.chunk, score) for item, score in zip(head, scores)]
        rescored.sort(key=lambda item: -item.score)  # stable, as in Teslov's port
        return rescored + tail


@dataclass(slots=True)
class OfflineRetrievalArms:
    """Two retrieval-only arms for a future clean qrels evaluation."""

    corpus: Sequence[TheoryChunk]
    dense_ranker: DenseRanker
    cross_encoder: GradeAwareTeslovCrossEncoder
    fetch_k: int = 200
    rrf_k: int = 60

    def bm25(
        self,
        query: str,
        *,
        subject: str | None = None,
        grade: int | str | None = None,
    ) -> list[RankedChunk]:
        return bm25_rank(
            query,
            self.corpus,
            subject=subject,
            grade=grade,
        )[: self.fetch_k]

    def teslov_rrf_cross_encoder(
        self,
        query: str,
        *,
        subject: str | None = None,
        grade: int | str | None = None,
    ) -> list[RankedChunk]:
        eligible = eligible_chunks(
            self.corpus,
            subject=subject,
            grade=grade,
        )
        lexical = bm25_rank(query, eligible)[: self.fetch_k]
        dense = list(self.dense_ranker(query, eligible))[: self.fetch_k]
        fused = reciprocal_rank_fusion([dense, lexical], rrf_k=self.rrf_k)
        return self.cross_encoder.rerank(query, fused, grade=grade)


def tuned_adapter_status(repo_root: Path) -> dict[str, str | bool]:
    """Report readiness without loading or downloading model artifacts."""

    path = (repo_root / TUNED_ADAPTER_RELATIVE_PATH).resolve()
    return {
        "relative_path": TUNED_ADAPTER_RELATIVE_PATH.as_posix(),
        "resolved_path": str(path),
        "present": path.is_dir(),
    }
