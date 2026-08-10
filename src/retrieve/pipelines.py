"""Именованные профили ретрива — одна точка сборки для прода и замеров.

Профили:
  bm25          — только лексический BM25 (классический бейзлайн);
  dense         — один плотный энкодер (текущий прод/бейзлайн, MiniLM);
  e5 / m3       — один плотный энкодер multilingual-e5 / bge-m3;
  hybrid        — RRF-ансамбль E5 + M3 + BM25;
  hybrid_rerank — тот же ансамбль плюс кросс-энкодер bge-reranker-v2-m3.

Каждый плотный ранкер получает СВОЙ каталог снапшота и СВОЙ namespace кэша
(иначе разные эмбеддеры затирают индекс/векторы друг друга). Профиль dense
сознательно оставлен в старом каталоге/кэше, чтобы прод не пересобирался.
"""

from __future__ import annotations

import re
from pathlib import Path

from .index import Index
from .pipeline import RetrievalPipeline

DEFAULT_PROFILE = "dense"
PROFILES = ("bm25", "dense", "e5", "m3", "hybrid", "hybrid_rerank")


def _safe(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def _dense_ranker(embedder, index: Index, index_root: Path | str | None, fetch_k: int):
    from .cache import EmbeddingCache
    from .rankers import DenseRanker

    name = getattr(embedder, "model_name", type(embedder).__name__)
    safe = _safe(name)
    return DenseRanker(
        embedder=embedder,
        index=index,
        embedding_cache=EmbeddingCache(namespace=safe),
        fetch_k=fetch_k,
        index_dir=(Path(index_root) / safe) if index_root is not None else None,
    )


def build_profile(
        profile: str,
        index: Index,
        *,
        index_root: Path | str | None = None,
        fetch_k: int = 200,
        rerank_top_n: int = 100,
) -> RetrievalPipeline:
    from .rankers import DEFAULT_RRF_K, BM25Ranker, ReciprocalRankFusion

    if profile == "bm25":
        return RetrievalPipeline([BM25Ranker(index, fetch_k=fetch_k)])

    if profile == "dense":
        # Прод-бейзлайн: та же модель, каталог и кэш, что и раньше.
        from .embedders import SentenceTransformerEmbedder
        from .rankers import DenseRanker

        return RetrievalPipeline([
            DenseRanker(
                embedder=SentenceTransformerEmbedder(),
                index=index,
                fetch_k=fetch_k,
                index_dir=index_root,
            )
        ])

    if profile == "e5":
        from .embedders import E5Embedder

        return RetrievalPipeline([_dense_ranker(E5Embedder(), index, index_root, fetch_k)])

    if profile == "m3":
        from .embedders import M3Embedder

        return RetrievalPipeline([_dense_ranker(M3Embedder(), index, index_root, fetch_k)])

    if profile in ("hybrid", "hybrid_rerank"):
        from .embedders import E5Embedder, M3Embedder

        rankers = [
            _dense_ranker(E5Embedder(), index, index_root, fetch_k),
            _dense_ranker(M3Embedder(), index, index_root, fetch_k),
            BM25Ranker(index, fetch_k=fetch_k),
        ]
        fusion = ReciprocalRankFusion(rankers, rrf_k=DEFAULT_RRF_K)
        if profile == "hybrid":
            return RetrievalPipeline([fusion])
        from .rankers import CrossEncoderRanker

        return RetrievalPipeline([fusion, CrossEncoderRanker(top_n=rerank_top_n)])

    raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")
