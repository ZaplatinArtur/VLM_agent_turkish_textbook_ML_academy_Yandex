"""Именованные профили ретрива — одна точка сборки для прода и замеров.

Имя перечисляет состав: `rrf_<ранкеры через _>`, суффикс `_cross-encoder` — хвост
bge-reranker-v2-m3, суффикс `_gate` — LLM-гейт последней ступенью. Полный список —
в PROFILES ниже, сравнение — в reports/tables/. Профиль выбирается переменной
RETRIEVE_PROFILE, без правки кода.

Реестр не сокращается: собирается только запрошенный профиль, а без остальных не
воспроизвести таблицы замеров.

ВНИМАНИЕ: у RRF своя шкала score (сумма 1/(60+позиция), максимум 0.033), порог
из retrieve.confidence к ней неприменим — нужен хвост-кросс-энкодер.

Каждый плотный ранкер получает свой каталог снапшота и свой namespace кэша, иначе
эмбеддеры затирают векторы друг друга. minilm намеренно оставлен в старом
каталоге, чтобы его снапшот не пересобирался.

TODO(qwen3): профили с Qwen3 не проверены ни разу. Под qwen3-embedding нет
индекса (другая размерность, нужна пересборка), а qwen3-reranker требует
поднятого рядом vLLM (RETRIEVE_RERANK_URL).
"""

from __future__ import annotations

import re
from pathlib import Path

from .index import Index
from .pipeline import RetrievalPipeline

E5_SMALL = "e5-small"
E5_BASE = "e5-base"
QWEN3 = "qwen3-embedding"
RRF_E5_SMALL = "rrf_e5-small_bm25"
RRF_E5_SMALL_CE = "rrf_e5-small_bm25_cross-encoder"
RRF_E5_SMALL_GATE = "rrf_e5-small_bm25_gate"
RRF_E5_SMALL_CE_GATE = "rrf_e5-small_bm25_cross-encoder_gate"
RRF_E5_SMALL_QWEN3_CE = "rrf_e5-small_bm25_qwen3-reranker"
RRF_QWEN3 = "rrf_qwen3-embedding_bm25"
RRF_E5_BASE = "rrf_e5-base_bm25"
RRF_E5_BASE_M3 = "rrf_e5-base_m3_bm25"
RRF_E5_BASE_M3_CE = "rrf_e5-base_m3_bm25_cross-encoder"
RRF_E5_SMALL_M3 = "rrf_e5-small_m3_bm25"
RRF_E5_SMALL_M3_CE = "rrf_e5-small_m3_bm25_cross-encoder"
RRF_M3_CE = "rrf_m3_bm25_cross-encoder"
RRF_M3_CE_GATE = "rrf_m3_bm25_cross-encoder_gate"
DEFAULT_PROFILE = RRF_E5_SMALL_CE
PROFILES = (
    "bm25", "minilm", E5_SMALL, E5_BASE, "m3", QWEN3,
    RRF_E5_SMALL, RRF_E5_SMALL_CE, RRF_E5_SMALL_GATE, RRF_E5_SMALL_CE_GATE,
    RRF_E5_SMALL_QWEN3_CE, RRF_QWEN3,
    RRF_E5_BASE, RRF_E5_BASE_M3, RRF_E5_BASE_M3_CE,
    RRF_E5_SMALL_M3, RRF_E5_SMALL_M3_CE, RRF_M3_CE, RRF_M3_CE_GATE,
)


def _safe(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def _dense_ranker(embedder, index: Index, index_root: Path | str | None, fetch_k: int):
    from .cache import EmbeddingCache
    from .rankers import DenseRanker

    safe = _safe(embedder.model_name)
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
        gate_top_n: int = 10,
) -> RetrievalPipeline:
    from .embedders import (
        E5_BASE_MODEL,
        E5_SMALL_MODEL,
        M3_MODEL,
        MINILM_MODEL,
        QWEN3_EMBEDDING_MODEL,
        E5Embedder,
        PlainEmbedder,
        Qwen3Embedder,
    )
    from .rankers import (
        DEFAULT_RRF_K,
        ReciprocalRankFusion,
        StemmedBM25Ranker as BM25Ranker,
    )

    if profile == "bm25":
        return RetrievalPipeline([BM25Ranker(index, fetch_k=fetch_k)])

    if profile == "minilm":
        # Каталог и кэш без namespace — как было до появления профилей.
        from .rankers import DenseRanker

        return RetrievalPipeline([
            DenseRanker(
                embedder=PlainEmbedder(MINILM_MODEL),
                index=index,
                fetch_k=fetch_k,
                index_dir=index_root,
            )
        ])

    if profile == E5_SMALL:
        embedder = E5Embedder(E5_SMALL_MODEL)
        return RetrievalPipeline([_dense_ranker(embedder, index, index_root, fetch_k)])

    if profile == E5_BASE:
        embedder = E5Embedder(E5_BASE_MODEL)
        return RetrievalPipeline([_dense_ranker(embedder, index, index_root, fetch_k)])

    if profile == "m3":
        embedder = PlainEmbedder(M3_MODEL)
        return RetrievalPipeline([_dense_ranker(embedder, index, index_root, fetch_k)])

    if profile == QWEN3:
        embedder = Qwen3Embedder(QWEN3_EMBEDDING_MODEL)
        return RetrievalPipeline([_dense_ranker(embedder, index, index_root, fetch_k)])

    rrf_pair = (RRF_E5_SMALL, RRF_E5_SMALL_CE, RRF_E5_SMALL_GATE, RRF_E5_SMALL_CE_GATE,
                RRF_E5_SMALL_QWEN3_CE, RRF_QWEN3, RRF_E5_BASE)
    if profile in rrf_pair:
        if profile == RRF_QWEN3:
            embedder = Qwen3Embedder(QWEN3_EMBEDDING_MODEL)
        else:
            embedder = E5Embedder(E5_BASE_MODEL if profile == RRF_E5_BASE else E5_SMALL_MODEL)
        stages = [ReciprocalRankFusion([
            _dense_ranker(embedder, index, index_root, fetch_k),
            BM25Ranker(index, fetch_k=fetch_k),
        ], rrf_k=DEFAULT_RRF_K)]

        if profile in (RRF_E5_SMALL_CE, RRF_E5_SMALL_CE_GATE):
            from .rankers import CrossEncoderRanker

            stages.append(CrossEncoderRanker(top_n=rerank_top_n))
        elif profile == RRF_E5_SMALL_QWEN3_CE:
            from .rankers import RerankApiRanker
            from .rankers.rerank_api import QWEN3_RERANKER_MODEL

            stages.append(RerankApiRanker(QWEN3_RERANKER_MODEL, top_n=rerank_top_n))

        if profile in (RRF_E5_SMALL_GATE, RRF_E5_SMALL_CE_GATE):
            from .gate import GateRanker

            stages.append(GateRanker(top_n=gate_top_n))
        return RetrievalPipeline(stages)

    # m3 + bm25 без e5: A/B к rrf_e5-small_bm25_cross-encoder ровно по эмбеддеру.
    if profile in (RRF_M3_CE, RRF_M3_CE_GATE):
        from .rankers import CrossEncoderRanker

        stages = [
            ReciprocalRankFusion([
                _dense_ranker(PlainEmbedder(M3_MODEL), index, index_root, fetch_k),
                BM25Ranker(index, fetch_k=fetch_k),
            ], rrf_k=DEFAULT_RRF_K),
            CrossEncoderRanker(top_n=rerank_top_n),
        ]
        if profile == RRF_M3_CE_GATE:
            from .gate import GateRanker

            stages.append(GateRanker(top_n=gate_top_n))
        return RetrievalPipeline(stages)

    triples = {
        RRF_E5_BASE_M3: (E5_BASE_MODEL, False),
        RRF_E5_BASE_M3_CE: (E5_BASE_MODEL, True),
        RRF_E5_SMALL_M3: (E5_SMALL_MODEL, False),
        RRF_E5_SMALL_M3_CE: (E5_SMALL_MODEL, True),
    }
    if profile in triples:
        e5_model, with_cross_encoder = triples[profile]
        fusion = ReciprocalRankFusion([
            _dense_ranker(E5Embedder(e5_model), index, index_root, fetch_k),
            _dense_ranker(PlainEmbedder(M3_MODEL), index, index_root, fetch_k),
            BM25Ranker(index, fetch_k=fetch_k),
        ], rrf_k=DEFAULT_RRF_K)
        if not with_cross_encoder:
            return RetrievalPipeline([fusion])
        from .rankers import CrossEncoderRanker

        return RetrievalPipeline([fusion, CrossEncoderRanker(top_n=rerank_top_n)])

    raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")
