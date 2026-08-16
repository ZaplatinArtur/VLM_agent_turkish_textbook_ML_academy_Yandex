"""Public retrieval API with lazy imports for optional ML dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_ATTRS = {
    "BM25Ranker": (".rankers", "BM25Ranker"),
    "CrossEncoderRanker": (".rankers", "CrossEncoderRanker"),
    "DenseRanker": (".rankers", "DenseRanker"),
    "E5Embedder": (".embedders", "E5Embedder"),
    "Embedder": (".embedders", "Embedder"),
    "M3Embedder": (".embedders", "M3Embedder"),
    "PlainEmbedder": (".embedders", "PlainEmbedder"),
    "Qwen3Embedder": (".embedders", "Qwen3Embedder"),
    "EmbeddingCache": (".cache", "EmbeddingCache"),
    "FaissVectorStore": (".storage.vector_store", "FaissVectorStore"),
    "Index": (".index", "Index"),
    "PrimaryCandidateUnion": (".rankers", "PrimaryCandidateUnion"),
    "MaximalMarginalRelevanceRanker": (
        ".rankers",
        "MaximalMarginalRelevanceRanker",
    ),
    "Ranker": (".rankers", "Ranker"),
    "RerankApiRanker": (".rankers", "RerankApiRanker"),
    "ReciprocalRankFusion": (".rankers", "ReciprocalRankFusion"),
    "StemmedBM25Ranker": (".rankers", "StemmedBM25Ranker"),
    "rescored": (".rankers", "rescored"),
    "build_profile": (".pipelines", "build_profile"),
    "evaluate_pipeline": (".evaluation.evaluate", "evaluate_pipeline"),
    "Relevance": (".confidence", "Relevance"),
    "RelevanceVerdict": (".confidence", "RelevanceVerdict"),
    "RetrievalPipeline": (".pipeline", "RetrievalPipeline"),
    "SentenceTransformerEmbedder": (
        ".embedders",
        "SentenceTransformerEmbedder",
    ),
    "assess_relevance": (".confidence", "assess_relevance"),
    "build_pipeline": (".service", "build_pipeline"),
    "get_pipeline": (".service", "get_pipeline"),
    "textbook_retrieve": (".service", "textbook_retrieve"),
    "textbook_retrieve_checked": (".service", "textbook_retrieve_checked"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
