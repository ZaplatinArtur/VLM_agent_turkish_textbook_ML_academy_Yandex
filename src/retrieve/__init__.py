"""Public retrieval API with lazy imports for optional ML dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_ATTRS = {
    "DenseRanker": (".rankers", "DenseRanker"),
    "Embedder": (".embedders", "Embedder"),
    "EmbeddingCache": (".cache", "EmbeddingCache"),
    "FaissVectorStore": (".vector_store", "FaissVectorStore"),
    "Index": (".index", "Index"),
    "Ranker": (".rankers", "Ranker"),
    "ReciprocalRankFusion": (".rankers", "ReciprocalRankFusion"),
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
