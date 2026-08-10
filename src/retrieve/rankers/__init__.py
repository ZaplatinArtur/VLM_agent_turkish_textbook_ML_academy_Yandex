from __future__ import annotations

from typing import Any

from .base import Ranker
from .fusion import (
    DEFAULT_PRIMARY_CANDIDATE_WEIGHT,
    DEFAULT_RRF_K,
    DEFAULT_SEMANTIC_CANDIDATE_WEIGHT,
    PrimaryCandidateUnion,
    ReciprocalRankFusion,
)
from .graph import GraphExpansionRanker
from .lexical import BM25Ranker
from .rerank import KnowledgeReranker

__all__ = [
    "BM25Ranker",
    "CrossEncoderRanker",
    "DEFAULT_PRIMARY_CANDIDATE_WEIGHT",
    "DEFAULT_RRF_K",
    "DEFAULT_SEMANTIC_CANDIDATE_WEIGHT",
    "DenseRanker",
    "GraphExpansionRanker",
    "KnowledgeReranker",
    "PrimaryCandidateUnion",
    "Ranker",
    "ReciprocalRankFusion",
]

_LAZY = {
    "DenseRanker": ("dense", "DenseRanker"),
    "CrossEncoderRanker": ("reranker", "CrossEncoderRanker"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
