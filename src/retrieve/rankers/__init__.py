from __future__ import annotations

from typing import Any

from .base import Ranker
from .bm25 import BM25Ranker
from .dense import DenseRanker
from .reranker import CrossEncoderRanker
from .fusion import DEFAULT_RRF_K, ReciprocalRankFusion

__all__ = [
    "DEFAULT_RRF_K",
    "BM25Ranker",
    "CrossEncoderRanker",
    "DenseRanker",
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
