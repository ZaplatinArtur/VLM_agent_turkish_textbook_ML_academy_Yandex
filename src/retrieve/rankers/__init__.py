"""Ranker interfaces with lazy loading of dense retrieval dependencies."""

from __future__ import annotations

from typing import Any

from .base import Ranker
from .fusion import DEFAULT_RRF_K, ReciprocalRankFusion

__all__ = ["DEFAULT_RRF_K", "DenseRanker", "Ranker", "ReciprocalRankFusion"]


def __getattr__(name: str) -> Any:
    if name != "DenseRanker":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .dense import DenseRanker

    globals()[name] = DenseRanker
    return DenseRanker
