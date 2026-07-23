"""Embedding interfaces; model-backed implementations load on demand."""

from __future__ import annotations

from typing import Any

from .base import Embedder, SymmetricTextEmbedder

__all__ = [
    "Embedder",
    "SymmetricTextEmbedder",
    "SentenceTransformerEmbedder",
]


def __getattr__(name: str) -> Any:
    if name != "SentenceTransformerEmbedder":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .sentence_transformer import SentenceTransformerEmbedder

    globals()[name] = SentenceTransformerEmbedder
    return SentenceTransformerEmbedder
