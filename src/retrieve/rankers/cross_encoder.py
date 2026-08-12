"""Teslov-style cross-encoder reranking for the current retrieval API.

This is a narrow port of D. Teslov's ``CrossEncoderRanker`` from commit
2116cb1288a8f773e1ccd929dac05382d319327e.  Heavy ML dependencies and model
weights are loaded lazily, so merely importing the retrieval package remains
cheap.  The optional LoRA adapter is deliberately explicit: the adapter used
for Teslov's tuned measurements was not committed to this repository.
"""

from __future__ import annotations

import math
import os
import inspect
import threading
from pathlib import Path
from typing import Any, Protocol

from schemas.retrieve import RetrievedChunk

from .base import Ranker, rescored


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
ADAPTER_ENV = "RETRIEVE_RERANKER_ADAPTER"


class CrossEncoderLike(Protocol):
    def predict(self, pairs: list[list[str]], batch_size: int = 32) -> Any: ...


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _needs_sigmoid(model: CrossEncoderLike) -> bool:
    """Return whether ``predict`` appears to expose logits, not probabilities."""
    activation = getattr(model, "default_activation_function", None)
    return "sigmoid" not in type(activation).__name__.lower()


class CrossEncoderRanker(Ranker):
    """Rerank a bounded candidate head with BGE or a compatible cross-encoder."""

    def __init__(
            self,
            model_name: str = DEFAULT_RERANKER_MODEL,
            *,
            revision: str | None = DEFAULT_RERANKER_REVISION,
            top_n: int = 100,
            batch_size: int = 32,
            cross_encoder: CrossEncoderLike | None = None,
            activation: str = "auto",
            adapter_path: Path | str | None = None,
            local_files_only: bool = True,
            device: str | None = None,
    ) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_name must not be empty")
        if revision is not None and not revision.strip():
            raise ValueError("revision must not be blank")
        if type(top_n) is not int or top_n <= 0:
            raise ValueError("top_n must be positive")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if activation not in {"auto", "sigmoid", "none"}:
            raise ValueError("activation must be 'auto', 'sigmoid', or 'none'")
        if type(local_files_only) is not bool:
            raise ValueError("local_files_only must be a boolean")

        configured_adapter = adapter_path or os.environ.get(ADAPTER_ENV) or None
        if configured_adapter is not None:
            raise RuntimeError(
                "the tuned reranker adapter is unavailable and unattested; "
                "use the pinned base model until an immutable adapter manifest "
                "is supplied"
            )
        self.model_name = model_name
        self.revision = revision.strip() if revision is not None else None
        self.top_n = top_n
        self.batch_size = batch_size
        self.activation = activation
        self.adapter_path: Path | None = None
        self.local_files_only = local_files_only
        self.device = device.strip() if device else None
        self._model = cross_encoder
        self._load_lock = threading.Lock()

    @property
    def model(self) -> CrossEncoderLike:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import CrossEncoder

                    device = self.device or (
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    dtype = torch.float16 if device.startswith("cuda") else torch.float32
                    cross_encoder_parameters = inspect.signature(
                        CrossEncoder.__init__
                    ).parameters
                    model_arguments = {"torch_dtype": dtype}
                    optional_arguments: dict[str, Any] = {}
                    if "model_kwargs" in cross_encoder_parameters:
                        optional_arguments["model_kwargs"] = model_arguments
                    elif "automodel_args" in cross_encoder_parameters:
                        # sentence-transformers 3.x used this older keyword.
                        optional_arguments["automodel_args"] = model_arguments
                    model = CrossEncoder(
                        self.model_name,
                        revision=self.revision,
                        local_files_only=self.local_files_only,
                        trust_remote_code=False,
                        device=device,
                        **optional_arguments,
                    )
                    self._model = model
        return self._model

    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
            grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        del subject, grade
        if not chunks:
            return list(chunks or [])
        head = chunks[:self.top_n]
        tail = chunks[self.top_n:]
        pairs = [[query, chunk.text] for chunk in head]
        scores = self.score_pairs(pairs)
        return rescored(head, tail, scores)

    def score_pairs(self, pairs: list[list[str]]) -> list[float]:
        """Score query/document pairs in one model call for batched diagnostics."""

        if any(
            type(pair) is not list
            or len(pair) != 2
            or any(type(value) is not str or not value for value in pair)
            for pair in pairs
        ):
            raise ValueError("each cross-encoder pair must contain two strings")
        if not pairs:
            return []
        model = self.model
        raw_scores = model.predict(pairs, batch_size=self.batch_size)
        values = [float(score) for score in raw_scores]
        if len(values) != len(pairs) or any(
            not math.isfinite(score) for score in values
        ):
            raise ValueError("cross-encoder returned malformed scores")
        squash = (
            self.activation == "sigmoid"
            or (self.activation == "auto" and _needs_sigmoid(model))
        )
        return [_sigmoid(value) for value in values] if squash else values
