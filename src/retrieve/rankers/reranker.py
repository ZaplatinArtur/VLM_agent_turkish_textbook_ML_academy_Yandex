from __future__ import annotations

import math
import threading
from typing import Any, Protocol

from schemas.retrieve import RetrievedChunk

from .base import Ranker

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderLike(Protocol):
    def predict(self, pairs: list[list[str]], batch_size: int = 32) -> Any: ...


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


class CrossEncoderRanker(Ranker):
    def __init__(
            self,
            model_name: str = DEFAULT_RERANKER_MODEL,
            top_n: int = 100,
            batch_size: int = 32,
            cross_encoder: CrossEncoderLike | None = None,
    ) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self.batch_size = batch_size
        self._model = cross_encoder
        self._load_lock = threading.Lock()

    @property
    def model(self) -> CrossEncoderLike:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name)
        return self._model

    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
            grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return list(chunks or [])
        head = chunks[: self.top_n]
        tail = chunks[self.top_n:]
        pairs = [[query, chunk.text] for chunk in head]
        raw_scores = self.model.predict(pairs, batch_size=self.batch_size)
        scored = [
            chunk.model_copy(update={"score": _sigmoid(float(score))})
            for chunk, score in zip(head, raw_scores)
        ]
        scored.sort(key=lambda chunk: -chunk.score)
        return scored + tail
