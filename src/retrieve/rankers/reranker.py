from __future__ import annotations

import math
import os
import inspect
import threading
from typing import Any, Protocol

from paths import RERANKER_ADAPTER_DIR

from schemas.retrieve import RetrievedChunk

from .base import Ranker, rescored

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# Ревизия пинится, иначе обновление весов на Hub разойдётся с замерами.
DEFAULT_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
# LoRA-адаптер дообученного реранкера: не отдельный профиль — состав пайплайна
# тот же, меняются веса одной ступени. Лежит в репозитории и берётся сам;
# переменной можно указать другой, значением "none" — отключить.
ADAPTER_ENV = "RETRIEVE_RERANKER_ADAPTER"


class CrossEncoderLike(Protocol):
    def predict(self, pairs: list[list[str]], batch_size: int = 32) -> Any: ...


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _resolve_adapter(adapter_path: str | None) -> str | None:
    """Явный путь важнее переменной, переменная — репозиторного адаптера."""
    if adapter_path:
        return adapter_path
    from_env = (os.environ.get(ADAPTER_ENV) or "").strip()
    if from_env:
        return None if from_env.casefold() == "none" else from_env
    return str(RERANKER_ADAPTER_DIR) if RERANKER_ADAPTER_DIR.is_dir() else None


def _needs_sigmoid(model: CrossEncoderLike) -> bool:
    """True, если predict отдаёт логиты, а не готовую вероятность.

    CrossEncoder применяет default_activation_function сам; вторая сигмоида
    поверх сплющит шкалу в [0.5, 0.73] и сломает порог по score.
    """
    activation = getattr(model, "default_activation_function", None)
    return "sigmoid" not in type(activation).__name__.lower()


class CrossEncoderRanker(Ranker):
    def __init__(
            self,
            model_name: str = DEFAULT_RERANKER_MODEL,
            top_n: int = 100,
            batch_size: int = 32,
            cross_encoder: CrossEncoderLike | None = None,
            activation: str = "auto",
            adapter_path: str | None = None,
            revision: str | None = DEFAULT_RERANKER_REVISION,
    ) -> None:
        if activation not in ("auto", "sigmoid", "none"):
            raise ValueError(f"unknown activation {activation!r}: auto | sigmoid | none")
        self.model_name = model_name
        self.revision = revision
        self.top_n = top_n
        self.batch_size = batch_size
        self.activation = activation
        self.adapter_path = _resolve_adapter(adapter_path)
        self._model = cross_encoder
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def model(self) -> CrossEncoderLike:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import CrossEncoder

                    # На карте fp16 (вчетверо быстрее, метрики те же), на CPU fp32.
                    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    parameters = inspect.signature(CrossEncoder).parameters
                    model_arguments = {"torch_dtype": dtype}
                    optional_arguments: dict[str, Any] = {}
                    if "model_kwargs" in parameters:
                        optional_arguments["model_kwargs"] = model_arguments
                    elif "automodel_args" in parameters:
                        optional_arguments["automodel_args"] = model_arguments
                    if self.revision is not None and "revision" in parameters:
                        optional_arguments["revision"] = self.revision
                    model = CrossEncoder(self.model_name, **optional_arguments)
                    if self.adapter_path:
                        from peft import PeftModel

                        # merge_and_unload вливает B·A обратно в веса, чтобы на
                        # инференсе адаптер ничего не стоил.
                        merged = PeftModel.from_pretrained(model.model, self.adapter_path)
                        model.model = merged.merge_and_unload()
                        model.model.eval()
                    self._model = model
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
        return rescored(head, tail, self.score_pairs([[query, c.text] for c in head]))

    def score_pairs(self, pairs: list[list[str]]) -> list[float]:
        """Оценивает пары (запрос, текст) одним вызовом модели."""
        if not pairs:
            return []
        model = self.model
        with self._predict_lock:
            raw_scores = model.predict(pairs, batch_size=self.batch_size)
        squash = _sigmoid if self.activation == "sigmoid" or (
            self.activation == "auto" and _needs_sigmoid(model)
        ) else float
        return [squash(float(score)) for score in raw_scores]
