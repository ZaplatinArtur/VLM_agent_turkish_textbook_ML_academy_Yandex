from __future__ import annotations

import threading
from typing import Any

from schemas.retrieve import RetrievedChunk

from .lexical import tokenize


class KnowledgeReranker:
    """Rerank fused candidates with quality signals and an optional cross-encoder."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        top_n: int = 40,
        batch_size: int = 8,
        max_length: int = 512,
        min_graph_theory_chars: int = 70,
    ) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_graph_theory_chars = max(0, min_graph_theory_chars)
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._inference_lock = threading.Lock()

    def _load_model(self) -> None:
        if not self.model_name or self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._model.eval()
        try:
            self._model.to("cuda")
        except Exception:
            pass

    @staticmethod
    def _heuristic(query: str, chunk: RetrievedChunk, base_score: float) -> float:
        query_tokens = set(tokenize(query))
        document = str(chunk.metadata.get("retrieval_text") or chunk.text)
        document_tokens = set(tokenize(document))
        overlap = len(query_tokens & document_tokens) / max(1, len(query_tokens))
        quality = 0.0
        # Graph connectivity describes what can be expanded after retrieval;
        # it is not evidence that the anchor matches this query. In particular,
        # solution/theory bonuses used to let a richly connected but unrelated
        # exercise outrank a lexically relevant theory block.
        length = len(document)
        if length < 24:
            quality -= 0.35
        elif 60 <= length <= 2_000:
            quality += 0.08
        return 0.48 * overlap + 0.32 * base_score + quality

    def _cross_scores(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[float] | None:
        if not self.model_name:
            return None
        with self._inference_lock:
            self._load_model()
            assert self._model is not None and self._tokenizer is not None
            import torch

            scores: list[float] = []
            device = next(self._model.parameters()).device
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start : start + self.batch_size]
                queries = [query] * len(batch)
                documents = [
                    str(chunk.metadata.get("retrieval_text") or chunk.text)
                    for chunk in batch
                ]
                encoded = self._tokenizer(
                    queries,
                    text_pair=documents,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.no_grad():
                    logits = self._model(**encoded).logits.reshape(-1)
                scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
            return scores

    def rank(
        self,
        query: str,
        chunks: list[RetrievedChunk] | None = None,
        subject: str | None = None,
        grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        del subject, grade
        candidates = [
            chunk
            for chunk in chunks or []
            if not (
                self.min_graph_theory_chars
                and chunk.metadata.get("knowledge_graph_node") is True
                and str(chunk.metadata.get("unit_kind")) == "theory"
                and len(
                    str(chunk.metadata.get("retrieval_text") or chunk.text).strip()
                )
                < self.min_graph_theory_chars
            )
        ]
        if not candidates:
            return []
        selected = candidates[: self.top_n]
        base_values = [float(chunk.score) for chunk in selected]
        minimum = min(base_values)
        maximum = max(base_values)
        span = maximum - minimum
        normalized = [
            (value - minimum) / span if span > 1e-12 else 1.0
            for value in base_values
        ]
        heuristic = [
            self._heuristic(query, chunk, base_score)
            for chunk, base_score in zip(selected, normalized)
        ]
        cross = self._cross_scores(query, selected)
        final = heuristic if cross is None else [
            0.78 * cross_score + 0.22 * heuristic_score
            for cross_score, heuristic_score in zip(cross, heuristic)
        ]
        reranked = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(selected, final)
        ]
        reranked.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        return reranked + candidates[self.top_n :]
