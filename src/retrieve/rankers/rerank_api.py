"""Реранкер по HTTP: vLLM отдаёт /v1/rerank для любой score-модели.

Нужен ради Qwen3-Reranker: тот не обычный кросс-энкодер, а causal LM со скором
из логитов yes/no, и sentence_transformers.CrossEncoder его не поднимет. Через
vLLM это деталь сервера, клиенту приходит тот же relevance_score в [0, 1], что и
у bge, — значит порог из confidence.py общий.

Адрес — RETRIEVE_RERANK_URL, по умолчанию совпадает с Settings.rerank_url, чтобы
эта тула и deep_search смотрели в один сервер.
"""

from __future__ import annotations

import json
import os
import urllib.request

from schemas.retrieve import RetrievedChunk

from .base import Ranker, rescored

QWEN3_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANK_URL_ENV = "RETRIEVE_RERANK_URL"
DEFAULT_RERANK_URL = "http://127.0.0.1:8002"


class RerankApiRanker(Ranker):
    def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            top_n: int = 100,
            timeout: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = (
            base_url or os.environ.get(RERANK_URL_ENV) or DEFAULT_RERANK_URL
        ).rstrip("/")
        self.top_n = top_n
        self.timeout = timeout

    def _scores(self, query: str, documents: list[str]) -> list[float]:
        body = json.dumps(
            {"model": self.model_name, "query": query, "documents": documents},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/rerank",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        # Сервер возвращает результаты отсортированными и с индексом исходного
        # документа — раскладываем обратно по позициям.
        scores = [0.0] * len(documents)
        for item in payload.get("results", []):
            scores[int(item["index"])] = float(item["relevance_score"])
        return scores

    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return list(chunks or [])
        head = chunks[: self.top_n]
        return rescored(head, chunks[self.top_n:], self._scores(query, [c.text for c in head]))
