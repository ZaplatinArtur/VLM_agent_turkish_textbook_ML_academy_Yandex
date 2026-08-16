"""Гибрид: текстовый и визуальный ретрив на один запрос, слияние по рангам.

Ветки ошибаются по-разному — текстовая ищет по словам после OCR, визуальная по
вёрстке, формуле и рисунку, — поэтому объединение может дать то, чего нет ни у
одной. Проверяется это замером, а не верой: гибрид имеет смысл включать, только
если обе чистые ветки показали ненулевой вклад.

Сливаем **по рангам, а не по счёту**: у кросс-энкодера шкала [0, 1], у MaxSim —
сумма по токенам запроса, уходящая за двадцать. Складывать их напрямую нельзя,
и нормировать тоже — распределения разной формы. RRF от шкалы не зависит.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from .textbook_search import TextbookSearchError, TextbookSearchInput

# Тот же rrf_k, что у слияния внутри текстового пайплайна (rankers/fusion.py):
# ранг важнее источника, и обе ветки входят с равным весом.
DEFAULT_RRF_K = 60

HYBRID_PROFILE = "hybrid_text_visual"


class HybridSearchClient:
    """Отдаёт ту же форму ответа, что текстовый и визуальный клиенты."""

    def __init__(
        self,
        text_client: Any,
        visual_client: Any,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        fetch_multiplier: int = 2,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if fetch_multiplier < 1:
            raise ValueError("fetch_multiplier must be at least 1")
        self.text_client = text_client
        self.visual_client = visual_client
        self.rrf_k = rrf_k
        # Каждая ветка отдаёт с запасом: после слияния часть позиций займёт
        # соседняя, и без запаса top_k выродился бы в выдачу одной из них.
        self.fetch_multiplier = fetch_multiplier

    def get_page_image(self, page_id: str) -> bytes | None:
        """Картинки живут в визуальной ветке и переживают слияние."""
        getter = getattr(self.visual_client, "get_page_image", None)
        return getter(page_id) if callable(getter) else None

    @staticmethod
    def _branch(client: Any, query: str, **kwargs: Any) -> dict[str, Any]:
        """Отказ одной ветки не роняет запрос: вторая ещё может ответить."""
        try:
            return client.search(query, **kwargs)
        except TextbookSearchError as exc:
            return {"hits": [], "relevance": {"label": "error", "is_useful": False,
                                              "top_score": None, "reason": str(exc)}}

    @staticmethod
    def _usable(branch: dict[str, Any]) -> list[dict[str, Any]]:
        """Слабую выдачу в слияние не пускаем — гейт уже сказал, что ей не верить."""
        relevance = branch.get("relevance") or {}
        if not relevance.get("is_useful"):
            return []
        hits = branch.get("hits")
        return [hit for hit in hits if isinstance(hit, dict)] if isinstance(hits, list) else []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        subject: str | None = None,
        grade: int | str | None = None,
        mode: Literal["or", "and"] = "or",
    ) -> dict[str, Any]:
        arguments = TextbookSearchInput(
            query=query, top_k=top_k, subject=subject, grade=grade, mode=mode
        )
        started = time.perf_counter()
        depth = arguments.top_k * self.fetch_multiplier
        branches = {
            "text": self._branch(self.text_client, arguments.query, top_k=depth,
                                 subject=arguments.subject, grade=arguments.grade,
                                 mode=arguments.mode),
            "visual": self._branch(self.visual_client, arguments.query, top_k=depth,
                                   subject=arguments.subject, grade=arguments.grade,
                                   mode=arguments.mode),
        }

        scores: dict[str, float] = {}
        seen: dict[str, dict[str, Any]] = {}
        sources: dict[str, list[str]] = {}
        for name, branch in branches.items():
            for rank, hit in enumerate(self._usable(branch), start=1):
                key = str(hit.get("chunk_id") or hit.get("page_id") or "")
                if not key:
                    continue
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                seen.setdefault(key, hit)
                sources.setdefault(key, []).append(name)

        ordered = sorted(seen, key=lambda key: -scores[key])[: arguments.top_k]
        hits = []
        for rank, key in enumerate(ordered, start=1):
            hit = dict(seen[key])
            hit["rank"] = rank
            hit["score"] = round(scores[key], 6)
            # Кто нашёл — видно в трассе: без этого вклад веток не разобрать.
            hit["found_by"] = sources[key]
            hits.append(hit)

        labels = {name: (branch.get("relevance") or {}).get("label")
                  for name, branch in branches.items()}
        if hits:
            relevance = {"label": "confident", "is_useful": True,
                         "top_score": hits[0]["score"],
                         "reason": f"слияние рангов, ветки: {labels}"}
        else:
            useful = [name for name, branch in branches.items() if self._usable(branch)]
            label = "error" if all(v == "error" for v in labels.values()) else "empty"
            relevance = {"label": label, "is_useful": False, "top_score": None,
                         "reason": f"ни одна ветка не дала выдачи, ветки: {labels}"
                         if not useful else f"пусто после слияния: {labels}"}

        return {
            "query": arguments.query,
            "top_k": arguments.top_k,
            "context_order": "score",
            "mode": arguments.mode,
            "profile": HYBRID_PROFILE,
            "filters": {"subject": arguments.subject, "grade": arguments.grade},
            "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
            "retrieved": len(seen),
            "returned": len(hits),
            "branches": {name: len(self._usable(branch))
                         for name, branch in branches.items()},
            "relevance": relevance,
            "hits": hits,
        }
