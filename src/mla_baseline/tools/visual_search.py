"""Визуальный ретрив под протоколом TextbookSearchBackend.

ColQwen ищет текстовым запросом по картинкам страниц, поэтому подменяется
целиком бэкенд одной и той же тулы: схема, промпт и цикл агента не меняются,
режим остаётся переменной эксперимента, а не решением модели.

Транспорты:
    inprocess — индекс в процессе агента, раскладка «воркер на карту»;
    http      — сервис на своей карте, когда воркеров больше, чем карт.

Байты страниц в выдачу не кладутся: format_search_result_for_model переносит
metadata в промпт дословно, и base64 уехал бы прямо в контекст модели. Хит
несёт page_id, а картинку по нему отдаёт get_page_image.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Literal

from .textbook_search import TextbookSearchError, TextbookSearchInput

# Профиль зовётся так же, как его репортит visual_retrive.search: трассы прогона
# и записи о пороге должны сходиться по одному имени.
VISUAL_PROFILE = "visual_colqwen25_cascade"

DEFAULT_URL = "http://127.0.0.1:8780"
DEFAULT_TIMEOUT = (5, 60)
# Держим ровно столько страниц, сколько может понадобиться на одну задачу:
# два вызова поиска по top_k, дальше картинки уже не нужны.
DEFAULT_IMAGE_CACHE = 16


class VisualSearchClient:
    """Отдаёт ту же форму ответа, что LocalTextbookSearchClient."""

    def __init__(
        self,
        *,
        transport: Literal["inprocess", "http"] = "inprocess",
        url: str | None = None,
        index_dir: str | None = None,
        min_score: float | None = None,
        timeout: tuple[int, int] = DEFAULT_TIMEOUT,
        image_cache_size: int = DEFAULT_IMAGE_CACHE,
        retriever: Any | None = None,
    ) -> None:
        if transport not in ("inprocess", "http"):
            raise ValueError(f"unknown transport {transport!r}: inprocess | http")
        if min_score is None:
            raise ValueError(
                "визуальному профилю не задан порог уверенности. Шкала MaxSim — "
                "сумма максимумов косинуса по токенам запроса, она уходит далеко "
                "за единицу, поэтому дефолтный порог пропустил бы любую выдачу. "
                "Снимите порог калибровкой и передайте MLA_VISUAL_MIN_SCORE."
            )
        if image_cache_size < 1:
            raise ValueError("image_cache_size must be positive")
        self.transport = transport
        self.url = (url or DEFAULT_URL).rstrip("/")
        self.index_dir = index_dir
        self.min_score = float(min_score)
        self.timeout = timeout
        self._retriever = retriever
        self._images: OrderedDict[str, bytes] = OrderedDict()
        self._images_lock = threading.Lock()
        self._image_cache_size = image_cache_size

    # --- транспорты -------------------------------------------------------

    def _get_retriever(self):
        # Импорт ленивый: поднимать torch и колпали при импорте агента не нужно.
        if self._retriever is None:
            try:
                from visual_retrive.service import visual_page_retrieve
            except ImportError as exc:
                raise TextbookSearchError(
                    "визуальный ретрив недоступен: не установлен colpali-engine "
                    "или его зависимости. Поставьте pip install -e '.[visual]'"
                ) from exc
            self._retriever = visual_page_retrieve
        return self._retriever

    def _search_inprocess(
        self,
        query: str,
        *,
        top_k: int,
        subject: str | None,
        grade: int | str | None,
    ) -> dict[str, Any]:
        retrieve = self._get_retriever()
        try:
            return retrieve(
                query,
                k=top_k,
                subject=subject,
                grade=grade,
                index_dir=self.index_dir,
            )
        except FileNotFoundError as exc:
            raise TextbookSearchError(
                f"индекс визуального ретрива не найден: {exc}. "
                "Укажите MLA_VISUAL_INDEX_DIR"
            ) from exc
        except Exception as exc:
            raise TextbookSearchError(f"visual retrieval failed: {exc}") from exc

    def _search_http(
        self,
        query: str,
        *,
        top_k: int,
        subject: str | None,
        grade: int | str | None,
    ) -> dict[str, Any]:
        import requests

        payload = {"query": query, "top_k": top_k, "subject": subject, "grade": grade}
        try:
            response = requests.post(
                f"{self.url}/api/search",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise TextbookSearchError(
                f"visual search service at {self.url} failed: {exc}"
            ) from exc

    # --- картинки ---------------------------------------------------------

    def _remember_image(self, page_id: str, encoded: str) -> None:
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception:
            return
        with self._images_lock:
            self._images[page_id] = raw
            self._images.move_to_end(page_id)
            while len(self._images) > self._image_cache_size:
                self._images.popitem(last=False)

    def get_page_image(self, page_id: str) -> bytes | None:
        """Байты страницы из последних выдач; None — картинки не было."""
        with self._images_lock:
            raw = self._images.get(page_id)
            if raw is not None:
                self._images.move_to_end(page_id)
            return raw

    # --- протокол ---------------------------------------------------------

    def _as_hit(self, raw: dict[str, Any], rank: int) -> dict[str, Any]:
        page_id = str(raw.get("page_id") or "")
        encoded = raw.get("page_image_b64")
        if page_id and isinstance(encoded, str) and encoded:
            self._remember_image(page_id, encoded)

        metadata = {
            "page_id": page_id,
            "book_slug": raw.get("book_slug"),
            "page_number": raw.get("page_number"),
            "subject": raw.get("subject"),
            "grade": raw.get("grade"),
            "has_solution": raw.get("has_solution"),
            # Путь оставляем строкой ради воспроизводимости трассы; сами байты
            # в промпт не идут, их забирает get_page_image по page_id.
            "page_image": raw.get("page_image"),
            "has_page_image": bool(encoded),
        }
        hit = {
            "chunk_id": page_id,
            "page_id": page_id,
            "text": str(raw.get("answer_text") or ""),
            "score": raw.get("score"),
            "rank": rank,
            "subject": raw.get("subject"),
            "grade": raw.get("grade"),
            "book_id": raw.get("book_slug"),
            "page_number": raw.get("page_number"),
            "metadata": {k: v for k, v in metadata.items() if v is not None},
        }
        return {k: v for k, v in hit.items() if v is not None}

    def _verdict(self, hits: list[dict[str, Any]]) -> dict[str, Any]:
        from retrieve.confidence import Relevance, RelevanceVerdict, assess_relevance
        from schemas.retrieve import RetrievedChunk

        if not hits:
            verdict = RelevanceVerdict(Relevance.EMPTY, None, "выдача пуста")
        else:
            chunks = [
                RetrievedChunk(
                    chunk_id=str(hit.get("chunk_id") or ""),
                    text=str(hit.get("text") or ""),
                    score=float(hit.get("score") or 0.0),
                    metadata={},
                )
                for hit in hits
            ]
            # min_score задаём явно: визуального профиля нет в реестре порогов
            # и быть не должно — его калибруют под свой индекс.
            verdict = assess_relevance(chunks, min_score=self.min_score)
        return {
            "label": str(verdict.relevance.value),
            "is_useful": bool(verdict.is_useful),
            "top_score": verdict.top_score,
            "reason": str(verdict.reason),
        }

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
            query=query,
            top_k=top_k,
            subject=subject,
            grade=grade,
            mode=mode,
        )
        started = time.perf_counter()
        search = (
            self._search_inprocess
            if self.transport == "inprocess"
            else self._search_http
        )
        raw = search(
            arguments.query,
            top_k=arguments.top_k,
            subject=arguments.subject,
            grade=arguments.grade,
        )
        raw_hits = raw.get("hits") if isinstance(raw.get("hits"), list) else []
        hits = [
            self._as_hit(hit, rank)
            for rank, hit in enumerate(raw_hits, 1)
            if isinstance(hit, dict)
        ]
        relevance = self._verdict(hits)
        # Слабой выдачи агент не видит — то же правило, что у текстового клиента.
        visible = hits if relevance["is_useful"] else []
        return {
            "query": arguments.query,
            "top_k": arguments.top_k,
            "context_order": "score",
            # mode лексический, у визуального поиска смысла не имеет: возвращаем
            # как пришло, чтобы трасса не врала о том, что фильтр применялся.
            "mode": arguments.mode,
            "profile": VISUAL_PROFILE,
            "transport": self.transport,
            "filters": {"subject": arguments.subject, "grade": arguments.grade},
            "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
            "retrieved": len(hits),
            "returned": len(visible),
            "relevance": relevance,
            "hits": visible,
        }


def visual_client_from_env(settings: Any | None = None) -> VisualSearchClient:
    """Собирает клиента из переменных окружения, не трогая умолчания в файлах."""
    min_score_raw = os.environ.get("MLA_VISUAL_MIN_SCORE", "").strip()
    if not min_score_raw and settings is not None:
        configured = getattr(settings, "visual_min_score", None)
        min_score_raw = "" if configured is None else str(configured)
    if not min_score_raw:
        raise ValueError(
            "MLA_VISUAL_MIN_SCORE не задан: визуальный профиль без калибровки "
            "не запускается. Снимите порог и пропишите его в окружении."
        )
    transport = os.environ.get("MLA_VISUAL_TRANSPORT", "inprocess").strip() or "inprocess"
    return VisualSearchClient(
        transport=transport,  # type: ignore[arg-type]
        url=os.environ.get("MLA_VISUAL_URL") or None,
        index_dir=os.environ.get("MLA_VISUAL_INDEX_DIR") or None,
        min_score=float(min_score_raw),
    )
