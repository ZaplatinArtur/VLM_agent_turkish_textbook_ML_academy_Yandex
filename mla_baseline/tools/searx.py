# -*- coding: utf-8 -*-
"""Устойчивый клиент SearXNG: ретраи, диагностика, размыкатель, кэш.

Зачем. В прогонах b1 веб-поиск возвращал пустоту в 48–71% вызовов, и разбор
логов (`scripts/diagnose_web_search.py`) показал, что дело не в формулировках:
доля пустоты одинакова во всех корзинах по длине запроса (76–82%), а по ходу
прогона идёт окнами по 100% пустых. Так выглядит не «не нашлось», а неработающий
бэкенд — движки в бане/таймауте, инстанс троттлит.

Что делает клиент:
  * различает «ничего не нашлось» и «поиск не работает» — по признаку
    unresponsive_engines в ответе SearXNG и по транспортным ошибкам;
  * ретраит с паузой то, что имеет смысл ретраить (сеть, 429/5xx, пустота при
    отвалившихся движках), и не ретраит честную пустоту;
  * эскалирует запрос: язык → без языка → резервный список движков;
  * размыкает цепь после серии отказов, чтобы не платить таймаутами за
    заведомо мёртвый бэкенд, и сам пробует восстановиться после паузы;
  * держит паузу между запросами и кэширует выдачу — бурст запросов от
    четырёх параллельных задач и есть то, за что движки банят;
  * возвращает диагностику (номер попытки, http-статус, отвалившиеся движки),
    которая ложится в лог прогона — без неё причину пустоты не восстановить.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from ..config import Settings
from . import ToolUnavailable

_HDRS = {"User-Agent": "Mozilla/5.0 (compatible; mla-baseline/0.2)"}

OK = "ok"
EMPTY = "empty"
UNAVAILABLE = "unavailable"

# Сообщения модели: они РАЗНЫЕ по смыслу и ведут к разным действиям.
MSG_EMPTY = "Sonuç bulunamadı. Sorguyu değiştir veya aramasız devam et."
MSG_UNAVAILABLE = "Arama servisi şu anda çalışmıyor. Aramadan, kendi bilginle çöz."


@dataclass
class SearxResponse:
    status: str                 # ok | empty | unavailable
    results: list[dict] = field(default_factory=list)
    diag: dict = field(default_factory=dict)


class _Breaker:
    """Размыкатель цепи на инстанс: N отказов подряд → пауза без сети."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fails: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def is_open(self, key: str) -> bool:
        with self._lock:
            return time.monotonic() < self._open_until.get(key, 0.0)

    def record(self, key: str, ok: bool, streak: int, cooldown_s: float) -> None:
        with self._lock:
            if ok:
                self._fails[key] = 0
                self._open_until.pop(key, None)
                return
            self._fails[key] = self._fails.get(key, 0) + 1
            if self._fails[key] >= streak:
                # цепь разомкнута; после паузы пропустим одну пробу и,
                # если она удачна, счётчик обнулится в ветке ok
                self._open_until[key] = time.monotonic() + cooldown_s
                self._fails[key] = 0


class _Alive:
    """Память об успехах инстанса и о том, какие движки их приносят.

    Нужна, чтобы не путать «не нашлось» с «поиск сломан». Отвалившиеся движки
    сами по себе ничего не доказывают: на живом инстансе с широким пулом
    несколько движков лежат почти всегда (проба показала brave и startpage в
    капче на всех 25 запросах при 24 успешных выдачах).

    Но и обратное неверно. Замер на 540 запросах: когда duckduckgo был жив,
    пустых 0 из 240, когда он уходил в капчу — 277 из 300. Пустота при живом
    инстансе означала не «не нашлось», а «слёг единственный движок, который
    что-то приносил». Поэтому помним, кто именно даёт выдачу, и если в бане
    оказались все они разом — это поломка, а не отсутствие результата.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._at: dict[str, float] = {}
        self._engines: dict[str, set[str]] = {}

    def mark(self, key: str, engines: set[str]) -> None:
        with self._lock:
            self._at[key] = time.monotonic()
            if engines:
                self._engines[key] = engines

    def recent(self, key: str, window_s: float) -> bool:
        with self._lock:
            return time.monotonic() - self._at.get(key, -1e9) < window_s

    def all_productive_dead(self, key: str, dead: list[str]) -> bool:
        """Все известные добытчики выдачи лежат?"""
        with self._lock:
            productive = self._engines.get(key)
        return bool(productive) and productive <= set(dead)


class _Pacer:
    """Минимальный интервал между исходящими запросами к инстансу."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self, interval_s: float) -> None:
        if interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + interval_s
        if delay:
            time.sleep(delay)


class _Cache:
    """Кэш выдачи с TTL: повторные запросы не бьют по движкам."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, tuple[float, SearxResponse]] = {}

    def get(self, key: str, ttl_s: float) -> SearxResponse | None:
        if ttl_s <= 0:
            return None
        with self._lock:
            item = self._data.get(key)
            if item is None or time.monotonic() - item[0] > ttl_s:
                return None
            cached = item[1]
        return SearxResponse(cached.status, list(cached.results),
                             {**cached.diag, "cache_hit": True})

    def put(self, key: str, value: SearxResponse, ttl_s: float) -> None:
        if ttl_s <= 0 or value.status == UNAVAILABLE:
            return  # недоступность не кэшируем: за неё отвечает размыкатель
        with self._lock:
            self._data[key] = (time.monotonic(), value)


_breaker = _Breaker()
_pacer = _Pacer()
_cache = _Cache()
_alive = _Alive()


def reset_state() -> None:
    """Сброс размыкателя/кэша/паузы — для тестов и повторных прогонов."""
    global _breaker, _pacer, _cache, _alive
    _breaker, _pacer, _cache, _alive = _Breaker(), _Pacer(), _Cache(), _Alive()


def _fetch(base_url: str, params: dict, timeout_s: float) -> tuple[int, dict]:
    url = base_url.rstrip("/") + "/search?" + urllib.parse.urlencode(
        {"q": params["q"], "format": "json",
         **{k: v for k, v in params.items() if k != "q" and v}})
    req = urllib.request.Request(url, headers=_HDRS)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        status = getattr(resp, "status", 200)
        return status, json.loads(resp.read())


def _probes(settings: Settings, query: str) -> list[dict]:
    """Лестница попыток: как просили → без языка → резервные движки."""
    probes = [{"q": query, "language": settings.searx_language}]
    if settings.searx_language:
        probes.append({"q": query})
    engines = [e.strip() for e in settings.searx_fallback_engines.split(",") if e.strip()]
    if engines:
        probes.append({"q": query, "engines": ",".join(engines)})
    return probes


def search(settings: Settings, query: str) -> SearxResponse:
    """Ищет с ретраями и эскалацией. Не бросает исключений — см. status."""
    base = settings.searxng_url.rstrip("/")
    cache_key = f"{base}|{settings.searx_language}|{query.casefold()}"
    cached = _cache.get(cache_key, settings.searx_cache_ttl_s)
    if cached is not None:
        return cached

    if _breaker.is_open(base):
        return SearxResponse(UNAVAILABLE, [], {"breaker": "open", "attempts": 0})

    attempts: list[dict] = []
    degraded = False          # видели признаки нерабочего бэкенда
    # инстанс, который недавно отдавал выдачу, считаем живым: его пустота —
    # это «не нашлось», а не поломка, и ретраить её незачем
    alive = _alive.recent(base, settings.searx_alive_window_s)
    for probe_no, probe in enumerate(_probes(settings, query), 1):
        for attempt in range(1, settings.searx_retries + 2):
            _pacer.wait(settings.searx_min_interval_s)
            record: dict = {"probe": probe_no, "attempt": attempt,
                            "params": {k: v for k, v in probe.items() if k != "q"}}
            try:
                status, data = _fetch(base, probe, settings.searx_timeout_s)
            except urllib.error.HTTPError as exc:
                record.update(http_status=exc.code, error="HTTPError")
                attempts.append(record)
                degraded = True
                if exc.code in (429, 500, 502, 503, 504) and \
                        attempt <= settings.searx_retries:
                    time.sleep(settings.searx_backoff_s * attempt)
                    continue
                break
            except Exception as exc:
                record.update(error=type(exc).__name__)
                attempts.append(record)
                degraded = True
                if attempt <= settings.searx_retries:
                    time.sleep(settings.searx_backoff_s * attempt)
                    continue
                break

            results = [r for r in (data.get("results") or []) if r.get("url")]
            dead = [e[0] if isinstance(e, (list, tuple)) and e else str(e)
                    for e in (data.get("unresponsive_engines") or [])]
            record.update(http_status=status, results=len(results), dead_engines=dead)
            attempts.append(record)

            if results:
                out = SearxResponse(OK, results, {"attempts": attempts})
                _cache.put(cache_key, out, settings.searx_cache_ttl_s)
                _alive.mark(base, {str(r.get("engine") or "") for r in results
                                   if r.get("engine")})
                _breaker.record(base, True, settings.searx_unavailable_streak,
                                settings.searx_cooldown_s)
                return out
            if dead and (not alive or _alive.all_productive_dead(base, dead)):
                # пусто и при этом либо инстанс давно молчит, либо в бане все
                # движки, которые вообще приносили выдачу, — это поломка
                degraded = True
                if attempt <= settings.searx_retries:
                    time.sleep(settings.searx_backoff_s * attempt)
                    continue
            break  # честная пустота — переходим к следующей ступени лестницы

    if degraded:
        _breaker.record(base, False, settings.searx_unavailable_streak,
                        settings.searx_cooldown_s)
        return SearxResponse(UNAVAILABLE, [], {"attempts": attempts})
    out = SearxResponse(EMPTY, [], {"attempts": attempts})
    _cache.put(cache_key, out, settings.searx_cache_ttl_s)
    _breaker.record(base, True, settings.searx_unavailable_streak,
                    settings.searx_cooldown_s)
    return out


def search_or_raise(settings: Settings, query: str) -> SearxResponse:
    """Как search, но недоступность бэкенда — исключение для тул-цикла."""
    response = search(settings, query)
    if response.status == UNAVAILABLE:
        raise ToolUnavailable(MSG_UNAVAILABLE, response.diag)
    return response
