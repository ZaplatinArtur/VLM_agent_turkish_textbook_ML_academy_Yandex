# -*- coding: utf-8 -*-
"""Проверки устойчивого клиента SearXNG без сети и без vLLM.

Подменяем `searx._fetch` и проверяем то, ради чего клиент написан: пустота
при отвалившихся движках лечится ретраем, честная пустота — нет, мёртвый
бэкенд снимает инструмент, а не заставляет модель искать снова.

Запуск: python tests/test_searx_client.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mla_baseline.config import Settings  # noqa: E402
from mla_baseline.tools import ToolUnavailable  # noqa: E402
from mla_baseline.tools import searx  # noqa: E402

OK_BODY = {"results": [{"url": "https://a.tr", "title": "A", "content": "metin"}],
           "unresponsive_engines": []}
EMPTY_BODY = {"results": [], "unresponsive_engines": []}
DEAD_BODY = {"results": [], "unresponsive_engines": [["google", "timeout"],
                                                     ["bing", "CAPTCHA"]]}

checks = 0


def check(condition: bool, label: str) -> None:
    global checks
    checks += 1
    if not condition:
        raise AssertionError(label)
    print(f"  ok: {label}")


def settings(**over) -> Settings:
    base = dict(searx_backoff_s=0.0, searx_min_interval_s=0.0, searx_cache_ttl_s=0.0,
                searx_retries=2, searx_fallback_engines="", searx_language="tr")
    base.update(over)
    return Settings(**base)


def fake(responses: list) -> tuple:
    """responses: элементы (status, body) или исключения. Возвращает (fn, calls)."""
    calls: list[dict] = []

    def _fetch(base_url, params, timeout_s):
        calls.append(dict(params))
        item = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    return _fetch, calls


def run(responses: list, **over):
    searx.reset_state()
    searx._fetch, calls = fake(responses)
    return searx.search(settings(**over), "kimya simya"), calls


def main() -> int:
    print("Клиент SearXNG")

    # 1. Обычная выдача — одна попытка, статус ok
    resp, calls = run([(200, OK_BODY)])
    check(resp.status == searx.OK and len(calls) == 1, "выдача есть → ok за один запрос")

    # 2. Честная пустота: движки живы — не ретраим, но проходим лестницу
    resp, calls = run([(200, EMPTY_BODY)])
    langs = [c.get("language") for c in calls]
    check(resp.status == searx.EMPTY, "пусто при живых движках → empty")
    check(len(calls) == 2 and langs == ["tr", None],
          "честная пустота не ретраится, но пробуется без языка")

    # 3. Пустота с отвалившимися движками — это не «не нашлось»
    resp, calls = run([(200, DEAD_BODY)])
    check(resp.status == searx.UNAVAILABLE, "пусто при мёртвых движках → unavailable")
    check(len(calls) == 6, "две ступени × три попытки (1 + 2 ретрая)")
    check(resp.diag["attempts"][0]["dead_engines"] == ["google", "bing"],
          "в диагностику попали имена отвалившихся движков")

    # 4. Ретрай спасает: первая попытка мертва, вторая приносит результат
    resp, calls = run([(200, DEAD_BODY), (200, OK_BODY)])
    check(resp.status == searx.OK and len(calls) == 2, "ретрай после отказа даёт выдачу")

    # 5. Транспортная ошибка = недоступность, а не пустота
    resp, calls = run([TimeoutError("timed out")])
    check(resp.status == searx.UNAVAILABLE, "таймаут → unavailable")
    check(resp.diag["attempts"][0]["error"] == "TimeoutError",
          "тип ошибки виден в диагностике")

    # 6. Третья ступень лестницы — резервные движки
    resp, calls = run([(200, EMPTY_BODY)], searx_fallback_engines="duckduckgo,brave")
    check(len(calls) == 3 and calls[-1].get("engines") == "duckduckgo,brave",
          "при пустоте пробуются резервные движки")

    # 7. Размыкатель: после серии отказов сеть не трогаем
    searx.reset_state()
    searx._fetch, calls = fake([TimeoutError("timed out")])
    cfg = settings(searx_unavailable_streak=2, searx_cooldown_s=60.0)
    for _ in range(2):
        searx.search(cfg, "kimya")
    before = len(calls)
    resp = searx.search(cfg, "başka sorgu")
    check(resp.status == searx.UNAVAILABLE and len(calls) == before,
          "разомкнутая цепь отвечает без сетевых запросов")
    check(resp.diag.get("breaker") == "open", "в логе видно, что сработал размыкатель")

    # 8. Кэш: повтор запроса не идёт в сеть
    searx.reset_state()
    searx._fetch, calls = fake([(200, OK_BODY)])
    cfg = settings(searx_cache_ttl_s=900.0)
    searx.search(cfg, "aynı sorgu")
    # регистр не важен; casefold в Python не турецкий (I→i, не I→ı),
    # поэтому берём вариант, где различие только в первых буквах слов
    resp = searx.search(cfg, "Aynı Sorgu")
    check(len(calls) == 1 and resp.status == searx.OK,
          "повтор запроса берётся из кэша (26% вызовов в прогонах — повторы)")

    # 9. На живом инстансе отвалившиеся движки не делают пустоту поломкой.
    #    Живая проба: brave и startpage были в капче на всех 25 запросах,
    #    при этом 24 из них вернули результаты (reports/searxng_probe.txt).
    searx.reset_state()
    searx._fetch, calls = fake([(200, OK_BODY), (200, DEAD_BODY)])
    cfg = settings()
    searx.search(cfg, "первый запрос")            # инстанс доказал, что живой
    before = len(calls)
    resp = searx.search(cfg, "точная цитата которой нет в сети")
    check(resp.status == searx.EMPTY,
          "пустота на недавно ответившем инстансе → empty, а не поломка")
    check(len(calls) - before == 2, "такую пустоту не ретраим, только лестница")

    # 10. Недоступность бэкенда — исключение для тул-цикла
    searx.reset_state()
    searx._fetch, _ = fake([(200, DEAD_BODY)])
    try:
        searx.search_or_raise(settings(), "kimya")
        raised = False
    except ToolUnavailable as exc:
        raised = exc.message_for_model == searx.MSG_UNAVAILABLE
    check(raised, "search_or_raise бросает ToolUnavailable с текстом для модели")

    # 10. Пустота исключением не является: запрос можно переформулировать
    searx.reset_state()
    searx._fetch, _ = fake([(200, EMPTY_BODY)])
    check(searx.search_or_raise(settings(), "kimya").status == searx.EMPTY,
          "честная пустота исключения не вызывает")

    print(f"\nвсе {checks} проверок прошли")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
