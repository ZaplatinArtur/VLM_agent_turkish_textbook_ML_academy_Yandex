"""Веб-поиск для B1: self-hosted SearXNG (JSON API).

Инструмент возвращает модели пронумерованный список результатов (заголовок,
URL, сниппет). Сеть, ретраи и различение «не нашлось» / «поиск не работает»
живут в `searx.py`; здесь только форматирование выдачи для модели.
"""

from ..config import Settings
from .searx import MSG_EMPTY, SearxResponse, search_or_raise


def format_results(response: SearxResponse, k: int) -> str:
    results = response.results[:k]
    if not results:
        return MSG_EMPTY
    lines = []
    for i, r in enumerate(results, 1):
        snippet = (r.get("content") or "").strip()
        lines.append(f"{i}. {r.get('title', '').strip()}\n   {r.get('url', '')}"
                     + (f"\n   {snippet}" if snippet else ""))
    return "\n".join(lines)


def searxng_search(settings: Settings, query: str) -> str:
    """Сниппеты по запросу. ToolUnavailable — если бэкенд не отвечает."""
    return format_results(search_or_raise(settings, query), settings.search_k)


# JSON-схема инструмента для bind_tools (OpenAI-формат)
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": ("Web'de arama yapar (Türkçe). Formüller, tanımlar, "
                        "tarihler ve emin olmadığın bilgiler için kullan."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Arama sorgusu (Türkçe)"},
            },
            "required": ["query"],
        },
    },
}
