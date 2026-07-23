"""Веб-поиск для B1: self-hosted SearXNG (JSON API).

Инструмент возвращает модели пронумерованный список результатов (заголовок,
URL, сниппет). Ошибки поиска не роняют прогон — модель получает текст ошибки
и может продолжить без поиска.
"""

import json
import urllib.parse
import urllib.request

from ..config import Settings


def searxng_search(settings: Settings, query: str) -> str:
    url = (f"{settings.searxng_url.rstrip('/')}/search?"
           + urllib.parse.urlencode({"q": query, "format": "json", "language": "tr"}))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mla-baseline/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return f"Arama hatası: {type(exc).__name__}. Aramasız devam et."

    results = data.get("results", [])[: settings.search_k]
    if not results:
        return "Sonuç bulunamadı. Sorguyu değiştir veya aramasız devam et."
    lines = []
    for i, r in enumerate(results, 1):
        snippet = (r.get("content") or "").strip()
        lines.append(f"{i}. {r.get('title', '').strip()}\n   {r.get('url', '')}"
                     + (f"\n   {snippet}" if snippet else ""))
    return "\n".join(lines)


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
