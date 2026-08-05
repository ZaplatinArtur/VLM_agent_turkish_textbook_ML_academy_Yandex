"""Поиск по корпусу учебников: HTTP-клиент BM25-сервера команды ретрива.

Контракт — docs/retrieval_tool_contract.md ветки feature/judge-agent-e2e:
POST {url}/api/search {"query", "top_k", "subject", "grade", "mode"} ->
{"hits": [{chunk_id, text, score, book_id, page_number, source_url, ...}]}.

Модели отдаём пронумерованный список фрагментов с провенансом (в стиле
searxng_search); chunk_id сохраняем в строке — он нужен для трассировки.
Ошибки поиска не роняют прогон.
"""

import json
import urllib.request

from ..config import Settings
from . import ToolUnavailable

# лейблы предметов в корпусе — ascii-слаги нижнего регистра ("turkce",
# "matematik"); модель же шлёт "Türkçe"/"Matematik" — нормализуем
_TR_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")


def _norm_subject(value: str) -> str:
    return value.translate(_TR_ASCII).casefold().strip()


def _post_search(settings: Settings, body: dict) -> dict:
    req = urllib.request.Request(
        f"{settings.textbook_search_url.rstrip('/')}/api/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "mla-baseline/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def textbook_search(settings: Settings, query: str, top_k: int | None = None,
                    subject: str | None = None, grade: int | str | None = None,
                    mode: str = "or") -> str:
    body: dict = {"query": query, "top_k": top_k or settings.rag_top_k, "mode": mode}
    if subject:
        body["subject"] = _norm_subject(str(subject))
    if grade is not None and grade != "":
        body["grade"] = grade
    try:
        data = _post_search(settings, body)
        hits = data.get("hits") or []
        if not hits and ("subject" in body or "grade" in body):
            # фильтры точные, а корпус — только 1-8 классы и слаг-лейблы:
            # пустой результат с фильтрами перепроверяем без них
            data = _post_search(settings, {k: v for k, v in body.items()
                                           if k in ("query", "top_k", "mode")})
    except Exception as exc:
        # сервер ретрива не отвечает — переформулировка не поможет, тул снимут
        raise ToolUnavailable(
            "Ders kitabı arama servisi çalışmıyor. Aramadan, kendi bilginle çöz.",
            {"error": type(exc).__name__, "url": settings.textbook_search_url},
        ) from exc

    hits = data.get("hits") or []
    if not hits:
        return ("Ders kitaplarında sonuç bulunamadı. Sorguyu kısalt/değiştir "
                "veya aramadan kendi bilginle çöz.")

    remaining = settings.rag_max_context_chars
    lines = []
    for i, h in enumerate(hits, 1):
        if remaining <= 0:
            break
        src = " · ".join(str(x) for x in (
            h.get("book_id"), h.get("subject"),
            f"sayfa {h.get('page_number')}" if h.get("page_number") is not None else None,
        ) if x) or str(h.get("page_id") or "")
        text = (h.get("text") or "").strip()[:remaining]
        remaining -= len(text)
        lines.append(f"[{i}] ({src or 'kaynak yok'}; chunk {h.get('chunk_id')})\n{text}")
    return "\n\n".join(lines)


# JSON-схема инструмента для bind_tools (OpenAI-формат, по контракту ретрива)
TEXTBOOK_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_textbooks",
        "description": (
            "Onaylı Türkçe ders kitabı ve çözüm korpusunda arama yapar: teori, "
            "formüller, çözümlü örnekler, benzer alıştırmalar. Sonuçlar kanıttır; "
            "cevabı körü körüne kopyalamak için değil."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": ("Kısa Türkçe sorgu: konu + işlem, formül "
                                          "veya ayırt edici terimler. Soru metnini "
                                          "kopyalama.")},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "subject": {"type": "string",
                            "description": "Ders adı (biliniyorsa; tahmin etme)"},
                "grade": {"type": "integer",
                          "description": "Sınıf (biliniyorsa)"},
                "mode": {"type": "string", "enum": ["or", "and"],
                         "description": "or = geniş arama (önerilen), and = dar"},
            },
            "required": ["query"],
        },
    },
}
