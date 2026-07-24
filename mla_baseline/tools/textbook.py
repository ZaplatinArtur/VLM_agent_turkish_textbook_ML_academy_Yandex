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


def textbook_search(settings: Settings, query: str, top_k: int | None = None,
                    subject: str | None = None, grade: int | str | None = None,
                    mode: str = "or") -> str:
    body: dict = {"query": query, "top_k": top_k or settings.rag_top_k, "mode": mode}
    if subject:
        body["subject"] = subject
    if grade is not None and grade != "":
        body["grade"] = grade
    try:
        req = urllib.request.Request(
            f"{settings.textbook_search_url.rstrip('/')}/api/search",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "mla-baseline/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return f"Arama hatası: {type(exc).__name__}. Aramasız devam et."

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
        ) if x)
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
