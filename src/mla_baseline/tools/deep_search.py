"""Deep-search: SearXNG -> полные страницы (trafilatura) -> реранкинг (bge-v2-m3).

Открытый стек уровня deep-research-реализаций вместо сырых сниппетов:
модель получает верхние фрагменты полных текстов с указанием источников.
Все компоненты self-hosted: SearXNG (docker), bge-reranker-v2-m3 (vLLM
score-API, мультиязычный, турецкий поддерживается), trafilatura.
"""

import concurrent.futures
import json
import urllib.parse
import urllib.request

from ..config import Settings
from .searx import MSG_EMPTY, search_or_raise

_HDRS = {"User-Agent": "Mozilla/5.0 (compatible; mla-baseline/0.1)"}


def _searx_urls(settings: Settings, query: str, k: int) -> list[dict]:
    """Ссылки для полнотекстового стека; сеть и ретраи — в searx.py."""
    return search_or_raise(settings, query).results[:k]


def _fetch_text(url: str, timeout: float = 10.0) -> str:
    import trafilatura
    try:
        req = urllib.request.Request(url, headers=_HDRS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read(1_500_000).decode(resp.headers.get_content_charset() or "utf-8",
                                               errors="replace")
        return trafilatura.extract(html, include_comments=False) or ""
    except Exception:
        return ""


def _chunks(text: str, size: int = 700, overlap: int = 100) -> list[str]:
    out = []
    step = size - overlap
    for i in range(0, len(text), step):
        c = text[i:i + size].strip()
        if len(c) > 120:
            out.append(c)
        if len(out) >= 40:  # достаточно кандидатов с одной страницы
            break
    return out


def _rerank(settings: Settings, query: str, docs: list[str]) -> list[float]:
    body = json.dumps({"model": "BAAI/bge-reranker-v2-m3", "query": query,
                       "documents": docs}).encode()
    req = urllib.request.Request(
        f"{settings.rerank_url.rstrip('/')}/v1/rerank", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    scores = [0.0] * len(docs)
    for item in data.get("results", []):
        scores[item["index"]] = item["relevance_score"]
    return scores


def deep_search(settings: Settings, query: str) -> str:
    """Возвращает топ-фрагменты полных страниц по запросу, с источниками."""
    # ToolUnavailable намеренно не глушим: цикл снимет инструмент, а не
    # заставит модель искать заново в мёртвый бэкенд
    hits = _searx_urls(settings, query, settings.deep_search_pages)
    if not hits:
        return MSG_EMPTY

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        texts = list(pool.map(lambda h: _fetch_text(h.get("url", "")), hits))

    cands: list[tuple[str, str]] = []  # (chunk, url)
    for h, text in zip(hits, texts):
        src = h.get("url", "")
        if text:
            for c in _chunks(text):
                cands.append((c, src))
        elif h.get("content"):           # страница не открылась — хотя бы сниппет
            cands.append((h["content"].strip(), src))
    if not cands:
        return "Sayfalar açılamadı. Aramasız devam et."

    try:
        scores = _rerank(settings, query, [c for c, _ in cands])
        ranked = sorted(zip(scores, cands), key=lambda x: -x[0])
    except Exception:                    # реранкер недоступен — порядок поиска
        ranked = [(0.0, c) for c in cands]

    out, used = [], set()
    for _score, (chunk, src) in ranked:
        if len(out) >= settings.deep_search_chunks:
            break
        key = chunk[:80]
        if key in used:
            continue
        used.add(key)
        out.append(f"[{len(out) + 1}] Kaynak: {src}\n{chunk}")
    return "\n\n".join(out)
