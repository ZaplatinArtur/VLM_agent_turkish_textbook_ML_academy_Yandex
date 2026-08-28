"""High-level ColModern visual page search (agent + offline eval)."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from .maxsim_index import DEFAULT_MAXSIM_INDEX_DIR, MaxSimPageIndex
from .page_index import DEFAULT_INDEX_DIR, ColModernPageIndex
from .colqwen_index import DEFAULT_COLQWEN_INDEX_DIR, ColQwenCascadeIndex


def _default_index_dir() -> Path:
    env = os.environ.get("MLA_VISUAL_INDEX_DIR")
    if env:
        return Path(env)
    if (DEFAULT_COLQWEN_INDEX_DIR / "meta.json").is_file():
        return DEFAULT_COLQWEN_INDEX_DIR
    if (DEFAULT_MAXSIM_INDEX_DIR / "meta.json").is_file():
        return DEFAULT_MAXSIM_INDEX_DIR
    return DEFAULT_INDEX_DIR


@lru_cache(maxsize=2)
def get_page_index(
    index_dir: str | None = None,
    *,
    load_model: bool = True,
) -> MaxSimPageIndex | ColModernPageIndex | ColQwenCascadeIndex:
    path = Path(index_dir) if index_dir else _default_index_dir()
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(
            f"ColModern page index not found at {path}. "
            "Build MaxSim index with: python -m visual_retrive.scripts.build_maxsim_index "
            "--devices 0,1,3"
        )
    meta = (path / "meta.json").read_text(encoding="utf-8")
    if '"scoring": "pooled_then_maxsim"' in meta:
        return ColQwenCascadeIndex.load(path, load_model=load_model)
    if '"scoring": "maxsim"' in meta:
        return MaxSimPageIndex.load(path, load_model=load_model)
    return ColModernPageIndex.load(path, load_model=load_model)


def search_pages(
    query: str,
    *,
    top_k: int = 5,
    subject: str | None = None,
    grade: int | str | None = None,
    index_dir: str | Path | None = None,
    index: MaxSimPageIndex | ColModernPageIndex | ColQwenCascadeIndex | None = None,
) -> dict[str, Any]:
    """Return a textbook_search-compatible payload for visual page hits."""
    started = time.perf_counter()
    idx = index or get_page_index(str(index_dir) if index_dir else None, load_model=True)
    hits = idx.search(query, top_k=top_k, subject=subject, grade=grade)
    mode = ("visual_colqwen25_cascade" if isinstance(idx, ColQwenCascadeIndex)
            else "visual_maxsim" if isinstance(idx, MaxSimPageIndex) else "visual_pooled")
    return {
        "query": query,
        "mode": mode,
        "filters": {"subject": subject, "grade": grade},
        "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
        "returned": len(hits),
        "index": str(index_dir or _default_index_dir()),
        "hits": hits,
    }


def format_visual_hits_for_model(
    result: dict[str, Any],
    *,
    answer_chars: int = 1_200,
) -> str:
    """Compact context block for the agent LLM."""
    hits = result.get("hits") or []
    if not hits:
        return "No relevant textbook pages found."
    parts = [
        f"Visual retrieval ({result.get('returned', 0)} hits, "
        f"{result.get('latency_ms', 0)} ms):"
    ]
    for hit in hits:
        ans = str(hit.get("answer_text") or "").strip()
        if len(ans) > answer_chars:
            ans = ans[: answer_chars - 1] + "…"
        parts.append(
            f"[{hit.get('rank')}] score={float(hit.get('score') or 0):.4f} "
            f"page_id={hit.get('page_id')} "
            f"grade={hit.get('grade')} subject={hit.get('subject')}\n"
            f"image={hit.get('page_image')}\n"
            f"{ans or '(no answer text)'}"
        )
    return "\n\n".join(parts)
