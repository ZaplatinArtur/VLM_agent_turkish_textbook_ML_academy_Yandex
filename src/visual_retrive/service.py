"""Process-local visual page retrieval (mirrors ``retrieve.service`` for text RAG)."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .search import get_page_index, search_pages

_index: Any | None = None
_index_path: str | None = None
_index_lock = threading.Lock()


def get_visual_index(
    index_dir: str | Path | None = None,
    *,
    load_model: bool = True,
) -> Any:
    """Return a process-wide page index (MaxSim preferred, pooled fallback)."""
    global _index, _index_path
    configured = index_dir or os.environ.get("MLA_VISUAL_INDEX_DIR")
    path = str(Path(configured).expanduser()) if configured else ""
    # Empty → let search._default_index_dir choose MaxSim if present.
    key = path or "__default__"
    with _index_lock:
        if _index is None or _index_path != key:
            _index = get_page_index(path or None, load_model=load_model)
            _index_path = key
        elif load_model and hasattr(_index, "ensure_encoder"):
            _index.ensure_encoder()
        return _index


def reset_visual_index() -> None:
    global _index, _index_path
    with _index_lock:
        _index = None
        _index_path = None
    get_page_index.cache_clear()


def visual_page_retrieve(
    query: str,
    *,
    k: int = 5,
    subject: str | None = None,
    grade: int | str | None = None,
    index_dir: str | Path | None = None,
) -> dict[str, Any]:
    index = get_visual_index(index_dir, load_model=True)
    return search_pages(
        query,
        top_k=k,
        subject=subject,
        grade=grade,
        index=index,
        index_dir=index_dir,
    )
