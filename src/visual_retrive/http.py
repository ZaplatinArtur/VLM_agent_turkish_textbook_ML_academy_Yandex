"""Shared HTTP helpers for OdevJet scraping."""

from __future__ import annotations

import threading
from typing import Any

import requests

BASE_URL = "https://www.odevjet.com"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MLA-visual-retrive/0.1; "
        "+https://github.com/local/mla-visual-retrive)"
    ),
    "Accept-Language": "tr,en;q=0.8",
}
DEFAULT_TIMEOUT = (10, 45)

_thread_local = threading.local()


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(REQUEST_HEADERS)
        _thread_local.session = session
    return session


def fetch_text(url: str, *, timeout: tuple[int, int] = DEFAULT_TIMEOUT) -> str:
    response = get_session().get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def fetch_bytes(
    url: str,
    *,
    timeout: tuple[int, int] = DEFAULT_TIMEOUT,
) -> tuple[int, bytes, str | None]:
    response = get_session().get(url, timeout=timeout)
    content_type = response.headers.get("Content-Type")
    return response.status_code, response.content, content_type


def head_ok(url: str, *, timeout: tuple[int, int] = DEFAULT_TIMEOUT) -> bool:
    try:
        response = get_session().head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            return True
        # Some hosts reject HEAD; fall back to a tiny GET.
        if response.status_code in {403, 405}:
            status, content, _ = fetch_bytes(url, timeout=timeout)
            return status == 200 and len(content) > 0
        return False
    except requests.RequestException:
        return False


def log(message: str) -> None:
    print(message, flush=True)


def json_dump_line(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
