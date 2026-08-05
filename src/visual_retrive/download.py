"""Download textbook page images and answer assets from OdevJet."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image

from .http import BASE_URL, DEFAULT_TIMEOUT, fetch_bytes, fetch_text, get_session, log
from .page_parse import parse_answer_page
from .paths import (
    LEGACY_BOOKS_DIR,
    VISUAL_RETRIVE_DIR,
    answers_dir,
    ensure_visual_retrive_dirs,
    pages_dir,
)

MIN_IMAGE_BYTES = 1000


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _rel(path: Path) -> str:
    return path.relative_to(VISUAL_RETRIVE_DIR).as_posix()


def page_image_path(book_slug: str, page_number: int) -> Path:
    return pages_dir(book_slug) / f"{page_number:04d}.jpg"


def answer_meta_path(book_slug: str, page_number: int) -> Path:
    return answers_dir(book_slug) / f"{page_number:04d}.json"


def legacy_page_image_path(book_slug: str, page_number: int) -> Path | None:
    legacy = LEGACY_BOOKS_DIR / book_slug / f"{page_number:04d}.jpg"
    if legacy.is_file() and legacy.stat().st_size > 0:
        return legacy
    return None


def download_page_image(
    book_slug: str,
    page_number: int,
    *,
    reuse_legacy: bool = True,
) -> str:
    """Download one textbook page image. Returns status token."""

    ensure_visual_retrive_dirs()
    output = page_image_path(book_slug, page_number)
    if output.exists() and output.stat().st_size > 0:
        return "already_exists"

    if reuse_legacy:
        legacy = legacy_page_image_path(book_slug, page_number)
        if legacy is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(legacy.read_bytes())
            return "copied_legacy"

    image_bytes: bytes | None = None
    for extension in ("jpg", "webp", "png"):
        image_url = (
            f"{BASE_URL}/download/sayfalar/{book_slug}/{page_number}.{extension}"
        )
        try:
            status, content, _ = fetch_bytes(image_url)
        except requests.RequestException:
            continue
        if status == 200 and len(content) > MIN_IMAGE_BYTES:
            image_bytes = content
            break

    if not image_bytes:
        return "error"

    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.save(output, "JPEG", quality=90)
    return "downloaded"


def _suffix_from_url(url: str) -> str:
    path_suffix = Path(urlparse(url).path).suffix.casefold()
    if path_suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if path_suffix == ".jpeg" else path_suffix
    return ".bin"


def download_answer_image(url: str, destination: Path) -> str:
    if destination.exists() and destination.stat().st_size > 0:
        return "already_exists"
    try:
        status, content, _ = fetch_bytes(url)
    except requests.RequestException:
        return "error"
    if status != 200 or len(content) <= MIN_IMAGE_BYTES:
        return "error"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.casefold() in {".jpg", ".jpeg"}:
        image = Image.open(BytesIO(content)).convert("RGB")
        image.save(destination, "JPEG", quality=90)
    else:
        destination.write_bytes(content)
    return "downloaded"


def scrape_answer_page(
    book_slug: str,
    page_number: int,
    *,
    page_url: str | None = None,
    force: bool = False,
    download_page_image_flag: bool = False,
) -> dict[str, Any]:
    """Fetch page HTML, save answer text/images, optionally page image."""

    ensure_visual_retrive_dirs()
    meta_path = answer_meta_path(book_slug, page_number)
    if meta_path.exists() and not force:
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("scraped_at"):
                return {**existing, "status": "already_exists"}
        except json.JSONDecodeError:
            pass

    url = page_url or f"{BASE_URL}/{book_slug}-sayfa-{page_number}.html"
    try:
        html = fetch_text(url)
    except requests.RequestException as exc:
        payload = {
            "book_slug": book_slug,
            "page_number": page_number,
            "page_url": url,
            "status": "error",
            "error": str(exc),
            "scraped_at": _utc_now(),
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return payload

    parsed = parse_answer_page(html)
    answer_dir = answers_dir(book_slug)
    answer_dir.mkdir(parents=True, exist_ok=True)

    saved_answer_images: list[str] = []
    image_statuses: list[str] = []
    image_urls = list(parsed["answer_image_urls"])
    for index, image_url in enumerate(image_urls, 1):
        suffix = _suffix_from_url(image_url)
        if len(image_urls) == 1:
            dest = answer_dir / f"{page_number:04d}{suffix}"
        else:
            dest = answer_dir / f"{page_number:04d}_{index}{suffix}"
        status = download_answer_image(image_url, dest)
        image_statuses.append(status)
        if status in {"downloaded", "already_exists"}:
            saved_answer_images.append(_rel(dest))

    page_status = None
    page_relpath = None
    if download_page_image_flag:
        page_status = download_page_image(book_slug, page_number)
        page_file = page_image_path(book_slug, page_number)
        if page_file.exists():
            page_relpath = _rel(page_file)

    text_path = None
    if parsed["answer_text"]:
        text_file = answer_dir / f"{page_number:04d}.txt"
        text_file.write_text(parsed["answer_text"] + "\n", encoding="utf-8")
        text_path = _rel(text_file)

    payload: dict[str, Any] = {
        "book_slug": book_slug,
        "page_number": page_number,
        "page_url": url,
        "has_solution": parsed["has_solution"],
        "no_solution": parsed["no_solution"],
        "answer_kinds": parsed["answer_kinds"],
        "answer_text": parsed["answer_text"],
        "answer_text_path": text_path,
        "answer_image_urls": image_urls,
        "answer_image_paths": saved_answer_images,
        "answer_image_statuses": image_statuses,
        "page_image_urls": parsed["page_image_urls"],
        "page_image_path": page_relpath,
        "page_image_status": page_status,
        "scraped_at": _utc_now(),
        "status": "scraped",
    }
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def warm_session() -> None:
    """Touch the shared session so DNS / TLS happen before a worker pool."""

    get_session()
    log(f"HTTP session ready (timeout={DEFAULT_TIMEOUT})")
