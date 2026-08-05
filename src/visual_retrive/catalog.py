"""Discover OdevJet books and page URLs from public sitemaps / book pages."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from retrieve.metadata import infer_textbook_metadata

from .http import BASE_URL, fetch_text, log
from .paths import CATALOG_DIR, ensure_visual_retrive_dirs

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
PAGE_URL_RE = re.compile(
    r"^(?P<slug>.+)-sayfa-(?P<page>\d+)(?:\.html)?/?$",
    flags=re.IGNORECASE,
)


def book_slug_from_url(book_url: str) -> str:
    path = urlparse(book_url).path.strip("/")
    if path.endswith(".html"):
        path = path[: -len(".html")]
    return path.split("/")[-1]


def page_ref_from_url(page_url: str) -> tuple[str, int] | None:
    path = urlparse(page_url).path.strip("/")
    if path.endswith(".html"):
        path = path[: -len(".html")]
    match = PAGE_URL_RE.match(path)
    if not match:
        return None
    return match.group("slug"), int(match.group("page"))


def page_url_for(book_slug: str, page_number: int) -> str:
    return f"{BASE_URL}/{book_slug}-sayfa-{page_number}.html"


def get_book_urls() -> list[str]:
    xml_text = fetch_text(f"{BASE_URL}/kitap-sitemap.xml")
    root = ET.fromstring(xml_text)
    urls = [
        node.text.strip()
        for node in root.findall(".//sm:loc", SITEMAP_NS)
        if node.text
    ]
    # Stable unique order.
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def iter_page_sitemap_urls() -> list[str]:
    index_xml = fetch_text(f"{BASE_URL}/sitemap_index.xml")
    root = ET.fromstring(index_xml)
    urls = [
        node.text.strip()
        for node in root.findall(".//sm:loc", SITEMAP_NS)
        if node.text and "kitap-sayfasi-sitemap" in node.text
    ]
    return sorted(set(urls), key=lambda u: (len(u), u))


def get_all_page_urls() -> list[str]:
    page_urls: list[str] = []
    seen: set[str] = set()
    for sitemap_url in iter_page_sitemap_urls():
        xml_text = fetch_text(sitemap_url)
        root = ET.fromstring(xml_text)
        for node in root.findall(".//sm:loc", SITEMAP_NS):
            if not node.text:
                continue
            url = node.text.strip()
            if url in seen:
                continue
            seen.add(url)
            page_urls.append(url)
    return page_urls


def get_page_numbers_from_book_html(html: str) -> list[int]:
    page_numbers: list[int] = []
    for script in re.findall(
        r'<script type="application/ld\+json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    ):
        try:
            data = json.loads(script)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("@type") != "LearningResource":
                continue
            main_entity = item.get("mainEntity") or {}
            for element in main_entity.get("itemListElement") or []:
                if not isinstance(element, dict):
                    continue
                page_url = ((element.get("item") or {}).get("url")) or ""
                ref = page_ref_from_url(str(page_url))
                if ref is not None:
                    page_numbers.append(ref[1])
    return sorted(set(page_numbers))


def get_page_numbers(book_url: str) -> list[int]:
    html = fetch_text(book_url)
    return get_page_numbers_from_book_html(html)


def build_book_record(book_url: str, page_numbers: list[int]) -> dict[str, Any]:
    slug = book_slug_from_url(book_url)
    inferred = infer_textbook_metadata(slug)
    return {
        "book_slug": slug,
        "book_url": book_url,
        "grade": inferred.get("grade"),
        "subject": inferred.get("subject"),
        "page_count": len(page_numbers),
        "page_numbers": page_numbers,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def discover_catalog(
    *,
    max_books: int | None = None,
    include_page_sitemap: bool = True,
) -> dict[str, Path]:
    """Build catalog files under data/visual_retrive/catalog."""

    ensure_visual_retrive_dirs()
    book_urls = get_book_urls()
    if max_books is not None:
        book_urls = book_urls[: max(0, max_books)]

    log(f"Discovering {len(book_urls)} books from kitap-sitemap.xml")
    books: list[dict[str, Any]] = []
    for index, book_url in enumerate(book_urls, 1):
        slug = book_slug_from_url(book_url)
        try:
            page_numbers = get_page_numbers(book_url)
        except Exception as exc:  # noqa: BLE001 - continue catalog build
            log(f"[{index}/{len(book_urls)}] {slug}: page list failed ({exc})")
            books.append(
                {
                    "book_slug": slug,
                    "book_url": book_url,
                    "grade": infer_textbook_metadata(slug).get("grade"),
                    "subject": infer_textbook_metadata(slug).get("subject"),
                    "page_count": 0,
                    "page_numbers": [],
                    "error": str(exc),
                }
            )
            continue
        record = build_book_record(book_url, page_numbers)
        books.append(record)
        log(f"[{index}/{len(book_urls)}] {slug}: {len(page_numbers)} pages")

    books_path = CATALOG_DIR / "books.jsonl"
    write_jsonl(books_path, books)

    pages_path = CATALOG_DIR / "pages_index.jsonl"
    page_rows: list[dict[str, Any]] = []
    if include_page_sitemap:
        log("Fetching kitap-sayfasi sitemaps...")
        for page_url in get_all_page_urls():
            ref = page_ref_from_url(page_url)
            if ref is None:
                continue
            slug, page_number = ref
            page_rows.append(
                {
                    "book_slug": slug,
                    "page_number": page_number,
                    "page_url": page_url,
                }
            )
        page_rows.sort(key=lambda row: (row["book_slug"], row["page_number"]))
        write_jsonl(pages_path, page_rows)
        log(f"Wrote {len(page_rows)} page URLs -> {pages_path}")
    else:
        for book in books:
            slug = book["book_slug"]
            for page_number in book.get("page_numbers") or []:
                page_rows.append(
                    {
                        "book_slug": slug,
                        "page_number": int(page_number),
                        "page_url": page_url_for(slug, int(page_number)),
                    }
                )
        write_jsonl(pages_path, page_rows)
        log(f"Wrote {len(page_rows)} page URLs from book pages -> {pages_path}")

    summary = {
        "books": len(books),
        "pages": len(page_rows),
        "books_with_pages": sum(1 for book in books if book.get("page_count")),
        "books_path": str(books_path),
        "pages_path": str(pages_path),
    }
    summary_path = CATALOG_DIR / "discover_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"Catalog summary: {summary}")
    return {"books": books_path, "pages": pages_path, "summary": summary_path}
