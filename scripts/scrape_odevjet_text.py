"""Текстовый корпус ÖdevJet для BM25-индекса ретрива.

Обходит kitap-sitemap.xml -> страницы книг -> текст решений + метаданные.
Выход — JSONL в формате prepare-corpus судейского CLI:
{"id", "content", "metadata": {book_id, subject, grade, page_number,
 source_url, image_urls}}.

Резюмируемый: уже записанные id пропускаются (append-режим).
"""

import html as html_lib
import json
import re
import sys
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

BASE_URL = "https://www.odevjet.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) mla-corpus/0.1"}
TIMEOUT = (10, 30)
WORKERS = 8
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# шаблонные строки темы сайта — на каждой странице, в корпус не нужны
BOILERPLATE_LINE_MARKERS = (
    "anlık görüntüleyici", "henüz görsel eklenmemiş",
    "onaylanmış öğrenci çözümü yok", "ilk çözümü sen paylaş",
    "kendi çözümünü paylaş", "jpg, png veya webp",
    "yalnızca kontrol ve öğrenme amacıyla", "velilerimiz de bu içerikleri",
    "fotoğrafını yükle", "çerez", "gizlilik politikası",
)

_local = threading.local()


def session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        _local.s = s
    return s


def fetch(url: str) -> str:
    for attempt in (1, 2, 3):
        try:
            r = session().get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception:
            if attempt == 3:
                raise
    return ""


def book_urls() -> list[str]:
    root = ET.fromstring(fetch(f"{BASE_URL}/kitap-sitemap.xml"))
    return [n.text.strip() for n in root.findall(".//sm:loc", SITEMAP_NS) if n.text]


def slug_of(url: str) -> str:
    return urlparse(url).path.strip("/").removesuffix(".html")


def parse_slug(slug: str) -> tuple[int | None, str | None]:
    """'8-sinif-matematik-ders-kitabi-...' -> (8, 'matematik')."""
    m = re.match(r"^(\d+)-sinif-(.+)$", slug)
    if not m:
        return None, None
    grade = int(m.group(1))
    rest = m.group(2)
    rest = re.split(r"-(?:ders|calisma|beceri|soru|meb|test)\b", rest)[0]
    return grade, rest.replace("-", " ").strip() or None


def page_links(book_html: str, slug: str) -> list[tuple[int, str]]:
    # два стиля URL на сайте: ...-sayfa-N/ и ...-sayfa-N.html
    pat = re.escape(BASE_URL) + "/" + re.escape(slug) + r"-sayfa-(\d+)(?:/|\.html)"
    seen = {}
    for m in re.finditer(pat, book_html):
        seen[int(m.group(1))] = m.group(0)
    return sorted(seen.items())


def extract(page_html: str) -> tuple[str, list[str]]:
    m = re.search(r"<article[^>]*>(.*?)</article>", page_html, re.S)
    body = m.group(1) if m else page_html
    imgs = [u for u in re.findall(r"<img[^>]+src=\"([^\"]+)\"", body)
            if "wp-content/uploads" in u]
    text = re.sub(r"<(script|style|form|nav|footer).*?</\1>", " ", body, flags=re.S)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = html_lib.unescape(text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        low = line.casefold()
        if any(mark in low for mark in BOILERPLATE_LINE_MARKERS):
            continue
        lines.append(line)
    return "\n".join(lines), imgs


def scrape_page(slug: str, grade, subject, page_no: int, url: str) -> dict:
    content, imgs = extract(fetch(url))
    return {
        "id": f"{slug}-sayfa-{page_no}",
        "content": content,
        "metadata": {
            "book_id": slug, "subject": subject, "grade": grade,
            "page_number": page_no, "source_url": url, "image_urls": imgs,
        },
    }


def main() -> int:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/odevjet_corpus.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    print(f"уже собрано: {len(done)}", flush=True)

    books = book_urls()
    print(f"книг в sitemap: {len(books)}", flush=True)
    written = errors = 0
    lock = threading.Lock()

    with out_path.open("a", encoding="utf-8") as out:
        for bi, burl in enumerate(books, 1):
            slug = slug_of(burl)
            grade, subject = parse_slug(slug)
            try:
                pages = page_links(fetch(burl), slug)
            except Exception as exc:
                print(f"[{bi}/{len(books)}] {slug}: список страниц не взят ({exc})",
                      flush=True)
                continue
            todo = [(n, u) for n, u in pages if f"{slug}-sayfa-{n}" not in done]
            if not todo:
                continue
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futs = {pool.submit(scrape_page, slug, grade, subject, n, u): n
                        for n, u in todo}
                for fut in as_completed(futs):
                    try:
                        row = fut.result()
                    except Exception:
                        errors += 1
                        continue
                    with lock:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
            out.flush()
            if bi % 10 == 0:
                print(f"[{bi}/{len(books)}] страниц записано {written}, "
                      f"ошибок {errors}", flush=True)
    print(f"ГОТОВО: страниц {written}, ошибок {errors} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
