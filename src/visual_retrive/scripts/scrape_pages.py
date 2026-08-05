"""Download textbook page images for all (or selected) OdevJet books."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_retrive.catalog import (  # noqa: E402
    page_url_for,
    read_jsonl,
)
from visual_retrive.download import download_page_image, warm_session  # noqa: E402
from visual_retrive.http import log  # noqa: E402
from visual_retrive.paths import CATALOG_DIR, ensure_visual_retrive_dirs  # noqa: E402


def _load_targets(
    *,
    book_slug: str | None,
    max_books: int | None,
    max_pages: int | None,
) -> list[tuple[str, int]]:
    pages_path = CATALOG_DIR / "pages_index.jsonl"
    books_path = CATALOG_DIR / "books.jsonl"
    targets: list[tuple[str, int]] = []

    if pages_path.is_file():
        rows = read_jsonl(pages_path)
        for row in rows:
            slug = str(row.get("book_slug") or "")
            page = int(row.get("page_number") or 0)
            if not slug or page < 1:
                continue
            if book_slug and slug != book_slug:
                continue
            targets.append((slug, page))
    elif books_path.is_file():
        books = read_jsonl(books_path)
        for book in books:
            slug = str(book.get("book_slug") or "")
            if not slug:
                continue
            if book_slug and slug != book_slug:
                continue
            for page in book.get("page_numbers") or []:
                targets.append((slug, int(page)))
    else:
        raise FileNotFoundError(
            f"Missing catalog. Run discover_catalog.py first "
            f"(expected {pages_path} or {books_path})."
        )

    if max_books is not None:
        allowed: list[str] = []
        seen: set[str] = set()
        for slug, _ in targets:
            if slug in seen:
                continue
            seen.add(slug)
            allowed.append(slug)
            if len(allowed) >= max_books:
                break
        allowed_set = set(allowed)
        targets = [item for item in targets if item[0] in allowed_set]

    if max_pages is not None:
        targets = targets[: max(0, max_pages)]
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download OdevJet textbook page images into data/visual_retrive.",
    )
    parser.add_argument("--book-slug", default=None)
    parser.add_argument("--max-books", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="Do not copy already-scraped images from data/books.",
    )
    args = parser.parse_args(argv)

    ensure_visual_retrive_dirs()
    warm_session()
    targets = _load_targets(
        book_slug=args.book_slug,
        max_books=args.max_books,
        max_pages=args.max_pages,
    )
    log(f"Page image targets: {len(targets)}")

    counts = {
        "downloaded": 0,
        "already_exists": 0,
        "copied_legacy": 0,
        "error": 0,
    }

    def _job(item: tuple[str, int]) -> str:
        slug, page = item
        return download_page_image(
            slug,
            page,
            reuse_legacy=not args.no_legacy,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_job, item) for item in targets]
        for index, future in enumerate(as_completed(futures), 1):
            try:
                status = future.result()
            except Exception:  # noqa: BLE001
                status = "error"
            counts[status] = counts.get(status, 0) + 1
            if index % 100 == 0 or index == len(futures):
                log(
                    f"[{index}/{len(futures)}] "
                    f"downloaded={counts.get('downloaded', 0)} "
                    f"exists={counts.get('already_exists', 0)} "
                    f"legacy={counts.get('copied_legacy', 0)} "
                    f"errors={counts.get('error', 0)}"
                )

    summary_path = CATALOG_DIR / "scrape_pages_summary.json"
    summary = {
        "targets": len(targets),
        "counts": counts,
        "sample_page_url": page_url_for(*targets[0]) if targets else None,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"Done: {summary}")
    return 0 if counts.get("error", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
