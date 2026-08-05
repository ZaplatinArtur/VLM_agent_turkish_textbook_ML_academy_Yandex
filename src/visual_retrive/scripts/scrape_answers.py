"""Scrape OdevJet answers (text solutions and/or answer images) for every page."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_retrive.catalog import read_jsonl  # noqa: E402
from visual_retrive.download import scrape_answer_page, warm_session  # noqa: E402
from visual_retrive.http import log  # noqa: E402
from visual_retrive.paths import CATALOG_DIR, ensure_visual_retrive_dirs  # noqa: E402


def _load_targets(
    *,
    book_slug: str | None,
    max_books: int | None,
    max_pages: int | None,
) -> list[dict[str, Any]]:
    pages_path = CATALOG_DIR / "pages_index.jsonl"
    books_path = CATALOG_DIR / "books.jsonl"
    targets: list[dict[str, Any]] = []

    if pages_path.is_file():
        for row in read_jsonl(pages_path):
            slug = str(row.get("book_slug") or "")
            page = int(row.get("page_number") or 0)
            if not slug or page < 1:
                continue
            if book_slug and slug != book_slug:
                continue
            targets.append(
                {
                    "book_slug": slug,
                    "page_number": page,
                    "page_url": row.get("page_url"),
                }
            )
    elif books_path.is_file():
        from visual_retrive.catalog import page_url_for

        for book in read_jsonl(books_path):
            slug = str(book.get("book_slug") or "")
            if not slug:
                continue
            if book_slug and slug != book_slug:
                continue
            for page in book.get("page_numbers") or []:
                page_number = int(page)
                targets.append(
                    {
                        "book_slug": slug,
                        "page_number": page_number,
                        "page_url": page_url_for(slug, page_number),
                    }
                )
    else:
        raise FileNotFoundError(
            f"Missing catalog. Run discover_catalog.py first "
            f"(expected {pages_path} or {books_path})."
        )

    if max_books is not None:
        allowed: list[str] = []
        seen: set[str] = set()
        for row in targets:
            slug = row["book_slug"]
            if slug in seen:
                continue
            seen.add(slug)
            allowed.append(slug)
            if len(allowed) >= max_books:
                break
        allowed_set = set(allowed)
        targets = [row for row in targets if row["book_slug"] in allowed_set]

    if max_pages is not None:
        targets = targets[: max(0, max_pages)]
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape OdevJet answers: text from text-solution-content and/or "
            "images from /download/cevaplar/."
        ),
    )
    parser.add_argument("--book-slug", default=None)
    parser.add_argument("--max-books", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape pages even if answers/*.json already exists.",
    )
    parser.add_argument(
        "--with-page-images",
        action="store_true",
        help="Also download textbook page images while scraping answers.",
    )
    args = parser.parse_args(argv)

    ensure_visual_retrive_dirs()
    warm_session()
    targets = _load_targets(
        book_slug=args.book_slug,
        max_books=args.max_books,
        max_pages=args.max_pages,
    )
    log(f"Answer scrape targets: {len(targets)}")

    counts = {
        "scraped": 0,
        "already_exists": 0,
        "error": 0,
        "with_text": 0,
        "with_image": 0,
        "no_solution": 0,
    }

    def _job(row: dict[str, Any]) -> dict[str, Any]:
        return scrape_answer_page(
            row["book_slug"],
            int(row["page_number"]),
            page_url=row.get("page_url"),
            force=args.force,
            download_page_image_flag=args.with_page_images,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_job, row) for row in targets]
        for index, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                counts["error"] += 1
                log(f"worker error: {exc}")
                continue

            status = str(result.get("status") or "error")
            counts[status] = counts.get(status, 0) + 1
            kinds = result.get("answer_kinds") or []
            if "text" in kinds:
                counts["with_text"] += 1
            if "image" in kinds:
                counts["with_image"] += 1
            if result.get("no_solution"):
                counts["no_solution"] += 1

            if index % 50 == 0 or index == len(futures):
                log(
                    f"[{index}/{len(futures)}] "
                    f"scraped={counts.get('scraped', 0)} "
                    f"exists={counts.get('already_exists', 0)} "
                    f"text={counts['with_text']} "
                    f"image={counts['with_image']} "
                    f"empty={counts['no_solution']} "
                    f"errors={counts.get('error', 0)}"
                )

    summary = {"targets": len(targets), "counts": counts}
    summary_path = CATALOG_DIR / "scrape_answers_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"Done: {summary}")
    return 0 if counts.get("error", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
