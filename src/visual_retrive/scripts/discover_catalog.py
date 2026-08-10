"""Build the OdevJet book/page catalog under data/visual_retrive/catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_retrive.catalog import discover_catalog  # noqa: E402
from visual_retrive.http import log  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover all OdevJet textbook books and page URLs.",
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=None,
        help="Optional limit for smoke runs.",
    )
    parser.add_argument(
        "--from-book-pages-only",
        action="store_true",
        help="Skip kitap-sayfasi sitemaps; use ld+json page lists from book pages.",
    )
    args = parser.parse_args(argv)

    paths = discover_catalog(
        max_books=args.max_books,
        include_page_sitemap=not args.from_book_pages_only,
    )
    log(f"Done. Catalog files: {paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
