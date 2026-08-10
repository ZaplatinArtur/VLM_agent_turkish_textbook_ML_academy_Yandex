"""Orchestrate full OdevJet scrape: catalog -> page images -> answers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_retrive.http import log  # noqa: E402
from visual_retrive.scripts.discover_catalog import main as discover_main  # noqa: E402
from visual_retrive.scripts.scrape_answers import main as answers_main  # noqa: E402
from visual_retrive.scripts.scrape_pages import main as pages_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Full OdevJet scrape into data/visual_retrive: "
            "discover catalog, download page images, scrape answers."
        ),
    )
    parser.add_argument("--max-books", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--book-slug", default=None)
    parser.add_argument("--page-workers", type=int, default=6)
    parser.add_argument("--answer-workers", type=int, default=4)
    parser.add_argument(
        "--skip-discover",
        action="store_true",
        help="Reuse existing catalog files.",
    )
    parser.add_argument(
        "--skip-pages",
        action="store_true",
        help="Skip textbook page image download.",
    )
    parser.add_argument(
        "--skip-answers",
        action="store_true",
        help="Skip answer scrape.",
    )
    parser.add_argument(
        "--from-book-pages-only",
        action="store_true",
        help="Discover pages from book ld+json only (no sayfa sitemaps).",
    )
    parser.add_argument("--force-answers", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_discover:
        discover_argv: list[str] = []
        if args.max_books is not None:
            discover_argv.extend(["--max-books", str(args.max_books)])
        if args.from_book_pages_only:
            discover_argv.append("--from-book-pages-only")
        code = discover_main(discover_argv)
        if code != 0:
            return code
    else:
        log("Skipping discover step")

    if not args.skip_pages:
        pages_argv: list[str] = ["--workers", str(args.page_workers)]
        if args.book_slug:
            pages_argv.extend(["--book-slug", args.book_slug])
        if args.max_books is not None:
            pages_argv.extend(["--max-books", str(args.max_books)])
        if args.max_pages is not None:
            pages_argv.extend(["--max-pages", str(args.max_pages)])
        code = pages_main(pages_argv)
        if code not in {0, 2}:
            return code
    else:
        log("Skipping page image download")

    if not args.skip_answers:
        answers_argv: list[str] = ["--workers", str(args.answer_workers)]
        if args.book_slug:
            answers_argv.extend(["--book-slug", args.book_slug])
        if args.max_books is not None:
            answers_argv.extend(["--max-books", str(args.max_books)])
        if args.max_pages is not None:
            answers_argv.extend(["--max-pages", str(args.max_pages)])
        if args.force_answers:
            answers_argv.append("--force")
        code = answers_main(answers_argv)
        if code not in {0, 2}:
            return code
    else:
        log("Skipping answer scrape")

    log("Full scrape orchestration finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
