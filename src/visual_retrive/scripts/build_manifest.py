"""Build page_bundles.jsonl from scraped books."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_retrive.http import log  # noqa: E402
from visual_retrive.manifest import build_manifest  # noqa: E402
from visual_retrive.paths import CATALOG_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build visual_retrive page-bundle manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=CATALOG_DIR / "page_bundles.jsonl",
    )
    parser.add_argument("--require-page-image", action="store_true")
    parser.add_argument("--require-solution", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    summary = build_manifest(
        output_path=args.output,
        require_page_image=args.require_page_image,
        require_solution=args.require_solution,
        limit=args.limit,
    )
    log(f"Manifest ready: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
