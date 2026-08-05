"""Generate query→page training pairs from page bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_retrive.http import log  # noqa: E402
from visual_retrive.manifest import build_manifest, read_manifest  # noqa: E402
from visual_retrive.paths import CATALOG_DIR, ensure_visual_retrive_dirs  # noqa: E402
from visual_retrive.query_gen import (  # noqa: E402
    OpenAICompatibleQueryGenerator,
    generate_training_rows,
)
from visual_retrive.train.dataset import split_by_page, write_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate short Turkish retrieval queries for page fine-tuning. "
            "Use --mode heuristic offline, or --mode llm against vLLM/OpenRouter."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CATALOG_DIR / "page_bundles.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CATALOG_DIR / "train_queries.jsonl",
    )
    parser.add_argument("--mode", choices=("heuristic", "llm"), default="heuristic")
    parser.add_argument("--n-queries", type=int, default=3)
    parser.add_argument("--hard-negatives-k", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-pages", type=int, default=None)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--require-page-image", action="store_true", default=True)
    parser.add_argument("--no-require-page-image", action="store_true")
    parser.add_argument("--write-splits", action="store_true")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    ensure_visual_retrive_dirs()
    if args.rebuild_manifest or not args.manifest.is_file():
        log("Building page-bundle manifest first...")
        build_manifest(output_path=args.manifest)

    bundles = read_manifest(args.manifest)
    if args.limit_pages is not None:
        bundles = bundles[: max(0, args.limit_pages)]
    log(f"Loaded {len(bundles)} page bundles")

    llm = None
    if args.mode == "llm":
        llm = OpenAICompatibleQueryGenerator(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
        )

    rows = generate_training_rows(
        bundles,
        mode=args.mode,
        n_queries=args.n_queries,
        hard_negatives_k=args.hard_negatives_k,
        workers=args.workers,
        require_page_image=not args.no_require_page_image,
        llm=llm,
    )
    write_jsonl(args.output, rows)
    summary = {
        "pairs": len(rows),
        "unique_pages": len({row["positive_page_id"] for row in rows}),
        "mode": args.mode,
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"Wrote train queries: {summary}")

    if args.write_splits:
        splits = split_by_page(rows)
        split_dir = args.output.parent / "train_splits"
        for name, split_rows in splits.items():
            write_jsonl(split_dir / f"{name}.jsonl", split_rows)
        log(f"Wrote splits under {split_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
