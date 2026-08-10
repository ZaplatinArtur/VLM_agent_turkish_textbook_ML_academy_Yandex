from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mla_baseline.config import get_settings
from vlm_judge.backends import OpenAICompatibleBackend

from .qrels_annotation import annotate_candidate_pool, build_candidate_pool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and annotate pooled retrieval qrels")
    commands = parser.add_subparsers(dest="command", required=True)

    pool = commands.add_parser("build-pool")
    pool.add_argument("--qrels", required=True, type=Path)
    pool.add_argument("--tasks", required=True, type=Path)
    pool.add_argument("--run", required=True, type=Path)
    pool.add_argument("--output", required=True, type=Path)
    pool.add_argument("--bm25-source", type=Path, default=Path("artifacts/retrieval/bm25_chunks.jsonl"))
    pool.add_argument("--bm25-index", type=Path, default=Path("artifacts/retrieval/bm25.sqlite"))
    pool.add_argument("--dense-k", type=int, default=200)
    pool.add_argument("--bm25-k", type=int, default=100)
    pool.add_argument("--text-chars", type=int, default=400)

    annotate = commands.add_parser("run")
    annotate.add_argument("--pool", required=True, type=Path)
    annotate.add_argument("--output", required=True, type=Path)
    annotate.add_argument("--limit", type=int, default=0)
    annotate.add_argument("--model")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-pool":
        summary = build_candidate_pool(
            args.qrels,
            args.tasks,
            args.run,
            args.output,
            bm25_source_path=args.bm25_source,
            bm25_index_path=args.bm25_index,
            dense_k=args.dense_k,
            bm25_k=args.bm25_k,
            text_chars=args.text_chars,
        )
    else:
        settings = get_settings()
        backend = OpenAICompatibleBackend(
            settings.openrouter_base_url,
            args.model or settings.openrouter_model_name,
            api_key=settings.llm_api_key,
            timeout=settings.request_timeout_s,
            temperature=0.0,
            max_tokens=1500,
            seed=20260804,
            provider="openrouter",
            enable_thinking=False,
        )
        summary = annotate_candidate_pool(args.pool, args.output, backend, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
