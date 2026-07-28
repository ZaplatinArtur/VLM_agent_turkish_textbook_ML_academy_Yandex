from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve.service import build_pipeline


def _serialize_hit(chunk: Any, rank: int) -> dict[str, Any]:
    retrieval_text = str(chunk.metadata.get("retrieval_text") or chunk.text)
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "score": round(float(chunk.score), 8),
        "unit_kind": chunk.metadata.get("unit_kind"),
        "retrieval_chars": len(retrieval_text),
        "textbook": chunk.metadata.get("textbook"),
        "page": chunk.metadata.get("source_page", chunk.metadata.get("page")),
        "preview": " ".join(retrieval_text.split())[:300],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect candidates before and after the knowledge reranker."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--subject")
    parser.add_argument("--grade")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    pipeline = build_pipeline()
    if len(pipeline.rankers) < 2:
        raise SystemExit("Configured pipeline has fewer than two ranking stages")

    fused = pipeline.rankers[0].rank(
        args.query,
        subject=args.subject,
        grade=args.grade,
    )
    reranked = pipeline.rankers[1].rank(
        args.query,
        fused,
        subject=args.subject,
        grade=args.grade,
    )
    top_n = max(1, args.top_n)
    payload = {
        "query": args.query,
        "subject": args.subject,
        "grade": args.grade,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "fused_count": len(fused),
        "reranked_count": len(reranked),
        "fused": [
            _serialize_hit(chunk, rank)
            for rank, chunk in enumerate(fused[:top_n], start=1)
        ],
        "reranked": [
            _serialize_hit(chunk, rank)
            for rank, chunk in enumerate(reranked[:top_n], start=1)
        ],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
