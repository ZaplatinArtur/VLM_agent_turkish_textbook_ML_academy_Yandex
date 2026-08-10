from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve.service import build_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test graph-expanded RAG.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--subject")
    parser.add_argument("--grade")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    pipeline = build_pipeline()
    startup_seconds = time.perf_counter() - started
    run_seconds: list[float] = []
    hits = []
    for _ in range(max(1, args.runs)):
        query_started = time.perf_counter()
        hits = pipeline.run(
            args.query,
            k=args.k,
            subject=args.subject,
            grade=args.grade,
        )
        run_seconds.append(time.perf_counter() - query_started)
    payload = {
        "query": args.query,
        "subject": args.subject,
        "grade": args.grade,
        "startup_seconds": round(startup_seconds, 3),
        "query_seconds": [round(value, 3) for value in run_seconds],
        "warm_query_seconds": round(run_seconds[-1], 3),
        "hits": [
            {
                "chunk_id": hit.chunk_id,
                "score": hit.score,
                "anchor_id": hit.metadata.get("graph_anchor_id"),
                "anchor_kind": hit.metadata.get("graph_anchor_kind"),
                "subject": hit.metadata.get("subject"),
                "grade": hit.metadata.get("grade"),
                "has_theory": hit.metadata.get("has_theory"),
                "has_example": hit.metadata.get("has_example"),
                "has_solution": hit.metadata.get("has_solution"),
                "graph_paths": hit.metadata.get("graph_paths"),
                "text_preview": hit.text[:1_500],
            }
            for hit in hits
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
