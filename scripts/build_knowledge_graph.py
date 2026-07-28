from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve.graph import KnowledgeGraph, KnowledgeGraphBuilder
from retrieve.metadata import enrich_chunk_metadata
from schemas.retrieve import RetrievedChunk


def read_chunks(path: Path) -> list[RetrievedChunk]:
    with path.open(encoding="utf-8") as source:
        return [
            enrich_chunk_metadata(RetrievedChunk.model_validate_json(line))
            for line in source
            if line.strip()
        ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a typed educational graph: exercise -> theory, "
            "worked example and solution."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit-books", type=int)
    parser.add_argument("--context-page-radius", type=int, default=2)
    parser.add_argument("--max-theory-edges", type=int, default=2)
    parser.add_argument("--max-example-edges", type=int, default=1)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.jsonl"))
    if args.limit_books is not None:
        paths = paths[: args.limit_books]
    if not paths:
        raise SystemExit(f"No JSONL books found in {args.input_dir}")

    started = time.perf_counter()
    builder = KnowledgeGraphBuilder(
        context_page_radius=args.context_page_radius,
        max_theory_edges=args.max_theory_edges,
        max_example_edges=args.max_example_edges,
    )
    graph = KnowledgeGraph()
    source_units = 0
    for position, path in enumerate(paths, 1):
        chunks = read_chunks(path)
        book_graph = builder.build(chunks)
        for node in book_graph.nodes.values():
            graph.add_node(node)
        for edge in book_graph.edges.values():
            graph.add_edge(edge)
        source_units += len(chunks)
        if position % 20 == 0 or position == len(paths):
            print(
                f"[{position}/{len(paths)}] nodes={len(graph.nodes)}, "
                f"edges={len(graph.edges)}",
                flush=True,
            )

    manifest = graph.save(args.output_dir)
    report = {
        **manifest,
        "books": len(paths),
        "source_units": source_units,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    report_path = args.report or args.output_dir / "build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
