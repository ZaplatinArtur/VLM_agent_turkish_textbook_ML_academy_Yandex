"""Validate the textbook corpus and build/load its persistent FAISS index.

Examples:
    python -m retrieve.build_index --dry-run
    python -m retrieve.build_index --sample-query "dikdörtgen alan formülü"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from paths import CHUNKS_JSONL_DIR, INDEX_DIR


def pipeline_index_directory(pipeline: Any) -> Path:
    """Return the snapshot directory built by the active candidate arm."""
    rankers = getattr(pipeline, "rankers", ())
    candidate = rankers[0] if rankers else None
    semantic = getattr(candidate, "semantic", None)
    semantic_index_dir = getattr(semantic, "index_dir", None)
    return Path(semantic_index_dir) if semantic_index_dir is not None else INDEX_DIR

def corpus_inventory(chunks: Sequence[Any]) -> dict[str, Any]:
    chunk_ids = [str(chunk.chunk_id) for chunk in chunks]
    duplicate_ids = len(chunk_ids) - len(set(chunk_ids))
    empty_texts = sum(not str(chunk.text).strip() for chunk in chunks)
    books = Counter(
        str(chunk.metadata.get("textbook") or chunk.chunk_id.split(":", 1)[0])
        for chunk in chunks
    )
    subjects = Counter(
        str(chunk.metadata.get("subject") or "unknown") for chunk in chunks
    )
    grades = Counter(str(chunk.metadata.get("grade") or "unknown") for chunk in chunks)
    return {
        "chunks": len(chunks),
        "books": len(books),
        "duplicate_chunk_ids": duplicate_ids,
        "empty_texts": empty_texts,
        "subjects": dict(sorted(subjects.items())),
        "grades": dict(sorted(grades.items())),
        "chunks_dir": str(CHUNKS_JSONL_DIR),
        "index_dir": str(INDEX_DIR),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Проверить корпус чанков и построить постоянный FAISS-индекс"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="только проверить JSONL и вывести состав корпуса",
    )
    parser.add_argument(
        "--sample-query",
        default=None,
        help="после сборки выполнить тестовый запрос",
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    graph_dir_value = os.environ.get("MLA_KNOWLEDGE_GRAPH_DIR", "").strip()
    graph_dir = Path(graph_dir_value).expanduser() if graph_dir_value else None
    if (
        graph_dir is not None
        and (graph_dir / "nodes.jsonl").is_file()
        and (graph_dir / "edges.jsonl").is_file()
    ):
        from ..graph import KnowledgeGraph

        chunks = KnowledgeGraph.load(graph_dir).searchable_nodes()
    else:
        from .parsing import get_retrieved_chunks

        chunks = get_retrieved_chunks()
    inventory = corpus_inventory(chunks)
    print(json.dumps({"corpus": inventory}, ensure_ascii=False, indent=2))
    if not chunks:
        raise SystemExit(f"Корпус пуст: {CHUNKS_JSONL_DIR}")
    if inventory["duplicate_chunk_ids"]:
        raise SystemExit(
            f"В корпусе повторяются chunk_id: {inventory['duplicate_chunk_ids']}"
        )
    if args.dry_run:
        return 0

    # Delay optional ML imports until after the dependency-free corpus check.
    from ..storage.persistence import load_manifest
    from ..service import build_pipeline

    started = time.perf_counter()
    pipeline = build_pipeline(chunks)
    for ranker in pipeline.rankers:
        build = getattr(ranker, "build", None)
        if callable(build):
            build()
    pipeline.persist()
    built_index_dir = pipeline_index_directory(pipeline)
    elapsed = round(time.perf_counter() - started, 3)
    print(
        json.dumps(
            {
                "index": load_manifest(built_index_dir),
                "index_dir": str(built_index_dir),
                "build_or_load_seconds": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.sample_query:
        results = pipeline.run(args.sample_query, k=args.k)
        print(
            json.dumps(
                {
                    "query": args.sample_query,
                    "hits": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "score": chunk.score,
                            "metadata": chunk.metadata,
                            "text_preview": " ".join(chunk.text.split())[:240],
                        }
                        for chunk in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
