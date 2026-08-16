from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve.embedders.sentence_transformer import DEFAULT_MODEL
from retrieve.graph import KnowledgeGraph
from retrieve.storage.persistence import save_index
from retrieve.storage.vector_store import FaissVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive an exact graph-node index from an existing unit index."
    )
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    source = FaissVectorStore.load(args.source_index)
    if source is None:
        raise SystemExit(f"Missing source index: {args.source_index}")
    graph = KnowledgeGraph.load(args.graph_dir)
    searchable = graph.searchable_nodes()
    chunk_ids = [chunk.chunk_id for chunk in searchable]
    source_positions = {
        chunk_id: position for position, chunk_id in enumerate(source.chunk_ids)
    }
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in source_positions]
    if missing:
        raise SystemExit(
            f"Source index misses {len(missing)} graph nodes; first={missing[:3]}"
        )

    positions = np.asarray(
        [source_positions[chunk_id] for chunk_id in chunk_ids],
        dtype="int64",
    )
    vectors = source._index.reconstruct_batch(positions)
    exact = faiss.IndexFlatIP(vectors.shape[1])
    exact.add(vectors)
    store = FaissVectorStore(chunk_ids, exact)
    manifest = save_index(args.output_index, store, DEFAULT_MODEL)
    books = {
        str(
            node.metadata.get("textbook")
            or node.metadata.get("book_id")
            or "unknown"
        )
        for node in searchable
    }
    manifest["n_books"] = len(books)
    (args.output_index / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "source_vectors": source.size,
        "selected_vectors": store.size,
        "missing": len(missing),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "manifest": manifest,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
