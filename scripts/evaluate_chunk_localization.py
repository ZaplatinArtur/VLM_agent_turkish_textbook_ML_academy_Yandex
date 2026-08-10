from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schemas.retrieve import RetrievedChunk


def _read_chunks(root: Path) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunks.append(RetrievedChunk.model_validate_json(line))
    return chunks


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    faiss.normalize_L2(matrix)
    return matrix


def _index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _metrics(
    indices: np.ndarray,
    expected_parent_ids: list[str],
    candidate_parent_ids: list[str],
) -> dict[str, float]:
    ranks: list[int | None] = []
    for row, expected in zip(indices, expected_parent_ids):
        rank = next(
            (
                position
                for position, candidate_index in enumerate(row, 1)
                if candidate_parent_ids[int(candidate_index)] == expected
            ),
            None,
        )
        ranks.append(rank)
    return {
        "hit_at_1": round(sum(rank == 1 for rank in ranks) / len(ranks), 4),
        "hit_at_5": round(sum(rank is not None for rank in ranks) / len(ranks), 4),
        "mrr_at_5": round(
            sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks),
            4,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic: retrieve an exercise's parent page from page chunks versus "
            "task-aware chunks. This measures localization, not downstream QA."
        )
    )
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--units-dir", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    pages = [chunk for chunk in _read_chunks(args.pages_dir) if chunk.text.strip()]
    units = [chunk for chunk in _read_chunks(args.units_dir) if chunk.text.strip()]
    page_by_id = {page.chunk_id: page for page in pages}
    eligible = [
        unit
        for unit in units
        if unit.metadata.get("unit_kind") == "exercise"
        and len(unit.text.split()) >= 12
        and len(page_by_id.get(str(unit.metadata.get("parent_chunk_id")), unit).text.split())
        > 128
    ]
    random.Random(args.seed).shuffle(eligible)
    queries = eligible[: args.queries]
    if len(queries) < args.queries:
        raise SystemExit(f"Only {len(queries)} eligible exercise queries")

    model = SentenceTransformer(args.model)
    page_vectors = _normalize(
        model.encode(
            [page.text for page in pages],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
    )
    unit_vectors = _normalize(
        model.encode(
            [unit.text for unit in units],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
    )
    query_vectors = _normalize(
        model.encode(
            [unit.text for unit in queries],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )

    _, page_indices = _index(page_vectors).search(query_vectors, 5)
    _, unit_indices = _index(unit_vectors).search(query_vectors, 5)
    expected = [str(unit.metadata["parent_chunk_id"]) for unit in queries]
    page_parent_ids = [page.chunk_id for page in pages]
    unit_parent_ids = [
        str(unit.metadata.get("parent_chunk_id") or unit.chunk_id) for unit in units
    ]
    report: dict[str, Any] = {
        "diagnostic": (
            "Exact/near-exact exercise localization on pages longer than the "
            "embedding model's 128-token limit; not a downstream QA metric."
        ),
        "model": args.model,
        "queries": len(queries),
        "page_candidates": len(pages),
        "unit_candidates": len(units),
        "page_chunking": _metrics(page_indices, expected, page_parent_ids),
        "hybrid_task_chunking": _metrics(unit_indices, expected, unit_parent_ids),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
