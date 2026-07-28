from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve.knowledge import KnowledgeBaseBuilder
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
        description="Build relation-aware theory/exercise/solution knowledge cards."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit-books", type=int)
    parser.add_argument("--min-retrieval-chars", type=int, default=70)
    args = parser.parse_args()

    started = time.perf_counter()
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if args.limit_books is not None:
        paths = paths[: args.limit_books]
    if not paths:
        raise SystemExit(f"No JSONL books found in {args.input_dir}")

    builder = KnowledgeBaseBuilder(
        min_retrieval_chars=args.min_retrieval_chars
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_kinds: Counter[str] = Counter()
    card_kinds: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    source_chunks = cards_count = 0
    retrieval_lengths: list[int] = []
    card_lengths: list[int] = []

    for index, input_path in enumerate(paths, 1):
        chunks = read_chunks(input_path)
        cards = builder.build(chunks)
        output_path = args.output_dir / input_path.name
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for card in cards:
                output.write(card.model_dump_json() + "\n")
        source_chunks += len(chunks)
        cards_count += len(cards)
        source_kinds.update(
            str(chunk.metadata.get("unit_kind") or "other") for chunk in chunks
        )
        for card in cards:
            card_kinds[str(card.metadata.get("knowledge_kind") or "unknown")] += 1
            for relation in ("has_theory", "has_example", "has_solution"):
                relation_counts[relation] += int(bool(card.metadata.get(relation)))
            retrieval_lengths.append(
                len(str(card.metadata.get("retrieval_text") or ""))
            )
            card_lengths.append(len(card.text))
        if index % 20 == 0 or index == len(paths):
            print(
                f"[{index}/{len(paths)}] source={source_chunks}, cards={cards_count}",
                flush=True,
            )

    report = {
        "books": len(paths),
        "source_chunks": source_chunks,
        "knowledge_cards": cards_count,
        "reduction_ratio": round(cards_count / source_chunks, 4)
        if source_chunks
        else 0.0,
        "source_unit_kinds": dict(source_kinds.most_common()),
        "knowledge_kinds": dict(card_kinds.most_common()),
        "relations": dict(relation_counts),
        "retrieval_text_chars": {
            "median": statistics.median(retrieval_lengths)
            if retrieval_lengths
            else 0,
            "under_100": sum(length < 100 for length in retrieval_lengths),
            "under_100_rate": round(
                sum(length < 100 for length in retrieval_lengths)
                / len(retrieval_lengths),
                4,
            )
            if retrieval_lengths
            else 0.0,
        },
        "card_chars": {
            "median": statistics.median(card_lengths) if card_lengths else 0,
            "p90": (
                sorted(card_lengths)[int(0.9 * (len(card_lengths) - 1))]
                if card_lengths
                else 0
            ),
        },
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
