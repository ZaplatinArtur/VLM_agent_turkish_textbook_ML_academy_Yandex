from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve.ingest.chunking import QwenEducationalRefiner
from schemas.retrieve import RetrievedChunk


def _load_books(
    input_dir: Path,
) -> tuple[dict[Path, list[RetrievedChunk]], list[tuple[Path, str]]]:
    books: dict[Path, list[RetrievedChunk]] = {}
    candidates: list[tuple[Path, str]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        chunks: list[RetrievedChunk] = []
        ambiguous_parents: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk = RetrievedChunk.model_validate_json(line)
                chunks.append(chunk)
                if chunk.metadata.get("low_confidence"):
                    ambiguous_parents.add(
                        str(chunk.metadata.get("parent_chunk_id") or "")
                    )
        books[path] = chunks
        candidates.extend((path, parent_id) for parent_id in ambiguous_parents)
    return books, candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply Qwen only to low-confidence pages from hybrid chunking."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    books, candidates = _load_books(args.input_dir)
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.max_pages] if args.max_pages else candidates
    selected_set = set(selected)

    grouped: dict[tuple[Path, str], list[RetrievedChunk]] = defaultdict(list)
    for path, chunks in books.items():
        for chunk in chunks:
            key = (path, str(chunk.metadata.get("parent_chunk_id") or ""))
            if key in selected_set:
                grouped[key].append(chunk)

    refiner = QwenEducationalRefiner(base_url=args.base_url, model=args.model)
    decisions_by_page: dict[tuple[Path, str], dict[int, Any]] = {}
    failures: list[dict[str, str]] = []

    def run(
        item: tuple[tuple[Path, str], list[RetrievedChunk]],
    ) -> tuple[tuple[Path, str], dict[int, Any]]:
        key, units = item
        result = refiner.refine(key[1], units)
        return key, {decision.index: decision for decision in result.decisions}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run, item): item[0] for item in grouped.items()
        }
        completed = 0
        for future in as_completed(futures):
            key = futures[future]
            try:
                result_key, decisions = future.result()
                decisions_by_page[result_key] = decisions
            except Exception as exc:
                failures.append(
                    {
                        "book": key[0].stem,
                        "parent_chunk_id": key[1],
                        "error": str(exc),
                    }
                )
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"[{completed}/{len(futures)}] refined, "
                    f"failures={len(failures)}",
                    flush=True,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    refined_units = changed_units = 0

    for input_path, chunks in books.items():
        page_indices: Counter[str] = Counter()
        output_path = args.output_dir / input_path.name
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for chunk in chunks:
                original_kind = str(chunk.metadata.get("unit_kind") or "other")
                before[original_kind] += 1
                parent_id = str(chunk.metadata.get("parent_chunk_id") or "")
                key = (input_path, parent_id)
                index = page_indices[parent_id]
                page_indices[parent_id] += 1
                decision = decisions_by_page.get(key, {}).get(index)
                final_kind = original_kind
                metadata = dict(chunk.metadata)
                if decision is not None:
                    final_kind = decision.kind.value
                    refined_units += 1
                    changed_units += int(final_kind != original_kind)
                    transitions[f"{original_kind} -> {final_kind}"] += 1
                    metadata.update(
                        {
                            "rule_unit_kind": original_kind,
                            "unit_kind": final_kind,
                            "qwen_refined": True,
                            "qwen_confidence": decision.confidence,
                            "qwen_reason": decision.reason,
                        }
                    )
                else:
                    metadata["qwen_refined"] = False
                after[final_kind] += 1
                output.write(
                    chunk.model_copy(update={"metadata": metadata}).model_dump_json()
                    + "\n"
                )

    runtime = round(time.perf_counter() - started, 3)
    report = {
        "books": len(books),
        "candidate_pages": len(candidates),
        "selected_pages": len(selected),
        "successful_pages": len(decisions_by_page),
        "failures": failures,
        "refined_units": refined_units,
        "changed_units": changed_units,
        "change_rate": round(changed_units / refined_units, 4)
        if refined_units
        else 0.0,
        "kind_counts_before": dict(sorted(before.items())),
        "kind_counts_after": dict(sorted(after.items())),
        "transitions": dict(transitions.most_common()),
        "runtime_seconds": runtime,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
