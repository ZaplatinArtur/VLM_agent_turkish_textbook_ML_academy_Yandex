from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve.metadata import subjects_match
from retrieve.service import build_pipeline


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay real agent search queries against the configured retriever."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--agent-results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit-calls", type=int)
    args = parser.parse_args()

    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    calls: list[tuple[str, str]] = []
    for result in read_jsonl(args.agent_results):
        task_id = str(result["task_id"])
        for call in result.get("tool_calls") or []:
            query = str((call.get("args") or {}).get("query") or "").strip()
            if query:
                calls.append((task_id, query))
    if args.limit_calls:
        calls = calls[: args.limit_calls]
    if not calls:
        raise SystemExit("No textbook-search queries found")

    pipeline = build_pipeline()
    trace_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    hit_lengths: list[int] = []
    retrieval_lengths: list[int] = []
    top1_subject_matches = 0
    hit_subject_matches = 0
    calls_with_subject_match = 0
    empty_calls = 0
    card_kinds: Counter[str] = Counter()
    relation_hits: Counter[str] = Counter()

    for position, (task_id, query) in enumerate(calls, 1):
        task = tasks[task_id]
        subject = task.get("subject")
        grade = task.get("grade")
        started = time.perf_counter()
        hits = pipeline.run(
            query,
            k=args.k,
            subject=str(subject) if subject else None,
            grade=grade,
        )
        latency_ms = (time.perf_counter() - started) * 1_000
        latencies.append(latency_ms)
        if not hits:
            empty_calls += 1

        subject_flags = [
            subjects_match(hit.metadata.get("subject"), subject)
            if subject
            else True
            for hit in hits
        ]
        if subject_flags and subject_flags[0]:
            top1_subject_matches += 1
        if any(subject_flags):
            calls_with_subject_match += 1
        hit_subject_matches += sum(subject_flags)

        serialized_hits = []
        for hit in hits:
            hit_lengths.append(len(hit.text))
            retrieval_lengths.append(
                len(str(hit.metadata.get("retrieval_text") or hit.text))
            )
            kind = str(hit.metadata.get("knowledge_kind") or "unknown")
            card_kinds[kind] += 1
            for relation in ("has_theory", "has_example", "has_solution"):
                relation_hits[relation] += int(bool(hit.metadata.get(relation)))
            serialized_hits.append(
                {
                    "chunk_id": hit.chunk_id,
                    "anchor_id": hit.metadata.get("graph_anchor_id"),
                    "score": hit.score,
                    "subject": hit.metadata.get("subject"),
                    "grade": hit.metadata.get("grade"),
                    "textbook": hit.metadata.get("textbook"),
                    "source": hit.metadata.get("source"),
                    "knowledge_kind": kind,
                    "anchor_kind": hit.metadata.get("graph_anchor_kind"),
                    "has_theory": bool(hit.metadata.get("has_theory")),
                    "has_example": bool(hit.metadata.get("has_example")),
                    "has_solution": bool(hit.metadata.get("has_solution")),
                    "retrieval_chars": len(
                        str(hit.metadata.get("retrieval_text") or hit.text)
                    ),
                    "card_chars": len(hit.text),
                }
            )
        trace_rows.append(
            {
                "task_id": task_id,
                "query": query,
                "scope": {"subject": subject, "grade": grade},
                "latency_ms": round(latency_ms, 3),
                "hits": serialized_hits,
            }
        )
        if position % 25 == 0 or position == len(calls):
            print(f"[{position}/{len(calls)}] replayed", flush=True)

    hit_count = len(hit_lengths)
    report = {
        "calls": len(calls),
        "hits": hit_count,
        "empty_calls": empty_calls,
        "calls_with_subject_match": calls_with_subject_match,
        "call_subject_match_rate": round(
            calls_with_subject_match / len(calls), 4
        ),
        "top1_subject_match_rate": round(
            top1_subject_matches / len(calls), 4
        ),
        "hit_subject_match_rate": round(
            hit_subject_matches / hit_count, 4
        )
        if hit_count
        else 0.0,
        "card_chars": {
            "median": statistics.median(hit_lengths) if hit_lengths else 0,
            "p10": percentile([float(x) for x in hit_lengths], 0.1),
            "under_100": sum(length < 100 for length in hit_lengths),
            "under_100_rate": round(
                sum(length < 100 for length in hit_lengths) / hit_count, 4
            )
            if hit_count
            else 0.0,
        },
        "retrieval_text_chars": {
            "median": statistics.median(retrieval_lengths)
            if retrieval_lengths
            else 0,
            "under_100_rate": round(
                sum(length < 100 for length in retrieval_lengths) / hit_count,
                4,
            )
            if hit_count
            else 0.0,
        },
        "knowledge_kinds": dict(card_kinds.most_common()),
        "relation_hit_rates": {
            key: round(value / hit_count, 4) if hit_count else 0.0
            for key, value in relation_hits.items()
        },
        "latency_ms": {
            "median": round(statistics.median(latencies), 3),
            "p90": round(percentile(latencies, 0.9), 3),
            "mean": round(statistics.fmean(latencies), 3),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with args.trace.open("w", encoding="utf-8", newline="\n") as output:
            for row in trace_rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
