"""Офлайн-оценка ретрива по qrels: сравнение профилей с бейзлайном.

Метрики совпадают с лексическим бейзлайном (vlm_judge.retrieval_eval), чтобы
числа были сопоставимы: hit_rate@k, recall@k, nDCG@k, MRR@k + latency.

Формат qrels — JSONL, по строке на запрос:
    {"query_id": "...", "query": "...", "subject": null, "grade": null,
     "relevant_chunk_ids": ["book:12", ...], "relevant_page_ids": []}
relevant_chunk_ids приоритетнее; если их нет — матчим по page_id.

Примеры:
    # быстрый прогон на одной книге, три системы, дельты к первой (bm25)
    python -m retrieve.evaluate --qrels data/qrels.jsonl \
        --book 8-sinif-inkilap-tarihi-ders-kitabi-cevaplari-meb-yayinlari \
        --systems bm25 dense hybrid hybrid_rerank --k 1 5 10

    # на всём корпусе, со снапшотами индексов, в файл
    python -m retrieve.evaluate --qrels data/qrels.jsonl \
        --systems dense hybrid_rerank --k 1 5 10 \
        --index-root data/cache/eval_index --output reports/retrieval_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from schemas.retrieve import RetrievedChunk

from .index import Index
from .pipelines import PROFILES, build_profile


def _read_qrels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = str(record.get("query_id") or record.get("task_id") or line_number)
            if query_id in seen:
                raise ValueError(f"duplicate query_id: {query_id}")
            seen.add(query_id)
            if not str(record.get("query") or "").strip():
                raise ValueError(f"query {query_id} is empty")
            record["query_id"] = query_id
            rows.append(record)
    if not rows:
        raise ValueError("qrels file is empty")
    return rows


def _page_id(chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata or {}
    return str(metadata.get("page_id") or metadata.get("page") or chunk.chunk_id)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def evaluate_pipeline(
        pipeline,
        qrels: list[dict[str, Any]],
        ks: list[int],
) -> dict[str, Any]:
    """Прогоняет пайплайн по qrels и усредняет метрики по запросам."""
    cutoffs = sorted(set(int(k) for k in ks))
    depth = cutoffs[-1]
    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    for qrel in qrels:
        query = str(qrel["query"]).strip()
        subject = qrel.get("subject") or None
        relevant_chunks = {str(v) for v in qrel.get("relevant_chunk_ids", []) if str(v)}
        relevant_pages = {str(v) for v in qrel.get("relevant_page_ids", []) if str(v)}
        if relevant_chunks:
            relevant, target = relevant_chunks, "chunk_id"
        elif relevant_pages:
            relevant, target = relevant_pages, "page_id"
        else:
            raise ValueError(f"query {qrel['query_id']} has no relevant ids")

        started = time.perf_counter()
        results = pipeline.run(query, k=depth, subject=subject)
        latencies.append((time.perf_counter() - started) * 1000)

        ranked = [
            chunk.chunk_id if target == "chunk_id" else _page_id(chunk)
            for chunk in results
        ]
        first_hit = next(
            (rank for rank, tid in enumerate(ranked, start=1) if tid in relevant),
            None,
        )
        metrics: dict[str, dict[str, float]] = {}
        for cutoff in cutoffs:
            found = relevant.intersection(ranked[:cutoff])
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, tid in enumerate(ranked[:cutoff], start=1)
                if tid in relevant
            )
            ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), cutoff) + 1))
            metrics[str(cutoff)] = {
                "hit": float(bool(found)),
                "recall": len(found) / len(relevant),
                "ndcg": dcg / ideal if ideal else 0.0,
                "mrr": 1.0 / first_hit if first_hit is not None and first_hit <= cutoff else 0.0,
            }
        per_query.append({"query_id": qrel["query_id"], "metrics": metrics})

    aggregate: dict[str, float] = {}
    for cutoff in cutoffs:
        key = str(cutoff)
        for name in ("hit", "recall", "ndcg", "mrr"):
            label = f"{'hit_rate' if name == 'hit' else name}_at_{cutoff}"
            aggregate[label] = _mean(row["metrics"][key][name] for row in per_query)
    return {
        "queries": len(per_query),
        "cutoffs": cutoffs,
        "metrics": aggregate,
        "latency_ms": {
            "mean": round(_mean(latencies), 2),
            "median": round(statistics.median(latencies), 2) if latencies else 0.0,
            "p95": round(_percentile(latencies, 0.95), 2),
        },
    }


def load_corpus(book: str | None) -> list[RetrievedChunk]:
    if book:
        from paths import CHUNKS_JSONL_DIR

        path = CHUNKS_JSONL_DIR / f"{book}.jsonl"
        if not path.exists():
            raise SystemExit(f"Нет файла: {path}")
        with path.open("r", encoding="utf-8") as handle:
            chunks = [RetrievedChunk.model_validate_json(line) for line in handle if line.strip()]
    else:
        from .parsing import get_retrieved_chunks

        chunks = get_retrieved_chunks()
    return [chunk for chunk in chunks if chunk.text.strip()]


def _format_report(systems: list[str], results: dict[str, dict], cutoffs: list[int]) -> str:
    baseline = systems[0]
    rows = [f"hit_rate_at_{k}" for k in cutoffs]
    rows += [f"recall_at_{k}" for k in cutoffs]
    rows += [f"ndcg_at_{k}" for k in cutoffs]
    rows += [f"mrr_at_{k}" for k in cutoffs]
    width = max(len("metric"), *(len(row) for row in rows))
    header = "metric".ljust(width) + "".join(f"{name:>22}" for name in systems)
    lines = [f"Бейзлайн для дельт: {baseline}", header, "-" * len(header)]
    for row in rows:
        cells = []
        for system in systems:
            value = results[system]["metrics"].get(row, 0.0) * 100
            if system == baseline:
                cells.append(f"{value:>21.1f}%")
            else:
                delta = value - results[baseline]["metrics"].get(row, 0.0) * 100
                cells.append(f"{value:>10.1f}% ({delta:+.1f})")
        lines.append(row.ljust(width) + "".join(cells))
    lat = ["latency mean ms", "latency p95 ms"]
    lines.append("-" * len(header))
    for label, key in zip(lat, ("mean", "p95")):
        cells = [f"{results[s]['latency_ms'][key]:>21.1f}" for s in systems]
        lines.append(label.ljust(width) + "".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сравнить профили ретрива на qrels")
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--systems", nargs="+", default=["bm25", "dense", "hybrid_rerank"],
                        choices=PROFILES, help="первый — бейзлайн для дельт")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10], dest="ks")
    parser.add_argument("--book", default=None, help="прогон на одной книге (иначе весь корпус)")
    parser.add_argument("--index-root", default=None, type=Path,
                        help="каталог снапшотов dense-индексов (иначе строятся в памяти)")
    parser.add_argument("--fetch-k", type=int, default=200)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args(argv)

    qrels = _read_qrels(args.qrels)
    corpus = load_corpus(args.book)
    index = Index(corpus)
    print(f"Корпус: {len(corpus)} чанков | запросов: {len(qrels)} | системы: {args.systems}\n")

    results: dict[str, dict] = {}
    for system in args.systems:
        started = time.perf_counter()
        pipeline = build_profile(system, index, index_root=args.index_root, fetch_k=args.fetch_k)
        for ranker in pipeline.rankers:
            build = getattr(ranker, "build", None)
            if callable(build):
                build()
        report = evaluate_pipeline(pipeline, qrels, args.ks)
        report["build_and_eval_seconds"] = round(time.perf_counter() - started, 2)
        results[system] = report
        print(f"[{system}] готово за {report['build_and_eval_seconds']}s")

    cutoffs = sorted(set(args.ks))
    print("\n" + _format_report(args.systems, results, cutoffs))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"corpus_chunks": len(corpus), "queries": len(qrels), "systems": results},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nОтчёт: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
