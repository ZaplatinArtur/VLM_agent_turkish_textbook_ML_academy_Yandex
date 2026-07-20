from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from .retrieval import search_bm25


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} of {path} is not an object")
            values.append(value)
    return values


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def evaluate_retrieval(
    index_path: Path,
    qrels_path: Path,
    *,
    ks: Iterable[int] = (1, 5, 10),
    mode: str = "or",
    low_information_weight: float = 0.25,
) -> dict[str, Any]:
    cutoffs = sorted(set(int(value) for value in ks))
    if not cutoffs or cutoffs[0] < 1 or cutoffs[-1] > 100:
        raise ValueError("retrieval cutoffs must be between 1 and 100")
    qrels = _read_jsonl(qrels_path)
    if not qrels:
        raise ValueError("qrels file is empty")

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    seen_query_ids: set[str] = set()
    for index, qrel in enumerate(qrels, start=1):
        query_id = str(qrel.get("query_id") or qrel.get("task_id") or index)
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen_query_ids.add(query_id)
        query = str(qrel.get("query") or "").strip()
        if not query:
            raise ValueError(f"query {query_id} is empty")
        relevant_chunks = {str(value) for value in qrel.get("relevant_chunk_ids", []) if str(value)}
        relevant_pages = {str(value) for value in qrel.get("relevant_page_ids", []) if str(value)}
        if relevant_chunks:
            target_field = "chunk_id"
            relevant = relevant_chunks
        elif relevant_pages:
            target_field = "page_id"
            relevant = relevant_pages
        else:
            raise ValueError(f"query {query_id} has no relevant_chunk_ids or relevant_page_ids")

        result = search_bm25(
            index_path,
            query,
            top_k=cutoffs[-1],
            subject=str(qrel["subject"]) if qrel.get("subject") not in (None, "") else None,
            grade=qrel.get("grade"),
            mode=mode,
            low_information_weight=low_information_weight,
        )
        latencies.append(float(result["latency_ms"]))
        ranked_targets = [str(hit.get(target_field) or "") for hit in result["hits"]]
        first_relevant_rank = next(
            (rank for rank, target in enumerate(ranked_targets, start=1) if target in relevant),
            None,
        )
        per_cutoff = {}
        for cutoff in cutoffs:
            retrieved_relevant = relevant.intersection(ranked_targets[:cutoff])
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, target in enumerate(ranked_targets[:cutoff], start=1)
                if target in relevant
            )
            ideal_dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, min(len(relevant), cutoff) + 1)
            )
            per_cutoff[str(cutoff)] = {
                "hit": bool(retrieved_relevant),
                "recall": len(retrieved_relevant) / len(relevant),
                "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
                "reciprocal_rank": (
                    1.0 / first_relevant_rank
                    if first_relevant_rank is not None and first_relevant_rank <= cutoff
                    else 0.0
                ),
            }
        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "target_type": target_field,
                "relevant_count": len(relevant),
                "first_relevant_rank": first_relevant_rank,
                "latency_ms": result["latency_ms"],
                "metrics": per_cutoff,
                "top_hits": [
                    {
                        "rank": hit["rank"],
                        "chunk_id": hit["chunk_id"],
                        "page_id": hit["page_id"],
                        "source_url": hit.get("source_url"),
                    }
                    for hit in result["hits"]
                ],
            }
        )

    metrics = {}
    for cutoff in cutoffs:
        key = str(cutoff)
        metrics[f"hit_rate_at_{cutoff}"] = _mean(float(row["metrics"][key]["hit"]) for row in rows)
        metrics[f"recall_at_{cutoff}"] = _mean(row["metrics"][key]["recall"] for row in rows)
        metrics[f"ndcg_at_{cutoff}"] = _mean(row["metrics"][key]["ndcg"] for row in rows)
        metrics[f"mrr_at_{cutoff}"] = _mean(row["metrics"][key]["reciprocal_rank"] for row in rows)

    return {
        "schema_version": "retrieval-eval-v1",
        "index_path": str(index_path),
        "qrels_path": str(qrels_path),
        "mode": mode,
        "low_information_weight": low_information_weight,
        "cutoffs": cutoffs,
        "queries": len(rows),
        "metrics": metrics,
        "latency_ms": {
            "mean": round(_mean(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "p95": round(_percentile(latencies, 0.95) or 0.0, 3),
        },
        "per_query": rows,
    }


def prepare_qrels_template(benchmark_path: Path, output_path: Path) -> dict[str, Any]:
    tasks = _read_jsonl(benchmark_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_queries = 0
    needs_manual_query = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
            question_text = str(task.get("question_text") or "").strip()
            if question_text:
                query = question_text
                text_queries += 1
            else:
                query = " ".join(
                    str(metadata.get(field) or "").strip()
                    for field in ("topic_area", "sub_topic")
                    if metadata.get(field)
                )
                needs_manual_query += 1
            value = {
                "query_id": str(task.get("task_id") or ""),
                "task_id": str(task.get("task_id") or ""),
                "query": query,
                "subject": task.get("subject"),
                "grade": task.get("grade"),
                "relevant_page_ids": [],
                "relevant_chunk_ids": [],
                "needs_manual_query": not bool(question_text),
                "annotation_notes": "",
            }
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    return {
        "records": len(tasks),
        "query_from_question_text": text_queries,
        "needs_manual_query": needs_manual_query,
        "output": str(output_path),
    }
