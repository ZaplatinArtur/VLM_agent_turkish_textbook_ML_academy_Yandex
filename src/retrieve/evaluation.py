from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from schemas.retrieve import RetrievedChunk

from .metadata import canonical_subject


DEFAULT_CUTOFFS = (1, 5, 10, 50)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} of {path} is not an object")
            rows.append(row)
    return rows


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def score_ranking(
    ranked_ids: Sequence[str],
    relevant_ids: set[str],
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
) -> dict[str, dict[str, float]]:
    """Score one ranked list against binary relevance labels.

    AP@k uses ``min(number_of_relevant_documents, k)`` as its denominator.
    Duplicate target IDs are ignored after their first occurrence.
    """

    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")
    unique_ranked = _unique(ranked_ids)
    result: dict[str, dict[str, float]] = {}
    for cutoff in _validate_cutoffs(cutoffs):
        hits = 0
        precision_sum = 0.0
        first_relevant_rank: int | None = None
        for rank, target_id in enumerate(unique_ranked[:cutoff], start=1):
            if target_id not in relevant_ids:
                continue
            hits += 1
            precision_sum += hits / rank
            if first_relevant_rank is None:
                first_relevant_rank = rank
        result[str(cutoff)] = {
            "hit": float(hits > 0),
            "recall": hits / len(relevant_ids),
            "average_precision": precision_sum / min(len(relevant_ids), cutoff),
            "reciprocal_rank": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        }
    return result


def _validate_cutoffs(cutoffs: Iterable[int]) -> list[int]:
    values = sorted({int(value) for value in cutoffs})
    if not values or values[0] < 1:
        raise ValueError("cutoffs must contain positive integers")
    return values


def _page_id(chunk: RetrievedChunk) -> str:
    explicit = chunk.metadata.get("page_id")
    if explicit not in (None, ""):
        return str(explicit)
    textbook = str(chunk.metadata.get("textbook") or chunk.chunk_id.split(":", 1)[0])
    page = chunk.metadata.get("page")
    try:
        page_token = f"{int(page):04d}"
    except (TypeError, ValueError):
        page_token = str(page or chunk.chunk_id.rsplit(":", 1)[-1])
    return f"{textbook}:{page_token}"


def _target_id(chunk: RetrievedChunk, target_type: str) -> str:
    return chunk.chunk_id if target_type == "chunk_id" else _page_id(chunk)


def _qrel_targets(qrel: dict[str, Any]) -> tuple[str, set[str]]:
    chunk_ids = {str(value) for value in qrel.get("relevant_chunk_ids", []) if str(value)}
    page_ids = {str(value) for value in qrel.get("relevant_page_ids", []) if str(value)}
    if chunk_ids:
        return "chunk_id", chunk_ids
    return "page_id", page_ids


def _candidate_view(chunk: RetrievedChunk, rank: int, text_chars: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "page_id": _page_id(chunk),
        "score": chunk.score,
        "subject": chunk.metadata.get("subject"),
        "grade": chunk.metadata.get("grade"),
        "textbook": chunk.metadata.get("textbook"),
        "page": chunk.metadata.get("page"),
        "text": chunk.text[:text_chars],
    }


def _aggregate(
    rows: Sequence[dict[str, Any]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    metrics: dict[str, float | None] = {}
    for cutoff in cutoffs:
        key = str(cutoff)
        metrics[f"hit_rate_at_{cutoff}"] = _mean(
            row["metrics"][key]["hit"] for row in rows
        )
        metrics[f"recall_at_{cutoff}"] = _mean(
            row["metrics"][key]["recall"] for row in rows
        )
        metrics[f"map_at_{cutoff}"] = _mean(
            row["metrics"][key]["average_precision"] for row in rows
        )
        metrics[f"mrr_at_{cutoff}"] = _mean(
            row["metrics"][key]["reciprocal_rank"] for row in rows
        )
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "queries": len(rows),
        "metrics": metrics,
        "latency_ms": {
            "mean": _mean(latencies),
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
        },
    }


def _subject_aggregates(
    rows: Sequence[dict[str, Any]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["subject"]].append(row)
    return {
        subject: _aggregate(subject_rows, cutoffs)
        for subject, subject_rows in sorted(grouped.items())
    }


def _coverage_status(
    declared: set[str],
    corpus_targets: set[str],
    *,
    annotation_complete: bool = False,
) -> tuple[str, set[str], set[str]]:
    if not declared:
        return ("uncovered" if annotation_complete else "unannotated"), set(), set()
    present = declared & corpus_targets
    missing = declared - corpus_targets
    if not present:
        return "uncovered", present, missing
    if missing:
        return "partially_covered", present, missing
    return "covered", present, missing


def evaluate_dense_mmr(
    qrels_path: Path,
    *,
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
    fetch_k: int = 50,
    mmr_lambda: float = 0.5,
    use_subject_filter: bool = True,
    candidate_text_chars: int = 500,
    chunks: list[RetrievedChunk] | None = None,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the current Dense stage and its optional MMR reranker.

    Metrics are macro-averaged only over qrels with at least one labelled
    relevant target present in the current non-empty corpus. For partially
    covered qrels, the denominator is restricted to relevant targets that are
    actually present in the corpus.
    """

    cutoffs = _validate_cutoffs(cutoffs)
    if fetch_k < cutoffs[-1]:
        raise ValueError("fetch_k must be at least the largest cutoff")
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("mmr_lambda must be between 0 and 1")
    if candidate_text_chars < 0:
        raise ValueError("candidate_text_chars must not be negative")

    from .parsing import get_retrieved_chunks
    from .service import build_pipeline

    raw_corpus = get_retrieved_chunks() if chunks is None else chunks
    corpus = [chunk for chunk in raw_corpus if chunk.text.strip()]
    if not corpus:
        raise ValueError("retrieval corpus is empty")
    if pipeline is None:
        pipeline = build_pipeline(corpus, fetch_k=fetch_k, mmr_lambda=mmr_lambda)
    if len(pipeline.rankers) != 2:
        raise ValueError("evaluation pipeline must contain Dense and MMR rankers")
    dense_ranker, mmr_ranker = pipeline.rankers

    corpus_chunk_ids = {chunk.chunk_id for chunk in corpus}
    corpus_page_ids = {_page_id(chunk) for chunk in corpus}
    qrels = _read_jsonl(qrels_path)
    if not qrels:
        raise ValueError("qrels file is empty")

    seen_query_ids: set[str] = set()
    coverage_counts: dict[str, int] = defaultdict(int)
    dense_rows: list[dict[str, Any]] = []
    mmr_rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []

    for index, qrel in enumerate(qrels, start=1):
        query_id = str(qrel.get("query_id") or qrel.get("task_id") or index)
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen_query_ids.add(query_id)
        query = str(qrel.get("query") or "").strip()
        target_type, declared_relevant = _qrel_targets(qrel)
        corpus_targets = corpus_chunk_ids if target_type == "chunk_id" else corpus_page_ids
        subject = canonical_subject(qrel.get("subject")) or "unknown"
        retrieval_subject_value = (
            qrel.get("retrieval_subject")
            if "retrieval_subject" in qrel
            else qrel.get("subject")
        )
        retrieval_subject = canonical_subject(retrieval_subject_value)
        if not query:
            coverage_counts["missing_query"] += 1
            per_query.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "subject": subject,
                    "retrieval_subject": retrieval_subject,
                    "grade": qrel.get("grade"),
                    "target_type": target_type,
                    "coverage_status": "missing_query",
                    "declared_relevant_ids": sorted(declared_relevant),
                    "evaluated_relevant_ids": [],
                    "missing_relevant_ids": sorted(declared_relevant),
                    "dense": {"latency_ms": None, "metrics": None, "candidates": []},
                    "dense_mmr": {
                        "latency_ms": None,
                        "rerank_latency_ms": None,
                        "metrics": None,
                        "candidates": [],
                    },
                }
            )
            continue
        status, relevant, missing = _coverage_status(
            declared_relevant,
            corpus_targets,
            annotation_complete=qrel.get("annotation_status") == "complete",
        )
        coverage_counts[status] += 1

        subject_filter = retrieval_subject if use_subject_filter else None
        dense_started = time.perf_counter()
        dense_chunks = dense_ranker.rank(query, subject=subject_filter)[:fetch_k]
        dense_latency_ms = (time.perf_counter() - dense_started) * 1000
        mmr_started = time.perf_counter()
        mmr_chunks = mmr_ranker.rank(query, dense_chunks, subject=subject_filter)[:fetch_k]
        mmr_latency_ms = (time.perf_counter() - mmr_started) * 1000

        dense_ids = [_target_id(chunk, target_type) for chunk in dense_chunks]
        mmr_ids = [_target_id(chunk, target_type) for chunk in mmr_chunks]
        dense_metrics = score_ranking(dense_ids, relevant, cutoffs) if relevant else None
        mmr_metrics = score_ranking(mmr_ids, relevant, cutoffs) if relevant else None
        query_result = {
            "query_id": query_id,
            "query": query,
            "subject": subject,
            "retrieval_subject": retrieval_subject,
            "grade": qrel.get("grade"),
            "target_type": target_type,
            "coverage_status": status,
            "declared_relevant_ids": sorted(declared_relevant),
            "evaluated_relevant_ids": sorted(relevant),
            "missing_relevant_ids": sorted(missing),
            "dense": {
                "latency_ms": round(dense_latency_ms, 3),
                "metrics": dense_metrics,
                "candidates": [
                    _candidate_view(chunk, rank, candidate_text_chars)
                    for rank, chunk in enumerate(dense_chunks, start=1)
                ],
            },
            "dense_mmr": {
                "latency_ms": round(dense_latency_ms + mmr_latency_ms, 3),
                "rerank_latency_ms": round(mmr_latency_ms, 3),
                "metrics": mmr_metrics,
                "candidates": [
                    _candidate_view(chunk, rank, candidate_text_chars)
                    for rank, chunk in enumerate(mmr_chunks, start=1)
                ],
            },
        }
        per_query.append(query_result)
        if not relevant:
            continue
        dense_rows.append(
            {
                "query_id": query_id,
                "subject": subject,
                "latency_ms": dense_latency_ms,
                "metrics": dense_metrics,
            }
        )
        mmr_rows.append(
            {
                "query_id": query_id,
                "subject": subject,
                "latency_ms": dense_latency_ms + mmr_latency_ms,
                "metrics": mmr_metrics,
            }
        )

    dense_report = _aggregate(dense_rows, cutoffs)
    mmr_report = _aggregate(mmr_rows, cutoffs)
    deltas = {
        metric: (
            None
            if dense_report["metrics"][metric] is None
            else mmr_report["metrics"][metric] - dense_report["metrics"][metric]
        )
        for metric in dense_report["metrics"]
    }
    return {
        "schema_version": "dense-mmr-retrieval-eval-v1",
        "qrels_path": str(qrels_path),
        "configuration": {
            "cutoffs": cutoffs,
            "fetch_k": fetch_k,
            "mmr_lambda": mmr_lambda,
            "subject_filter": use_subject_filter,
            "candidate_text_chars": candidate_text_chars,
        },
        "metric_definitions": {
            "recall_at_k": "relevant in top-k / relevant present in corpus",
            "hit_rate_at_k": "fraction of queries with at least one relevant target in top-k",
            "map_at_k": "macro mean of AP@k with denominator min(relevant, k)",
            "mrr_at_k": "macro mean reciprocal rank of the first relevant target within k",
        },
        "coverage": {
            "total_qrels": len(qrels),
            "scored_qrels": len(dense_rows),
            "missing_query_qrels": coverage_counts["missing_query"],
            "unannotated_qrels": coverage_counts["unannotated"],
            "uncovered_qrels": coverage_counts["uncovered"],
            "partially_covered_qrels": coverage_counts["partially_covered"],
            "fully_covered_qrels": coverage_counts["covered"],
        },
        "variants": {
            "dense": {
                **dense_report,
                "by_subject": _subject_aggregates(dense_rows, cutoffs),
            },
            "dense_mmr": {
                **mmr_report,
                "by_subject": _subject_aggregates(mmr_rows, cutoffs),
            },
        },
        "dense_mmr_minus_dense": deltas,
        "per_query": per_query,
    }


def prepare_qrels_from_agent_run(
    tasks_path: Path,
    run_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze the first textbook-search query per task for manual qrels labeling."""

    tasks = {str(row.get("task_id")): row for row in _read_jsonl(tasks_path)}
    runs = _read_jsonl(run_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with_query = 0
    without_query = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for run in runs:
            task_id = str(run.get("task_id") or "")
            if not task_id:
                raise ValueError("run row has no task_id")
            task = tasks.get(task_id, {})
            query = ""
            retrieval_subject = None
            retrieval_grade = None
            for call in run.get("tool_calls", []):
                if call.get("tool") != "search_textbooks":
                    continue
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                query = str(args.get("query") or "").strip()
                if query:
                    retrieval_subject = args.get("subject")
                    retrieval_grade = args.get("grade")
                    break
            if query:
                with_query += 1
            else:
                without_query += 1
            qrel = {
                "query_id": task_id,
                "task_id": task_id,
                "query": query,
                "subject": task.get("subject"),
                "grade": task.get("grade"),
                "retrieval_subject": retrieval_subject,
                "retrieval_grade": retrieval_grade,
                "relevant_chunk_ids": [],
                "needs_manual_query": not bool(query),
                "needs_manual_relevance": True,
                "annotation_notes": "",
            }
            handle.write(json.dumps(qrel, ensure_ascii=False) + "\n")
    return {
        "records": len(runs),
        "with_agent_query": with_query,
        "needs_manual_query": without_query,
        "output": str(output_path),
    }
