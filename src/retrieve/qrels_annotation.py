from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from schemas.retrieve import RetrievedChunk
from vlm_judge.backends import BackendResponse
from vlm_judge.prompts import JudgeRequest
from vlm_judge.retrieval import build_bm25_index, search_bm25

from .parsing import get_retrieved_chunks
from .service import build_pipeline


ANNOTATION_SYSTEM_PROMPT = """You label textbook chunks for retrieval evaluation.
Decide whether each chunk contains evidence that materially helps solve the exact
school task. Broad topical similarity is not enough. A relevant chunk must state
a needed fact, rule, formula, definition, or directly useful worked example.
Use the reference answer only to disambiguate the task; do not label a chunk
relevant merely because it contains the answer letter or an unrelated occurrence
of the same word.

Return one JSON object with exactly these keys:
relevant_chunk_ids, uncertain_chunk_ids, confidence, rationale.
Both ID fields must be arrays containing only IDs from the supplied candidates.
Put borderline or merely partial evidence in uncertain_chunk_ids, not in
relevant_chunk_ids. confidence must be between 0 and 1. Keep rationale under 100
words. If no chunk is useful, return empty arrays."""


class AnnotationBackend(Protocol):
    model: str

    def complete(self, request: JudgeRequest) -> BackendResponse: ...


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


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _bm25_record(chunk: RetrievedChunk) -> dict[str, Any]:
    metadata = dict(chunk.metadata)
    textbook = str(metadata.get("textbook") or chunk.chunk_id.split(":", 1)[0])
    page = metadata.get("page")
    metadata.update(
        {
            "book_id": metadata.get("book_id") or textbook,
            "page_number": metadata.get("page_number") or page,
        }
    )
    return {
        "kind": "text",
        "chunk_id": chunk.chunk_id,
        "page_id": chunk.chunk_id,
        "text": chunk.text,
        "metadata": metadata,
    }


def _write_bm25_source(chunks: list[RetrievedChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            if chunk.text.strip():
                handle.write(json.dumps(_bm25_record(chunk), ensure_ascii=False) + "\n")


def _fallback_query(run: dict[str, Any]) -> str:
    evidence = run.get("image_evidence")
    if not isinstance(evidence, list):
        return ""
    return " ".join(str(value).strip() for value in evidence if str(value).strip())[:500]


def build_candidate_pool(
    qrels_path: Path,
    tasks_path: Path,
    run_path: Path,
    output_path: Path,
    *,
    bm25_source_path: Path,
    bm25_index_path: Path,
    dense_k: int = 200,
    bm25_k: int = 100,
    text_chars: int = 400,
    chunks: list[RetrievedChunk] | None = None,
    dense_ranker: Any | None = None,
) -> dict[str, Any]:
    """Build an all-corpus Dense+BM25 pool wider than the evaluated top-50."""

    if dense_k < 1 or not 1 <= bm25_k <= 100:
        raise ValueError("dense_k must be positive and bm25_k must be between 1 and 100")
    if text_chars < 1:
        raise ValueError("text_chars must be positive")
    corpus = [
        chunk
        for chunk in (get_retrieved_chunks() if chunks is None else chunks)
        if chunk.text.strip()
    ]
    if not corpus:
        raise ValueError("retrieval corpus is empty")
    if dense_ranker is None:
        dense_ranker = build_pipeline(corpus, fetch_k=dense_k).rankers[0]

    _write_bm25_source(corpus, bm25_source_path)
    bm25_summary = build_bm25_index(bm25_source_path, bm25_index_path)
    qrels = _read_jsonl(qrels_path)
    tasks = {str(row.get("task_id")): row for row in _read_jsonl(tasks_path)}
    runs = {str(row.get("task_id")): row for row in _read_jsonl(run_path)}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus}

    output_rows: list[dict[str, Any]] = []
    fallback_queries = 0
    for qrel in qrels:
        task_id = str(qrel.get("task_id") or qrel.get("query_id") or "")
        task = tasks.get(task_id, {})
        run = runs.get(task_id, {})
        query = str(qrel.get("query") or "").strip()
        query_source = "agent_tool_call"
        if not query:
            query = _fallback_query(run)
            query_source = "image_evidence_fallback"
            fallback_queries += 1
        if not query:
            raise ValueError(f"cannot build a query for task {task_id}")

        dense_chunks = dense_ranker.rank(query, subject=None)[:dense_k]
        lexical = search_bm25(bm25_index_path, query, top_k=bm25_k, mode="or")
        dense_by_id = {chunk.chunk_id: (rank, chunk) for rank, chunk in enumerate(dense_chunks, 1)}
        bm25_by_id = {str(hit["chunk_id"]): hit for hit in lexical["hits"]}
        candidate_ids = list(dense_by_id)
        candidate_ids.extend(chunk_id for chunk_id in bm25_by_id if chunk_id not in dense_by_id)
        candidates = []
        for chunk_id in candidate_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            dense_entry = dense_by_id.get(chunk_id)
            lexical_entry = bm25_by_id.get(chunk_id)
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "dense_rank": dense_entry[0] if dense_entry else None,
                    "dense_score": dense_entry[1].score if dense_entry else None,
                    "bm25_rank": lexical_entry.get("rank") if lexical_entry else None,
                    "bm25_score": lexical_entry.get("lexical_score") if lexical_entry else None,
                    "subject": chunk.metadata.get("subject"),
                    "grade": chunk.metadata.get("grade"),
                    "textbook": chunk.metadata.get("textbook"),
                    "page": chunk.metadata.get("page"),
                    "text": chunk.text[:text_chars],
                }
            )
        output_rows.append(
            {
                "task_id": task_id,
                "query": query,
                "query_source": query_source,
                "subject": task.get("subject") or qrel.get("subject"),
                "grade": task.get("grade") if "grade" in task else qrel.get("grade"),
                "retrieval_subject": qrel.get("retrieval_subject"),
                "retrieval_grade": qrel.get("retrieval_grade"),
                "reference_answer": task.get("reference_answer"),
                "image_evidence": run.get("image_evidence") or [],
                "pool": {"dense_k": dense_k, "bm25_k": bm25_k},
                "candidates": candidates,
            }
        )
    _write_jsonl_atomic(output_path, output_rows)
    return {
        "tasks": len(output_rows),
        "fallback_queries": fallback_queries,
        "mean_candidates": sum(len(row["candidates"]) for row in output_rows) / len(output_rows),
        "bm25": bm25_summary,
        "output": str(output_path),
    }


def _annotation_request(row: dict[str, Any]) -> JudgeRequest:
    payload = {
        "subject": row.get("subject"),
        "query": row.get("query"),
        "reference_answer": row.get("reference_answer"),
        "task_evidence": row.get("image_evidence"),
        "candidates": [
            {"chunk_id": candidate["chunk_id"], "text": candidate["text"]}
            for candidate in row.get("candidates", [])
        ],
    }
    return JudgeRequest(
        system_prompt=ANNOTATION_SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        image_urls=(),
    )


def parse_annotation(text: str, allowed_ids: set[str]) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.removesuffix("```").strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("annotation response is not a JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("annotation response is not an object")

    def validate_ids(field: str) -> list[str]:
        raw = value.get(field)
        if not isinstance(raw, list):
            raise ValueError(f"{field} must be an array")
        ids = list(dict.fromkeys(str(item) for item in raw))
        unknown = set(ids) - allowed_ids
        if unknown:
            raise ValueError(f"{field} contains unknown IDs: {sorted(unknown)[:3]}")
        return ids

    relevant = validate_ids("relevant_chunk_ids")
    uncertain = [
        chunk_id
        for chunk_id in validate_ids("uncertain_chunk_ids")
        if chunk_id not in set(relevant)
    ]
    confidence = float(value.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rationale = str(value.get("rationale") or "").strip()
    if not rationale:
        raise ValueError("rationale must not be empty")
    return {
        "relevant_chunk_ids": relevant,
        "uncertain_chunk_ids": uncertain,
        "confidence": confidence,
        "rationale": rationale,
    }


def annotate_candidate_pool(
    pool_path: Path,
    output_path: Path,
    backend: AnnotationBackend,
    *,
    limit: int = 0,
) -> dict[str, Any]:
    """Annotate a candidate pool with atomic resume after every task."""

    pool = _read_jsonl(pool_path)
    existing_rows = _read_jsonl(output_path) if output_path.exists() else []
    annotations = {str(row.get("task_id")): row for row in existing_rows}
    completed_before = sum(row.get("annotation_status") == "complete" for row in annotations.values())
    processed = 0
    errors = 0
    for row in pool:
        task_id = str(row.get("task_id") or "")
        if annotations.get(task_id, {}).get("annotation_status") == "complete":
            continue
        if limit and processed >= limit:
            break
        try:
            response = backend.complete(_annotation_request(row))
            allowed_ids = {str(candidate["chunk_id"]) for candidate in row.get("candidates", [])}
            annotation = parse_annotation(response.text, allowed_ids)
            annotations[task_id] = {
                "query_id": task_id,
                "task_id": task_id,
                "query": row.get("query"),
                "subject": row.get("subject"),
                "grade": row.get("grade"),
                "retrieval_subject": row.get("retrieval_subject"),
                "retrieval_grade": row.get("retrieval_grade"),
                "relevant_chunk_ids": annotation["relevant_chunk_ids"],
                "uncertain_chunk_ids": annotation["uncertain_chunk_ids"],
                "annotation_status": "complete",
                "annotation_method": "llm_assisted_dense200_bm25_100_v1",
                "annotation_confidence": annotation["confidence"],
                "annotation_rationale": annotation["rationale"],
                "annotation_model": response.model,
                "annotation_usage": response.metadata.get("usage"),
                "pool": row.get("pool"),
            }
        except Exception as exc:
            errors += 1
            annotations[task_id] = {
                "query_id": task_id,
                "task_id": task_id,
                "query": row.get("query"),
                "subject": row.get("subject"),
                "grade": row.get("grade"),
                "retrieval_subject": row.get("retrieval_subject"),
                "retrieval_grade": row.get("retrieval_grade"),
                "relevant_chunk_ids": [],
                "annotation_status": "error",
                "annotation_error": f"{type(exc).__name__}: {exc}",
            }
        processed += 1
        ordered = [annotations[str(item["task_id"])] for item in pool if str(item["task_id"]) in annotations]
        _write_jsonl_atomic(output_path, ordered)
    completed = sum(row.get("annotation_status") == "complete" for row in annotations.values())
    return {
        "pool_tasks": len(pool),
        "completed_before": completed_before,
        "processed": processed,
        "completed": completed,
        "errors": errors,
        "output": str(output_path),
    }
