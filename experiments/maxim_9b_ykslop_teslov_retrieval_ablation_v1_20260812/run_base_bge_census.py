"""Run an outcome-free base-BGE context census over the frozen YKS DEV inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retrieve.rankers.cross_encoder import CrossEncoderRanker

from experiments.maxim_9b_ykslop_teslov_retrieval_ablation_v1_20260812.teslov_retrieval import (
    TheoryChunk,
    bm25_rank,
    subject_filter,
)


PUBLIC = (
    REPO_ROOT
    / "experiments/maxim_9b_ykslop_generic_content_pipeline_v5_20260811"
    / "frozen/benchmark_public_dev.jsonl"
)
CORPUS = (
    REPO_ROOT
    / "experiments/maxim_9b_ykslop_no_overlap_theory_v6_20260811"
    / "frozen/local_textbook_strict_theory_corpus.jsonl"
)
EXPECTED_PUBLIC_SHA256 = "eebfda230a10ef98f07c53b5d7ab55cca24a718c8439074192d4f29156acc47c"
EXPECTED_CORPUS_SHA256 = "cc7d236bdff91eba94022795d5bae0aeb5e32196581e88181033147c7a4edb75"
OUTPUT = HERE / "BASE_BGE_CONTEXT_CENSUS.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _query(row: dict[str, Any]) -> str:
    return row["question"] + "\n" + "\n".join(
        row["choices"][label] for label in "ABCDE"
    )


def _select(
    ranked: Iterable[tuple[TheoryChunk, float]],
    *,
    k: int = 4,
    max_chars: int = 5200,
) -> list[tuple[TheoryChunk, float]]:
    selected: list[tuple[TheoryChunk, float]] = []
    total_chars = 0
    for chunk, score in ranked:
        if total_chars + len(chunk.text) > max_chars and selected:
            continue
        selected.append((chunk, score))
        total_chars += len(chunk.text)
        if len(selected) == k:
            break
    return selected


def run(*, device: str, output: Path) -> dict[str, Any]:
    if _sha256(PUBLIC) != EXPECTED_PUBLIC_SHA256:
        raise RuntimeError("public benchmark descriptor mismatch")
    if _sha256(CORPUS) != EXPECTED_CORPUS_SHA256:
        raise RuntimeError("theory corpus descriptor mismatch")

    public_rows = _read_jsonl(PUBLIC)
    raw_corpus = _read_jsonl(CORPUS)
    if len(public_rows) != 185 or len(raw_corpus) != 75:
        raise RuntimeError("frozen row count mismatch")
    corpus = [
        TheoryChunk(
            row["chunk_id"],
            row["text"],
            row.get("grade"),
            row.get("subject"),
        )
        for row in raw_corpus
    ]

    queries = [_query(row) for row in public_rows]
    eligible = [subject_filter(corpus, row["subject"]) for row in public_rows]
    pairs = [
        [query, chunk.text]
        for query, pool in zip(queries, eligible)
        for chunk in pool
    ]
    started = time.perf_counter()
    reranker = CrossEncoderRanker(
        top_n=100,
        batch_size=32,
        device=device,
        local_files_only=True,
    )
    scores = reranker.score_pairs(pairs)
    elapsed = time.perf_counter() - started

    score_offset = 0
    baseline_nonempty = 0
    treatment_nonempty = 0
    top1_changed = 0
    top2_set_changed = 0
    treatment_counts: dict[str, int] = {}
    for row, query, pool in zip(public_rows, queries, eligible):
        baseline = _select(
            (item.chunk, item.score)
            for item in bm25_rank(query, corpus, subject=row["subject"])
        )
        pool_scores = scores[score_offset: score_offset + len(pool)]
        score_offset += len(pool)
        treatment = _select(
            sorted(
                zip(pool, pool_scores),
                key=lambda item: (-item[1], item[0].chunk_id),
            )
        )
        baseline_ids = [item[0].chunk_id for item in baseline]
        treatment_ids = [item[0].chunk_id for item in treatment]
        baseline_nonempty += bool(baseline_ids)
        treatment_nonempty += bool(treatment_ids)
        top1_changed += bool(baseline_ids and treatment_ids and baseline_ids[0] != treatment_ids[0])
        top2_set_changed += set(baseline_ids[:2]) != set(treatment_ids[:2])
        treatment_counts[str(len(treatment_ids))] = (
            treatment_counts.get(str(len(treatment_ids)), 0) + 1
        )
    if score_offset != len(scores):
        raise RuntimeError("pair score accounting mismatch")

    result = {
        "schema_version": "yks185-base-bge-context-census-v1",
        "status": "OUTCOME_FREE_CONTEXT_CENSUS_NOT_QREL_METRICS",
        "inputs": {
            "public_rows": 185,
            "public_sha256": EXPECTED_PUBLIC_SHA256,
            "theory_chunks": 75,
            "theory_sha256": EXPECTED_CORPUS_SHA256,
        },
        "model": {
            "id": reranker.model_name,
            "revision": reranker.revision,
            "device": device,
            "local_files_only": True,
            "tuned_adapter_used": False,
        },
        "policy": {
            "subject_filter": "exact_frozen_yks_map",
            "grade_filter": "none_public_grade_absent",
            "candidate_pool": "all_same_subject_strict_theory_chunks",
            "final_k": 4,
            "max_context_chars": 5200,
        },
        "aggregates": {
            "query_chunk_pairs": len(pairs),
            "baseline_nonempty_rows": baseline_nonempty,
            "treatment_nonempty_rows": treatment_nonempty,
            "top1_changed_where_both_nonempty": top1_changed,
            "top2_set_changed_rows": top2_set_changed,
            "treatment_hit_count_histogram": dict(sorted(treatment_counts.items())),
            "elapsed_seconds": round(elapsed, 6),
        },
        "claims": {
            "gold_read": False,
            "prior_predictions_or_outcomes_read": False,
            "per_row_results_persisted": False,
            "hit_recall_mrr_ndcg_claimed": False,
            "qa_accuracy_claimed": False,
        },
    }
    data = (
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(output, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(device=args.device, output=args.output)
    print(json.dumps(result["aggregates"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
