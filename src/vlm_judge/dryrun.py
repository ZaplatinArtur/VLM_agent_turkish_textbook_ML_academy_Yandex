from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .aggregation import aggregate_results
from .arena import build_pairwise_records
from .ingest import read_records
from .metrics import deterministic_match
from .normalization import normalize_multiple_choice, parse_numeric
from .reporting import render_experiment_report
from .validation import validate_experiment_runs


SETUP_CORRECTNESS = {
    "no_tools": 0.56,
    "web_search": 0.68,
    "textbook_retrieval": 0.80,
}


def _unit_interval(seed: str, setup: str, task_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{setup}:{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def _wrong_answer(task: dict[str, Any]) -> str:
    reference = str(task.get("reference_answer") or "")
    answer_type = str(task.get("answer_type") or "unknown")
    if answer_type == "multiple_choice":
        correct = normalize_multiple_choice(reference)
        wrong = next((choice for choice in "ABCDE" if choice != correct), "A")
        return f"Final answer: {wrong}"
    if answer_type == "numeric":
        number = parse_numeric(reference)
        if number is not None:
            shifted: Fraction = number + 1
            if shifted.denominator == 1:
                return f"Final answer: {shifted.numerator}"
            return f"Final answer: {shifted.numerator}/{shifted.denominator}"
    return "[SYNTHETIC_INCORRECT_RESPONSE]"


def _candidate_answer(task: dict[str, Any], correct: bool) -> str:
    if not correct:
        return _wrong_answer(task)
    reference = str(task.get("reference_answer") or "")
    if task.get("answer_type") in {"multiple_choice", "numeric"}:
        return f"Final answer: {reference}"
    return reference


def _synthetic_verdict(correct: bool) -> dict[str, Any]:
    return {
        "label": "fully_correct" if correct else "incorrect",
        "score": 4 if correct else 0,
        "strict_correct": correct,
        "final_answer_correct": correct,
        "reasoning_correct": None,
        "complete": correct,
        "confidence": 1.0,
        "error_types": [] if correct else ["wrong_final_answer"],
        "rationale": "Synthetic oracle label for pipeline testing only.",
        "reference_quality_issue": False,
    }


def run_synthetic_experiment(
    benchmark_path: Path,
    output_dir: Path,
    *,
    seed: str = "synthetic-experiment-v1",
    limit: int | None = None,
) -> dict[str, Any]:
    tasks = [record for record in read_records(benchmark_path) if record.get("reference_answer")]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("benchmark has no text-reference tasks for a synthetic dry run")
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / "benchmark_subset.jsonl"
    with subset_path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    run_paths: dict[str, Path] = {}
    all_candidates: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []

    for setup, target_rate in SETUP_CORRECTNESS.items():
        run_path = output_dir / f"{setup}.jsonl"
        run_paths[setup] = run_path
        with run_path.open("w", encoding="utf-8", newline="\n") as handle:
            for task in tasks:
                task_id = str(task["task_id"])
                expected_correct = _unit_interval(seed, setup, task_id) < target_rate
                metadata = dict(task.get("metadata") or {})
                metadata.update(
                    {
                        "run_id": f"{seed}-{setup}",
                        "agent_model": "synthetic-generator",
                        "agent_prompt_version": "synthetic-v1",
                        "seed": seed,
                        "latency_ms": 100 + int(_unit_interval(seed + "-latency", setup, task_id) * 900),
                        "input_tokens": 100,
                        "output_tokens": 12,
                        "synthetic_dry_run": True,
                        "synthetic_expected_correct": expected_correct,
                    }
                )
                if setup == "textbook_retrieval":
                    metadata.update(
                        {
                            "retrieval_config_hash": "synthetic-bm25-config",
                            "retrieved_chunk_ids": [f"synthetic-chunk-{task_id}"],
                            "retrieval_calls": 1,
                        }
                    )
                candidate = {
                    **task,
                    "setup": setup,
                    "candidate_answer": _candidate_answer(task, expected_correct),
                    "metadata": metadata,
                }
                handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                all_candidates.append(candidate)
                exact = deterministic_match(
                    candidate.get("reference_answer"),
                    candidate["candidate_answer"],
                    str(candidate.get("answer_type") or "unknown"),
                    acceptable_answers=candidate.get("acceptable_answers") or [],
                )
                scored.append(
                    {
                        "task_id": task_id,
                        "setup": setup,
                        "subject": candidate.get("subject"),
                        "grade": candidate.get("grade"),
                        "answer_type": candidate.get("answer_type"),
                        "metadata": metadata,
                        "deterministic": {
                            "applicable": exact.applicable,
                            "matched": exact.matched,
                            "method": exact.method,
                            "normalized_reference": exact.normalized_reference,
                            "normalized_candidate": exact.normalized_candidate,
                        },
                        "verdict": _synthetic_verdict(expected_correct),
                        "judge": {
                            "backend": "synthetic-oracle",
                            "model": "not-a-model",
                            "attempts": 1,
                            "cache_hit": False,
                            "error": None,
                            "valid_for_quality_claims": False,
                        },
                    }
                )

    scored_path = output_dir / "scored_records.jsonl"
    with scored_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in scored:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    validation = validate_experiment_runs(subset_path, run_paths)
    validation_path = output_dir / "validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = aggregate_results(scored)
    summary.update(
        {
            "synthetic_smoke_test": True,
            "valid_for_quality_claims": False,
            "seed": seed,
            "target_correctness": SETUP_CORRECTNESS,
            "benchmark": str(benchmark_path),
            "benchmark_subset": str(subset_path),
            "tasks_with_text_reference": len(tasks),
            "validation_ready": validation["ready_for_experiment"],
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pairs = build_pairwise_records(
        all_candidates,
        "no_tools",
        "textbook_retrieval",
        seed=seed,
        mirrored=True,
    )
    pairs_path = output_dir / "arena_pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    report_path = output_dir / "report.html"
    render_experiment_report(summary, report_path)
    return {
        "synthetic_smoke_test": True,
        "valid_for_quality_claims": False,
        "tasks": len(tasks),
        "candidate_records": len(all_candidates),
        "scored_records": len(scored),
        "arena_pairs": len(pairs),
        "validation_ready": validation["ready_for_experiment"],
        "outputs": {
            "runs": {setup: str(path) for setup, path in run_paths.items()},
            "benchmark_subset": str(subset_path),
            "scored": str(scored_path),
            "validation": str(validation_path),
            "summary": str(summary_path),
            "arena_pairs": str(pairs_path),
            "html_report": str(report_path),
        },
    }
