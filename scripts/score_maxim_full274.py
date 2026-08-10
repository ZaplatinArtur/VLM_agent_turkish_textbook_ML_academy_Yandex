#!/usr/bin/env python3
"""Score one completed 274-row solver run without exposing gold during generation.

This is deliberately a post-generation, standard-library-only scorer.  It uses
the frozen page-RAG judge file to define the immutable 177/97 scoring partition
and the frozen comparison verdicts:

* 177 reference-text rows use the historical deterministic matching semantics;
* 97 reference-image rows consume a separately produced strict judge JSONL;
* solver failures remain in the fixed denominator and are always incorrect.

Neither reference answers nor candidate answers are copied into the reports.
The script also rejects solver outputs that contain obvious top-level gold
fields or explicitly declare generation.gold_access != false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "maxim-full274-score-v1"
DEFAULT_EXPECTED_ROWS = 274
DEFAULT_EXPECTED_DETERMINISTIC = 177
DEFAULT_EXPECTED_IMAGE_JUDGE = 97

# These byte-level pins identify the frozen benchmark and page-RAG comparison.
FROZEN_BENCHMARK_SHA256 = (
    "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
)
FROZEN_BASELINE_JUDGE_SHA256 = (
    "59dcc93454b29dfc65b0a9b1243a177d472b6c0a13cbe46fb5c98079810a73f4"
)

FORBIDDEN_SOLVER_GOLD_FIELDS = {
    "reference_answer",
    "reference_solution",
    "gold_answer",
    "gold_solution",
}


class ScoringError(ValueError):
    """Raised when an input violates the fixed evaluation protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ScoringError(f"{label}: file does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoringError(
                    f"{label}: invalid JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ScoringError(
                    f"{label}: line {line_number} must be a JSON object"
                )
            records.append(record)
    if not records:
        raise ScoringError(f"{label}: no JSONL records")
    return records


def _index_by_task(
    records: Iterable[dict[str, Any]], label: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for position, record in enumerate(records, start=1):
        task_id = str(record.get("task_id") or "").strip()
        if not task_id:
            raise ScoringError(f"{label}: record {position} has no task_id")
        if task_id in indexed:
            raise ScoringError(f"{label}: duplicate task_id {task_id}")
        indexed[task_id] = record
        order.append(task_id)
    return indexed, order


def _assert_same_ids(
    actual: set[str], expected: set[str], label: str
) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise ScoringError(
        f"{label}: task-ID mismatch; missing={missing[:20]}"
        f"{'...' if len(missing) > 20 else ''}, extra={extra[:20]}"
        f"{'...' if len(extra) > 20 else ''}"
    )


def _check_sha(path: Path, actual: str, expected: str | None, label: str) -> None:
    if expected is None:
        return
    normalized = expected.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ScoringError(f"{label}: expected SHA256 must be 64 hex characters")
    if actual != normalized:
        raise ScoringError(
            f"{label}: SHA256 mismatch for {path}; expected={normalized}, actual={actual}"
        )


def _baseline_source(record: dict[str, Any], task_id: str) -> str:
    metadata = record.get("metadata")
    metadata_source = metadata.get("score_source") if isinstance(metadata, dict) else None
    raw_source = record.get("score_source") or metadata_source
    source = str(raw_source or "").strip().casefold()
    if source in {"exact", "deterministic", "reference_text"}:
        return "deterministic"
    if source in {"vlm_image_judge", "image_judge", "reference_image"}:
        return "image_judge"
    raise ScoringError(
        f"baseline judge: task {task_id} has unsupported score_source={raw_source!r}"
    )


def _strict_correct(record: dict[str, Any], label: str, task_id: str) -> bool:
    # The first shape is the project's canonical judge JSONL.  The additional
    # two shapes make the scorer usable with a minimal post-processed judge file.
    containers: list[dict[str, Any]] = []
    verdict = record.get("verdict")
    if isinstance(verdict, dict):
        containers.append(verdict)
    for key in ("parsed_verdict", "judge_verdict"):
        candidate = record.get(key)
        if isinstance(candidate, dict):
            containers.append(candidate)
    containers.append(record)
    for container in containers:
        value = container.get("strict_correct")
        if isinstance(value, bool):
            return value
    raise ScoringError(
        f"{label}: task {task_id} has no boolean verdict.strict_correct"
    )


def _judge_error(record: dict[str, Any]) -> Any:
    if record.get("error"):
        return record.get("error")
    judge = record.get("judge")
    if isinstance(judge, dict) and judge.get("error"):
        return judge.get("error")
    return None


# Historical deterministic semantics from src/mla_baseline/eval.py.  Keeping
# them local makes this script standalone and protects the frozen metric from
# future project-import changes.
def _norm_choice(value: str) -> str | None:
    match = re.search(r"[A-Ea-e]", value.strip())
    return match.group(0).upper() if match else None


def _norm_number(value: str) -> float | None:
    value = value.strip().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    try:
        return float(match.group(0)) if match else None
    except ValueError:
        return None


def _norm_text(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value.casefold(), flags=re.UNICODE).strip()


def deterministic_match(
    candidate: str, reference: str, answer_type: str
) -> tuple[bool | None, str]:
    """Return the historical deterministic verdict and its matching method."""
    if reference.startswith("http") or answer_type == "free_form":
        return None, "needs_judge"
    if answer_type == "choice":
        return _norm_choice(candidate) == _norm_choice(reference), "choice_first_a_e"
    if answer_type == "numeric":
        candidate_number = _norm_number(candidate)
        reference_number = _norm_number(reference)
        matched = (
            candidate_number is not None
            and reference_number is not None
            and abs(candidate_number - reference_number) < 1e-6
        )
        return matched, "numeric_abs_lt_1e-6"
    return _norm_text(candidate) == _norm_text(reference), "normalized_short_text"


def _missing_answer(record: dict[str, Any]) -> bool:
    answer = record.get("final_answer")
    return answer is None or not str(answer).strip()


def _solver_error(record: dict[str, Any]) -> str | None:
    value = record.get("error")
    return str(value) if value else None


def _validate_no_gold(solver: dict[str, dict[str, Any]]) -> None:
    for task_id, record in solver.items():
        present = sorted(FORBIDDEN_SOLVER_GOLD_FIELDS.intersection(record))
        if present:
            raise ScoringError(
                f"solver results: task {task_id} contains forbidden gold fields {present}"
            )
        generation = record.get("generation")
        if isinstance(generation, dict) and "gold_access" in generation:
            if generation["gold_access"] is not False:
                raise ScoringError(
                    f"solver results: task {task_id} declares "
                    f"generation.gold_access={generation['gold_access']!r}; expected false"
                )


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ScoringError(f"{label}: boolean is not a valid integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ScoringError(f"{label}: expected a non-negative integer, got {value!r}") from exc
    if parsed < 0 or str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise ScoringError(f"{label}: expected a non-negative integer, got {value!r}")
    return parsed


def _optional_nonnegative_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ScoringError(f"{label}: boolean is not a valid number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoringError(f"{label}: expected a non-negative number, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ScoringError(f"{label}: expected a finite non-negative number, got {value!r}")
    return parsed


def _usage_for_task(record: dict[str, Any], task_id: str) -> dict[str, Any]:
    usage = record.get("usage")
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise ScoringError(f"solver results: task {task_id} usage must be an object")
    input_tokens = _optional_nonnegative_int(
        usage.get("input_tokens"), f"solver results: task {task_id} usage.input_tokens"
    )
    output_tokens = _optional_nonnegative_int(
        usage.get("output_tokens"), f"solver results: task {task_id} usage.output_tokens"
    )
    latency_s = _optional_nonnegative_float(
        usage.get("latency_s"), f"solver results: task {task_id} usage.latency_s"
    )
    generation = record.get("generation")
    call_count_raw = generation.get("call_count") if isinstance(generation, dict) else None
    call_count = _optional_nonnegative_int(
        call_count_raw, f"solver results: task {task_id} generation.call_count"
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s": latency_s,
        "call_count": call_count,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _summary(outcomes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    denominator = len(outcomes)
    new_correct = sum(item["new_correct"] is True for item in outcomes)
    baseline_correct = sum(item["baseline_correct"] is True for item in outcomes)
    fixed = sum(item["transition"] == "fixed" for item in outcomes)
    regressed = sum(item["transition"] == "regressed" for item in outcomes)
    return {
        "n": denominator,
        "new_correct": new_correct,
        "new_accuracy": _ratio(new_correct, denominator),
        "baseline_correct": baseline_correct,
        "baseline_accuracy": _ratio(baseline_correct, denominator),
        "delta_correct": new_correct - baseline_correct,
        "delta_pp": round(100.0 * (new_correct - baseline_correct) / denominator, 3)
        if denominator
        else None,
        "fixed": fixed,
        "regressed": regressed,
        "both_correct": sum(item["transition"] == "both_correct" for item in outcomes),
        "both_wrong": sum(item["transition"] == "both_wrong" for item in outcomes),
        "deterministic_n": sum(
            item["score_source"] == "deterministic" for item in outcomes
        ),
        "image_judge_n": sum(item["score_source"] == "image_judge" for item in outcomes),
        "solver_errors": sum(bool(item["solver_error"]) for item in outcomes),
        "missing_answers": sum(item["missing_final_answer"] for item in outcomes),
    }


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _operational_summary(outcomes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = [
        item["usage"]["input_tokens"]
        for item in outcomes
        if item["usage"]["input_tokens"] is not None
    ]
    output_tokens = [
        item["usage"]["output_tokens"]
        for item in outcomes
        if item["usage"]["output_tokens"] is not None
    ]
    latencies = [
        item["usage"]["latency_s"]
        for item in outcomes
        if item["usage"]["latency_s"] is not None
    ]
    call_counts = [
        item["usage"]["call_count"]
        for item in outcomes
        if item["usage"]["call_count"] is not None
    ]
    error_ids = [item["task_id"] for item in outcomes if item["solver_error"]]
    missing_ids = [
        item["task_id"] for item in outcomes if item["missing_final_answer"]
    ]
    failure_ids = [
        item["task_id"]
        for item in outcomes
        if item["solver_error"] or item["missing_final_answer"]
    ]
    judge_override_ids = [
        item["task_id"]
        for item in outcomes
        if item.get("judge_correct_before_failure_override") is True
        and (item["solver_error"] or item["missing_final_answer"])
    ]
    return {
        "errors": {
            "solver_error_count": len(error_ids),
            "solver_error_task_ids": error_ids,
            "missing_final_answer_count": len(missing_ids),
            "missing_final_answer_task_ids": missing_ids,
            "generation_failure_union_count": len(failure_ids),
            "generation_failure_union_task_ids": failure_ids,
            "forced_answer_count": sum(item["forced_answer"] for item in outcomes),
            "forced_answer_task_ids": [
                item["task_id"] for item in outcomes if item["forced_answer"]
            ],
            "judge_true_overridden_by_generation_failure_count": len(judge_override_ids),
            "judge_true_overridden_by_generation_failure_task_ids": judge_override_ids,
        },
        "tokens": {
            "input_token_rows": len(input_tokens),
            "input_tokens_total": sum(input_tokens),
            "input_tokens_mean_per_reported_row": round(
                statistics.fmean(input_tokens), 3
            )
            if input_tokens
            else None,
            "output_token_rows": len(output_tokens),
            "output_tokens_total": sum(output_tokens),
            "output_tokens_mean_per_reported_row": round(
                statistics.fmean(output_tokens), 3
            )
            if output_tokens
            else None,
            "combined_tokens_total": sum(input_tokens) + sum(output_tokens),
        },
        "latency": {
            "reported_rows": len(latencies),
            "latency_s_total": round(sum(latencies), 3),
            "latency_s_mean": round(statistics.fmean(latencies), 3)
            if latencies
            else None,
            "latency_s_median": round(statistics.median(latencies), 3)
            if latencies
            else None,
            "latency_s_p95_nearest_rank": round(
                _nearest_rank_percentile(latencies, 0.95) or 0.0, 3
            )
            if latencies
            else None,
            "latency_s_max": round(max(latencies), 3) if latencies else None,
        },
        "model_calls": {
            "reported_rows": len(call_counts),
            "call_count_total": sum(call_counts),
            "call_count_mean_per_reported_row": round(
                statistics.fmean(call_counts), 3
            )
            if call_counts
            else None,
        },
    }


def _transition(baseline_correct: bool, new_correct: bool) -> str:
    if baseline_correct and new_correct:
        return "both_correct"
    if not baseline_correct and new_correct:
        return "fixed"
    if baseline_correct and not new_correct:
        return "regressed"
    return "both_wrong"


def build_report(
    *,
    benchmark_path: Path,
    solver_results_path: Path,
    image_judge_path: Path,
    baseline_judge_path: Path,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
    expected_deterministic: int = DEFAULT_EXPECTED_DETERMINISTIC,
    expected_image_judge: int = DEFAULT_EXPECTED_IMAGE_JUDGE,
    expected_benchmark_sha256: str | None = FROZEN_BENCHMARK_SHA256,
    expected_baseline_judge_sha256: str | None = FROZEN_BASELINE_JUDGE_SHA256,
    label: str | None = None,
) -> dict[str, Any]:
    if expected_rows <= 0:
        raise ScoringError("expected_rows must be positive")
    if expected_deterministic < 0 or expected_image_judge < 0:
        raise ScoringError("expected source counts must be non-negative")
    if expected_deterministic + expected_image_judge != expected_rows:
        raise ScoringError(
            "expected_deterministic + expected_image_judge must equal expected_rows"
        )

    paths = {
        "benchmark": benchmark_path.resolve(),
        "solver_results": solver_results_path.resolve(),
        "image_judge": image_judge_path.resolve(),
        "frozen_page_rag_judge": baseline_judge_path.resolve(),
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    _check_sha(
        paths["benchmark"],
        hashes["benchmark"],
        expected_benchmark_sha256,
        "benchmark",
    )
    _check_sha(
        paths["frozen_page_rag_judge"],
        hashes["frozen_page_rag_judge"],
        expected_baseline_judge_sha256,
        "frozen page-RAG judge",
    )

    benchmark_records = _read_jsonl(paths["benchmark"], "benchmark")
    solver_records = _read_jsonl(paths["solver_results"], "solver results")
    image_judge_records = _read_jsonl(paths["image_judge"], "image judge")
    baseline_records = _read_jsonl(
        paths["frozen_page_rag_judge"], "frozen page-RAG judge"
    )

    benchmark, task_order = _index_by_task(benchmark_records, "benchmark")
    solver, _ = _index_by_task(solver_records, "solver results")
    image_judge, _ = _index_by_task(image_judge_records, "image judge")
    baseline, _ = _index_by_task(baseline_records, "frozen page-RAG judge")

    if len(task_order) != expected_rows:
        raise ScoringError(
            f"benchmark: expected {expected_rows} rows, found {len(task_order)}"
        )
    benchmark_ids = set(task_order)
    _assert_same_ids(set(solver), benchmark_ids, "solver results")
    _assert_same_ids(set(baseline), benchmark_ids, "frozen page-RAG judge")
    _validate_no_gold(solver)

    partition: dict[str, list[str]] = {"deterministic": [], "image_judge": []}
    baseline_scores: dict[str, bool] = {}
    for task_id in task_order:
        baseline_row = baseline[task_id]
        source = _baseline_source(baseline_row, task_id)
        partition[source].append(task_id)
        if _judge_error(baseline_row):
            raise ScoringError(
                f"frozen page-RAG judge: task {task_id} has judge error "
                f"{_judge_error(baseline_row)!r}"
            )
        baseline_scores[task_id] = _strict_correct(
            baseline_row, "frozen page-RAG judge", task_id
        )

    if len(partition["deterministic"]) != expected_deterministic:
        raise ScoringError(
            "frozen partition mismatch: expected "
            f"{expected_deterministic} deterministic rows, found "
            f"{len(partition['deterministic'])}"
        )
    if len(partition["image_judge"]) != expected_image_judge:
        raise ScoringError(
            "frozen partition mismatch: expected "
            f"{expected_image_judge} image-judge rows, found "
            f"{len(partition['image_judge'])}"
        )

    deterministic_ids = set(partition["deterministic"])
    image_ids = set(partition["image_judge"])
    judge_ids = set(image_judge)
    # A dedicated 97-row file is preferred.  A complete 274-row hybrid judge
    # file is also accepted so the frozen baseline can replay itself exactly.
    if judge_ids not in (image_ids, benchmark_ids):
        missing = sorted(image_ids - judge_ids)
        unexpected = sorted(judge_ids - image_ids)
        raise ScoringError(
            "image judge: IDs must be exactly the 97 image rows or the complete "
            f"benchmark; missing_image_ids={missing[:20]}, unexpected_ids={unexpected[:20]}"
        )

    image_scores: dict[str, bool] = {}
    for task_id in partition["image_judge"]:
        judge_row = image_judge[task_id]
        error = _judge_error(judge_row)
        if error:
            raise ScoringError(f"image judge: task {task_id} has judge error {error!r}")
        image_scores[task_id] = _strict_correct(judge_row, "image judge", task_id)

    # Validate that every frozen deterministic row is actually deterministically
    # scoreable under the historical matcher before examining candidate answers.
    for task_id in partition["deterministic"]:
        task = benchmark[task_id]
        reference = str(task.get("reference_answer") or "")
        answer_type = str(task.get("answer_type") or "short_text")
        if not reference.strip():
            raise ScoringError(f"benchmark: deterministic task {task_id} has empty reference")
        comparable, _ = deterministic_match("", reference, answer_type)
        if comparable is None:
            raise ScoringError(
                f"frozen partition marks task {task_id} deterministic, but its "
                f"answer_type/reference requires a judge"
            )

    outcomes: list[dict[str, Any]] = []
    for task_id in task_order:
        task = benchmark[task_id]
        result = solver[task_id]
        source = "deterministic" if task_id in deterministic_ids else "image_judge"
        error = _solver_error(result)
        missing = _missing_answer(result)
        failed = bool(error) or missing
        score_method: str
        judge_before_override: bool | None = None
        if source == "deterministic":
            matched, score_method = deterministic_match(
                str(result.get("final_answer") or ""),
                str(task.get("reference_answer") or ""),
                str(task.get("answer_type") or "short_text"),
            )
            if matched is None:
                raise ScoringError(
                    f"benchmark: deterministic task {task_id} unexpectedly requires judge"
                )
            new_correct = bool(matched) and not failed
        else:
            score_method = "strict_image_judge"
            judge_before_override = image_scores[task_id]
            new_correct = bool(judge_before_override) and not failed

        baseline_correct = baseline_scores[task_id]
        usage = _usage_for_task(result, task_id)
        outcomes.append(
            {
                "task_id": task_id,
                "subject": str(task.get("subject") or "<missing>"),
                "answer_type": str(task.get("answer_type") or "short_text"),
                "score_source": source,
                "score_method": score_method,
                "new_correct": new_correct,
                "baseline_correct": baseline_correct,
                "transition": _transition(baseline_correct, new_correct),
                "solver_error": error,
                "missing_final_answer": missing,
                "forced_answer": bool(result.get("forced_answer")),
                "judge_correct_before_failure_override": judge_before_override,
                "usage": usage,
            }
        )

    by_subject_outcomes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_outcomes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        by_subject_outcomes[outcome["subject"]].append(outcome)
        by_source_outcomes[outcome["score_source"]].append(outcome)

    overall = _summary(outcomes)
    fixed_ids = [item["task_id"] for item in outcomes if item["transition"] == "fixed"]
    regressed_ids = [
        item["task_id"] for item in outcomes if item["transition"] == "regressed"
    ]
    conditions = sorted(
        {
            str(record.get("condition") or record.get("prompt_version") or "").strip()
            for record in solver.values()
            if str(record.get("condition") or record.get("prompt_version") or "").strip()
        }
    )
    models = sorted(
        {
            str(record.get("model") or "").strip()
            for record in solver.values()
            if str(record.get("model") or "").strip()
        }
    )
    report_label = label or (conditions[0] if len(conditions) == 1 else "solver_run")
    scorer_path = Path(__file__).resolve()
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": report_label,
        "conditions": conditions,
        "models": models,
        "protocol": {
            "fixed_denominator": expected_rows,
            "deterministic_rows": expected_deterministic,
            "image_judge_rows": expected_image_judge,
            "partition_authority": "frozen_page_rag_judge.score_source",
            "deterministic_matcher": (
                "historical mla_baseline.eval semantics: first A-E; first numeric "
                "token with abs(delta)<1e-6; casefold/punctuation-normalized short text"
            ),
            "image_metric": "boolean verdict.strict_correct from separate judge JSONL",
            "solver_failure_policy": (
                "non-empty error or missing final_answer is incorrect and remains in denominator"
            ),
            "gold_isolation": (
                "benchmark references are read only by this post-generation scorer; "
                "top-level gold fields in solver JSONL are rejected"
            ),
        },
        "provenance": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in (
                "benchmark",
                "solver_results",
                "image_judge",
                "frozen_page_rag_judge",
            )
        }
        | {
            "scorer": {
                "path": str(scorer_path),
                "sha256": sha256_file(scorer_path),
            }
        },
        "guardrails": {
            "benchmark_rows_verified": len(benchmark),
            "solver_rows_verified": len(solver),
            "baseline_rows_verified": len(baseline),
            "image_judge_rows_supplied": len(image_judge),
            "image_judge_input_shape": "image_only"
            if judge_ids == image_ids
            else "full_hybrid",
            "task_id_sets_match": True,
            "duplicate_task_ids": 0,
            "forbidden_gold_fields_in_solver": 0,
            "explicit_nonfalse_generation_gold_access": 0,
            "frozen_sha_pins_checked": bool(
                expected_benchmark_sha256 and expected_baseline_judge_sha256
            ),
        },
        "overall": overall,
        "by_source": {
            source: _summary(by_source_outcomes[source])
            for source in ("deterministic", "image_judge")
        },
        "by_subject": {
            subject: _summary(by_subject_outcomes[subject])
            for subject in sorted(by_subject_outcomes, key=str.casefold)
        },
        "changes_vs_frozen_page_rag": {
            "fixed_count": len(fixed_ids),
            "fixed_task_ids": fixed_ids,
            "regressed_count": len(regressed_ids),
            "regressed_task_ids": regressed_ids,
            "net_correct_change": len(fixed_ids) - len(regressed_ids),
        },
        "operational": _operational_summary(outcomes),
        "task_outcomes": outcomes,
    }
    if overall["n"] != expected_rows:
        raise AssertionError("internal error: fixed denominator was not preserved")
    return report


def _format_pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def _format_number(value: Any) -> str:
    return "—" if value is None else str(value)


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any], json_sha256: str | None = None) -> str:
    overall = report["overall"]
    lines = [
        f"# Full-274 score: {report['label']}",
        "",
        (
            f"New: **{overall['new_correct']}/{overall['n']} "
            f"({_format_pct(overall['new_accuracy'])})**; frozen page-RAG: "
            f"**{overall['baseline_correct']}/{overall['n']} "
            f"({_format_pct(overall['baseline_accuracy'])})**; "
            f"delta **{overall['delta_correct']:+d} correct / {overall['delta_pp']:+.3f} pp**."
        ),
        "",
    ]
    math_summary = report["by_subject"].get("Math")
    if math_summary:
        lines.extend(
            [
                (
                    f"Math: **{math_summary['new_correct']}/{math_summary['n']} "
                    f"({_format_pct(math_summary['new_accuracy'])})** vs frozen "
                    f"**{math_summary['baseline_correct']}/{math_summary['n']} "
                    f"({_format_pct(math_summary['baseline_accuracy'])})**; "
                    f"delta **{math_summary['delta_correct']:+d} / "
                    f"{math_summary['delta_pp']:+.3f} pp**."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Score-source split",
            "",
            "| Source | n | New correct | Frozen correct | Delta pp | Fixed | Regressed |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source in ("deterministic", "image_judge"):
        summary = report["by_source"][source]
        lines.append(
            f"| {source} | {summary['n']} | {summary['new_correct']} | "
            f"{summary['baseline_correct']} | {summary['delta_pp']:+.3f} | "
            f"{summary['fixed']} | {summary['regressed']} |"
        )

    lines.extend(
        [
            "",
            "## By subject",
            "",
            "| Subject | n | New | New accuracy | Frozen | Frozen accuracy | Delta pp | Fixed | Regressed | Errors |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for subject, summary in report["by_subject"].items():
        lines.append(
            f"| {_escape_table(subject)} | {summary['n']} | {summary['new_correct']} | "
            f"{_format_pct(summary['new_accuracy'])} | {summary['baseline_correct']} | "
            f"{_format_pct(summary['baseline_accuracy'])} | {summary['delta_pp']:+.3f} | "
            f"{summary['fixed']} | {summary['regressed']} | {summary['solver_errors']} |"
        )

    operational = report["operational"]
    errors = operational["errors"]
    tokens = operational["tokens"]
    latency = operational["latency"]
    calls = operational["model_calls"]
    lines.extend(
        [
            "",
            "## Operational",
            "",
            f"- Solver errors: {errors['solver_error_count']}; missing answers: "
            f"{errors['missing_final_answer_count']}; generation-failure union: "
            f"{errors['generation_failure_union_count']}.",
            f"- Tokens: input {tokens['input_tokens_total']}, output "
            f"{tokens['output_tokens_total']}, combined {tokens['combined_tokens_total']} "
            f"(reported rows: {tokens['input_token_rows']}/{overall['n']} input, "
            f"{tokens['output_token_rows']}/{overall['n']} output).",
            f"- Latency: total {_format_number(latency['latency_s_total'])} s; "
            f"mean {_format_number(latency['latency_s_mean'])} s; median "
            f"{_format_number(latency['latency_s_median'])} s; p95 "
            f"{_format_number(latency['latency_s_p95_nearest_rank'])} s; "
            f"reported rows {latency['reported_rows']}/{overall['n']}.",
            f"- Model calls: {_format_number(calls['call_count_total'])}; "
            f"reported rows {calls['reported_rows']}/{overall['n']}.",
            "",
            "## Paired changes vs frozen page-RAG",
            "",
            f"- Fixed ({report['changes_vs_frozen_page_rag']['fixed_count']}): "
            + (", ".join(report["changes_vs_frozen_page_rag"]["fixed_task_ids"]) or "none"),
            f"- Regressed ({report['changes_vs_frozen_page_rag']['regressed_count']}): "
            + (", ".join(report["changes_vs_frozen_page_rag"]["regressed_task_ids"]) or "none"),
            "",
            "## Provenance",
            "",
            "| Input | SHA256 | Path |",
            "|---|---|---|",
        ]
    )
    for name, item in report["provenance"].items():
        lines.append(
            f"| {_escape_table(name)} | `{item['sha256']}` | "
            f"`{_escape_table(item['path'])}` |"
        )
    if json_sha256:
        lines.extend(["", f"Report JSON SHA256: `{json_sha256}`."])
    lines.extend(
        [
            "",
            "The benchmark reference is read only after generation. Candidate and reference "
            "answers are intentionally omitted from this report.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(text)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_reports(
    report: dict[str, Any],
    *,
    out_json: Path,
    out_md: Path,
    out_sha256: Path | None = None,
) -> dict[str, str]:
    output_paths = [out_json.resolve(), out_md.resolve()]
    if out_sha256 is not None:
        output_paths.append(out_sha256.resolve())
    if len(set(output_paths)) != len(output_paths):
        raise ScoringError("output paths must be distinct")

    json_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(out_json, json_text)
    json_sha = sha256_file(out_json.resolve())
    markdown_text = render_markdown(report, json_sha256=json_sha)
    _atomic_write_text(out_md, markdown_text)
    markdown_sha = sha256_file(out_md.resolve())
    output_hashes = {
        "json": json_sha,
        "markdown": markdown_sha,
    }
    if out_sha256 is not None:
        manifest = (
            f"{json_sha}  {out_json.resolve().name}\n"
            f"{markdown_sha}  {out_md.resolve().name}\n"
        )
        _atomic_write_text(out_sha256, manifest)
        output_hashes["sha256_manifest"] = sha256_file(out_sha256.resolve())
    return output_hashes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--solver-results", type=Path, required=True)
    parser.add_argument("--image-judge", type=Path, required=True)
    parser.add_argument("--baseline-judge", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-sha256", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    parser.add_argument(
        "--expected-deterministic", type=int, default=DEFAULT_EXPECTED_DETERMINISTIC
    )
    parser.add_argument(
        "--expected-image-judge", type=int, default=DEFAULT_EXPECTED_IMAGE_JUDGE
    )
    parser.add_argument(
        "--expected-benchmark-sha256", default=FROZEN_BENCHMARK_SHA256
    )
    parser.add_argument(
        "--expected-baseline-judge-sha256", default=FROZEN_BASELINE_JUDGE_SHA256
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_paths = {
        args.benchmark.resolve(),
        args.solver_results.resolve(),
        args.image_judge.resolve(),
        args.baseline_judge.resolve(),
    }
    output_paths = {args.out_json.resolve(), args.out_md.resolve()}
    if args.out_sha256:
        output_paths.add(args.out_sha256.resolve())
    overlap = sorted(str(path) for path in input_paths.intersection(output_paths))
    if overlap:
        print(f"SCORING ERROR: outputs may not overwrite inputs: {overlap}", file=sys.stderr)
        return 2
    try:
        report = build_report(
            benchmark_path=args.benchmark,
            solver_results_path=args.solver_results,
            image_judge_path=args.image_judge,
            baseline_judge_path=args.baseline_judge,
            expected_rows=args.expected_rows,
            expected_deterministic=args.expected_deterministic,
            expected_image_judge=args.expected_image_judge,
            expected_benchmark_sha256=args.expected_benchmark_sha256,
            expected_baseline_judge_sha256=args.expected_baseline_judge_sha256,
            label=args.label,
        )
        output_hashes = write_reports(
            report,
            out_json=args.out_json,
            out_md=args.out_md,
            out_sha256=args.out_sha256,
        )
    except (OSError, ScoringError) as exc:
        print(f"SCORING ERROR: {exc}", file=sys.stderr)
        return 2

    overall = report["overall"]
    math_summary = report["by_subject"].get("Math")
    result = {
        "label": report["label"],
        "new": f"{overall['new_correct']}/{overall['n']}",
        "frozen_page_rag": f"{overall['baseline_correct']}/{overall['n']}",
        "delta_correct": overall["delta_correct"],
        "fixed": report["changes_vs_frozen_page_rag"]["fixed_count"],
        "regressed": report["changes_vs_frozen_page_rag"]["regressed_count"],
        "math_new": (
            f"{math_summary['new_correct']}/{math_summary['n']}" if math_summary else None
        ),
        "math_frozen_page_rag": (
            f"{math_summary['baseline_correct']}/{math_summary['n']}"
            if math_summary
            else None
        ),
        "output_sha256": output_hashes,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
