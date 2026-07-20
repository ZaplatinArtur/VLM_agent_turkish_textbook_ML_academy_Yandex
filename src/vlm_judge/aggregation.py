from __future__ import annotations

import itertools
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from .schema import JudgeVerdict


def _score_detail(
    record: dict[str, Any],
    *,
    strategy: str = "hybrid",
) -> tuple[bool | None, float | None, str]:
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("agent_failure"):
        return False, 0.0, "agent_failure"

    deterministic = record.get("deterministic")
    deterministic_score: tuple[bool, float, str] | None = None
    if isinstance(deterministic, dict) and deterministic.get("applicable"):
        matched = deterministic.get("matched")
        if matched is not None:
            deterministic_score = bool(matched), 4.0 if matched else 0.0, "deterministic"

    verdict = record.get("verdict")
    judge_score: tuple[bool | None, float | None, str] | None = None
    if isinstance(verdict, dict):
        try:
            parsed = JudgeVerdict.from_dict(verdict)
        except (KeyError, TypeError, ValueError):
            judge_score = None, None, "evaluation_failure"
        else:
            if parsed.label == "unjudgeable":
                judge_score = None, None, "unjudgeable_reference"
            else:
                judge_score = parsed.strict_correct, float(parsed.score), "judge"

    if strategy == "deterministic":
        return deterministic_score or (None, None, "not_applicable")
    if strategy == "judge":
        return judge_score or (None, None, "evaluation_failure")
    if strategy != "hybrid":
        raise ValueError(f"unknown scoring strategy: {strategy}")
    return deterministic_score or judge_score or (None, None, "evaluation_failure")


def _score(record: dict[str, Any]) -> tuple[bool | None, float | None, bool]:
    correct, score, status = _score_detail(record)
    return correct, score, status in {"unjudgeable_reference", "evaluation_failure", "not_applicable"}


def _bootstrap_mean_ci(
    values: list[float],
    *,
    iterations: int = 2000,
    seed: int = 20260713,
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.mean(sample))
    estimates.sort()
    return [
        estimates[int(0.025 * (iterations - 1))],
        estimates[int(0.975 * (iterations - 1))],
    ]


def _summarize_group(
    records: list[dict[str, Any]],
    *,
    strategy: str = "hybrid",
) -> dict[str, Any]:
    correct_values: list[float] = []
    scores: list[float] = []
    statuses: Counter[str] = Counter()
    for record in records:
        correct, score, status = _score_detail(record, strategy=strategy)
        statuses[status] += 1
        if correct is not None:
            correct_values.append(float(correct))
        if score is not None:
            scores.append(score)
    return {
        "records": len(records),
        "scored": len(scores),
        "unjudgeable": statuses["unjudgeable_reference"],
        "evaluation_failures": statuses["evaluation_failure"],
        "not_applicable": statuses["not_applicable"],
        "agent_failures_counted_incorrect": statuses["agent_failure"],
        "score_sources": dict(statuses),
        "strict_accuracy": statistics.mean(correct_values) if correct_values else None,
        "strict_accuracy_ci95": _bootstrap_mean_ci(correct_values),
        "mean_score_0_4": statistics.mean(scores) if scores else None,
        "mean_score_ci95": _bootstrap_mean_ci(scores),
    }


def aggregate_results(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(records)
    by_setup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_subject_setup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_answer_type_setup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_task_setup: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_task_setup_units: list[str] = []
    for record in materialized:
        setup = str(record.get("setup") or "unknown")
        subject = str(record.get("subject") or "unknown")
        answer_type = str(record.get("answer_type") or "unknown")
        task_id = str(record.get("task_id") or "")
        by_setup[setup].append(record)
        by_subject_setup[(subject, setup)].append(record)
        by_answer_type_setup[(answer_type, setup)].append(record)
        if task_id:
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            agent_run = metadata.get("agent_run")
            pairing_id = metadata.get("pairing_id") or metadata.get("replicate_id")
            if not pairing_id and isinstance(agent_run, dict):
                pairing_id = agent_run.get("pairing_id") or agent_run.get("replicate_id")
            # Setup-specific run IDs must not break task pairing. Repeated runs
            # should provide a shared pairing_id/replicate_id across setups.
            unit_id = f"{task_id}::{pairing_id}" if pairing_id else task_id
            if setup in by_task_setup[unit_id]:
                duplicate_task_setup_units.append(f"{unit_id}::{setup}")
            by_task_setup[unit_id][setup] = record

    setups = sorted(by_setup)
    pairwise: dict[str, Any] = {}
    for setup_a, setup_b in itertools.combinations(setups, 2):
        deltas: list[float] = []
        wins_a = wins_b = ties = 0
        for task_records in by_task_setup.values():
            if setup_a not in task_records or setup_b not in task_records:
                continue
            _, score_a, bad_a = _score(task_records[setup_a])
            _, score_b, bad_b = _score(task_records[setup_b])
            if bad_a or bad_b or score_a is None or score_b is None:
                continue
            delta = score_b - score_a
            deltas.append(delta)
            if delta > 0:
                wins_b += 1
            elif delta < 0:
                wins_a += 1
            else:
                ties += 1
        total = len(deltas)
        pairwise[f"{setup_b}_vs_{setup_a}"] = {
            "paired_tasks": total,
            f"{setup_b}_wins": wins_b,
            f"{setup_a}_wins": wins_a,
            "ties": ties,
            f"{setup_b}_win_rate_all": wins_b / total if total else None,
            "mean_score_delta": statistics.mean(deltas) if deltas else None,
            "mean_score_delta_ci95": _bootstrap_mean_ci(deltas),
        }

    by_subject: dict[str, dict[str, Any]] = defaultdict(dict)
    for (subject, setup), group in sorted(by_subject_setup.items()):
        by_subject[subject][setup] = _summarize_group(group)

    by_answer_type: dict[str, dict[str, Any]] = defaultdict(dict)
    for (answer_type, setup), group in sorted(by_answer_type_setup.items()):
        by_answer_type[answer_type][setup] = _summarize_group(group)

    all_units = set(by_task_setup)
    coverage = {
        setup: {
            "task_run_units": len({unit for unit, values in by_task_setup.items() if setup in values}),
            "coverage_against_union": (
                sum(setup in values for values in by_task_setup.values()) / len(all_units)
                if all_units
                else None
            ),
            "missing_task_run_units": sorted(unit for unit, values in by_task_setup.items() if setup not in values),
        }
        for setup in setups
    }
    complete_grid = sum(all(setup in values for setup in setups) for values in by_task_setup.values())
    hybrid_by_setup = {setup: _summarize_group(group) for setup, group in sorted(by_setup.items())}

    return {
        "records": len(materialized),
        "by_setup": hybrid_by_setup,
        "metric_views": {
            "hybrid_primary": hybrid_by_setup,
            "deterministic_only": {
                setup: _summarize_group(group, strategy="deterministic")
                for setup, group in sorted(by_setup.items())
            },
            "judge_only": {
                setup: _summarize_group(group, strategy="judge")
                for setup, group in sorted(by_setup.items())
            },
        },
        "by_subject": dict(by_subject),
        "by_answer_type": dict(by_answer_type),
        "paired_comparisons": pairwise,
        "coverage": {
            "task_run_units_union": len(all_units),
            "complete_setup_grid_units": complete_grid,
            "by_setup": coverage,
            "duplicate_task_setup_units": sorted(set(duplicate_task_setup_units)),
        },
    }
