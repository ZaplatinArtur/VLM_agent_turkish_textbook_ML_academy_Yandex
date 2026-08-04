#!/usr/bin/env python3
"""Report source-family grouped diagnostics for a completed Evidence OS run.

This is evaluator-only code.  It consumes task outcomes after inference and
must never be imported by ``src/evidence_os/policy.py``.  Source families are
metadata-derived proxies, not authoritative book/edition IDs, so the result is
reported as a grouped diagnostic rather than an untouched-book benchmark.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.source_groups import (  # noqa: E402
    SourceGroupIndex,
    assign_group_folds,
)


SCHEMA = "maxim-evidence-os-source-family-diagnostic-v1"


class EvaluationError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvaluationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_bool(value: Any, source: str) -> bool:
    if type(value) is not bool:
        raise EvaluationError(f"{source}: expected strict JSON boolean")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise EvaluationError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _metric(values: list[bool]) -> dict[str, Any]:
    correct = sum(values)
    return {
        "correct": correct,
        "n": len(values),
        "accuracy": correct / len(values) if values else None,
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of empty values")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_bootstrap(
    family_values: dict[str, list[bool]],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    families = sorted(family_values)
    rng = random.Random(seed)
    micro: list[float] = []
    macro: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(families) for _ in families]
        flattened = [value for family in sampled for value in family_values[family]]
        micro.append(sum(flattened) / len(flattened))
        macro.append(
            statistics.fmean(
                sum(family_values[family]) / len(family_values[family])
                for family in sampled
            )
        )
    return {
        "micro_95_ci": [_percentile(micro, 0.025), _percentile(micro, 0.975)],
        "family_macro_95_ci": [_percentile(macro, 0.025), _percentile(macro, 0.975)],
    }


def evaluate(
    metadata_path: Path,
    score_path: Path,
    *,
    folds: int,
    bootstrap_samples: int,
    seed: int,
    expected_rows: int,
    expected_task_ids: frozenset[str] | None = None,
) -> dict[str, Any]:
    metadata = _load_jsonl(metadata_path)
    metadata_task_ids = [str(row.get("task_id") or "").strip() for row in metadata]
    if any(not task_id for task_id in metadata_task_ids) or len(set(metadata_task_ids)) != len(
        metadata_task_ids
    ):
        raise EvaluationError("metadata contains missing/duplicate task_id")
    source_index = SourceGroupIndex.from_records(metadata)
    score = json.loads(score_path.read_text(encoding="utf-8"))
    outcomes = score.get("task_outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise EvaluationError("score artifact has no task_outcomes")
    if len(outcomes) != expected_rows:
        raise EvaluationError(
            f"score must contain exactly {expected_rows} task outcomes, got {len(outcomes)}"
        )

    task_ids: list[str] = []
    correct: list[bool] = []
    subjects: list[str] = []
    seen: set[str] = set()
    family_values: dict[str, list[bool]] = defaultdict(list)
    for row in outcomes:
        if not isinstance(row, dict):
            raise EvaluationError("task outcome is not an object")
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in seen:
            raise EvaluationError("task outcomes contain missing/duplicate task_id")
        seen.add(task_id)
        value = _require_bool(
            row.get("new_correct"), f"task_outcomes[{task_id}].new_correct"
        )
        family = source_index.family_for(task_id)
        task_ids.append(task_id)
        correct.append(value)
        subjects.append(str(row.get("subject") or "Unknown"))
        family_values[family].append(value)
    frozen_task_ids = expected_task_ids or frozenset(metadata_task_ids)
    if set(task_ids) != set(frozen_task_ids):
        raise EvaluationError("score task set does not exactly match the frozen task set")
    if not set(task_ids) <= set(metadata_task_ids):
        raise EvaluationError("frozen metadata does not cover the complete score task set")

    assignment = assign_group_folds(
        task_ids,
        source_index,
        n_folds=folds,
        seed=f"evidence-os-v1:{seed}",
    )
    family_folds: dict[str, set[int]] = defaultdict(set)
    for task_id in task_ids:
        family_folds[source_index.family_for(task_id)].add(assignment[task_id])
    if any(len(values) != 1 for values in family_folds.values()):
        raise EvaluationError("one source family appears in multiple folds")

    fold_metrics: dict[str, Any] = {}
    for fold in range(folds):
        fold_values = [
            value
            for task_id, value in zip(task_ids, correct, strict=True)
            if assignment[task_id] == fold
        ]
        fold_metrics[str(fold)] = _metric(fold_values)

    math_values = [
        value
        for value, subject in zip(correct, subjects, strict=True)
        if subject.casefold() == "math"
    ]
    non_math_values = [
        value
        for value, subject in zip(correct, subjects, strict=True)
        if subject.casefold() != "math"
    ]
    family_accuracies = [sum(values) / len(values) for values in family_values.values()]
    result = {
        "schema_version": SCHEMA,
        "reporting_status": "source_family_grouped_diagnostic_not_untouched_book_holdout",
        "warning": (
            "Source URL families are proxies and all target rows were previously inspected; "
            "do not call this an independent book-disjoint production estimate."
        ),
        "overall": _metric(correct),
        "math": _metric(math_values),
        "non_math": _metric(non_math_values),
        "source_families": len(family_values),
        "family_macro_accuracy": statistics.fmean(family_accuracies),
        "family_min_accuracy": min(family_accuracies),
        "family_max_accuracy": max(family_accuracies),
        "folds": fold_metrics,
        "fold_family_disjoint": True,
        "bootstrap": _cluster_bootstrap(
            dict(family_values), samples=bootstrap_samples, seed=seed
        ),
        "inputs": {
            "expected_rows": expected_rows,
            "metadata": {
                "path": str(metadata_path),
                "sha256": _sha256_file(metadata_path),
            },
            "score": {
                "path": str(score_path),
                "sha256": _sha256_file(score_path),
            },
        },
    }
    return result


def _profile_expectations(
    profile_path: Path,
    metadata_path: Path,
) -> tuple[int, frozenset[str], dict[str, Any]]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise EvaluationError("frozen profile must be one JSON object")
    if profile.get("schema_version") != "maxim-evidence-os-frozen-profile-v1":
        raise EvaluationError("frozen profile schema mismatch")
    expected_rows = profile.get("expected_rows")
    if type(expected_rows) is not int or expected_rows <= 0:
        raise EvaluationError("frozen profile expected_rows must be a positive integer")
    evaluation = profile.get("evaluation")
    metadata_spec = (
        evaluation.get("source_family_metadata")
        if isinstance(evaluation, dict)
        else None
    )
    if not isinstance(metadata_spec, dict):
        raise EvaluationError("frozen profile has no source-family metadata binding")
    expected_sha = metadata_spec.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise EvaluationError("frozen metadata SHA-256 binding is invalid")
    configured_path = metadata_spec.get("path")
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise EvaluationError("frozen metadata path binding is invalid")
    try:
        profile_path.resolve().relative_to(REPO_ROOT)
    except ValueError as exc:
        raise EvaluationError("frozen profile must be inside the repository") from exc
    profile_repo_root = REPO_ROOT
    expected_path = (profile_repo_root / configured_path).resolve()
    if expected_path != metadata_path.resolve():
        raise EvaluationError("metadata path differs from frozen profile binding")
    actual_sha = _sha256_file(metadata_path)
    if actual_sha != expected_sha:
        raise EvaluationError(
            f"metadata SHA-256 differs from frozen profile: expected={expected_sha}, actual={actual_sha}"
        )
    public_spec = profile.get("public_tasks")
    if not isinstance(public_spec, dict):
        raise EvaluationError("frozen profile has no public task binding")
    public_path_value = public_spec.get("path")
    public_sha = public_spec.get("sha256")
    if not isinstance(public_path_value, str) or not public_path_value.strip():
        raise EvaluationError("frozen public task path binding is invalid")
    if not isinstance(public_sha, str) or len(public_sha) != 64:
        raise EvaluationError("frozen public task SHA-256 binding is invalid")
    public_path = (profile_repo_root / public_path_value).resolve()
    actual_public_sha = _sha256_file(public_path)
    if actual_public_sha != public_sha:
        raise EvaluationError(
            "public task SHA-256 differs from frozen profile: "
            f"expected={public_sha}, actual={actual_public_sha}"
        )
    public_rows = _load_jsonl(public_path)
    public_task_ids = [str(row.get("task_id") or "").strip() for row in public_rows]
    if (
        len(public_task_ids) != expected_rows
        or any(not task_id for task_id in public_task_ids)
        or len(set(public_task_ids)) != expected_rows
    ):
        raise EvaluationError("frozen public task artifact is incomplete or duplicated")
    audit = {
        "frozen_profile": {
            "path": str(profile_path.resolve()),
            "sha256": _sha256_file(profile_path),
        },
        "public_tasks": {
            "path": str(public_path),
            "sha256": actual_public_sha,
        },
    }
    return expected_rows, frozenset(public_task_ids), audit


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    overall = result["overall"]
    lines = [
        "# Evidence OS v1 — source-family diagnostic",
        "",
        f"- Micro: **{overall['correct']}/{overall['n']} = {overall['accuracy']:.6f}**",
        f"- Family-macro: **{result['family_macro_accuracy']:.6f}**",
        f"- Source families: **{result['source_families']}**",
        f"- Math: **{result['math']['correct']}/{result['math']['n']} = {result['math']['accuracy']:.6f}**",
        f"- Non-Math: **{result['non_math']['correct']}/{result['non_math']['n']} = {result['non_math']['accuracy']:.6f}**",
        "",
        f"> {result['warning']}",
        "",
        "| Fold | Correct | N | Accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for fold, metric in result["folds"].items():
        lines.append(
            f"| {fold} | {metric['correct']} | {metric['n']} | {metric['accuracy']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    try:
        expected_rows, expected_task_ids, profile_audit = _profile_expectations(
            args.profile_json.resolve(), args.metadata.resolve()
        )
        result = evaluate(
            args.metadata.resolve(),
            args.score.resolve(),
            folds=args.folds,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            expected_rows=expected_rows,
            expected_task_ids=expected_task_ids,
        )
        result["inputs"].update(profile_audit)
    except (EvaluationError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.out_md, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
