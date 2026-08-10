#!/usr/bin/env python3
"""Gold-blind integrity audit for the frozen Visual Sketchpad V2 run.

The audit reconstructs every emitted row from the frozen queue, fallback, and
the row's saved model payload.  It never reads benchmark references or scores.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import run_maxim_agent_ideas as core
import run_maxim_visual_sketchpad_v2 as sketchpad


EXPECTED_ROWS = 274
EXPECTED_ENDPOINT = "http://127.0.0.1:18021/v1"
EXPECTED_PROFILE_SHA256 = (
    "231885ec0968e6f8338bb287980f1c46942aacb69d07840db75a50badce22ca8"
)
EXPECTED_RUNNER_SHA256 = (
    "8fa649cebaf20650b8f205759ce2bbb620832070959615e274caa08f704dadaf"
)


class AuditError(ValueError):
    """Raised when a frozen-run invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AuditError(f"{label}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def index_rows(
    rows: list[dict[str, Any]], *, label: str
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order = [str(row.get("task_id") or "") for row in rows]
    if not all(order):
        raise AuditError(f"{label}: empty task_id")
    if len(set(order)) != len(order):
        raise AuditError(f"{label}: duplicate task_id")
    return order, dict(zip(order, rows, strict=True))


def expected_fallback_row(
    fallback: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, Any]:
    return sketchpad._fallback_result(
        fallback,
        plan=treatment.get("plan"),
        sketch_metadata=treatment.get("sketch_metadata"),
        solve=treatment.get("candidate_evidence"),
        calls=list(treatment.get("call_traces") or []),
        failures=list(treatment.get("gate_failures") or []),
        error=treatment.get("candidate_error"),
    )


def expected_candidate_row(
    task: dict[str, Any], row: dict[str, Any], treatment: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        solve = json.loads(str(row.get("raw_response") or ""))
    except json.JSONDecodeError as exc:
        raise AuditError(f"{task['task_id']}: invalid saved candidate raw_response") from exc
    if not isinstance(solve, dict):
        raise AuditError(f"{task['task_id']}: candidate raw_response is not an object")
    expected = sketchpad._candidate_result(
        task,
        plan=dict(treatment.get("plan") or {}),
        sketch_metadata=dict(treatment.get("sketch_metadata") or {}),
        solve=solve,
        calls=list(treatment.get("call_traces") or []),
    )
    return expected, solve


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.profile) != EXPECTED_PROFILE_SHA256:
        raise AuditError("preregistered profile SHA mismatch")
    if sha256_file(Path(sketchpad.__file__).resolve()) != EXPECTED_RUNNER_SHA256:
        raise AuditError("frozen runner SHA mismatch")
    if sha256_file(args.queue) != sketchpad.FROZEN_PUBLIC_QUEUE_SHA256:
        raise AuditError("public queue SHA mismatch")
    if sha256_file(args.fallback) != sketchpad.FROZEN_FALLBACK_SHA256:
        raise AuditError("frozen fallback SHA mismatch")

    queue_rows = load_jsonl(args.queue, label="queue")
    fallback_rows = load_jsonl(args.fallback, label="fallback")
    solver_rows = load_jsonl(args.solver, label="solver")
    if not (
        len(queue_rows) == len(fallback_rows) == len(solver_rows) == EXPECTED_ROWS
    ):
        raise AuditError(
            "row count mismatch: "
            f"queue={len(queue_rows)} fallback={len(fallback_rows)} "
            f"solver={len(solver_rows)}"
        )

    queue_order, queue_index = index_rows(queue_rows, label="queue")
    fallback_order, fallback_index = index_rows(fallback_rows, label="fallback")
    solver_order, _ = index_rows(solver_rows, label="solver")
    if fallback_order != queue_order:
        raise AuditError("fallback order differs from frozen public queue")
    if solver_order != queue_order:
        raise AuditError("solver order differs from frozen public queue")

    route_counts: collections.Counter[str] = collections.Counter()
    kind_counts: collections.Counter[str] = collections.Counter()
    failure_counts: collections.Counter[str] = collections.Counter()
    error_counts: collections.Counter[str] = collections.Counter()
    changed_task_ids: list[str] = []
    planner_calls = 0
    solver_calls = 0

    for task_id, raw_task, row in zip(queue_order, queue_rows, solver_rows, strict=True):
        sketchpad.assert_gold_blind(raw_task, location=f"queue:{task_id}")
        sketchpad.assert_gold_blind(row, location=f"solver:{task_id}")
        task = core._task_view(raw_task)
        fallback = fallback_index[task_id]
        generation = row.get("generation") or {}
        treatment = generation.get("visual_sketchpad_v2") or {}
        if row.get("condition") != sketchpad.CONDITION:
            raise AuditError(f"{task_id}: condition mismatch")
        if row.get("prompt_version") != sketchpad.CONDITION:
            raise AuditError(f"{task_id}: prompt_version mismatch")
        if row.get("model") != core.MODEL:
            raise AuditError(f"{task_id}: model mismatch")
        if generation.get("gold_access") is not False:
            raise AuditError(f"{task_id}: generation.gold_access is not false")
        if row.get("error") is not None:
            raise AuditError(f"{task_id}: top-level error is not null")
        if not str(row.get("final_answer") or "").strip():
            raise AuditError(f"{task_id}: empty final answer")
        if treatment.get("schema_version") != sketchpad.SCHEMA_VERSION:
            raise AuditError(f"{task_id}: treatment schema mismatch")

        route = str(treatment.get("selected_source") or "")
        if route not in {"frozen_active_crop_v2", "visual_sketchpad_candidate"}:
            raise AuditError(f"{task_id}: invalid selected_source {route!r}")
        route_counts[route] += 1
        plan = treatment.get("plan")
        if isinstance(plan, dict):
            kind_counts[str(plan.get("sketch_kind") or "missing")] += 1
        else:
            kind_counts["missing"] += 1
        failures = list(treatment.get("gate_failures") or [])
        failure_counts.update(str(value) for value in failures)
        candidate_error = treatment.get("candidate_error")
        if candidate_error is not None:
            error_counts[str(candidate_error).split(":", 1)[0]] += 1

        traces = list(treatment.get("call_traces") or [])
        if len(traces) not in {0, 1, 2}:
            raise AuditError(f"{task_id}: invalid model-call count")
        for trace in traces:
            if trace.get("endpoint") != EXPECTED_ENDPOINT:
                raise AuditError(f"{task_id}: endpoint binding mismatch")
            if int(trace.get("attempt") or 0) not in {1, 2}:
                raise AuditError(f"{task_id}: invalid attempt number")
        planner_calls += min(1, len(traces))
        solver_calls += max(0, len(traces) - 1)

        if route == "frozen_active_crop_v2":
            expected = expected_fallback_row(fallback, treatment)
            if candidate_error is not None:
                expected_failures = ["candidate_error"]
            elif not isinstance(plan, dict):
                raise AuditError(f"{task_id}: fallback row lacks a plan without error")
            elif treatment.get("candidate_evidence") is None:
                expected_failures = []
                if plan.get("sketch_kind") not in sketchpad.ELIGIBLE_SKETCH_KINDS:
                    expected_failures.append("ineligible_sketch_kind")
                if float(plan.get("confidence") or 0.0) < 0.85:
                    expected_failures.append("planner_confidence_below_0.85")
            else:
                expected_failures = sketchpad.gate_failures(
                    task, plan, treatment["candidate_evidence"], fallback
                )
            if failures != expected_failures:
                raise AuditError(
                    f"{task_id}: saved fallback gate failures are not policy output"
                )
        else:
            if failures or candidate_error is not None:
                raise AuditError(f"{task_id}: selected candidate has a failed gate")
            expected, solve = expected_candidate_row(task, row, treatment)
            if sketchpad.gate_failures(task, treatment["plan"], solve, fallback):
                raise AuditError(f"{task_id}: selected candidate does not pass frozen gates")
            changed_task_ids.append(task_id)

        if canonical(row) != canonical(expected):
            raise AuditError(f"{task_id}: row is not exact frozen-policy output")

    return {
        "schema_version": "maxim-visual-sketchpad-v2-integrity-audit-v1",
        "status": "PASS",
        "gold_or_score_access": False,
        "rows": EXPECTED_ROWS,
        "task_order": "PASS",
        "unique_task_ids": EXPECTED_ROWS,
        "bindings": {
            "preregistered_profile": {
                "path": str(args.profile),
                "sha256": sha256_file(args.profile),
            },
            "queue": {
                "path": str(args.queue),
                "sha256": sha256_file(args.queue),
            },
            "fallback": {
                "path": str(args.fallback),
                "sha256": sha256_file(args.fallback),
            },
            "solver": {
                "path": str(args.solver),
                "sha256": sha256_file(args.solver),
            },
            "runner": {
                "path": str(Path(sketchpad.__file__).resolve()),
                "sha256": sha256_file(Path(sketchpad.__file__).resolve()),
            },
        },
        "policy_reconstruction": "PASS_274_OF_274",
        "route_counts": dict(sorted(route_counts.items())),
        "changed_task_ids": changed_task_ids,
        "sketch_kind_counts": dict(sorted(kind_counts.items())),
        "gate_failure_counts": dict(sorted(failure_counts.items())),
        "candidate_error_type_counts": dict(sorted(error_counts.items())),
        "saved_successful_call_traces": {
            "planner": planner_calls,
            "solver": solver_calls,
        },
        "model": core.MODEL,
        "endpoint": EXPECTED_ENDPOINT,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
