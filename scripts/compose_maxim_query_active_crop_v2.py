"""Compose full274 from active-crop results with a conservative frozen gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    import prepare_maxim_query_active_crop_v2 as prepare
    import run_maxim_agent_ideas as core
except ModuleNotFoundError:  # Imported as scripts.compose_maxim_query_active_crop_v2.
    from scripts import prepare_maxim_query_active_crop_v2 as prepare
    from scripts import run_maxim_agent_ideas as core


CONDITION = prepare.CONDITION + "_failclosed_no_tools_v1"
MIN_VERIFIER_CONFIDENCE = 0.90
MIN_LOCATOR_CONFIDENCE = 0.80
MIN_REGION_CONFIDENCE = 0.70
CHOICE_RE = re.compile(r"^[A-E]$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_sha256(row: dict[str, Any]) -> str:
    return prepare.canonical_sha256(row)


def _index(rows: list[dict[str, Any]], label: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    index: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row_number, row in enumerate(rows, 1):
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError(f"{label} row {row_number}: missing task_id")
        if task_id in index:
            raise ValueError(f"{label}: duplicate task_id {task_id}")
        index[task_id] = row
        order.append(task_id)
    return index, order


def canonical_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _nonempty_list(value: Any, minimum: int) -> bool:
    return (
        isinstance(value, list)
        and len([item for item in value if str(item or "").strip()]) >= minimum
    )


def _valid_answer_type(answer: str, answer_type: Any) -> bool:
    value = str(answer or "").strip()
    if not value:
        return False
    if str(answer_type or "").casefold() == "choice":
        return bool(CHOICE_RE.fullmatch(value))
    return len(value) <= 120


def gate_decision(
    result: dict[str, Any], fallback: dict[str, Any], answer_type: Any
) -> tuple[bool, list[str]]:
    """Apply every preregistered conjunct; return decision and failed clauses."""
    failures: list[str] = []
    candidate = str(result.get("final_answer") or "").strip()
    baseline = str(fallback.get("final_answer") or "").strip()
    generation = result.get("generation")
    if not isinstance(generation, dict) or generation.get("gold_access") is not False:
        failures.append("missing_gold_blind_generation_provenance")
        generation = {}
    evidence = generation.get("selection_evidence")
    if not isinstance(evidence, dict):
        evidence = {}
        failures.append("missing_selection_evidence")
    locator = generation.get("locator")
    if not isinstance(locator, dict):
        locator = {}
        failures.append("missing_locator")
    if result.get("error"):
        failures.append("runner_error")
    if not candidate:
        failures.append("empty_candidate")
    if canonical_answer(candidate) == canonical_answer(baseline):
        failures.append("no_canonical_disagreement")
    if evidence.get("baseline_supported") is not False:
        failures.append("baseline_not_explicitly_refuted")
    try:
        verifier_confidence = float(evidence.get("confidence"))
    except (TypeError, ValueError):
        verifier_confidence = -1.0
    if verifier_confidence < MIN_VERIFIER_CONFIDENCE:
        failures.append("verifier_confidence_below_0_90")
    try:
        locator_confidence = float(locator.get("overall_confidence"))
    except (TypeError, ValueError):
        locator_confidence = -1.0
    if locator_confidence < MIN_LOCATOR_CONFIDENCE:
        failures.append("locator_confidence_below_0_80")
    regions = locator.get("used_regions")
    if not isinstance(regions, list) or not 1 <= len(regions) <= 2:
        failures.append("used_region_count_not_1_or_2")
    else:
        for region in regions:
            try:
                confidence = float(region.get("confidence"))
            except (AttributeError, TypeError, ValueError):
                confidence = -1.0
            if confidence < MIN_REGION_CONFIDENCE:
                failures.append("region_confidence_below_0_70")
                break
    for key in (
        "all_required_evidence_visible",
        "original_crop_consistent",
        "answer_format_verified",
    ):
        if evidence.get(key) is not True:
            failures.append(f"{key}_not_true")
    if not _nonempty_list(evidence.get("visible_facts"), 2):
        failures.append("fewer_than_two_visible_facts")
    if not _nonempty_list(evidence.get("verification_checks"), 2):
        failures.append("fewer_than_two_verification_checks")
    if not _valid_answer_type(candidate, answer_type):
        failures.append("candidate_answer_type_invalid")
    return not failures, failures


def compose(
    *, benchmark_rows: list[dict[str, Any]], fallback_rows: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, order = _index(benchmark_rows, "benchmark")
    fallback, fallback_order = _index(fallback_rows, "fallback")
    queue, queue_order = _index(queue_rows, "queue")
    results, result_order = _index(result_rows, "results")
    benchmark_ids = set(order)
    if set(fallback_order) != benchmark_ids:
        raise ValueError("fallback task-id set differs from benchmark")
    if not set(queue_order).issubset(benchmark_ids):
        raise ValueError("queue contains task IDs outside benchmark")
    if set(result_order) != set(queue_order):
        raise ValueError("result task-id set differs from frozen queue")

    output: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    fallback_gate_ids: list[str] = []
    nonroute_ids: list[str] = []
    failure_counts: dict[str, int] = {}
    decisions: list[dict[str, Any]] = []
    for task_id in order:
        baseline = fallback[task_id]
        if not str(baseline.get("final_answer") or "").strip():
            raise ValueError(f"fallback {task_id}: empty final_answer")
        if task_id not in queue:
            chosen = baseline
            selected = False
            failures = ["not_in_frozen_blind_route"]
            nonroute_ids.append(task_id)
            candidate_sha = None
        else:
            candidate = results[task_id]
            selected, failures = gate_decision(
                candidate, baseline, benchmark[task_id].get("answer_type")
            )
            chosen = candidate if selected else baseline
            candidate_sha = row_sha256(candidate)
            if selected:
                selected_ids.append(task_id)
            else:
                fallback_gate_ids.append(task_id)
            for failure in set(failures):
                failure_counts[failure] = failure_counts.get(failure, 0) + 1
        row = copy.deepcopy(chosen)
        generation = copy.deepcopy(row.get("generation") or {})
        generation["gold_access"] = False
        generation["active_crop_failclosed_composition"] = {
            "schema_version": "maxim-query-active-crop-composition-provenance-v2",
            "selected_source": "active_crop" if selected else "frozen_no_tools",
            "gate_passed": selected,
            "failed_clauses": failures,
            "fallback_row_sha256": row_sha256(baseline),
            "candidate_row_sha256": candidate_sha,
            "queue_request_sha256": queue.get(task_id, {}).get("request_sha256"),
        }
        row["condition"] = CONDITION
        row["prompt_version"] = CONDITION
        row["generation"] = generation
        row["error"] = None
        output.append(row)
        decisions.append({
            "task_id": task_id,
            "in_route": task_id in queue,
            "selected_source": "active_crop" if selected else "frozen_no_tools",
            "failed_clauses": failures,
        })
    stats = {
        "rows": len(output),
        "routed_rows": len(queue),
        "nonroute_fallback_rows": len(nonroute_ids),
        "gate_fallback_rows": len(fallback_gate_ids),
        "active_crop_selected_rows": len(selected_ids),
        "active_crop_selected_task_ids": selected_ids,
        "failure_clause_counts": dict(sorted(failure_counts.items())),
    }
    return output, {"stats": stats, "decisions": decisions}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--skip-frozen-sha-check", action="store_true")
    args = parser.parse_args(argv)
    if not args.skip_frozen_sha_check:
        if sha256_file(args.benchmark) != prepare.FROZEN_BENCHMARK_SHA256:
            raise SystemExit("benchmark SHA mismatch")
        if sha256_file(args.fallback) != prepare.FROZEN_NO_TOOLS_SHA256:
            raise SystemExit("fallback SHA mismatch")
    if sha256_file(args.queue) != args.queue_sha256:
        raise SystemExit("queue SHA mismatch")
    if sha256_file(args.profile) != args.profile_sha256:
        raise SystemExit("profile SHA mismatch")
    output, details = compose(
        benchmark_rows=core._load_jsonl(args.benchmark),
        fallback_rows=core._load_jsonl(args.fallback),
        queue_rows=core._load_jsonl(args.queue),
        result_rows=core._load_jsonl(args.results),
    )
    _write_jsonl(args.output, output)
    _write_json(args.decisions, {"schema_version": "maxim-query-active-crop-decisions-v2", **details})
    manifest = {
        "schema_version": "maxim-query-active-crop-composition-manifest-v2",
        "condition": CONDITION,
        "scoring_performed": False,
        "gold_access": False,
        "sources": {
            "benchmark": {"path": str(args.benchmark.resolve()), "sha256": sha256_file(args.benchmark)},
            "fallback": {"path": str(args.fallback.resolve()), "sha256": sha256_file(args.fallback)},
            "queue": {"path": str(args.queue.resolve()), "sha256": sha256_file(args.queue)},
            "profile": {"path": str(args.profile.resolve()), "sha256": sha256_file(args.profile)},
            "results": {"path": str(args.results.resolve()), "sha256": sha256_file(args.results)},
        },
        "code_sha256": sha256_file(Path(__file__)),
        "stats": details["stats"],
        "output": {"path": str(args.output.resolve()), "rows": len(output), "sha256": sha256_file(args.output)},
        "decisions": {"path": str(args.decisions.resolve()), "sha256": sha256_file(args.decisions)},
    }
    _write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
