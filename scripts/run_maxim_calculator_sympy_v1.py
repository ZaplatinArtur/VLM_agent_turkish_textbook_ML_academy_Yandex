"""Audit a frozen candidate with a gold-blind calculator/SymPy treatment.

The first model call sees only the task and must derive an answer plus a bounded
calculator program.  The frozen candidate is revealed only to a second audit
call, and only when the independent answer disagrees.  Every failure is
fail-closed: the frozen candidate remains the final output.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import run_maxim_agent_ideas as core
except ModuleNotFoundError:  # Imported as scripts.run_maxim_calculator_sympy_v1.
    from scripts import run_maxim_agent_ideas as core

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import mla_baseline.calculator_sympy as calculator_module
import vlm_judge.normalization as normalization_module
from mla_baseline.calculator_sympy import (
    GateDecision,
    ProgramResult,
    decide_calculator_switch,
    execute_program,
    sympy_available,
)


CONDITION = "maxim_calculator_sympy_v1"
FROZEN_BENCHMARK_SHA256 = (
    "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
)
FROZEN_BASELINE_RESULTS_SHA256 = (
    "62bc952c3802308bc0fbf8d8dc1f82ec523a3ab1e3264bae87a5f8828021d75d"
)
SYSTEM_PROMPT = (
    "You are an answer-blind verifier for Turkish school mathematics. There is "
    "no answer key, reference answer, gold label, or candidate answer in this "
    "stage. Read only the visible question. Never invent a missing number, unit, "
    "option, bound, or symbol. Return exactly the requested JSON schema."
)

CHECK_VALUES = ["pass", "fail", "not_applicable"]
CONFIDENCE_VALUES = ["high", "medium", "low"]

CALCULATOR_DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "applicable": {"type": "boolean"},
        "problem_kind": {
            "type": "string",
            "enum": [
                "arithmetic",
                "percent",
                "fraction",
                "equation_substitution",
                "geometry",
                "table_or_graph",
                "combinatorics",
                "other",
            ],
        },
        "reasoning": {"type": "string", "maxLength": 900},
        "solution_steps": {"type": "string", "maxLength": 1200},
        "verification_program": {"type": "string", "maxLength": 500},
        "predicted_program_value": {"type": "string", "maxLength": 120},
        "independent_answer": {"type": "string", "maxLength": 120},
        "option_mapping": {"type": "string", "maxLength": 500},
        "unit_check": {"type": "string", "enum": CHECK_VALUES},
        "constraint_check": {"type": "string", "enum": CHECK_VALUES},
        "confidence": {"type": "string", "enum": CONFIDENCE_VALUES},
    },
    "required": [
        "applicable",
        "problem_kind",
        "reasoning",
        "solution_steps",
        "verification_program",
        "predicted_program_value",
        "independent_answer",
        "option_mapping",
        "unit_check",
        "constraint_check",
        "confidence",
    ],
    "additionalProperties": False,
}

CALCULATOR_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "switch_recommended": {"type": "boolean"},
        "tool_consistent": {"type": "boolean"},
        "question_consistent": {"type": "boolean"},
        "unit_check": {"type": "string", "enum": CHECK_VALUES},
        "constraint_check": {"type": "string", "enum": CHECK_VALUES},
        "reasoning": {"type": "string", "maxLength": 900},
        "solution_steps": {"type": "string", "maxLength": 1200},
        "final_answer": {"type": "string", "maxLength": 120},
        "confidence": {"type": "string", "enum": CONFIDENCE_VALUES},
    },
    "required": [
        "switch_recommended",
        "tool_consistent",
        "question_consistent",
        "unit_check",
        "constraint_check",
        "reasoning",
        "solution_steps",
        "final_answer",
        "confidence",
    ],
    "additionalProperties": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_sha256() -> dict[str, str]:
    return {
        "runner": _sha256(Path(__file__)),
        "core_transport": _sha256(Path(core.__file__)),
        "calculator_gate": _sha256(Path(calculator_module.__file__)),
        "normalization": _sha256(Path(normalization_module.__file__)),
    }


def _index_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = core._load_jsonl(path)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError(f"row without task_id: {path}")
        if task_id in output:
            raise ValueError(f"duplicate task_id {task_id}: {path}")
        output[task_id] = row
    return output


def gold_blind_task(row: dict[str, Any]) -> dict[str, Any]:
    """Expose the same explicit allow-list used by the frozen experiment."""

    task = core._task_view(row)
    serialized = json.dumps(task, ensure_ascii=False).casefold()
    forbidden = ("reference_answer", "reference_solution", "gold_answer")
    if any(value in serialized for value in forbidden):
        raise ValueError("reference field leaked into calculator task")
    return task


def _is_math(task: dict[str, Any]) -> bool:
    return str(task.get("subject") or "").strip().casefold() in {
        "math",
        "mathematics",
        "matematik",
    }


def _first_messages(
    task: dict[str, Any],
    *,
    image_root: Path,
    image_url_root: str,
) -> list[dict[str, Any]]:
    instruction = (
        "Solve independently. Set applicable=false if the visible data do not "
        "support a meaningful deterministic calculation. Otherwise write one "
        "safe verification_program: either one arithmetic expression or "
        "`result = <expression>`. Allowed constructs are numeric literals, "
        "+ - * / // % **, parentheses, and abs/min/max/sum/product/mean/sqrt/"
        "percent/gcd/lcm/comb/perm. Do not use variables, imports, attributes, "
        "loops, indexing, assignments other than result, or multiple statements. "
        "The program must contain a real operation: never return the proposed "
        "answer as a constant or identity expression. For a numeric response the "
        "program must calculate the final numeric answer. The only exception is "
        "problem_kind=equation_substitution: substitute the independently derived "
        "candidate numerically and calculate a residual that must equal zero. "
        "predicted_program_value must be the exact expected tool output. Also "
        "check units, domain/bounds, integer/non-negative constraints, and answer "
        "format. For choice questions independent_answer must be only A-E and "
        "option_mapping must explain how the calculated value maps to that option."
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"{core._task_prompt(task)}\n\n{instruction}",
        }
    ]
    content.extend(
        core._image_blocks(
            task,
            image_root=image_root,
            image_url_root=image_url_root,
        )
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _audit_messages(
    task: dict[str, Any],
    baseline: dict[str, Any],
    draft: dict[str, Any],
    program: ProgramResult,
    *,
    image_root: Path,
    image_url_root: str,
) -> list[dict[str, Any]]:
    candidate = {
        "final_answer": baseline.get("final_answer"),
        "reasoning": str(baseline.get("reasoning") or "")[:1400],
        "solution_steps": str(baseline.get("solution_steps") or "")[:1800],
    }
    evidence = {
        "independent_draft": draft,
        "actual_tool_result": {
            "ok": program.ok,
            "value": program.value,
            "engine": program.engine,
            "operation_count": program.operation_count,
            "numeric_literal_count": program.numeric_literal_count,
            "nontrivial": program.nontrivial,
        },
        "frozen_candidate": candidate,
    }
    instruction = (
        "This is the only comparison stage. Re-read the visible question, then "
        "audit the independent calculation against the frozen candidate and the "
        "ACTUAL tool value below. Do not trust predicted values over the actual "
        "tool value. Recommend a switch only if the program genuinely represents "
        "the visible givens, the option mapping is exact, the answer respects the "
        "requested unit and all bounds/domain constraints, and the independent "
        "answer is clearly stronger. If anything is missing or ambiguous, keep "
        "the frozen candidate. For choice answers final_answer must be only A-E.\n"
        + json.dumps(evidence, ensure_ascii=False)
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"{core._task_prompt(task)}\n\n{instruction}",
        }
    ]
    content.extend(
        core._image_blocks(
            task,
            image_root=image_root,
            image_url_root=image_url_root,
        )
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative final auditor. No gold or reference answer "
                "exists. Prefer the frozen candidate unless independent visual and "
                "calculator evidence justifies a high-confidence correction."
            ),
        },
        {"role": "user", "content": content},
    ]


def _program_payload(program: ProgramResult | None) -> dict[str, Any] | None:
    if program is None:
        return None
    return {
        "ok": program.ok,
        "value": program.value,
        "error": program.error,
        "engine": program.engine,
        "operation_count": program.operation_count,
        "numeric_literal_count": program.numeric_literal_count,
        "nontrivial": program.nontrivial,
    }


def _compact_call(call: dict[str, Any]) -> dict[str, Any]:
    return core._compact_call(call)


def _sum_usage(
    baseline: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    audit_latency_s: float,
) -> dict[str, Any]:
    usage = baseline.get("usage") if isinstance(baseline.get("usage"), dict) else {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0)
        + sum(int(call.get("input_tokens") or 0) for call in calls),
        "output_tokens": int(usage.get("output_tokens") or 0)
        + sum(int(call.get("output_tokens") or 0) for call in calls),
        "latency_s": round(
            float(usage.get("latency_s") or 0.0) + audit_latency_s,
            3,
        ),
    }


def _compose_row(
    task: dict[str, Any],
    baseline: dict[str, Any],
    *,
    model: str,
    calls: list[dict[str, Any]],
    started: float,
    eligibility: str,
    draft: dict[str, Any] | None,
    program: ProgramResult | None,
    audit: dict[str, Any] | None,
    decision: GateDecision,
    audit_error: str | None,
    source_sha256: dict[str, str],
) -> dict[str, Any]:
    row = dict(baseline)
    baseline_generation = (
        dict(baseline.get("generation"))
        if isinstance(baseline.get("generation"), dict)
        else {}
    )
    baseline_call_count = int(baseline_generation.get("call_count") or 0)
    trace = {
        "strategy": "calculator_sympy_v1",
        "gold_access": False,
        "baseline_reused": True,
        "eligibility": eligibility,
        "audit_triggered": len(calls) > 0,
        "comparison_call_triggered": audit is not None,
        "switch_applied": decision.switch,
        "switch_reasons": list(decision.reasons),
        "draft": draft,
        "program": _program_payload(program),
        "audit": audit,
        "audit_error": audit_error,
        "seeds": [505, 606],
        "max_tokens_per_call": [1800, 1400],
        "call_traces": [_compact_call(call) for call in calls],
        "source_sha256": source_sha256,
        "implementation_sha256": _implementation_sha256(),
    }
    row["condition"] = CONDITION
    row["prompt_version"] = CONDITION
    row["model"] = model
    if decision.switch and audit is not None:
        row["final_answer"] = str(audit.get("final_answer") or "") or None
        row["reasoning"] = str(audit.get("reasoning") or "") or None
        row["solution_steps"] = str(audit.get("solution_steps") or "") or None
        row["raw_response"] = json.dumps(audit, ensure_ascii=False)
        row["forced_answer"] = False
    row["generation"] = {
        **baseline_generation,
        "idea": "calculator_sympy_substitution_units_constraints",
        "source_condition": baseline.get("condition"),
        "source_prompt_version": baseline.get("prompt_version"),
        "gold_access": False,
        "call_count": baseline_call_count + len(calls),
        "calculator_sympy": trace,
    }
    tool_calls = list(baseline.get("tool_calls") or [])
    if program is not None:
        tool_calls.append(
            {
                "tool": "bounded_calculator_sympy",
                "args": {
                    "program": str((draft or {}).get("verification_program") or "")
                },
                "result_preview": program.value,
                "returned_chunk_ids": [],
                "latency_ms": None,
                "error": program.error,
            }
        )
    row["tool_calls"] = tool_calls
    row["usage"] = _sum_usage(
        baseline,
        calls,
        audit_latency_s=(
            time.perf_counter() - started if calls or audit_error else 0.0
        ),
    )
    return row


def run_task(
    task: dict[str, Any],
    baseline: dict[str, Any],
    *,
    pool: core.EndpointPool,
    model: str,
    image_root: Path,
    image_url_root: str,
    source_sha256: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []
    draft: dict[str, Any] | None = None
    program: ProgramResult | None = None
    audit: dict[str, Any] | None = None
    audit_error: str | None = None
    eligibility = "eligible_math"
    decision = GateDecision(False, False, ("not_run",))

    if not _is_math(task):
        eligibility = "non_math_passthrough"
        decision = GateDecision(False, False, ("non_math_passthrough",))
    elif baseline.get("error") or not str(baseline.get("final_answer") or "").strip():
        eligibility = "invalid_frozen_candidate"
        decision = GateDecision(False, False, ("invalid_frozen_candidate",))
    elif not task.get("question_images"):
        eligibility = "missing_question_image"
        decision = GateDecision(False, False, ("missing_question_image",))
    else:
        try:
            first_call = pool.complete(
                messages=_first_messages(
                    task,
                    image_root=image_root,
                    image_url_root=image_url_root,
                ),
                schema_name="calculator_sympy_independent_draft",
                schema=CALCULATOR_DRAFT_SCHEMA,
                max_tokens=1800,
                temperature=0.0,
                seed=505,
            )
            calls.append(first_call)
            draft = first_call["parsed"]
            program = execute_program(str(draft.get("verification_program") or ""))
            decision = decide_calculator_switch(
                baseline_answer=baseline.get("final_answer"),
                answer_type=str(task.get("answer_type") or ""),
                draft=draft,
                program=program,
            )
            if decision.audit_required:
                audit_call = pool.complete(
                    messages=_audit_messages(
                        task,
                        baseline,
                        draft,
                        program,
                        image_root=image_root,
                        image_url_root=image_url_root,
                    ),
                    schema_name="calculator_sympy_candidate_audit",
                    schema=CALCULATOR_AUDIT_SCHEMA,
                    max_tokens=1400,
                    temperature=0.0,
                    seed=606,
                )
                calls.append(audit_call)
                audit = audit_call["parsed"]
                decision = decide_calculator_switch(
                    baseline_answer=baseline.get("final_answer"),
                    answer_type=str(task.get("answer_type") or ""),
                    draft=draft,
                    program=program,
                    audit=audit,
                )
        except Exception as exc:
            audit_error = f"{type(exc).__name__}: {exc}"
            decision = GateDecision(False, False, ("audit_failure",))

    return _compose_row(
        task,
        baseline,
        model=model,
        calls=calls,
        started=started,
        eligibility=eligibility,
        draft=draft,
        program=program,
        audit=audit,
        decision=decision,
        audit_error=audit_error,
        source_sha256=source_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-url-root", default="file:///images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", action="append")
    parser.add_argument("--model", default=core.MODEL)
    parser.add_argument("--task-concurrency", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-audit-errors", action="store_true")
    parser.add_argument("--allow-unfrozen-sources", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.task_concurrency <= 64:
        raise SystemExit("--task-concurrency must be in [1, 64]")
    if args.output.resolve() in {
        args.input.resolve(),
        args.baseline_results.resolve(),
    }:
        raise SystemExit("--output must not overwrite an input")

    raw_tasks = core._load_jsonl(args.input)
    tasks = [gold_blind_task(task) for task in raw_tasks]
    if args.task_id:
        requested = set(args.task_id)
        known = {str(task.get("task_id")) for task in tasks}
        missing = sorted(requested - known)
        if missing:
            raise SystemExit(f"unknown task IDs: {missing}")
        tasks = [task for task in tasks if str(task.get("task_id")) in requested]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    if not all(task_ids) or len(task_ids) != len(set(task_ids)):
        raise SystemExit("blank or duplicate task IDs")

    baseline = _index_rows(args.baseline_results)
    missing_candidates = sorted(set(task_ids) - set(baseline))
    if missing_candidates:
        raise SystemExit(f"missing frozen candidates: {missing_candidates}")
    source_sha256 = {
        "tasks": _sha256(args.input),
        "baseline_results": _sha256(args.baseline_results),
    }
    expected_source_sha256 = {
        "tasks": FROZEN_BENCHMARK_SHA256,
        "baseline_results": FROZEN_BASELINE_RESULTS_SHA256,
    }
    frozen_sources_verified = source_sha256 == expected_source_sha256
    if not frozen_sources_verified and not args.allow_unfrozen_sources:
        raise SystemExit(
            "frozen source SHA256 mismatch: "
            f"expected {expected_source_sha256}, got {source_sha256}; "
            "use --allow-unfrozen-sources only for explicit tests/debugging"
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "condition": CONDITION,
                    "tasks": len(tasks),
                    "math_tasks": sum(_is_math(task) for task in tasks),
                    "non_math_passthrough": sum(not _is_math(task) for task in tasks),
                    "gold_access": False,
                    "sympy_available": sympy_available(),
                    "source_sha256": source_sha256,
                    "implementation_sha256": _implementation_sha256(),
                    "frozen_sources_verified": frozen_sources_verified,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not args.base_url:
        raise SystemExit("at least one --base-url is required unless --dry-run")
    if not sympy_available():
        raise SystemExit("SymPy is required for the calculator_sympy treatment")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume explicitly: {args.output}")

    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        resume_rows = _index_rows(args.output)
        unexpected_resume_ids = sorted(set(resume_rows) - set(task_ids))
        if unexpected_resume_ids:
            raise ValueError(
                "resume output contains rows outside the selected task set; "
                f"refusing to truncate them: {unexpected_resume_ids[:10]}"
            )
        for task_id, row in resume_rows.items():
            if row.get("condition") != CONDITION:
                raise ValueError(f"resume row has wrong condition: {task_id}")
            if row.get("model") != args.model:
                raise ValueError(f"resume row has different model: {task_id}")
            trace = (
                row.get("generation", {}).get("calculator_sympy", {})
                if isinstance(row.get("generation"), dict)
                else {}
            )
            if trace.get("source_sha256") != source_sha256:
                raise ValueError(f"resume provenance mismatch: {task_id}")
            if trace.get("implementation_sha256") != _implementation_sha256():
                raise ValueError(f"resume implementation mismatch: {task_id}")
            if args.retry_audit_errors and trace.get("audit_error"):
                continue
            existing[task_id] = row

    pool = core.EndpointPool(args.base_url, model=args.model, timeout_s=args.timeout_s)
    output_rows = dict(existing)
    pending = [task for task in tasks if str(task["task_id"]) not in existing]
    write_lock = threading.Lock()

    def execute(task: dict[str, Any]) -> dict[str, Any]:
        return run_task(
            task,
            baseline[str(task["task_id"])],
            pool=pool,
            model=args.model,
            image_root=args.image_root,
            image_url_root=args.image_url_root,
            source_sha256=source_sha256,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.task_concurrency
    ) as executor:
        future_to_task = {executor.submit(execute, task): task for task in pending}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                row = future.result()
            except Exception as exc:
                row = _compose_row(
                    task,
                    baseline[str(task["task_id"])],
                    model=args.model,
                    calls=[],
                    started=time.perf_counter(),
                    eligibility="task_failure",
                    draft=None,
                    program=None,
                    audit=None,
                    decision=GateDecision(False, False, ("task_failure",)),
                    audit_error=f"{type(exc).__name__}: {exc}",
                    source_sha256=source_sha256,
                )
            with write_lock:
                output_rows[str(task["task_id"])] = row
                core._canonicalize_output(args.output, tasks, output_rows)
            completed += 1
            trace = row.get("generation", {}).get("calculator_sympy", {})
            print(
                f"[{completed}/{len(pending)}] {task['task_id']} "
                f"answer={row.get('final_answer')!r} "
                f"switch={trace.get('switch_applied')} "
                f"reasons={trace.get('switch_reasons')}",
                flush=True,
            )

    core._canonicalize_output(args.output, tasks, output_rows)
    ordered = [output_rows[str(task["task_id"])] for task in tasks]
    traces = [row.get("generation", {}).get("calculator_sympy", {}) for row in ordered]
    audit_failures = sum(bool(trace.get("audit_error")) for trace in traces)
    summary = {
        "condition": CONDITION,
        "rows": len(ordered),
        "math_tasks": sum(_is_math(task) for task in tasks),
        "audit_calls_added": sum(len(trace.get("call_traces") or []) for trace in traces),
        "comparison_calls": sum(bool(trace.get("comparison_call_triggered")) for trace in traces),
        "switches": sum(bool(trace.get("switch_applied")) for trace in traces),
        "audit_failures": audit_failures,
        "gold_access": False,
        "sympy_available": True,
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "source_sha256": source_sha256,
        "implementation_sha256": _implementation_sha256(),
        "frozen_sources_verified": frozen_sources_verified,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if audit_failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
