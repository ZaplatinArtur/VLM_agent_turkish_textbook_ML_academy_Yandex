"""Run the model-call variants from Maksim's eight-idea benchmark plan.

The runner is deliberately answer-blind: benchmark reference fields are removed
before a task reaches any prompt builder.  It reuses the frozen experiment's
OpenAI-compatible transport and image handling, but writes to a separate output
chosen by the caller.

Implemented here:

* ``solver_critic_repair`` -- independent draft, first-error critique, repair;
* ``two_pass_transcription`` -- exact visual transcription, then solve/verify;
* ``error_memory`` -- solve once with a frozen taxonomy of common failure modes.

The calculator/SymPy variant lives in a separate safe-tool module so arithmetic
never executes model-provided Python.  It is attached to this CLI only after its
tool contract has passed its own unit tests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

try:
    import run_maxim_agent_ideas as core
except ModuleNotFoundError:  # Imported as scripts.run_maxim_online_variants_v1.
    from scripts import run_maxim_agent_ideas as core


FROZEN_BENCHMARK_SHA256 = (
    "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
)

SYSTEM_PROMPT = (
    "You are a careful expert solving Turkish school textbook questions. "
    "There is no answer key, reference answer, or gold information. Use only "
    "the visible question and metadata supplied in the user message. Re-check "
    "negation, symbols, units, option-to-letter mapping, and requested output. "
    "For multiple choice, final_answer must be only the option letter. For a "
    "numeric or open response, final_answer must contain the actual answer. "
    "Return exactly the requested JSON schema."
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 1400},
        "solution_steps": {"type": "string", "maxLength": 1800},
        "final_answer": {"type": "string", "minLength": 1, "maxLength": 120},
    },
    "required": ["reasoning", "solution_steps", "final_answer"],
    "additionalProperties": False,
}

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "accept_draft": {"type": "boolean"},
        "error_type": {
            "type": "string",
            "enum": [
                "none",
                "visual_transcription",
                "question_intent",
                "reasoning",
                "calculation",
                "units_or_bounds",
                "option_mapping",
                "missing_subanswer",
                "answer_format",
            ],
        },
        "first_incorrect_step": {"type": "string", "maxLength": 700},
        "visible_evidence": {"type": "string", "maxLength": 900},
        "repair_instruction": {"type": "string", "maxLength": 900},
    },
    "required": [
        "accept_draft",
        "error_type",
        "first_incorrect_step",
        "visible_evidence",
        "repair_instruction",
    ],
    "additionalProperties": False,
}

TRANSCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question_stem": {"type": "string", "maxLength": 1800},
        "options_or_required_parts": {
            "type": "array",
            "items": {"type": "string", "maxLength": 500},
            "maxItems": 12,
        },
        "critical_values_symbols_units": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 16,
        },
        "table_graph_relations": {
            "type": "array",
            "items": {"type": "string", "maxLength": 500},
            "maxItems": 12,
        },
        "ambiguous_spans": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 8,
        },
    },
    "required": [
        "question_stem",
        "options_or_required_parts",
        "critical_values_symbols_units",
        "table_graph_relations",
        "ambiguous_spans",
    ],
    "additionalProperties": False,
}

MEMORY_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "checks_applied": {
            "type": "array",
            "items": {"type": "string", "maxLength": 180},
            "minItems": 1,
            "maxItems": 9,
        },
        "reasoning": {"type": "string", "maxLength": 1400},
        "solution_steps": {"type": "string", "maxLength": 1800},
        "final_answer": {"type": "string", "minLength": 1, "maxLength": 120},
    },
    "required": ["checks_applied", "reasoning", "solution_steps", "final_answer"],
    "additionalProperties": False,
}

ERROR_MEMORY = (
    "1) read every required subpart; 2) copy signs, exponents, decimal commas, "
    "labels and table axes exactly; 3) detect negative wording such as NOT/EXCEPT; "
    "4) keep option content separate from option letter; 5) check algebraic signs "
    "and arithmetic by substitution; 6) check domain, divisibility, bounds and "
    "units; 7) distinguish a fact asked by the problem from a merely similar fact; "
    "8) answer all blanks/subquestions; 9) emit only the requested final format."
)

CONDITIONS = {
    "solver_critic_repair": "maxim_solver_critic_repair_v1",
    "two_pass_transcription": "maxim_two_pass_transcription_v1",
    "error_memory": "maxim_error_memory_v1",
}


def _messages(
    task: dict[str, Any],
    instruction: str,
    *,
    image_root: Path,
    image_url_root: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"{core._task_prompt(task)}\n\n{instruction}",
        }
    ]
    content.extend(
        core._image_blocks(
            task, image_root=image_root, image_url_root=image_url_root
        )
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _call(
    pool: core.EndpointPool,
    task: dict[str, Any],
    instruction: str,
    *,
    image_root: Path,
    image_url_root: str,
    schema_name: str,
    schema: dict[str, Any],
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    return pool.complete(
        messages=_messages(
            task,
            instruction,
            image_root=image_root,
            image_url_root=image_url_root,
        ),
        schema_name=schema_name,
        schema=schema,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
        retries=1,
    )


def _compact_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": call.get("endpoint"),
        "finish_reason": call.get("finish_reason"),
        "attempt": call.get("attempt"),
        "latency_s": call.get("latency_s"),
        "input_tokens": call.get("input_tokens"),
        "output_tokens": call.get("output_tokens"),
        "recovered_partial": bool(call.get("recovered_partial")),
        "parse_error": call.get("parse_error"),
    }


def _result(
    task: dict[str, Any],
    *,
    pool: core.EndpointPool,
    condition: str,
    final: dict[str, Any] | None,
    calls: list[dict[str, Any]],
    started: float,
    generation: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    parsed = final or {}
    answer = str(parsed.get("final_answer") or "").strip() or None
    if error is None and answer is None:
        error = "ValueError: final stage returned an empty final_answer"
    return {
        "task_id": str(task["task_id"]),
        "condition": condition,
        "model": pool.model,
        "prompt_version": condition,
        "final_answer": answer,
        "solution_steps": str(parsed.get("solution_steps") or "").strip() or None,
        "reasoning": str(parsed.get("reasoning") or "").strip() or None,
        "forced_answer": False,
        "raw_response": json.dumps(parsed, ensure_ascii=False) if parsed else None,
        "generation": {
            "temperature": 0.0,
            "top_p": 0.95,
            "structured_mode": "response_format",
            "enable_thinking": False,
            "gold_access": False,
            "call_count": len(calls),
            "retry_calls": sum(
                max(0, int(call.get("attempt") or 1) - 1) for call in calls
            ),
            "call_traces": [_compact_call(call) for call in calls],
            **generation,
        },
        "tool_calls": [],
        "usage": core._usage(calls, time.perf_counter() - started),
        "error": error,
    }


def run_solver_critic_repair(
    task: dict[str, Any],
    *,
    pool: core.EndpointPool,
    image_root: Path,
    image_url_root: str,
) -> dict[str, Any]:
    """Solve, locate the first concrete error, then repair the answer."""

    condition = CONDITIONS["solver_critic_repair"]
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []
    draft: dict[str, Any] | None = None
    try:
        draft_call = _call(
            pool,
            task,
            (
                "Solve the problem independently. Make the reasoning auditable and "
                "derive the requested answer from the visible image."
            ),
            image_root=image_root,
            image_url_root=image_url_root,
            schema_name="maxim_scr_draft",
            schema=ANSWER_SCHEMA,
            max_tokens=2048,
            seed=4101,
        )
        calls.append(draft_call)
        draft = draft_call["parsed"]
        critique_call = _call(
            pool,
            task,
            (
                "Act only as a strict critic. Compare the draft with the original "
                "image and identify the FIRST concrete incorrect or unsupported "
                "step. Do not invent an error: accept the draft if every required "
                "part, calculation, unit and option mapping is supported.\nDRAFT:\n"
                + json.dumps(draft, ensure_ascii=False)
            ),
            image_root=image_root,
            image_url_root=image_url_root,
            schema_name="maxim_scr_critique",
            schema=CRITIQUE_SCHEMA,
            max_tokens=1400,
            seed=4102,
        )
        calls.append(critique_call)
        critique = critique_call["parsed"]
        repair_call = _call(
            pool,
            task,
            (
                "Produce the final answer after independently checking the image. "
                "Use the critique only when it is supported by visible evidence. If "
                "the critic accepted the draft, preserve it unless your own check "
                "finds a concrete contradiction. Correct the first bad step and all "
                "downstream consequences; answer every requested part.\nDRAFT:\n"
                + json.dumps(draft, ensure_ascii=False)
                + "\nCRITIQUE:\n"
                + json.dumps(critique, ensure_ascii=False)
            ),
            image_root=image_root,
            image_url_root=image_url_root,
            schema_name="maxim_scr_repair",
            schema=ANSWER_SCHEMA,
            max_tokens=2048,
            seed=4103,
        )
        calls.append(repair_call)
        return _result(
            task,
            pool=pool,
            condition=condition,
            final=repair_call["parsed"],
            calls=calls,
            started=started,
            generation={
                "idea": "solver_critic_repair",
                "max_tokens_per_call": [2048, 1400, 2048],
                "seeds": [4101, 4102, 4103],
                "draft": draft,
                "critique": critique,
                "fallback_stage": None,
            },
            error=None,
        )
    except Exception as exc:
        # A valid draft remains a legitimate answer if a later checking stage has
        # an operational failure.  The fallback is explicit in provenance.
        if draft and str(draft.get("final_answer") or "").strip():
            return _result(
                task,
                pool=pool,
                condition=condition,
                final=draft,
                calls=calls,
                started=started,
                generation={
                    "idea": "solver_critic_repair",
                    "max_tokens_per_call": [2048, 1400, 2048],
                    "seeds": [4101, 4102, 4103],
                    "draft": draft,
                    "fallback_stage": "draft",
                    "stage_error": f"{type(exc).__name__}: {exc}",
                },
                error=None,
            )
        return _result(
            task,
            pool=pool,
            condition=condition,
            final=None,
            calls=calls,
            started=started,
            generation={"idea": "solver_critic_repair"},
            error=f"{type(exc).__name__}: {exc}",
        )


def run_two_pass_transcription(
    task: dict[str, Any],
    *,
    pool: core.EndpointPool,
    image_root: Path,
    image_url_root: str,
) -> dict[str, Any]:
    """Transcribe without solving, then verify the transcript while solving."""

    condition = CONDITIONS["two_pass_transcription"]
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []
    try:
        transcription_call = _call(
            pool,
            task,
            (
                "PASS 1 -- TRANSCRIPTION ONLY. Do not solve and do not guess an "
                "answer. Precisely transcribe the question stem, every option or "
                "subpart, numbers, signs, exponents, labels, units, and relevant "
                "table/graph relations. Mark genuinely unreadable spans as ambiguous."
            ),
            image_root=image_root,
            image_url_root=image_url_root,
            schema_name="maxim_transcription",
            schema=TRANSCRIPTION_SCHEMA,
            max_tokens=1800,
            seed=4201,
        )
        calls.append(transcription_call)
        transcription = transcription_call["parsed"]
        solve_call = _call(
            pool,
            task,
            (
                "PASS 2 -- VERIFY AND SOLVE. First compare the supplied transcript "
                "against the original image and silently correct any mismatch. Then "
                "solve from the verified evidence. Never treat the transcript as an "
                "answer key. Resolve ambiguous spans from the image when possible.\n"
                "TRANSCRIPT:\n"
                + json.dumps(transcription, ensure_ascii=False)
            ),
            image_root=image_root,
            image_url_root=image_url_root,
            schema_name="maxim_transcription_solve",
            schema=ANSWER_SCHEMA,
            max_tokens=2048,
            seed=4202,
        )
        calls.append(solve_call)
        return _result(
            task,
            pool=pool,
            condition=condition,
            final=solve_call["parsed"],
            calls=calls,
            started=started,
            generation={
                "idea": "two_pass_transcription_then_solve",
                "max_tokens_per_call": [1800, 2048],
                "seeds": [4201, 4202],
                "transcription": transcription,
            },
            error=None,
        )
    except Exception as exc:
        return _result(
            task,
            pool=pool,
            condition=condition,
            final=None,
            calls=calls,
            started=started,
            generation={"idea": "two_pass_transcription_then_solve"},
            error=f"{type(exc).__name__}: {exc}",
        )


def run_error_memory(
    task: dict[str, Any],
    *,
    pool: core.EndpointPool,
    image_root: Path,
    image_url_root: str,
) -> dict[str, Any]:
    """Solve with a frozen, task-independent memory of typical mistakes."""

    condition = CONDITIONS["error_memory"]
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []
    try:
        answer_call = _call(
            pool,
            task,
            (
                "Solve the visible problem. Before finalizing, apply only the "
                "relevant checks from this frozen generic error memory. The memory "
                "contains no benchmark answers and may not override visible evidence.\n"
                f"ERROR MEMORY:\n{ERROR_MEMORY}"
            ),
            image_root=image_root,
            image_url_root=image_url_root,
            schema_name="maxim_error_memory_answer",
            schema=MEMORY_ANSWER_SCHEMA,
            max_tokens=2200,
            seed=4301,
        )
        calls.append(answer_call)
        parsed = answer_call["parsed"]
        return _result(
            task,
            pool=pool,
            condition=condition,
            final=parsed,
            calls=calls,
            started=started,
            generation={
                "idea": "frozen_generic_error_memory",
                "max_tokens_per_call": [2200],
                "seeds": [4301],
                "memory_version": "generic_error_taxonomy_v1",
                "checks_applied": parsed.get("checks_applied") or [],
            },
            error=None,
        )
    except Exception as exc:
        return _result(
            task,
            pool=pool,
            condition=condition,
            final=None,
            calls=calls,
            started=started,
            generation={"idea": "frozen_generic_error_memory"},
            error=f"{type(exc).__name__}: {exc}",
        )


RUNNERS: dict[
    str,
    Callable[..., dict[str, Any]],
] = {
    "solver_critic_repair": run_solver_critic_repair,
    "two_pass_transcription": run_two_pass_transcription,
    "error_memory": run_error_memory,
}


def _write_in_benchmark_order(
    output: Path,
    tasks: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as sink:
        for task in tasks:
            task_id = str(task["task_id"])
            if task_id in rows:
                sink.write(json.dumps(rows[task_id], ensure_ascii=False) + "\n")


def _implementation_sha256() -> dict[str, str]:
    return {
        "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_transport": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(RUNNERS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-url-root", default="file:///images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", action="append", default=[])
    parser.add_argument("--model", default=core.MODEL)
    parser.add_argument("--task-concurrency", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--allow-unfrozen-input", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.task_concurrency <= 64:
        raise SystemExit("--task-concurrency must be in [1, 64]")
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("--output must not overwrite --input")
    input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    implementation_sha256 = _implementation_sha256()
    frozen_benchmark_verified = input_sha256 == FROZEN_BENCHMARK_SHA256
    if not frozen_benchmark_verified and not args.allow_unfrozen_input:
        raise SystemExit(
            "benchmark SHA256 mismatch: "
            f"expected {FROZEN_BENCHMARK_SHA256}, got {input_sha256}; "
            "use --allow-unfrozen-input only for explicit tests/debugging"
        )
    tasks = [core._task_view(task) for task in core._load_jsonl(args.input)]
    if args.task_id:
        selected = set(args.task_id)
        tasks = [task for task in tasks if str(task.get("task_id")) in selected]
        missing = sorted(selected - {str(task.get("task_id")) for task in tasks})
        if missing:
            raise SystemExit(f"unknown task IDs: {missing}")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if len({str(task["task_id"]) for task in tasks}) != len(tasks):
        raise SystemExit("duplicate task IDs")

    dry_run_report = {
        "mode": args.mode,
        "condition": CONDITIONS[args.mode],
        "tasks": len(tasks),
        "task_ids": [task["task_id"] for task in tasks[:10]],
        "gold_access": False,
        "input_sha256": input_sha256,
        "implementation_sha256": implementation_sha256,
        "frozen_benchmark_verified": frozen_benchmark_verified,
    }
    if args.dry_run:
        print(json.dumps(dry_run_report, ensure_ascii=False, indent=2))
        return 0
    if not args.base_url:
        raise SystemExit("at least one --base-url is required unless --dry-run is used")

    condition = CONDITIONS[args.mode]
    if args.output.exists() and not (args.resume or args.retry_errors):
        raise FileExistsError(
            f"output exists; pass --resume explicitly: {args.output}"
        )
    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        selected_ids = {str(task["task_id"]) for task in tasks}
        seen_ids: set[str] = set()
        for row in core._load_jsonl(args.output):
            task_id = str(row.get("task_id") or "")
            if not task_id:
                raise ValueError("resume output contains a row without task_id")
            if task_id in seen_ids:
                raise ValueError(f"resume output has duplicate task_id: {task_id}")
            seen_ids.add(task_id)
            if task_id not in selected_ids:
                raise ValueError(
                    f"resume output contains task outside selected set: {task_id}"
                )
            if row.get("condition") != condition:
                raise SystemExit(
                    f"output contains condition {row.get('condition')!r}; "
                    f"expected {condition!r}"
                )
            if row.get("model") != args.model:
                raise ValueError(f"resume row has different model: {task_id}")
            generation = row.get("generation")
            provenance = (
                generation.get("source_sha256")
                if isinstance(generation, dict)
                else None
            )
            if provenance != {"tasks": input_sha256}:
                raise ValueError(f"resume provenance mismatch: {task_id}")
            if generation.get("implementation_sha256") != implementation_sha256:
                raise ValueError(f"resume implementation mismatch: {task_id}")
            if task_id and (not args.retry_errors or not row.get("error")):
                existing[task_id] = row

    pool = core.EndpointPool(args.base_url, model=args.model, timeout_s=args.timeout_s)
    output_rows = dict(existing)
    pending = [task for task in tasks if str(task["task_id"]) not in existing]
    runner = RUNNERS[args.mode]
    write_lock = threading.Lock()

    def execute(task: dict[str, Any]) -> dict[str, Any]:
        return runner(
            task,
            pool=pool,
            image_root=args.image_root,
            image_url_root=args.image_url_root,
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
            except Exception as exc:  # defensive task-level fail-closed record
                row = _result(
                    task,
                    pool=pool,
                    condition=condition,
                    final=None,
                    calls=[],
                    started=time.perf_counter(),
                    generation={"idea": args.mode},
                    error=f"{type(exc).__name__}: {exc}",
                )
            generation = (
                dict(row.get("generation"))
                if isinstance(row.get("generation"), dict)
                else {}
            )
            generation["source_sha256"] = {"tasks": input_sha256}
            generation["implementation_sha256"] = implementation_sha256
            row["generation"] = generation
            with write_lock:
                output_rows[str(task["task_id"])] = row
                _write_in_benchmark_order(args.output, tasks, output_rows)
            completed += 1
            print(
                f"[{completed}/{len(pending)}] {task['task_id']} "
                f"answer={row.get('final_answer')!r} error={row.get('error')!r}",
                flush=True,
            )

    _write_in_benchmark_order(args.output, tasks, output_rows)
    errors = sum(bool(row.get("error")) for row in output_rows.values())
    print(
        json.dumps(
            {
                **dry_run_report,
                "rows": len(output_rows),
                "errors": errors,
                "output": str(args.output),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
        )
    )
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
