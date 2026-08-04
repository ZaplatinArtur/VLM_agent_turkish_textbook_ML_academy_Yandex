"""Reasoning-first v2 of Maksim's matched direct/decompose/parallel8 experiment.

The v1 structured schema emitted ``final_answer`` before its bounded rationale.
That made recovery robust but created an answer-first autoregressive ablation.
This v2 keeps all other treatment mechanics and gold-isolation guardrails while
placing short, schema-bounded reasoning before the final answer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import run_maxim_agent_ideas as core


SYSTEM_PROMPT = (
    "Sen Türkçe okul sorularını çözen dikkatli bir uzmansın. Görseldeki soruyu "
    "doğrudan incele. Kaynak cevap, cevap anahtarı veya gold bilgi verilmemiştir. "
    "Yalnızca sorunun kendi kanıtlarını kullan. Gerekçeyi kısa tut; uzun deneme "
    "listeleri üretme, matematikte önce denklem, bölünebilirlik ve sınır kullan. "
    "Çoktan seçmeli soruda final_answer yalnızca seçenek harfi olsun; sayısal "
    "veya açık uçlu soruda gerçek cevabı yaz. Önce kısa analizi tamamla, sonra "
    "final_answer alanını yaz ve JSON şemasından çıkma."
)

SOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 700},
        "solution_steps": {"type": "string", "maxLength": 900},
        "final_answer": {"type": "string", "maxLength": 100},
    },
    "required": ["reasoning", "solution_steps", "final_answer"],
    "additionalProperties": False,
}

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 700},
        "final_answer": {"type": "string", "maxLength": 100},
    },
    "required": ["reasoning", "final_answer"],
    "additionalProperties": False,
}

SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 800},
        "solution_steps": {"type": "string", "maxLength": 900},
        "selected_index": {"type": "integer", "minimum": 1, "maximum": 8},
        "final_answer": {"type": "string", "maxLength": 100},
    },
    "required": [
        "reasoning",
        "solution_steps",
        "selected_index",
        "final_answer",
    ],
    "additionalProperties": False,
}

CONDITIONS = {
    "direct": "maxim_direct_reasoning_first_v2",
    "decompose": "maxim_decompose_reasoning_first_v2",
    "parallel8": "maxim_parallel8_judge_reasoning_first_v2",
}


def _install_reasoning_first_contract() -> None:
    core.SYSTEM_PROMPT = SYSTEM_PROMPT
    core.SOLVE_SCHEMA = SOLVE_SCHEMA
    core.CANDIDATE_SCHEMA = CANDIDATE_SCHEMA
    core.SELECT_SCHEMA = SELECT_SCHEMA


def run_direct(
    task: dict[str, Any],
    *,
    pool: core.EndpointPool,
    image_root: Path,
    image_url_root: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []
    try:
        call = pool.complete(
            messages=core._messages(
                task,
                (
                    "Soruyu tek bir kısa çözümle doğrudan çöz. Görseldeki kritik "
                    "kanıtı, soru kökünü ve seçenek eşlemesini kontrol et. Ayrı "
                    "alt-görev planı veya birden fazla aday üretme."
                ),
                image_root=image_root,
                image_url_root=image_url_root,
            ),
            schema_name="direct_reasoning_first_v2",
            schema=SOLVE_SCHEMA,
            max_tokens=1536,
            temperature=0.0,
            seed=303,
        )
        calls.append(call)
        return core._base_result(
            task,
            condition=CONDITIONS["direct"],
            final=call["parsed"],
            calls=calls,
            started=started,
            generation={
                "idea": "matched_direct_vision_control",
                "schema_order": "reasoning_first",
                "max_tokens_per_call": [1536],
                "call_traces": [core._compact_call(call)],
            },
            error=None,
        )
    except Exception as exc:
        return core._base_result(
            task,
            condition=CONDITIONS["direct"],
            final=None,
            calls=calls,
            started=started,
            generation={
                "idea": "matched_direct_vision_control",
                "schema_order": "reasoning_first",
            },
            error=f"{type(exc).__name__}: {exc}",
        )


def _retag(row: dict[str, Any], mode: str) -> dict[str, Any]:
    row["condition"] = CONDITIONS[mode]
    row["prompt_version"] = CONDITIONS[mode]
    generation = row.setdefault("generation", {})
    generation["schema_order"] = "reasoning_first"
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("direct", "decompose", "parallel8"), required=True
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-url-root", default="file:///images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", default=core.MODEL)
    parser.add_argument("--task-concurrency", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args(argv)

    _install_reasoning_first_contract()
    if not 1 <= args.task_concurrency <= 64:
        raise SystemExit("--task-concurrency must be in [1, 64]")
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

    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for row in core._load_jsonl(args.output):
            task_id = str(row.get("task_id") or "")
            if task_id and (not args.retry_errors or not row.get("error")):
                existing[task_id] = row

    pool = core.EndpointPool(
        args.base_url, model=args.model, timeout_s=args.timeout_s
    )
    output_rows = dict(existing)
    pending = [task for task in tasks if str(task["task_id"]) not in existing]
    write_lock = threading.Lock()

    runners: dict[str, Callable[..., dict[str, Any]]] = {
        "direct": run_direct,
        "decompose": core.run_decompose,
        "parallel8": core.run_parallel8,
    }

    def execute(task: dict[str, Any]) -> dict[str, Any]:
        row = runners[args.mode](
            task,
            pool=pool,
            image_root=args.image_root,
            image_url_root=args.image_url_root,
        )
        return _retag(row, args.mode)

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
                row = core._base_result(
                    task,
                    condition=CONDITIONS[args.mode],
                    final=None,
                    calls=[],
                    started=time.perf_counter(),
                    generation={"idea": args.mode, "schema_order": "reasoning_first"},
                    error=f"{type(exc).__name__}: {exc}",
                )
            with write_lock:
                output_rows[str(task["task_id"])] = row
                core._canonicalize_output(args.output, tasks, output_rows)
            completed += 1
            print(
                f"[{completed}/{len(pending)}] {task['task_id']} "
                f"answer={row.get('final_answer')!r} error={row.get('error')!r}",
                flush=True,
            )

    core._canonicalize_output(args.output, tasks, output_rows)
    errors = sum(bool(row.get("error")) for row in output_rows.values())
    print(
        json.dumps(
            {
                "mode": args.mode,
                "condition": CONDITIONS[args.mode],
                "rows": len(output_rows),
                "errors": errors,
                "output": str(args.output),
                "gold_access": False,
                "schema_order": "reasoning_first",
            },
            ensure_ascii=False,
        )
    )
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
