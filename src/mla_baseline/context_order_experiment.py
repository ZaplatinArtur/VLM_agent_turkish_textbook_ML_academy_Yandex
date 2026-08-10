"""Freeze retrieval once, then compare score and edge context ordering."""

from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .config import Settings, get_settings
from .contracts import Task
from .parsing import parse_solve_output
from .runner import load_done_ids, load_tasks
from .schemas import ImageTaskEvidence, SolveResult, Usage
from .solvers.agent_rag import AgentRag
from .tools.textbook_search import LocalTextbookSearchClient

ContextOrder = Literal["score", "edge"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def _effective_search_calls(seed: dict[str, Any]) -> list[dict[str, Any]]:
    calls = seed.get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [
        call
        for call in calls
        if isinstance(call, dict)
        and call.get("tool") == "search_textbooks"
        and not call.get("error")
    ]


def order_hits(hits: list[dict[str, Any]], order: ContextOrder) -> list[dict[str, Any]]:
    """Return the same hits in score or strongest-at-both-edges order."""

    copied = [copy.deepcopy(hit) for hit in hits]
    if order == "edge":
        copied = copied[::2] + copied[1::2][::-1]
    for position, hit in enumerate(copied, 1):
        if order == "edge":
            hit["context_position"] = position
        else:
            hit.pop("context_position", None)
    return copied


def payload_for_order(record: dict[str, Any], order: ContextOrder) -> dict[str, Any]:
    payload = copy.deepcopy(record["payload"])
    hits = payload.get("hits")
    hits = hits if isinstance(hits, list) else []
    payload["hits"] = order_hits(hits, order)
    payload["returned"] = len(payload["hits"])
    payload["context_order"] = order
    return payload


def freeze_retrieval_contexts(
    *,
    tasks: list[Task],
    seed_rows: list[dict[str, Any]],
    search_client: LocalTextbookSearchClient,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[Task]]:
    """Rehydrate full hit text for the exact post-conflict IDs in a seed run."""

    seeds_by_task = {str(row.get("task_id")): row for row in seed_rows}
    duplicate_ids = len(seeds_by_task) != len(seed_rows)
    if duplicate_ids:
        raise ValueError("seed results contain duplicate task_id values")

    records: list[dict[str, Any]] = []
    affected_tasks: list[Task] = []
    for task in tasks:
        seed = seeds_by_task.get(task.task_id)
        if seed is None:
            continue
        calls = _effective_search_calls(seed)
        if len(calls) > 1:
            raise ValueError(
                f"{task.task_id}: expected at most one effective retrieval call"
            )

        call = calls[0] if calls else {}
        arguments_value = call.get("args")
        arguments = (
            dict(arguments_value) if isinstance(arguments_value, dict) else {}
        )
        query = str(arguments.get("query") or "").strip()
        visible_ids_value = call.get("returned_chunk_ids")
        visible_ids = (
            [str(chunk_id) for chunk_id in visible_ids_value]
            if isinstance(visible_ids_value, list)
            else []
        )

        if visible_ids:
            retrieved = search_client.search(
                query,
                top_k=top_k,
                subject=(str(arguments["subject"]) if arguments.get("subject") else None),
                grade=arguments.get("grade"),
                mode=("and" if arguments.get("mode") == "and" else "or"),
            )
            retrieved_hits = retrieved.get("hits")
            retrieved_hits = retrieved_hits if isinstance(retrieved_hits, list) else []
            hits_by_id = {
                str(hit.get("chunk_id")): hit
                for hit in retrieved_hits
                if isinstance(hit, dict) and hit.get("chunk_id")
            }
            missing = [chunk_id for chunk_id in visible_ids if chunk_id not in hits_by_id]
            if missing:
                raise ValueError(
                    f"{task.task_id}: could not rehydrate frozen chunks: {missing}"
                )
            hits = [copy.deepcopy(hits_by_id[chunk_id]) for chunk_id in visible_ids]
            payload = copy.deepcopy(retrieved)
        else:
            hits = []
            payload = {
                "query": query,
                "top_k": top_k,
                "mode": arguments.get("mode", "or"),
                "filters": {
                    "subject": arguments.get("subject"),
                    "grade": arguments.get("grade"),
                },
                "retrieved": 0,
            }

        relevance_value = call.get("relevance")
        relevance = (
            copy.deepcopy(relevance_value)
            if isinstance(relevance_value, dict)
            else {
                "label": "empty",
                "is_useful": False,
                "top_score": None,
                "reason": "no visible chunks in preparation run",
            }
        )
        payload.update(
            {
                "query": query,
                "top_k": top_k,
                "context_order": "score",
                "returned": len(hits),
                "relevance": relevance,
                "hits": hits,
                "frozen": True,
            }
        )
        score_ids = [str(hit.get("chunk_id")) for hit in hits]
        edge_ids = [
            str(hit.get("chunk_id")) for hit in order_hits(hits, "edge")
        ]
        order_changes = score_ids != edge_ids
        structured_evidence = seed.get("image_evidence_structured")
        if not isinstance(structured_evidence, dict):
            evidence_values = seed.get("image_evidence")
            evidence_values = (
                evidence_values if isinstance(evidence_values, list) else []
            )
            topic = (query or task.subject or "task")[:200]
            structured_evidence = {
                "image_evidence": [str(value) for value in evidence_values],
                "question": task.question,
                "topic": topic,
                "unknown_concepts": [],
            }
        record = {
            "task_id": task.task_id,
            "arguments": {**arguments, "query": query},
            "image_evidence": copy.deepcopy(structured_evidence),
            "payload": payload,
            "retrieval_conflict": seed.get("retrieval_conflict"),
            "order_changes": order_changes,
        }
        records.append(record)
        if order_changes:
            affected_tasks.append(task)

    return records, affected_tasks


class FrozenContextOrderSolver(AgentRag):
    """Generate an answer from precomputed evidence and retrieval hits."""

    condition = "agent_rag_frozen_context"

    def __init__(
        self,
        settings: Settings,
        *,
        records: list[dict[str, Any]],
        order: ContextOrder,
        llm: Any | None = None,
    ) -> None:
        settings = settings.model_copy(update={"retrieval_context_order": order})
        super().__init__(settings, llm=llm)
        self.records = {str(record["task_id"]): record for record in records}
        self.order = order

    def solve(self, task: Task) -> SolveResult:
        usage = Usage()
        raw: str | None = None
        parsed = None
        error: str | None = None
        started = time.perf_counter()
        record = self.records.get(task.task_id)
        tool_logs = []
        relevance_label: str | None = None
        evidence: ImageTaskEvidence | None = None

        try:
            if record is None:
                raise ValueError(f"missing frozen context for {task.task_id}")
            evidence = ImageTaskEvidence.model_validate(record["image_evidence"])
            arguments = dict(record["arguments"])
            query = str(arguments.get("query") or "")
            payload = payload_for_order(record, self.order)
            relevance = payload.get("relevance")
            if isinstance(relevance, dict):
                relevance_label = str(relevance.get("label") or "") or None
            output = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

            messages = self.build_messages(task)
            messages.append(HumanMessage(content=self._evidence_note(evidence, query)))
            call_id = f"frozen-retrieval-{task.task_id}"
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": self.search_tool.name,
                            "args": arguments,
                            "id": call_id,
                        }
                    ],
                )
            )
            messages.append(
                ToolMessage(
                    content=output,
                    tool_call_id=call_id,
                    name=self.search_tool.name,
                )
            )
            tool_logs.append(
                self._tool_trace(
                    name=self.search_tool.name,
                    arguments=arguments,
                    output=output,
                )
            )

            raw = self._force_final_answer(messages, task, usage)
            parsed = parse_solve_output(raw)
            if parsed is None:
                error = "parse_error"
            else:
                verified_raw = self._verify_against_image(
                    messages=messages,
                    task=task,
                    evidence=evidence,
                    candidate=raw,
                    usage=usage,
                )
                if verified_raw is not None:
                    raw = verified_raw
                    parsed = parse_solve_output(raw)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        usage.latency_s = round(time.perf_counter() - started, 3)
        has_hits = bool(tool_logs and tool_logs[0].returned_chunk_ids)
        return SolveResult(
            task_id=task.task_id,
            condition=self.condition,
            model=self.settings.llm_model_name,
            prompt_version=self.settings.prompt_version,
            final_answer=parsed.final_answer if parsed else None,
            solution_steps=parsed.solution_steps if parsed else None,
            reasoning=parsed.reasoning if parsed else None,
            forced_answer=True,
            raw_response=raw,
            exit_reason=("frozen_context_answer" if parsed else "parse_error"),
            image_evidence=(evidence.image_evidence if evidence else []),
            image_evidence_structured=(evidence.model_dump() if evidence else None),
            retrieval_relevance=relevance_label,
            retrieval_conflict=record.get("retrieval_conflict") if record else None,
            answer_source=(
                "image_with_retrieval_support" if has_hits else "image_only"
            ),
            generation={
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "top_k": self.settings.top_k,
                "presence_penalty": self.settings.presence_penalty,
                "max_tokens": self.settings.max_tokens,
                "structured_mode": self.settings.structured_mode,
                "enable_thinking": self.settings.enable_thinking,
                "llm_provider": self.settings.llm_provider,
                "retrieval_strategy": "dense_frozen",
                "retrieval_fetch_k": self.settings.retrieval_fetch_k,
                "retrieval_context_order": self.order,
                "agent_strategy": "frozen_context_order_ablation_v1",
                "experiment_id": "context_order_v2",
            },
            tool_calls=tool_logs,
            usage=usage,
            error=error,
        )


def run_frozen_generation(
    *,
    tasks: list[Task],
    records: list[dict[str, Any]],
    order: ContextOrder,
    out_path: Path,
    settings: Settings,
    retry_errors: bool,
) -> tuple[int, int]:
    solver = FrozenContextOrderSolver(settings, records=records, order=order)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path, retry_errors=retry_errors)
    todo = [task for task in tasks if task.task_id not in done]
    print(
        f"Задач: {len(tasks)}, уже готово: {len(tasks) - len(todo)}, "
        f"к прогону: {len(todo)}"
    )

    errors = 0
    write_lock = threading.Lock()
    with out_path.open("a", encoding="utf-8") as destination:
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            futures = {pool.submit(solver.solve, task): task for task in todo}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result.error:
                    errors += 1
                with write_lock:
                    destination.write(result.model_dump_json() + "\n")
                    destination.flush()
                print(
                    f"[{index}/{len(todo)}] {result.task_id}: "
                    f"{result.error or 'ok'} ({result.usage.latency_s}s)"
                )
    print(f"Готово: {len(todo)} прогнано, ошибок: {errors}, результат: {out_path}")
    return len(todo), errors


def _prepare_command(args: argparse.Namespace) -> int:
    settings = get_settings()
    tasks = load_tasks(args.tasks)
    seed_rows = _read_jsonl(args.seed_results)
    client = LocalTextbookSearchClient(
        retrieval_fetch_k=settings.retrieval_fetch_k,
        mmr_lambda=None,
        context_order="score",
    )
    records, affected_tasks = freeze_retrieval_contexts(
        tasks=tasks,
        seed_rows=seed_rows,
        search_client=client,
        top_k=settings.retrieval_top_k,
    )
    if not affected_tasks:
        raise SystemExit("No tasks have a retrieval context whose order changes.")
    _write_jsonl(args.output, records)
    args.affected_tasks.parent.mkdir(parents=True, exist_ok=True)
    with args.affected_tasks.open("w", encoding="utf-8", newline="\n") as destination:
        for task in affected_tasks:
            destination.write(task.model_dump_json() + "\n")
    print(
        json.dumps(
            {
                "seed_tasks": len(seed_rows),
                "frozen_contexts": len(records),
                "affected_tasks": len(affected_tasks),
                "output": str(args.output),
                "affected_output": str(args.affected_tasks),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _run_command(args: argparse.Namespace) -> int:
    settings = get_settings()
    records = _read_jsonl(args.contexts)
    tasks = load_tasks(args.tasks)
    _, errors = run_frozen_generation(
        tasks=tasks,
        records=records,
        order=args.order,
        out_path=args.output,
        settings=settings,
        retry_errors=args.retry_errors,
    )
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="freeze the exact retrieval query and visible chunks from a seed run",
    )
    prepare.add_argument("--tasks", type=Path, required=True)
    prepare.add_argument("--seed-results", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--affected-tasks", type=Path, required=True)
    prepare.set_defaults(handler=_prepare_command)

    run = subparsers.add_parser("run", help="answer from a frozen retrieval context")
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--contexts", type=Path, required=True)
    run.add_argument("--order", choices=("score", "edge"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--retry-errors", action="store_true")
    run.set_defaults(handler=_run_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
