"""Adapters between ``mla_baseline`` JSONL and the text binary judge."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .ingest import read_records


CONDITION_TO_SETUP = {
    "b0_no_tools": "no_tools",
    "b1_search": "web_search",
    "agent_rag": "textbook_retrieval",
}

_ANSWER_TYPE_TO_BASELINE = {
    "multiple_choice": "choice",
    "numeric": "numeric",
    "short_text": "short_text",
    "multi_answer": "free_form",
    "open_ended": "free_form",
    "unknown": "free_form",
}
_MARKDOWN_ATTACHMENT = re.compile(r"!\[[^\]]*\]\(attachment://[^)]+\)")


def _unique_by_task(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = str(record.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"{label} record is missing task_id")
        if task_id in by_id:
            raise ValueError(f"duplicate task_id in {label}: {task_id}")
        by_id[task_id] = record
    return by_id


def candidate_text_from_solve_result(result: dict[str, Any]) -> str:
    """Preserve both reasoning and final answer for semantic judging."""
    parts: list[str] = []
    solution = str(result.get("solution_steps") or "").strip()
    final_answer = str(result.get("final_answer") or "").strip()
    raw_response = str(result.get("raw_response") or "").strip()
    if solution:
        parts.append(f"Solution:\n{solution}")
    if final_answer:
        parts.append(f"Final answer:\n{final_answer}")
    if not parts and raw_response:
        parts.append(raw_response)
    return "\n\n".join(parts) or "[EMPTY_RESPONSE]"


def prepare_text_judge_input(
    tasks_path: Path,
    results_path: Path,
    output_path: Path,
    *,
    require_all: bool = False,
) -> dict[str, Any]:
    """Join baseline tasks and solver results into text-binary judge records."""
    tasks = read_records(tasks_path)
    results = read_records(results_path)
    tasks_by_id = _unique_by_task(tasks, "tasks")
    results_by_id = _unique_by_task(results, "results")
    unknown_result_ids = sorted(set(results_by_id) - set(tasks_by_id))
    missing_result_ids = sorted(set(tasks_by_id) - set(results_by_id))
    if unknown_result_ids:
        raise ValueError(f"results contain unknown task_ids: {unknown_result_ids[:10]}")
    if require_all and missing_result_ids:
        raise ValueError(f"tasks without solver results: {missing_result_ids[:10]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failures = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for task in tasks:
            task_id = str(task["task_id"])
            result = results_by_id.get(task_id)
            if result is None:
                continue
            question = str(task.get("question") or task.get("question_text") or "").strip()
            reference = str(task.get("reference_answer") or "").strip()
            if not question or not reference:
                raise ValueError(f"task {task_id} requires question and reference_answer")
            condition = str(result.get("condition") or "unknown")
            setup = CONDITION_TO_SETUP.get(condition, condition)
            agent_error = result.get("error")
            if agent_error:
                failures += 1
            record = {
                "task_id": task_id,
                "question_text": question,
                "reference_answer": reference,
                "candidate_answer": candidate_text_from_solve_result(result),
                "condition": condition,
                "setup": setup,
                "subject": task.get("subject"),
                "grade": task.get("grade"),
                "answer_type": task.get("answer_type"),
                "agent_result": {
                    key: result.get(key)
                    for key in (
                        "model",
                        "prompt_version",
                        "generation",
                        "tool_calls",
                        "usage",
                        "error",
                    )
                },
            }
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return {
        "tasks": len(tasks),
        "results": len(results),
        "written": written,
        "agent_failures": failures,
        "missing_result_ids": missing_result_ids,
        "output": str(output_path),
    }


def build_seed_text_tasks(source_path: Path, output_path: Path) -> dict[str, Any]:
    """Extract reproducible text-only tasks from the eight real legacy failures."""
    source = read_records(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for record in source:
            candidate = str(record.get("candidate_answer") or "")
            question_block = candidate.split("### Solution", 1)[0]
            question = question_block.replace("### Question", "", 1).strip()
            question = _MARKDOWN_ATTACHMENT.sub("[image omitted in text-only smoke test]", question)
            reference = str(record.get("reference_answer") or "").strip()
            if not question or not reference:
                continue
            answer_type = _ANSWER_TYPE_TO_BASELINE.get(
                str(record.get("answer_type") or "unknown"), "free_form"
            )
            task = {
                "task_id": str(record["task_id"]),
                "subject": str(record.get("subject") or "unknown").casefold(),
                "grade": record.get("grade"),
                "question": question,
                "question_images": [],
                "reference_answer": reference,
                "answer_type": answer_type,
                "reference_solution": None,
            }
            destination.write(json.dumps(task, ensure_ascii=False) + "\n")
            written += 1
    return {"source_records": len(source), "written": written, "output": str(output_path)}
