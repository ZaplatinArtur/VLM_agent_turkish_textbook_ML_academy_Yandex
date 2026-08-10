"""Compose a routed evaluation from already generated E0 and E3 results."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .runner import load_tasks


def _read_unique(path: Path, label: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            task_id = str(value.get("task_id") or "").strip()
            if not task_id:
                raise ValueError(f"{label}:{line_number} is missing task_id")
            if task_id in records:
                raise ValueError(f"duplicate task_id in {label}: {task_id}")
            records[task_id] = value
    return records


def compose_routed_results(
    *,
    tasks_path: Path,
    no_tools_path: Path,
    rag_path: Path,
    output_path: Path,
    no_retrieval_subjects: str,
) -> dict[str, Any]:
    """Select the exact E0 or E3 row per subject, without another LLM call."""

    tasks = load_tasks(tasks_path)
    tasks_by_id = {task.task_id: task for task in tasks}
    no_tools = _read_unique(no_tools_path, "no-tools results")
    rag = _read_unique(rag_path, "RAG results")
    if set(no_tools) != set(rag):
        missing_no_tools = sorted(set(rag) - set(no_tools))
        missing_rag = sorted(set(no_tools) - set(rag))
        raise ValueError(
            "E0 and E3 task sets differ; "
            f"missing_no_tools={missing_no_tools[:10]}, missing_rag={missing_rag[:10]}"
        )
    unknown = sorted(set(no_tools) - set(tasks_by_id))
    if unknown:
        raise ValueError(f"results contain unknown task_ids: {unknown[:10]}")

    blocked = {
        subject.strip().casefold()
        for subject in no_retrieval_subjects.split(",")
        if subject.strip()
    }
    composed: list[dict[str, Any]] = []
    skipped = 0
    allowed = 0
    for task in tasks:
        if task.task_id not in no_tools:
            continue
        baseline = no_tools[task.task_id]
        retrieval = rag[task.task_id]
        if (
            baseline.get("model") != retrieval.get("model")
            or baseline.get("prompt_version") != retrieval.get("prompt_version")
        ):
            raise ValueError(f"E0/E3 model or prompt mismatch for {task.task_id}")

        route_skips = task.subject.strip().casefold() in blocked
        source = baseline if route_skips else retrieval
        row = copy.deepcopy(source)
        row["condition"] = "agent_rag_routed"
        generation = dict(row.get("generation") or {})
        generation.update(
            {
                "experiment_id": "e4_routed_image_first_rag_v1",
                "agent_strategy": "subject_routed_image_first_checked_retrieval_v1",
                "retrieval_route": "skip" if route_skips else "allow",
                "retrieval_route_reason": (
                    "subject_blocklist" if route_skips else "subject_allowed"
                ),
                "retrieval_route_subject": task.subject,
                "composed_from_condition": (
                    "b0_no_tools" if route_skips else "agent_rag"
                ),
            }
        )
        row["generation"] = generation
        if route_skips:
            skipped += 1
            row["tool_calls"] = []
            row["exit_reason"] = row.get("error") or "router_no_retrieval"
            row["retrieval_relevance"] = "not_attempted"
            row["retrieval_conflict"] = False
            row["answer_source"] = (
                "image_only_no_retrieval"
                if task.question_images
                else "text_only_no_retrieval"
            )
        else:
            allowed += 1
        composed.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in composed:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    return {
        "written": len(composed),
        "router_skips": skipped,
        "router_allows": allowed,
        "output": str(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compose routed E4 rows from paired E0 and E3 outputs"
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--no-tools", type=Path, required=True)
    parser.add_argument("--rag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-retrieval-subjects", default="Math")
    args = parser.parse_args(argv)
    report = compose_routed_results(
        tasks_path=args.tasks,
        no_tools_path=args.no_tools,
        rag_path=args.rag,
        output_path=args.output,
        no_retrieval_subjects=args.no_retrieval_subjects,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
