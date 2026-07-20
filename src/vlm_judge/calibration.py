from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


DEFAULT_QUOTAS = {
    "multiple_choice": 50,
    "multi_answer": 38,
    "numeric": 15,
    "open_ended": 15,
    "unknown": 2,
}


def build_synthetic_mc_stress(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create obvious judge smoke tests without pretending they are human calibration data."""
    choices = "ABCDE"
    records: list[dict[str, Any]] = []
    for task in tasks:
        if task.get("answer_type") != "multiple_choice":
            continue
        reference = str(task.get("reference_answer") or "").strip().upper()
        if reference not in choices:
            continue
        wrong = choices[(choices.index(reference) + 1) % len(choices)]
        cases = (
            (
                "correct_concise",
                f"Final answer: {reference}",
                ["fully_correct"],
            ),
            (
                "wrong_concise",
                f"Final answer: {wrong}",
                ["incorrect"],
            ),
            (
                "correct_choice_invalid_reasoning",
                f"Because 2 + 2 = 5, the final answer is {reference}.",
                ["mostly_correct", "partially_correct"],
            ),
        )
        for case_name, candidate, expected_labels in cases:
            metadata = dict(task.get("metadata") or {})
            metadata.update(
                {
                    "synthetic_case": case_name,
                    "expected_labels": expected_labels,
                    "synthetic_not_human_calibration": True,
                }
            )
            records.append(
                {
                    **task,
                    "task_id": f"{task['task_id']}__{case_name}",
                    "candidate_answer": candidate,
                    "setup": "unknown",
                    "metadata": metadata,
                }
            )
    return records


def _stable_key(task: dict[str, Any], seed: str) -> str:
    value = f"{seed}:{task.get('task_id', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _difficulty(task: dict[str, Any]) -> str:
    metadata = task.get("metadata") or {}
    return next(
        (name for name in ("easy", "medium", "hard") if metadata.get(name)),
        "unlabeled",
    )


def select_calibration_tasks(
    tasks: list[dict[str, Any]],
    *,
    quotas: dict[str, int] | None = None,
    seed: str = "calibration-v1",
) -> list[dict[str, Any]]:
    quotas = dict(quotas or DEFAULT_QUOTAS)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_type[str(task.get("answer_type") or "unknown")].append(task)

    selected: list[dict[str, Any]] = []
    for answer_type, quota in quotas.items():
        candidates = by_type.get(answer_type, [])
        if len(candidates) < quota:
            raise ValueError(
                f"not enough {answer_type} tasks: requested {quota}, found {len(candidates)}"
            )
        buckets: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
        for task in sorted(candidates, key=lambda value: _stable_key(value, seed)):
            key = (str(task.get("grade")), _difficulty(task))
            buckets[key].append(task)
        bucket_keys = sorted(buckets)
        picked = 0
        while picked < quota:
            made_progress = False
            for key in bucket_keys:
                if buckets[key] and picked < quota:
                    selected.append(buckets[key].popleft())
                    picked += 1
                    made_progress = True
            if not made_progress:
                raise RuntimeError(f"unable to satisfy quota for {answer_type}")
    return sorted(selected, key=lambda value: str(value.get("task_id")))


def prepare_calibration(
    benchmark_path: Path,
    output_dir: Path,
    *,
    seed: str = "calibration-v1",
) -> dict[str, Any]:
    with benchmark_path.open("r", encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    selected = select_calibration_tasks(tasks, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "calibration_tasks.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in selected:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    csv_path = output_dir / "calibration_labeling.csv"
    fields = [
        "task_id",
        "subject",
        "grade",
        "answer_type",
        "difficulty",
        "topic_area",
        "question_image_url",
        "reference_image_url",
        "reference_answer",
        "candidate_setup",
        "candidate_answer",
        "human_label",
        "human_score_0_4",
        "final_answer_correct",
        "reasoning_correct",
        "complete",
        "reference_quality_issue",
        "error_types",
        "rationale",
        "annotator",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in selected:
            metadata = task.get("metadata") or {}
            writer.writerow(
                {
                    "task_id": task.get("task_id"),
                    "subject": task.get("subject"),
                    "grade": task.get("grade"),
                    "answer_type": task.get("answer_type"),
                    "difficulty": _difficulty(task),
                    "topic_area": metadata.get("topic_area"),
                    "question_image_url": task.get("question_image_url"),
                    "reference_image_url": task.get("reference_image_url"),
                    "reference_answer": task.get("reference_answer"),
                }
            )

    stress_records = build_synthetic_mc_stress(selected)
    stress_path = output_dir / "synthetic_mc_stress.jsonl"
    with stress_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in stress_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_type = Counter(str(task.get("answer_type")) for task in selected)
    by_difficulty = Counter(_difficulty(task) for task in selected)
    by_grade = Counter(str(task.get("grade")) for task in selected)
    summary = {
        "seed": seed,
        "records": len(selected),
        "by_answer_type": dict(by_type.most_common()),
        "by_difficulty": dict(by_difficulty.most_common()),
        "by_grade": dict(sorted(by_grade.items(), key=lambda item: int(item[0]))),
        "jsonl": str(jsonl_path),
        "labeling_csv": str(csv_path),
        "synthetic_mc_stress_records": len(stress_records),
        "synthetic_mc_stress_jsonl": str(stress_path),
    }
    with (output_dir / "calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary
