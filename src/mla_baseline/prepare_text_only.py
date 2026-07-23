"""Build a clean text-only Task JSONL and remove all image references."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import Task
from .preflight import QUESTION_PLACEHOLDERS


IMAGE_DEPENDENT_MARKERS = (
    "[image omitted",
    "![image",
    "(условие на картинке)",
)


def _reason_to_reject(record: dict[str, Any]) -> str | None:
    question = str(record.get("question") or "").strip()
    if question.casefold() in QUESTION_PLACEHOLDERS:
        return "placeholder_question"
    if any(marker in question.casefold() for marker in IMAGE_DEPENDENT_MARKERS):
        return "image_dependent_question"
    if not str(record.get("reference_answer") or "").strip():
        return "missing_reference"
    return None


def prepare_text_only(
    records: list[dict[str, Any]],
) -> tuple[list[Task], dict[str, Any]]:
    tasks: list[Task] = []
    rejected: Counter[str] = Counter()
    for record in records:
        reason = _reason_to_reject(record)
        if reason is not None:
            rejected[reason] += 1
            continue
        normalized = dict(record)
        normalized["question_images"] = []
        tasks.append(Task.model_validate(normalized))
    return tasks, {
        "input_records": len(records),
        "written": len(tasks),
        "rejected": dict(sorted(rejected.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.input.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    tasks, report = prepare_text_only(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as destination:
        for task in tasks:
            destination.write(task.model_dump_json() + "\n")
    report["input"] = str(args.input)
    report["output"] = str(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not tasks:
        print("TEXT-ONLY DATASET IS EMPTY")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
