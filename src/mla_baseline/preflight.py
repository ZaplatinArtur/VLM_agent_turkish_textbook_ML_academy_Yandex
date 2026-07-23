"""Validate that an evaluation task file is runnable and judgeable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QUESTION_PLACEHOLDERS = {
    "",
    "(soru görselde)",
    "[image omitted in text-only smoke test]",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def inspect_tasks(tasks: list[dict[str, Any]], data_root: Path) -> dict[str, Any]:
    missing_local_images: list[dict[str, str]] = []
    local_images = 0
    remote_images = 0
    text_questions = 0
    missing_references = 0
    for task in tasks:
        question = str(task.get("question") or "").strip()
        if question.casefold() not in QUESTION_PLACEHOLDERS:
            text_questions += 1
        if not str(task.get("reference_answer") or "").strip():
            missing_references += 1
        images = task.get("question_images")
        images = images if isinstance(images, list) else []
        for image in images:
            if not isinstance(image, dict):
                continue
            if image.get("format") == "file_path":
                local_images += 1
                path = Path(str(image.get("data") or ""))
                if not path.is_absolute():
                    path = data_root / path
                if not path.is_file():
                    missing_local_images.append(
                        {
                            "task_id": str(task.get("task_id") or ""),
                            "path": str(path),
                        }
                    )
            elif image.get("format") in {"url", "base64"}:
                remote_images += 1
    return {
        "tasks": len(tasks),
        "text_questions": text_questions,
        "placeholder_questions": len(tasks) - text_questions,
        "local_image_refs": local_images,
        "remote_or_embedded_image_refs": remote_images,
        "missing_local_images": len(missing_local_images),
        "missing_local_image_examples": missing_local_images[:10],
        "missing_references": missing_references,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--require-question-text",
        action="store_true",
        help="fail if any task has only an image placeholder (required by text judge)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="ignore missing image assets but require real question text",
    )
    args = parser.parse_args(argv)
    report = inspect_tasks(_read_jsonl(args.tasks), args.data_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failures: list[str] = []
    if report["missing_references"]:
        failures.append(f"missing references: {report['missing_references']}")
    if report["missing_local_images"] and not args.text_only:
        failures.append(f"missing local images: {report['missing_local_images']}")
    if (args.require_question_text or args.text_only) and report["placeholder_questions"]:
        failures.append(
            f"question text placeholders: {report['placeholder_questions']}"
        )
    if failures:
        print("PRECHECK FAILED: " + "; ".join(failures))
        return 2
    print("PRECHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
