from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


LEGACY_TASK_ID = re.compile(r"validation_sheet1_r(\d{4})")
KNOWN_SETUP_NAMES = {
    "b0_no_tools": "no_tools",
    "b1_search": "web_search",
    "agent_rag": "textbook_retrieval",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_by_task(
    rows: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"{label} contains a row without task_id")
        if task_id in indexed:
            raise ValueError(f"duplicate task_id in {label}: {task_id}")
        indexed[task_id] = row
    return indexed


def canonical_legacy_id(task_id: str) -> str:
    match = LEGACY_TASK_ID.fullmatch(task_id)
    if match is None:
        raise ValueError(f"unexpected legacy task_id: {task_id}")
    return f"val_{match.group(1)}"


def candidate_text(result: dict[str, Any]) -> str:
    """Match the frozen judge-v2 adapter's candidate serialization exactly."""
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


def validate_template(row: dict[str, Any], label: str) -> None:
    required = {
        "task_id",
        "candidate_answer",
        "subject",
        "grade",
        "answer_type",
        "setup",
        "question_text",
        "question_image_url",
        "reference_answer",
        "reference_image_url",
        "acceptable_answers",
        "metadata",
    }
    if set(row) != required:
        raise ValueError(
            f"{label} template keys changed; missing={sorted(required - set(row))}, "
            f"extra={sorted(set(row) - required)}"
        )
    for field in ("question_image_url", "reference_image_url"):
        value = str(row.get(field) or "").strip()
        if not value or not Path(value).is_file():
            raise ValueError(f"{label} has missing {field}: {value}")
    if row.get("reference_answer") not in (None, ""):
        raise ValueError(f"{label} is not an image-reference template")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact frozen judge-v2 input shape for the common full274 "
            "image route: 80 legacy templates plus 17 delta templates."
        )
    )
    parser.add_argument("--app-manifest", type=Path, required=True)
    parser.add_argument("--legacy-template", type=Path, required=True)
    parser.add_argument("--delta-template", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setup", help="override output setup label for all 97 rows")
    parser.add_argument("--expected-common", type=int, default=274)
    parser.add_argument("--expected-legacy-images", type=int, default=80)
    parser.add_argument("--expected-delta-images", type=int, default=17)
    args = parser.parse_args()

    app_rows = read_jsonl(args.app_manifest)
    app_by_id = unique_by_task(app_rows, "app manifest")
    if len(app_rows) != args.expected_common:
        raise ValueError(
            f"expected {args.expected_common} common tasks, found {len(app_rows)}"
        )

    result_rows = read_jsonl(args.results)
    results_by_id = unique_by_task(result_rows, "results")
    missing_results = sorted(set(app_by_id) - set(results_by_id))
    unknown_results = sorted(set(results_by_id) - set(app_by_id))
    if missing_results or unknown_results:
        raise ValueError(
            "results must contain exactly the common benchmark IDs; "
            f"missing={missing_results[:10]}, unknown={unknown_results[:10]}"
        )

    templates_by_id: dict[str, dict[str, Any]] = {}
    legacy_count = 0
    for source in read_jsonl(args.legacy_template):
        if not source.get("reference_image_url"):
            continue
        task_id = canonical_legacy_id(str(source["task_id"]))
        if task_id not in app_by_id:
            raise ValueError(f"legacy template is outside common benchmark: {task_id}")
        row = dict(source)
        row["task_id"] = task_id
        validate_template(row, f"legacy {task_id}")
        templates_by_id[task_id] = row
        legacy_count += 1

    delta_rows = read_jsonl(args.delta_template)
    unique_by_task(delta_rows, "delta template")
    for source in delta_rows:
        task_id = str(source["task_id"])
        if task_id not in app_by_id:
            raise ValueError(f"delta template is outside common benchmark: {task_id}")
        if task_id in templates_by_id:
            raise ValueError(f"duplicate template task: {task_id}")
        row = dict(source)
        validate_template(row, f"delta {task_id}")
        templates_by_id[task_id] = row

    if legacy_count != args.expected_legacy_images:
        raise ValueError(
            f"expected {args.expected_legacy_images} legacy templates, found {legacy_count}"
        )
    if len(delta_rows) != args.expected_delta_images:
        raise ValueError(
            f"expected {args.expected_delta_images} delta templates, found {len(delta_rows)}"
        )

    ordered_ids = [str(row["task_id"]) for row in app_rows if row["task_id"] in templates_by_id]
    expected_images = args.expected_legacy_images + args.expected_delta_images
    if len(ordered_ids) != expected_images or len(set(ordered_ids)) != expected_images:
        raise ValueError(
            f"expected {expected_images} unique image tasks, found {len(ordered_ids)}"
        )
    if "val_0058" not in templates_by_id:
        raise ValueError("canonical image-route exception val_0058 is missing")

    output_rows: list[dict[str, Any]] = []
    failures = 0
    for task_id in ordered_ids:
        result = results_by_id[task_id]
        row = dict(templates_by_id[task_id])
        row["candidate_answer"] = candidate_text(result)
        condition = str(result.get("condition") or "unknown")
        row["setup"] = args.setup or KNOWN_SETUP_NAMES.get(condition, condition)
        output_rows.append(row)
        failures += int(bool(result.get("error")))

    write_jsonl(args.output, output_rows)
    report = {
        "common_tasks": len(app_rows),
        "deterministic_tasks": len(app_rows) - len(output_rows),
        "image_tasks": len(output_rows),
        "legacy_templates": legacy_count,
        "delta_templates": len(delta_rows),
        "canonical_exception": "val_0058",
        "agent_failures": failures,
        "first_image_task": ordered_ids[0],
        "last_image_task": ordered_ids[-1],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
