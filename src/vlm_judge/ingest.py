from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ANSWER_FIELD_CANDIDATES = (
    "candidate_answer",
    "answer",
    "response",
    "result",
    "output",
    "text",
)


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} of {path} is not an object")
                records.append(value)
        return records
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict):
            value = value.get("records") or value.get("results") or value.get("items")
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("JSON input must be an array of objects or contain records/results/items")
        return list(value)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    raise ValueError(f"unsupported input format: {path.suffix}")


def _answer_value(record: dict[str, Any], requested_field: str | None) -> tuple[str, str | None]:
    fields = (requested_field,) if requested_field else ANSWER_FIELD_CANDIDATES
    for field in fields:
        if field and field in record:
            value = record.get(field)
            return ("" if value is None else str(value), field)
    return "", None


def import_candidates(
    benchmark_path: Path,
    responses_path: Path,
    output_path: Path,
    *,
    setup: str,
    id_field: str = "task_id",
    answer_field: str | None = None,
) -> dict[str, Any]:
    benchmark_records = read_records(benchmark_path)
    response_records = read_records(responses_path)
    benchmark = {str(record.get("task_id")): record for record in benchmark_records}
    seen: set[str] = set()
    duplicate_response_ids: set[str] = set()
    unknown_task_ids: list[str] = []
    empty_responses = 0
    imported: list[dict[str, Any]] = []

    for response in response_records:
        task_id = str(response.get(id_field) or "").strip()
        if not task_id:
            raise ValueError(f"response record is missing id field {id_field!r}")
        if task_id in seen:
            duplicate_response_ids.add(task_id)
            continue
        seen.add(task_id)
        task = benchmark.get(task_id)
        if task is None:
            unknown_task_ids.append(task_id)
            continue
        answer, detected_field = _answer_value(response, answer_field)
        failed = not answer.strip()
        if failed:
            empty_responses += 1
            answer = "[EMPTY_RESPONSE]"
        metadata = dict(task.get("metadata") or {})
        metadata.update(
            {
                "agent_run": {
                    key: value
                    for key, value in response.items()
                    if key not in {id_field, detected_field}
                },
                "answer_source_field": detected_field,
                "agent_failure": failed,
            }
        )
        imported.append(
            {
                **task,
                "candidate_answer": answer,
                "setup": setup,
                "metadata": metadata,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in imported:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    missing_task_ids = sorted(set(benchmark) - seen)
    return {
        "benchmark_records": len(benchmark_records),
        "response_records": len(response_records),
        "imported": len(imported),
        "empty_responses_preserved": empty_responses,
        "duplicate_response_ids": sorted(duplicate_response_ids),
        "unknown_task_ids": sorted(unknown_task_ids),
        "missing_task_ids": missing_task_ids,
        "output": str(output_path),
    }

