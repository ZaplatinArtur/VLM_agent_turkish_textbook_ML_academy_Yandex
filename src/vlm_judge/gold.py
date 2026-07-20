from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ingest import read_records


_NON_APPLICABLE_QUALITIES = {"unknown", "incorrect", "unreadable"}


def _unique_strings(*groups: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for value in group:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
    return result


def apply_verified_gold(
    dataset_path: Path,
    gold_path: Path,
    output_path: Path,
    *,
    require_all: bool = False,
) -> dict[str, Any]:
    """Attach task-scoped verified transcriptions without discarding source images."""

    records = read_records(dataset_path)
    gold_records = read_records(gold_path)
    gold_by_task: dict[str, dict[str, Any]] = {}
    for value in gold_records:
        task_id = str(value.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("gold record is missing task_id")
        if task_id in gold_by_task:
            raise ValueError(f"duplicate gold task_id: {task_id}")
        gold_by_task[task_id] = value

    dataset_ids = {str(value.get("task_id") or "").strip() for value in records}
    unknown_gold_ids = sorted(set(gold_by_task) - dataset_ids)
    applied = 0
    missing_verified = 0
    invalid_quality = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for source in records:
            record = dict(source)
            task_id = str(record.get("task_id") or "").strip()
            gold = gold_by_task.get(task_id)
            metadata = dict(record.get("metadata") or {})
            if not gold or gold.get("status") != "verified":
                missing_verified += 1
            else:
                quality = str(gold.get("quality") or "unknown")
                metadata["gold_quality"] = quality
                notes = str(gold.get("notes") or "").strip()
                if notes:
                    metadata["reference_notes"] = notes
                subanswers = _unique_strings(list(gold.get("subanswers") or []))
                if subanswers:
                    metadata["required_subanswers"] = subanswers

                if quality in _NON_APPLICABLE_QUALITIES:
                    invalid_quality += 1
                else:
                    transcription = str(gold.get("transcription") or "").strip()
                    if transcription:
                        metadata["source_reference_answer_before_gold"] = record.get("reference_answer")
                        record["reference_answer"] = transcription
                    elif subanswers:
                        metadata["source_reference_answer_before_gold"] = record.get("reference_answer")
                        record["reference_answer"] = "\n".join(subanswers)
                    record["acceptable_answers"] = _unique_strings(
                        list(record.get("acceptable_answers") or []),
                        list(gold.get("acceptable_answers") or []),
                    )
                    applied += 1
                metadata["verified_gold_updated_at"] = gold.get("updated_at")
            record["metadata"] = metadata
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if require_all and missing_verified:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"{missing_verified} dataset records do not have verified gold")

    return {
        "dataset_records": len(records),
        "gold_records": len(gold_records),
        "applied_verified_gold": applied,
        "missing_verified_gold": missing_verified,
        "invalid_gold_quality_not_applied": invalid_quality,
        "unknown_gold_task_ids": len(unknown_gold_ids),
        "unknown_gold_task_id_examples": unknown_gold_ids[:25],
        "output": str(output_path),
    }
