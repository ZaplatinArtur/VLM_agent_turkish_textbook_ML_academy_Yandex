from __future__ import annotations

import json
import hashlib
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from .ingest import read_records


def _difficulty(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return next((value for value in ("easy", "medium", "hard") if metadata.get(value)), "unlabeled")


def _stable_key(record: dict[str, Any], seed: str) -> str:
    value = f"{seed}:{record.get('annotation_id') or record.get('task_id')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare_pointwise_annotation_dataset(
    input_paths: Iterable[Path],
    output_path: Path,
    *,
    seed: str = "pointwise-ui-v1",
) -> dict[str, Any]:
    """Combine and shuffle setup runs so the UI cannot reveal run blocks by order."""

    paths = list(input_paths)
    if not paths:
        raise ValueError("at least one input run is required")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    setup_counts: Counter[str] = Counter()
    for path in paths:
        for source in read_records(path):
            record = dict(source)
            task_id = str(record.get("task_id") or "").strip()
            setup = str(record.get("setup") or "unknown").strip()
            if not task_id:
                raise ValueError(f"record in {path} is missing task_id")
            if "candidate_answer" not in record:
                raise ValueError(f"record {task_id} in {path} is missing candidate_answer")
            identifier = str(record.get("annotation_id") or f"{task_id}::{setup}")
            if identifier in seen:
                raise ValueError(f"duplicate pointwise annotation_id: {identifier}")
            seen.add(identifier)
            record["annotation_id"] = identifier
            records.append(record)
            setup_counts[setup] += 1

    random.Random(seed).shuffle(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "input_files": [str(path) for path in paths],
        "records": len(records),
        "unique_annotation_ids": len(seen),
        "setup_counts": dict(sorted(setup_counts.items())),
        "seed": seed,
        "output": str(output_path),
    }


def sample_calibration_responses(
    input_paths: Iterable[Path],
    output_path: Path,
    *,
    size: int = 120,
    seed: str = "response-calibration-v1",
) -> dict[str, Any]:
    """Select a setup-balanced, stratified set of candidate answers for humans."""

    paths = list(input_paths)
    if size < 1:
        raise ValueError("size must be positive")
    if not paths:
        raise ValueError("at least one input run is required")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_setup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for source in read_records(path):
            record = dict(source)
            task_id = str(record.get("task_id") or "").strip()
            setup = str(record.get("setup") or "unknown").strip()
            if not task_id or "candidate_answer" not in record:
                raise ValueError(f"invalid candidate record in {path}: task_id/candidate_answer required")
            identifier = str(record.get("annotation_id") or f"{task_id}::{setup}")
            if identifier in seen:
                raise ValueError(f"duplicate pointwise annotation_id: {identifier}")
            seen.add(identifier)
            record["annotation_id"] = identifier
            records.append(record)
            by_setup[setup].append(record)
    if size > len(records):
        raise ValueError(f"requested {size} responses, only {len(records)} are available")

    setups = sorted(by_setup)
    base_quota, remainder = divmod(size, len(setups))
    quotas = {
        setup: base_quota + int(index < remainder)
        for index, setup in enumerate(setups)
    }
    selected: list[dict[str, Any]] = []
    for setup in setups:
        quota = quotas[setup]
        candidates = by_setup[setup]
        if len(candidates) < quota:
            raise ValueError(f"setup {setup} has {len(candidates)} records, needs {quota}")
        buckets: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
        for record in sorted(candidates, key=lambda value: _stable_key(value, seed)):
            key = (
                str(record.get("answer_type") or "unknown"),
                str(record.get("grade") or "unknown"),
                _difficulty(record),
            )
            buckets[key].append(record)
        bucket_keys = sorted(buckets)
        picked = 0
        while picked < quota:
            progress = False
            for key in bucket_keys:
                if buckets[key] and picked < quota:
                    selected.append(buckets[key].popleft())
                    picked += 1
                    progress = True
            if not progress:
                raise RuntimeError(f"unable to fill calibration quota for {setup}")

    selected.sort(key=lambda value: _stable_key(value, seed + ":final"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "input_files": [str(path) for path in paths],
        "available_records": len(records),
        "selected_records": len(selected),
        "setup_quotas": quotas,
        "selected_by_setup": dict(sorted(Counter(str(value.get("setup")) for value in selected).items())),
        "selected_by_answer_type": dict(sorted(Counter(str(value.get("answer_type")) for value in selected).items())),
        "seed": seed,
        "output": str(output_path),
    }
