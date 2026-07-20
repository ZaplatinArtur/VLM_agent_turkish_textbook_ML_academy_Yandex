from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .ingest import read_records


def _swap_decision(task_id: str, setup_a: str, setup_b: str, seed: str) -> bool:
    digest = hashlib.sha256(f"{seed}:{task_id}:{setup_a}:{setup_b}".encode("utf-8")).digest()
    return bool(digest[0] & 1)


def build_pairwise_records(
    records: list[dict[str, Any]],
    setup_a: str,
    setup_b: str,
    *,
    seed: str = "arena-v1",
    mirrored: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        task_id = str(record.get("task_id") or "").strip()
        setup = str(record.get("setup") or "").strip()
        if task_id and setup in {setup_a, setup_b}:
            if setup in grouped[task_id]:
                raise ValueError(f"duplicate record for task={task_id}, setup={setup}")
            grouped[task_id][setup] = record

    pairs: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        group = grouped[task_id]
        if setup_a not in group or setup_b not in group:
            continue
        first, second = group[setup_a], group[setup_b]
        swap = _swap_decision(task_id, setup_a, setup_b, seed)
        orientations = [swap, not swap] if mirrored else [swap]
        for orientation_index, should_swap in enumerate(orientations):
            left, right = (second, first) if should_swap else (first, second)
            pair_id = f"{task_id}__{setup_a}_vs_{setup_b}"
            if mirrored:
                pair_id += f"__{orientation_index + 1}"
            metadata = dict(first.get("metadata") or {})
            metadata.update(
                {
                    "candidate_a_setup": left.get("setup"),
                    "candidate_b_setup": right.get("setup"),
                    "comparison_setups": [setup_a, setup_b],
                    "side_swapped": should_swap,
                    "pair_seed": seed,
                    "mirrored": mirrored,
                }
            )
            pairs.append(
                {
                    "pair_id": pair_id,
                    "task_id": task_id,
                    "subject": first.get("subject"),
                    "grade": first.get("grade"),
                    "answer_type": first.get("answer_type"),
                    "question_text": first.get("question_text"),
                    "question_image_url": first.get("question_image_url"),
                    "reference_answer": first.get("reference_answer"),
                    "reference_image_url": first.get("reference_image_url"),
                    "candidate_a": left.get("candidate_answer"),
                    "candidate_b": right.get("candidate_answer"),
                    "metadata": metadata,
                }
            )
    return pairs


def prepare_pairwise(
    input_paths: list[Path],
    output_path: Path,
    *,
    setup_a: str,
    setup_b: str,
    seed: str = "arena-v1",
    mirrored: bool = False,
) -> dict[str, Any]:
    records = [record for path in input_paths for record in read_records(path)]
    pairs = build_pairwise_records(
        records,
        setup_a,
        setup_b,
        seed=seed,
        mirrored=mirrored,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return {
        "input_records": len(records),
        "pairs": len(pairs),
        "setup_a": setup_a,
        "setup_b": setup_b,
        "mirrored": mirrored,
        "output": str(output_path),
    }

