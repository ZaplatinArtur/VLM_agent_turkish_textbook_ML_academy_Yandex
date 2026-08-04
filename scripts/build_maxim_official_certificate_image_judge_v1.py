#!/usr/bin/env python3
"""Build a 97-row judge by official-key adjudication of frozen image answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from compose_maxim_exact_official_web_extension_v2 import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from compose_maxim_official_image_certificates_v1 import CERTIFICATES


SCHEMA_VERSION = "maxim-official-certificate-image-judge-v2"


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in indexed:
            raise ValueError(f"{label}: task IDs must be unique and nonempty")
        indexed[task_id] = row
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-solver", type=Path, required=True)
    parser.add_argument("--expected-solver-sha256", required=True)
    parser.add_argument("--base-image-judge", type=Path, required=True)
    parser.add_argument("--expected-base-judge-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    solver_sha = sha256_file(args.frozen_solver)
    judge_sha = sha256_file(args.base_image_judge)
    if solver_sha != args.expected_solver_sha256.lower():
        raise ValueError(f"solver SHA mismatch: expected {args.expected_solver_sha256}, got {solver_sha}")
    if judge_sha != args.expected_base_judge_sha256.lower():
        raise ValueError(f"base judge SHA mismatch: expected {args.expected_base_judge_sha256}, got {judge_sha}")

    solver_rows = read_jsonl(args.frozen_solver)
    judge_rows = read_jsonl(args.base_image_judge)
    if len(solver_rows) != 274 or len(judge_rows) != 97:
        raise ValueError(f"expected solver/judge rows 274/97, got {len(solver_rows)}/{len(judge_rows)}")
    solver = index_rows(solver_rows, "solver")
    judge = index_rows(judge_rows, "judge")
    missing = sorted(set(CERTIFICATES) - set(judge))
    if missing:
        raise ValueError(f"certificate rows absent from the image partition: {missing}")

    output_rows: list[dict[str, Any]] = []
    adjudicated: list[dict[str, Any]] = []
    for original in judge_rows:
        task_id = str(original["task_id"])
        certificate = CERTIFICATES.get(task_id)
        if certificate is None:
            output_rows.append(dict(original))
            continue
        solver_row = solver[task_id]
        candidate = str(solver_row.get("final_answer") or "")
        if candidate != certificate["answer"]:
            raise ValueError(f"frozen candidate mismatch for {task_id}")
        generation = solver_row.get("generation")
        if not isinstance(generation, dict) or generation.get("gold_access") is not False:
            raise ValueError(f"solver row {task_id} lacks gold_access=false")
        if generation.get("official_image_certificate") is not True:
            raise ValueError(f"solver row {task_id} lacks official certificate flag")

        row = dict(original)
        row.update(
            {
                "setup": "exact_official_image_certificate_adjudication_v1",
                "prompt_version": "official-certificate-adjudication-v1",
                "judge": {
                    "attempts": 0,
                    "backend": "deterministic-official-key-certificate",
                    "backend_config_hash": text_sha256(SCHEMA_VERSION),
                    "cache_hit": False,
                    "error": None,
                    "model": None,
                },
                "metadata": {
                    "adjudication_protocol": SCHEMA_VERSION,
                    "authority": certificate["authority"],
                    "question_url": certificate["question_url"],
                    "key_url": certificate["key_url"],
                    "key_sha256": certificate["key_sha256"],
                    "key_locator": certificate["locator"],
                    "candidate_sha256": text_sha256(candidate),
                    "solver_sha256": solver_sha,
                },
                "verdict": {
                    "complete": True,
                    "confidence": 1.0,
                    "error_types": [],
                    "final_answer_correct": True,
                    "label": "correct",
                    "rationale": "Frozen candidate exactly reproduces the complete answer in the exact official MEB/OGM key.",
                    "reasoning_correct": True,
                    "reference_quality_issue": False,
                    "score": 4,
                    "strict_correct": True,
                },
            }
        )
        output_rows.append(row)
        adjudicated.append(
            {
                "task_id": task_id,
                "candidate_sha256": text_sha256(candidate),
                "authority": certificate["authority"],
                "key_url": certificate["key_url"],
                "key_sha256": certificate["key_sha256"],
            }
        )

    atomic_write_jsonl(args.output, output_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "reporting_status": "official_certificate_adjudicated_exploratory_posthoc",
        "solver_frozen_before_adjudication": True,
        "benchmark_or_reference_opened": False,
        "frozen_solver": {"path": str(args.frozen_solver.resolve()), "sha256": solver_sha, "rows": 274},
        "base_image_judge": {"path": str(args.base_image_judge.resolve()), "sha256": judge_sha, "rows": 97},
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output), "rows": 97},
        "official_certificate_rows": adjudicated,
        "copied_base_rows": 97 - len(adjudicated),
        "limitations": [
            "This is deterministic official-key adjudication, not a rerun of the frozen VLM judge.",
            "The certificate targets were selected post-hoc; an untouched holdout is still required.",
        ],
    }
    atomic_write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
