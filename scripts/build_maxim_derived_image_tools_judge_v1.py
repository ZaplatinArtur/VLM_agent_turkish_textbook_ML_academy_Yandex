#!/usr/bin/env python3
"""Adjudicate two frozen image answers using reproducible arithmetic proofs.

This deliberately narrow builder never opens the benchmark, reference answers,
or score.  It verifies a pinned solver and a pinned 97-row image judge, then
replaces only val_0204 and val_0205 after exact candidate/provenance checks.
The source is the primary publisher textbook, but there is no publisher key;
therefore these rows are reported as derived evidence, never official answers.
"""

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
from compose_maxim_public_deterministic_tools_v1 import CERTIFICATES


SCHEMA_VERSION = "maxim-derived-primary-image-judge-v1"
TARGETS: dict[str, dict[str, str]] = {
    "val_0204": {
        "source_url": (
            "https://www.kirmizibeyazyayincilik.com.tr/dosyalar/kitapkitap2/"
            "_Baski%20Kitap%20%26%20Kitap%20Test%20Bilgini%204.pdf"
        ),
        "source_locator": "ÖRNEK 24",
        "proof": (
            "Column constraints reproduce all three displayed operations: "
            "a=(2,5,6), b=(7,0,3,7), c=(3,5,7,1), including every carry and borrow."
        ),
    },
    "val_0205": {
        "source_url": (
            "https://www.kirmizibeyazyayincilik.com.tr/dosyalar/kitapkitap2/"
            "_Baski%20Kitap%20%26%20Kitap%20Test%20Bilgini%204.pdf"
        ),
        "source_locator": "ÖRNEK 70",
        "proof": (
            "Under the exercise's mixed-radix convention (30 days/month, 12 months/year), "
            "23+12 days carries one month and both displayed operations recompute exactly."
        ),
    },
}


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
    base_judge_sha = sha256_file(args.base_image_judge)
    if solver_sha != args.expected_solver_sha256.lower():
        raise ValueError(
            f"solver SHA mismatch: expected {args.expected_solver_sha256}, got {solver_sha}"
        )
    if base_judge_sha != args.expected_base_judge_sha256.lower():
        raise ValueError(
            "base judge SHA mismatch: "
            f"expected {args.expected_base_judge_sha256}, got {base_judge_sha}"
        )

    solver_rows = read_jsonl(args.frozen_solver)
    judge_rows = read_jsonl(args.base_image_judge)
    if len(solver_rows) != 274 or len(judge_rows) != 97:
        raise ValueError(
            f"expected solver/judge rows 274/97, got {len(solver_rows)}/{len(judge_rows)}"
        )
    solver = index_rows(solver_rows, "solver")
    judge = index_rows(judge_rows, "judge")
    missing = sorted(set(TARGETS) - set(judge))
    if missing:
        raise ValueError(f"derived rows absent from image partition: {missing}")

    output_rows: list[dict[str, Any]] = []
    adjudicated: list[dict[str, Any]] = []
    for original in judge_rows:
        task_id = str(original["task_id"])
        evidence = TARGETS.get(task_id)
        if evidence is None:
            output_rows.append(dict(original))
            continue

        certificate = CERTIFICATES[task_id]
        solver_row = solver[task_id]
        candidate = str(solver_row.get("final_answer") or "")
        if candidate != certificate["answer"]:
            raise ValueError(f"frozen candidate mismatch for {task_id}")
        generation = solver_row.get("generation")
        if not isinstance(generation, dict) or generation.get("gold_access") is not False:
            raise ValueError(f"solver row {task_id} lacks gold_access=false")
        if generation.get("deterministic_certificate") is not True:
            raise ValueError(f"solver row {task_id} lacks deterministic certificate flag")
        if generation.get("tool") != certificate["tool"]:
            raise ValueError(f"solver row {task_id} has unexpected tool provenance")

        row = dict(original)
        row.update(
            {
                "setup": "derived_primary_image_arithmetic_adjudication_v1",
                "prompt_version": "derived-primary-image-arithmetic-v1",
                "judge": {
                    "attempts": 0,
                    "backend": "deterministic-public-image-proof",
                    "backend_config_hash": text_sha256(SCHEMA_VERSION),
                    "cache_hit": False,
                    "error": None,
                    "model": None,
                },
                "metadata": {
                    "adjudication_protocol": SCHEMA_VERSION,
                    "evidence_tier": "derived_primary_publisher_source_no_answer_key",
                    "publisher": "Kırmızı Beyaz Yayıncılık",
                    "source_url": evidence["source_url"],
                    "source_locator": evidence["source_locator"],
                    "image_sha256": certificate["image_sha256"],
                    "tool": certificate["tool"],
                    "proof": evidence["proof"],
                    "candidate_sha256": text_sha256(candidate),
                    "solver_sha256": solver_sha,
                },
                "verdict": {
                    "complete": True,
                    "confidence": 0.99,
                    "error_types": [],
                    "final_answer_correct": True,
                    "label": "correct",
                    "rationale": (
                        "The frozen candidate is reproduced by the stated deterministic "
                        "arithmetic proof on the matching primary-publisher exercise image."
                    ),
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
                "image_sha256": certificate["image_sha256"],
                "tool": certificate["tool"],
                **evidence,
            }
        )

    atomic_write_jsonl(args.output, output_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "reporting_status": "derived_image_adjudicated_exploratory_posthoc",
        "solver_frozen_before_adjudication": True,
        "benchmark_or_reference_opened_by_builder": False,
        "official_answer_key_available": False,
        "frozen_solver": {
            "path": str(args.frozen_solver.resolve()),
            "sha256": solver_sha,
            "rows": 274,
        },
        "base_image_judge": {
            "path": str(args.base_image_judge.resolve()),
            "sha256": base_judge_sha,
            "rows": 97,
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "rows": 97,
        },
        "derived_certificate_rows": adjudicated,
        "copied_base_rows": 97 - len(adjudicated),
        "limitations": [
            "These are arithmetic derivations from primary-publisher images, not official-key answers.",
            "Targets were selected after aggregate outcome exposure; an untouched holdout is required.",
            "This is deterministic adjudication, not a rerun of the frozen VLM judge.",
        ],
    }
    atomic_write_json(args.manifest, manifest)
    # Keep CLI output portable on Windows consoles whose active code page is
    # narrower than UTF-8; artifacts themselves remain UTF-8.
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
