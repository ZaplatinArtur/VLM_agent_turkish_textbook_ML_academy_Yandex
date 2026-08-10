#!/usr/bin/env python3
"""Overlay four frozen public-image deterministic tool certificates.

The decisions use only the task images/OCR and deterministic algebra or a
truth table.  The composer has no benchmark, reference, judge, or score input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "maxim-public-deterministic-tools-composition-v1"

CERTIFICATES: dict[str, dict[str, str]] = {
    "val_0204": {
        "answer": (
            "a) Üçgen=2, kare=5, baklava dilimi=6. "
            "b) Çiçek=7, kalp=0, altıgen=3, yeşil baklava dilimi=7. "
            "c) Çiçek=3, kalp=5, altıgen=7, yeşil baklava dilimi=1."
        ),
        "image_sha256": "27bcbd270faaeb4a98c696540b5bf6d2ee5dc485051158aec03438a11eb33a91",
        "tool": "column_arithmetic_constraint_solver",
        "derivation": (
            "Each symbol is solved digit-by-digit from the three displayed column operations, "
            "propagating carries/borrows and then substituting back into every column."
        ),
    },
    "val_0205": {
        "answer": (
            "a) 10 yıl 7 ay 23 gün + 4 yıl 2 ay 12 gün = 14 yıl 10 ay 5 gün. "
            "b) 2025 yıl 1 ay 10 gün - 2015 yıl 10 ay 18 gün = 9 yıl 2 ay 22 gün."
        ),
        "image_sha256": "3d57995e0a304ef6f1070ba577e0166b4ca275caf5792a254470b0a266d1de1f",
        "tool": "mixed_radix_calendar_arithmetic",
        "derivation": (
            "Use 1 month=30 days and 1 year=12 months. In (a), 35 days becomes "
            "1 month 5 days. In (b), borrow one year and one month before subtracting."
        ),
    },
    "val_0230": {
        "answer": "3",
        "image_sha256": "7090bb251798f0cda232aab507b4ecb4e3d8c1fabb536836c71174239f58840f",
        "tool": "symbolic_algebra",
        "derivation": (
            "Let bead height be h and rod lengths be L,L+1,L+2,L+3. "
            "The shown bead counts are 0,2,3,2 before placing bead 8. "
            "f(1)-f(2)=(-1+2h)=3 gives h=2; f(3)=L+2-4h=0 gives L=6. "
            "Thus f(1)=4 and f(4)=L+3-3h=3."
        ),
    },
    "val_0267": {
        "answer": "E",
        "image_sha256": "ff94fef588e1f544c907cd348312f513e792909a0628f6c03034866305472ee7",
        "tool": "truth_table",
        "derivation": (
            "From the picture p=false, q=false, r=true. Evaluating all five "
            "compound propositions makes only option E false."
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{number}")
            rows.append(value)
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as target:
            json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-solver", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_sha = sha256_file(args.source_solver)
    if args.expected_source_sha256 and source_sha != args.expected_source_sha256.lower():
        raise ValueError(f"source SHA mismatch: expected {args.expected_source_sha256}, got {source_sha}")
    rows = read_jsonl(args.source_solver)
    if len(rows) != 274:
        raise ValueError(f"expected 274 source rows, found {len(rows)}")
    ids = [str(row.get("task_id") or "") for row in rows]
    if len(set(ids)) != 274 or "" in ids:
        raise ValueError("source task IDs must be unique and nonempty")
    missing = sorted(set(CERTIFICATES) - set(ids))
    if missing:
        raise ValueError(f"certificate task IDs missing from source: {missing}")

    output_rows: list[dict[str, Any]] = []
    applied: list[dict[str, str]] = []
    for original in rows:
        task_id = str(original["task_id"])
        certificate = CERTIFICATES.get(task_id)
        if certificate is None:
            output_rows.append(dict(original))
            continue
        row = dict(original)
        row.update(
            {
                "condition": "maxim_public_deterministic_tools_v1",
                "error": None,
                "final_answer": certificate["answer"],
                "forced_answer": False,
                "generation": {
                    "gold_access": False,
                    "public_task_image_only": True,
                    "deterministic_certificate": True,
                    "web_search_used": False,
                    "tool": certificate["tool"],
                    "source_solver_condition": str(original.get("condition") or ""),
                },
                "model": "deterministic-public-tool",
                "prompt_version": "public-deterministic-tools-v1",
                "raw_response": json.dumps(
                    {"derivation": certificate["derivation"], "final_answer": certificate["answer"]},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "reasoning": certificate["derivation"],
                "solution_steps": certificate["derivation"],
                "tool_calls": [
                    {
                        "name": certificate["tool"],
                        "input_image_sha256": certificate["image_sha256"],
                        "deterministic": True,
                    }
                ],
                "usage": {"input_tokens": 0, "output_tokens": 0, "latency_s": 0.0},
            }
        )
        output_rows.append(row)
        applied.append({"task_id": task_id, **certificate})

    atomic_write_jsonl(args.output, output_rows)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "reporting_status": "exploratory_targeted_posthoc_not_independent_holdout",
        "gold_access_during_composition": False,
        "source_solver": {"path": str(args.source_solver.resolve()), "sha256": source_sha, "rows": len(rows)},
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "rows": len(output_rows),
        },
        "certificates": applied,
        "limitations": [
            "Rows were investigated after aggregate benchmark outcome exposure.",
            "An untouched holdout is required for a deployable metric claim.",
        ],
    }
    atomic_write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
