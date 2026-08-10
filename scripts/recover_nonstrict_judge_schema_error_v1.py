#!/usr/bin/env python3
"""Recover one provably non-strict frozen-judge schema failure.

The frozen judge validates ``strict_correct == (label == "fully_correct")``
before it can raise the specific error handled here.  Consequently the error

    mostly_correct cannot have an explicitly incorrect final answer

proves that the rejected verdict was non-strict.  This utility performs an
explicit, fail-closed canonicalization for the binary benchmark metric.  It
does not re-adjudicate the answer and cannot turn a failure into a correct
result.  The original artifact remains untouched and a JSON audit sidecar is
written next to the recovered artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_ERROR = (
    "ValueError: mostly_correct cannot have an explicitly incorrect final answer"
)
FROZEN_SCHEMA_SHA256 = (
    "d86fed78c297d0479d00ceb2d54fad502dfbfe8b49cd7be08fed2d25fc7a4e7c"
)
SCHEMA_VERSION = "maxim-judge-nonstrict-schema-recovery-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            task_id = str(value.get("task_id") or "").strip()
            if not task_id:
                raise ValueError(f"{path}:{line_number}: missing task_id")
            rows.append(value)
    task_ids = [str(row["task_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{path}: duplicate task_id")
    return rows


def canonical_fail_closed_verdict() -> dict[str, Any]:
    return {
        "label": "partially_correct",
        "score": 2,
        "strict_correct": False,
        "final_answer_correct": False,
        "reasoning_correct": None,
        "complete": None,
        "confidence": 0.0,
        "error_types": ["judge_schema_recovery", "incorrect_final_answer"],
        "rationale": (
            "Mechanical fail-closed recovery of a frozen-judge schema error. "
            "The original validation path proves strict_correct=false; richer "
            "rubric fields are not re-adjudicated."
        ),
        "reference_quality_issue": False,
    }


def recover(
    *,
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    schema_path: Path,
    expected_input_sha256: str,
    expected_task_id: str,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {audit_path}")

    actual_input_sha256 = sha256_file(input_path)
    if actual_input_sha256 != expected_input_sha256:
        raise ValueError(
            "input SHA256 mismatch: "
            f"expected={expected_input_sha256}, actual={actual_input_sha256}"
        )
    actual_schema_sha256 = sha256_file(schema_path)
    if actual_schema_sha256 != FROZEN_SCHEMA_SHA256:
        raise ValueError(
            "frozen schema SHA256 mismatch: "
            f"expected={FROZEN_SCHEMA_SHA256}, actual={actual_schema_sha256}"
        )

    rows = load_jsonl(input_path)
    failures: list[tuple[str, str | None, Any]] = []
    for row in rows:
        judge = row.get("judge")
        judge_error = judge.get("error") if isinstance(judge, dict) else None
        if row.get("error") or judge_error or not isinstance(row.get("verdict"), dict):
            failures.append((str(row["task_id"]), judge_error, row.get("verdict")))
    expected_failure = [(expected_task_id, EXPECTED_ERROR, None)]
    if failures != expected_failure:
        raise ValueError(
            "refusing recovery: failure set is not the one pinned non-strict "
            f"schema error; expected={expected_failure!r}, actual={failures!r}"
        )

    target = next(row for row in rows if str(row["task_id"]) == expected_task_id)
    judge = target["judge"]
    assert isinstance(judge, dict)  # established by exact failure-set check
    original_response_metadata = judge.get("response_metadata")
    original_attempts = judge.get("attempts")
    target["verdict"] = canonical_fail_closed_verdict()
    judge["error"] = None
    judge["schema_recovery"] = {
        "schema_version": SCHEMA_VERSION,
        "original_error": EXPECTED_ERROR,
        "frozen_schema_sha256": FROZEN_SCHEMA_SHA256,
        "inferred_strict_correct": False,
        "inference": (
            "The frozen schema checks label/score and strict_correct/label "
            "consistency before raising this exact mostly_correct/final-answer "
            "error; therefore the rejected verdict cannot be fully_correct."
        ),
        "policy": "fail_closed_no_positive_credit",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp-recovery")
    with temporary_output.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary_output, output_path)
    output_sha256 = sha256_file(output_path)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "policy": "fail_closed_no_positive_credit",
        "input": {
            "path": str(input_path.resolve()),
            "sha256": actual_input_sha256,
            "rows": len(rows),
        },
        "frozen_schema": {
            "path": str(schema_path.resolve()),
            "sha256": actual_schema_sha256,
        },
        "recovered_task": {
            "task_id": expected_task_id,
            "original_error": EXPECTED_ERROR,
            "original_attempts": original_attempts,
            "original_response_metadata": original_response_metadata,
            "inferred_strict_correct": False,
            "canonical_verdict": target["verdict"],
        },
        "metric_effect": {
            "strict_correct_credit": 0,
            "optimistic_credit_added": False,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": output_sha256,
            "rows": len(rows),
        },
    }
    temporary_audit = audit_path.with_name(audit_path.name + ".tmp-recovery")
    temporary_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_audit, audit_path)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-task-id", required=True)
    args = parser.parse_args()
    audit = recover(
        input_path=args.input,
        output_path=args.output,
        audit_path=args.audit,
        schema_path=args.schema,
        expected_input_sha256=args.expected_input_sha256,
        expected_task_id=args.expected_task_id,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
