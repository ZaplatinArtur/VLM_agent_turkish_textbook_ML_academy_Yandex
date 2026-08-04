#!/usr/bin/env python3
"""Fail-close one pinned frozen-judge invalid-JSON result and cache it.

The recovery never adjudicates the candidate and always grants zero strict
credit.  It exists only so a completed frozen evaluation is not blocked by a
repeatable transport/serialization failure from the judge backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_ERROR = "ValueError: judge did not return valid JSON"
SCHEMA_VERSION = "maxim-judge-invalid-json-failclosed-recovery-v1"
POLICY = "fail_closed_no_positive_credit"
FROZEN_BACKEND_HASH = (
    "e3f71b4af7fa8ad8a6db755d43bdf4a895d087b701436c105da8c5416804fbd9"
)


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
            rows.append(value)
    ids = [str(row.get("task_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("missing or duplicate task_id")
    return rows


def verdict() -> dict[str, Any]:
    return {
        "label": "incorrect",
        "score": 0,
        "strict_correct": False,
        "final_answer_correct": False,
        "reasoning_correct": False,
        "complete": False,
        "confidence": 0.0,
        "error_types": ["judge_transport_recovery", "judge_invalid_json"],
        "rationale": (
            "Mechanical fail-closed recovery of a repeatable frozen-judge "
            "invalid-JSON failure; the candidate receives no positive credit."
        ),
        "reference_quality_issue": False,
    }


def recover(
    *,
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    cache_dir: Path,
    expected_input_sha256: str,
    expected_task_id: str,
) -> dict[str, Any]:
    for target in (output_path, audit_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite: {target}")
    actual_sha = sha256_file(input_path)
    if actual_sha != expected_input_sha256:
        raise ValueError(
            f"input SHA256 mismatch: expected={expected_input_sha256}, actual={actual_sha}"
        )
    rows = load_jsonl(input_path)
    failures: list[tuple[str, Any, Any]] = []
    for row in rows:
        judge = row.get("judge")
        error = judge.get("error") if isinstance(judge, dict) else None
        if row.get("error") or error or not isinstance(row.get("verdict"), dict):
            failures.append((str(row["task_id"]), error, row.get("verdict")))
    expected = [(expected_task_id, EXPECTED_ERROR, None)]
    if failures != expected:
        raise ValueError(f"refusing recovery: expected failures={expected!r}, actual={failures!r}")

    row = next(row for row in rows if row["task_id"] == expected_task_id)
    judge = row["judge"]
    assert isinstance(judge, dict)
    if judge.get("backend_config_hash") != FROZEN_BACKEND_HASH:
        raise ValueError("frozen backend hash mismatch")
    cache_key = str(judge.get("cache_key") or "")
    if len(cache_key) != 64 or any(c not in "0123456789abcdef" for c in cache_key):
        raise ValueError("invalid cache key")
    recovered_verdict = verdict()
    original_metadata = judge.get("response_metadata")
    original_attempts = judge.get("attempts")
    row["verdict"] = recovered_verdict
    judge["error"] = None
    judge["transport_recovery"] = {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "original_error": EXPECTED_ERROR,
        "strict_correct_credit": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp-recovery")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for value in rows:
            sink.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, output_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        raise FileExistsError(f"refusing to overwrite cache: {cache_path}")
    cache_record = {
        "request_id": row.get("request_id"),
        "prompt_version": row.get("prompt_version"),
        "backend": judge.get("backend"),
        "model": judge.get("model"),
        "backend_config": judge.get("backend_config"),
        "backend_config_hash": judge.get("backend_config_hash"),
        "raw_response": json.dumps(recovered_verdict, ensure_ascii=False),
        "backend_metadata": original_metadata,
        "verdict": recovered_verdict,
        "transport_recovery": judge["transport_recovery"],
    }
    cache_tmp = cache_path.with_name(cache_path.name + ".tmp-recovery")
    cache_tmp.write_text(
        json.dumps(cache_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(cache_tmp, cache_path)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "input": {"path": str(input_path.resolve()), "sha256": actual_sha, "rows": len(rows)},
        "task": {
            "task_id": expected_task_id,
            "original_error": EXPECTED_ERROR,
            "original_attempts": original_attempts,
            "original_response_metadata": original_metadata,
        },
        "metric_effect": {"strict_correct_credit": 0, "positive_credit_added": False},
        "output": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path)},
        "cache": {"path": str(cache_path.resolve()), "sha256": sha256_file(cache_path)},
    }
    audit_tmp = audit_path.with_name(audit_path.name + ".tmp-recovery")
    audit_tmp.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(audit_tmp, audit_path)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-task-id", required=True)
    args = parser.parse_args()
    result = recover(
        input_path=args.input,
        output_path=args.output,
        audit_path=args.audit,
        cache_dir=args.cache_dir,
        expected_input_sha256=args.expected_input_sha256,
        expected_task_id=args.expected_task_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
