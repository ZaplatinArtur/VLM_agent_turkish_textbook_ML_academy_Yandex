#!/usr/bin/env python3
"""Materialize a fail-closed schema recovery as a frozen judge cache entry.

This utility is intentionally narrow.  It accepts only an artifact produced by
``recover_nonstrict_judge_schema_error_v1.py``, verifies the recovery audit and
the exact recovered row, and writes the corresponding response-cache entry.
The recovered verdict is non-strict, so materialization can never add positive
credit to the binary benchmark metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RECOVERY_SCHEMA = "maxim-judge-nonstrict-schema-recovery-v1"
RECOVERY_POLICY = "fail_closed_no_positive_credit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def materialize(
    *,
    recovered_path: Path,
    recovery_audit_path: Path,
    cache_dir: Path,
    materialization_audit_path: Path,
    expected_recovered_sha256: str,
    expected_task_id: str,
) -> dict[str, Any]:
    if materialization_audit_path.exists():
        raise FileExistsError(
            f"refusing to overwrite audit: {materialization_audit_path}"
        )
    actual_recovered_sha256 = sha256_file(recovered_path)
    if actual_recovered_sha256 != expected_recovered_sha256:
        raise ValueError(
            "recovered SHA256 mismatch: "
            f"expected={expected_recovered_sha256}, actual={actual_recovered_sha256}"
        )

    recovery_audit = read_object(recovery_audit_path)
    if recovery_audit.get("schema_version") != RECOVERY_SCHEMA:
        raise ValueError("recovery audit schema mismatch")
    if recovery_audit.get("policy") != RECOVERY_POLICY:
        raise ValueError("recovery audit policy is not fail-closed")
    audit_output = recovery_audit.get("output")
    if not isinstance(audit_output, dict) or (
        audit_output.get("sha256") != actual_recovered_sha256
    ):
        raise ValueError("recovery audit does not bind the recovered artifact")
    audit_task = recovery_audit.get("recovered_task")
    if not isinstance(audit_task, dict) or (
        audit_task.get("task_id") != expected_task_id
        or audit_task.get("inferred_strict_correct") is not False
    ):
        raise ValueError("recovery audit does not bind the expected non-strict task")

    rows = read_jsonl(recovered_path)
    targets = [row for row in rows if row.get("task_id") == expected_task_id]
    if len(targets) != 1:
        raise ValueError("recovered artifact must contain the expected task exactly once")
    row = targets[0]
    verdict = row.get("verdict")
    judge = row.get("judge")
    if not isinstance(verdict, dict) or not isinstance(judge, dict):
        raise ValueError("recovered task has no verdict/judge provenance")
    schema_recovery = judge.get("schema_recovery")
    if not isinstance(schema_recovery, dict) or (
        schema_recovery.get("schema_version") != RECOVERY_SCHEMA
        or schema_recovery.get("policy") != RECOVERY_POLICY
    ):
        raise ValueError("task is not a pinned fail-closed schema recovery")
    if verdict.get("strict_correct") is not False:
        raise ValueError("refusing to cache a recovery with positive strict credit")
    if judge.get("error") is not None:
        raise ValueError("recovered judge error is not cleared")

    cache_key = str(judge.get("cache_key") or "")
    if len(cache_key) != 64 or any(c not in "0123456789abcdef" for c in cache_key):
        raise ValueError("invalid recovered cache key")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        raise FileExistsError(f"refusing to overwrite cache entry: {cache_path}")

    cache_record = {
        "request_id": row.get("request_id"),
        "prompt_version": row.get("prompt_version"),
        "backend": judge.get("backend"),
        "model": judge.get("model"),
        "backend_config": judge.get("backend_config"),
        "backend_config_hash": judge.get("backend_config_hash"),
        "raw_response": json.dumps(verdict, ensure_ascii=False),
        "backend_metadata": judge.get("response_metadata"),
        "verdict": verdict,
        "schema_recovery": schema_recovery,
    }
    temporary_cache = cache_path.with_name(cache_path.name + ".tmp-recovery")
    temporary_cache.write_text(
        json.dumps(cache_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_cache, cache_path)

    audit = {
        "schema_version": "maxim-judge-recovery-cache-materialization-v1",
        "policy": RECOVERY_POLICY,
        "metric_effect": {
            "strict_correct_credit": 0,
            "positive_credit_added": False,
        },
        "source": {
            "recovered_path": str(recovered_path.resolve()),
            "recovered_sha256": actual_recovered_sha256,
            "recovery_audit_path": str(recovery_audit_path.resolve()),
            "recovery_audit_sha256": sha256_file(recovery_audit_path),
        },
        "task_id": expected_task_id,
        "cache": {
            "path": str(cache_path.resolve()),
            "sha256": sha256_file(cache_path),
            "cache_key": cache_key,
        },
    }
    materialization_audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_audit = materialization_audit_path.with_name(
        materialization_audit_path.name + ".tmp-recovery"
    )
    temporary_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_audit, materialization_audit_path)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--recovery-audit", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-recovered-sha256", required=True)
    parser.add_argument("--expected-task-id", required=True)
    args = parser.parse_args()
    audit = materialize(
        recovered_path=args.recovered,
        recovery_audit_path=args.recovery_audit,
        cache_dir=args.cache_dir,
        materialization_audit_path=args.audit,
        expected_recovered_sha256=args.expected_recovered_sha256,
        expected_task_id=args.expected_task_id,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
