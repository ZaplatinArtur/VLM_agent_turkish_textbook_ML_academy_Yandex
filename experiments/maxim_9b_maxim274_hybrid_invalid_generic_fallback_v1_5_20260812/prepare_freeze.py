"""Freeze fallback V1.5 before the candidate's atomic completion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fallback_compose as fallback


HERE = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": Path(os.path.relpath(path, HERE)).as_posix(),
        "sha256": fallback.sha256(path),
        "size": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    if any(
        path.exists()
        for path in (
            fallback.FREEZE,
            fallback.FREEZE_SHA_FILE,
            fallback.AUDIT,
            fallback.OUTPUT,
            fallback.MANIFEST,
        )
    ):
        raise RuntimeError("fallback V1.5 freeze/runtime output exists")
    if fallback.COMPLETION.exists() or fallback.PREDICTIONS.exists():
        raise RuntimeError("candidate atomic completion exists before fallback V1.5 freeze")
    fixed = {
        fallback.CANDIDATE / "EXECUTION_FREEZE.json": fallback.CANDIDATE_FREEZE_SHA,
        fallback.CANDIDATE / "INDEPENDENT_AUDIT.json": fallback.CANDIDATE_AUDIT_SHA,
        fallback.ALIGNMENT: fallback.ALIGNMENT_SHA,
        fallback.QUEUE: fallback.QUEUE_SHA,
        fallback.DECISIONS: fallback.DECISIONS_SHA,
        fallback.BASE240: fallback.BASE240_SHA,
        fallback.V14 / "FALLBACK_RULE_FREEZE.json": fallback.V14_FREEZE_SHA,
        fallback.V14 / "INDEPENDENT_AUDIT.json": fallback.V14_AUDIT_SHA,
        fallback.V14 / "fallback_compose.py": fallback.V14_IMPLEMENTATION_SHA,
    }
    if any(fallback.sha256(path) != digest for path, digest in fixed.items()):
        raise RuntimeError("fallback V1.5 static ancestry mismatch")
    v14_audit = fallback.read_json(fallback.V14 / "INDEPENDENT_AUDIT.json")
    if v14_audit.get("status") != "PASS" or v14_audit.get("freeze_sha256") != fallback.V14_FREEZE_SHA:
        raise RuntimeError("V1.4 PASS audit lineage mismatch")
    if (fallback.V14 / "FALLBACK_COMPOSITION_COMPLETION.json").exists() or (
        fallback.V14 / "runs" / "hybrid_solver_274_invalid_fallback.jsonl"
    ).exists():
        raise RuntimeError("V1.4 was not superseded preuse")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "test_fallback.py"],
        cwd=HERE,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fallback V1.5 tests failed: {result.stdout}{result.stderr}")
    if fallback.COMPLETION.exists() or fallback.PREDICTIONS.exists():
        raise RuntimeError("candidate atomic completion appeared during freeze")
    implementations = {
        path.stem: descriptor(path)
        for path in (
            HERE / "fallback_compose.py",
            HERE / "prepare_freeze.py",
            HERE / "test_fallback.py",
            HERE / "README.md",
            HERE / "INDEPENDENT_AUDIT_TEMPLATE.json",
        )
    }
    artifacts = {
        f"artifact_{number:02d}": descriptor(path)
        for number, path in enumerate(fixed, 1)
    }
    value = {
        "schema_version": "maxim-invalid-generic-fallback-rule-freeze-v1.5",
        "state": "frozen_before_candidate_completion",
        "created_utc": utc_now(),
        "rows": 274,
        "rule": {
            "certified_noid": "preserve frozen Hybrid V3.1 certified answer",
            "generic_invalid": "copy exact raw base240 row",
            "generic_valid_choice": "emit exact validated option_label A-E for official scorer",
            "generic_valid_nonchoice": "emit stripped final_answer",
            "invalid_definition": "any error, empty/oversize answer, wrong answer contract, wrong generation provenance, or malformed row content; task ID/order/set mismatch aborts globally",
            "valid_semantic_correctness_claimed": False,
        },
        "binary_exact_contract": {
            "exclusive_write_mode": "O_BINARY_when_available",
            "terminal_newline": "LF",
            "carriage_returns": 0,
            "postwrite_full_payload_byte_equality": True,
            "postwrite_full_payload_sha_equality": True,
            "postwrite_each_invalid_raw_row_byte_equality": True,
        },
        "pre_outcome_contract": {
            "frozen_before_atomic_candidate_completion": True,
            "invalid_transport_or_schema_regression_protected_by_base240": True,
            "valid_but_wrong_generic_regression_protected": False,
            "success_correct_at_least": 240,
            "total": 274,
        },
        "supersession": {
            "v1_4_freeze_sha256": fallback.V14_FREEZE_SHA,
            "v1_4_independent_audit_sha256": fallback.V14_AUDIT_SHA,
            "v1_4_audit_status": "PASS",
            "v1_4_runtime_outputs_absent": True,
            "reason": "Windows text-mode descriptors can translate LF to CRLF and violate the exact raw base240 row byte contract",
        },
        "guards": {
            "api_called": False,
            "candidate_completion_opened": False,
            "candidate_predictions_opened": False,
            "gold_opened": False,
            "outcomes_opened": False,
            "score_opened": False,
            "independent_pass_audit_required": True,
        },
        "tests": {
            "command": "python -B -m unittest test_fallback.py",
            "passed_immediately_before_freeze": True,
        },
        "artifacts": artifacts,
        "implementation": implementations,
    }
    data = fallback.canonical(value)
    expected = fallback.sha256_bytes(data)
    fallback.exclusive_bytes(fallback.FREEZE, data)
    fallback.exclusive_bytes(
        fallback.FREEZE_SHA_FILE, (expected + "\n").encode("ascii")
    )
    if fallback.sha256(fallback.FREEZE) != expected:
        raise RuntimeError("fallback V1.5 actual freeze SHA mismatch")
    print(
        json.dumps(
            {
                "schema_version": value["schema_version"],
                "freeze_sha256": expected,
                "actual_freeze_sha256": fallback.sha256(fallback.FREEZE),
                "freeze_size": fallback.FREEZE.stat().st_size,
                "sidecar_sha256": fallback.sha256(fallback.FREEZE_SHA_FILE),
                "exact_bytes_verified": True,
                "state": value["state"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
