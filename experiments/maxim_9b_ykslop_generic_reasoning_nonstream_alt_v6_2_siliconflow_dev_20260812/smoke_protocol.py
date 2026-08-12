"""Bounded post-freeze wire smoke on a fixed non-benchmark question.

The persisted result excludes response content and the selected answer.  It
records only the strict JSON-contract verdict, routing
validation, and usage metadata.  Exact retry policy matches the frozen runner.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from generic_candidate import CandidateError, build_request, canonical_json_bytes, fixed_smoke_row, sha256_bytes, validate_model_content
from run_dev import validate_independent_audit, verify_freeze
from nonstream_protocol import (
    MAX_ATTEMPTS,
    RETRY_DELAYS_SECONDS,
    call_openrouter_once,
    exclusive_json,
    prompt_api_key,
    sha256_file,
    utc_now,
)


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "PROTOCOL_SMOKE_RESULT.json"
SMOKE_ATTEMPT_PATH = HERE / "PROTOCOL_SMOKE_ATTEMPT.json"
SMOKE_JOURNAL_DIR = HERE / "runs" / "smoke_structure_journal"


def _sanitized_attempt(envelope: dict, number: int) -> dict:
    result = envelope.get("result") if envelope.get("success") is True else None
    json_contract_valid = False
    wire = None
    if type(result) is dict:
        try:
            validate_model_content(result["content"])
            json_contract_valid = True
        except CandidateError:
            json_contract_valid = False
        wire = {
            "returned_model": result["returned_model"],
            "returned_provider": result["returned_provider"],
            "finish_reason": result["finish_reason"],
            "usage": result["usage"],
            "routing_validation": result["routing_validation"],
            "cache_status": result["cache_status"],
        }
    return {
        "schema_version": "generic-medium-nonstream-smoke-sanitized-attempt-v6",
        "attempt_number": number,
        "recorded_utc": utc_now(),
        "success": envelope["success"],
        "retryable": envelope["retryable"],
        "error_kind": envelope["error_kind"],
        "json_contract_valid": json_contract_valid,
        "wire": wire,
        "response_content_persisted": False,
        "selected_answer_persisted": False,
    }


def run(
    expected_freeze_sha256: str,
    expected_authorization_sha256: str,
    *,
    independent_audit: Path,
    expected_independent_audit_sha256: str,
) -> dict:
    freeze, _queue = verify_freeze(expected_freeze_sha256, expected_authorization_sha256)
    audit_sha = validate_independent_audit(
        independent_audit,
        expected_independent_audit_sha256,
        freeze=freeze,
        freeze_sha256=expected_freeze_sha256,
        authorization_sha256=expected_authorization_sha256,
    )
    if RESULT_PATH.exists():
        raise RuntimeError("smoke result already exists")
    request, _aliases = build_request(fixed_smoke_row())
    api_key = prompt_api_key()
    if SMOKE_ATTEMPT_PATH.exists() or SMOKE_JOURNAL_DIR.exists():
        raise RuntimeError("smoke attempt/journal exists; replay forbidden")
    top_attempt = {
        "schema_version": "generic-medium-nonstream-smoke-attempt-v6",
        "created_utc": utc_now(),
        "freeze_sha256": expected_freeze_sha256,
        "authorization_sha256": expected_authorization_sha256,
        "independent_audit_sha256": audit_sha,
        "request_sha256": sha256_bytes(canonical_json_bytes(request)),
        "max_attempts": MAX_ATTEMPTS,
        "response_content_persisted": False,
        "selected_answer_persisted": False,
    }
    exclusive_json(SMOKE_ATTEMPT_PATH, top_attempt, api_key=api_key)
    terminal: dict | None = None
    attempt_count = 0
    for number in range(1, MAX_ATTEMPTS + 1):
        intent = {
            "schema_version": "generic-medium-nonstream-smoke-dispatch-intent-v6",
            "created_utc": utc_now(),
            "top_attempt_sha256": sha256_file(SMOKE_ATTEMPT_PATH),
            "request_sha256": top_attempt["request_sha256"],
            "attempt_number": number,
            "replay_after_ambiguous_power_loss": False,
        }
        intent_path = SMOKE_JOURNAL_DIR / f"attempt-{number}.intent.json"
        exclusive_json(intent_path, intent, api_key=api_key)
        envelope = call_openrouter_once(request, api_key=api_key)
        sanitized = _sanitized_attempt(envelope, number)
        sanitized["intent_sha256"] = sha256_file(intent_path)
        exclusive_json(
            SMOKE_JOURNAL_DIR / f"attempt-{number}.result.json",
            sanitized,
            api_key=api_key,
        )
        terminal = sanitized
        attempt_count = number
        if sanitized["retryable"] is not True or number == MAX_ATTEMPTS:
            break
        time.sleep(RETRY_DELAYS_SECONDS[number - 1])
    if terminal is None:
        raise AssertionError("bounded smoke retry loop did not execute")
    summary = {
        "schema_version": "generic-medium-nonstream-protocol-smoke-v6",
        "created_utc": utc_now(),
        "freeze_sha256": expected_freeze_sha256,
        "authorization_sha256": expected_authorization_sha256,
        "independent_audit_sha256": audit_sha,
        "smoke_attempt_sha256": sha256_file(SMOKE_ATTEMPT_PATH),
        "attempt_count": attempt_count,
        "provider_call_count_upper_bound": attempt_count,
        "success": terminal["success"],
        "retryable": terminal["retryable"],
        "error_kind": terminal["error_kind"],
        "json_contract_valid": terminal["json_contract_valid"],
        "wire": terminal["wire"],
        "response_content_persisted": False,
        "selected_answer_persisted": False,
        "gold_accessed": False,
        "final_accessed": False,
    }
    exclusive_json(RESULT_PATH, summary, api_key=api_key)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--independent-audit", required=True)
    parser.add_argument("--expected-independent-audit-sha256", required=True)
    args = parser.parse_args()
    result = run(
        args.expected_freeze_sha256,
        args.expected_authorization_sha256,
        independent_audit=Path(args.independent_audit),
        expected_independent_audit_sha256=args.expected_independent_audit_sha256,
    )
    public = dict(result)
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
