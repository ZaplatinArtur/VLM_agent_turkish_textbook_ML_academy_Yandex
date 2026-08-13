"""Post-freeze fixed non-benchmark smoke; response content is never persisted."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from generic_candidate import build_request, canonical_json_bytes, fixed_smoke_row, sha256_bytes, validate_model_content
from nonstream_protocol import MAX_ATTEMPTS, RETRY_DELAYS_SECONDS, call_openrouter_once, exclusive_json, prompt_api_key, sha256_file, utc_now
from run_candidate import AUTH_PATH, HERE, verify_audit, verify_freeze

RESULT = HERE / "PROTOCOL_SMOKE_RESULT.json"
ATTEMPT = HERE / "PROTOCOL_SMOKE_ATTEMPT.json"
JOURNAL = HERE / "runs" / "smoke_journal"


def run(freeze_sha: str, auth_sha: str, audit_sha: str) -> dict:
    verify_freeze(freeze_sha, auth_sha)
    verify_audit(audit_sha, freeze_sha, auth_sha)
    if RESULT.exists() or ATTEMPT.exists() or JOURNAL.exists():
        raise RuntimeError("smoke state exists; replay forbidden")
    request, _ = build_request(fixed_smoke_row())
    key = prompt_api_key()
    exclusive_json(ATTEMPT, {"schema_version": "maxim256-smoke-attempt-v1", "created_utc": utc_now(), "freeze_sha256": freeze_sha, "authorization_sha256": auth_sha, "independent_audit_sha256": audit_sha, "request_sha256": sha256_bytes(canonical_json_bytes(request)), "max_attempts": MAX_ATTEMPTS, "response_content_persisted": False, "selected_answer_persisted": False}, api_key=key)
    terminal = None
    for number in range(1, MAX_ATTEMPTS + 1):
        intent_path = JOURNAL / f"attempt-{number}.intent.json"
        exclusive_json(intent_path, {"schema_version": "maxim256-smoke-intent-v1", "created_utc": utc_now(), "attempt_number": number, "request_sha256": sha256_bytes(canonical_json_bytes(request)), "top_attempt_sha256": sha256_file(ATTEMPT), "replay_after_ambiguous_power_loss": False}, api_key=key)
        envelope = call_openrouter_once(request, api_key=key)
        valid = False
        wire = None
        if envelope["success"] is True:
            try:
                validate_model_content(envelope["result"]["content"], "choice")
                valid = True
            except Exception:
                valid = False
            result = envelope["result"]
            wire = {key_name: result[key_name] for key_name in ("returned_model", "returned_provider", "finish_reason", "usage", "routing_validation", "cache_status")}
        terminal = {"schema_version": "maxim256-smoke-sanitized-attempt-v1", "recorded_utc": utc_now(), "attempt_number": number, "intent_sha256": sha256_file(intent_path), "success": envelope["success"], "retryable": envelope["retryable"], "error_kind": envelope["error_kind"], "json_contract_valid": valid, "wire": wire, "response_content_persisted": False, "selected_answer_persisted": False}
        exclusive_json(JOURNAL / f"attempt-{number}.result.json", terminal, api_key=key)
        if terminal["retryable"] is not True or number == MAX_ATTEMPTS:
            break
        time.sleep(RETRY_DELAYS_SECONDS[number - 1])
    if terminal is None:
        raise AssertionError("smoke loop empty")
    summary = {"schema_version": "maxim256-nonstream-smoke-v1", "created_utc": utc_now(), "freeze_sha256": freeze_sha, "authorization_sha256": auth_sha, "independent_audit_sha256": audit_sha, "smoke_attempt_sha256": sha256_file(ATTEMPT), "attempt_count": terminal["attempt_number"], "success": terminal["success"], "retryable": terminal["retryable"], "error_kind": terminal["error_kind"], "json_contract_valid": terminal["json_contract_valid"], "wire": terminal["wire"], "response_content_persisted": False, "selected_answer_persisted": False, "gold_opened": False, "final_opened": False}
    exclusive_json(RESULT, summary, api_key=key)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--expected-independent-audit-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.expected_freeze_sha256, args.expected_authorization_sha256, args.expected_independent_audit_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
