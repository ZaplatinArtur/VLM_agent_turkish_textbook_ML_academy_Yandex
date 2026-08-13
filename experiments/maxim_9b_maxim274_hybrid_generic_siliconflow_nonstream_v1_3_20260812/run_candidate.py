"""Run the frozen 256-row Hybrid V3.1 generic branch without selector IDs on wire."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generic_candidate import MODEL_ID, PROVIDER_NAME, PROVIDER_QUANTIZATION, build_request, canonical_json_bytes, content_projection, fixed_smoke_row, sha256_bytes, validate_model_content
from nonstream_protocol import MAX_ATTEMPTS, RETRYABLE_KINDS, SECRET_PATTERN, TIMEOUT_SECONDS, WORKERS, cached_call, exclusive_bytes, exclusive_json, load_cached_call, parse_utc, prompt_api_key, read_json, sha256_file, stable_bytes, utc_now

HERE = Path(__file__).resolve().parent
FREEZE_PATH = HERE / "EXECUTION_FREEZE.json"
FREEZE_SIDECAR = HERE / "EXECUTION_FREEZE_SHA256.txt"
AUTH_PATH = HERE / "USER_AUTHORIZATION.json"
QUEUE_PATH = HERE / "frozen" / "queue_content_only_256.jsonl"
ALIGNMENT_PATH = HERE / "frozen" / "outer_alignment_256.jsonl"
SMOKE_REQUEST_PATH = HERE / "frozen" / "smoke_request.json"
AUDIT_PATH = HERE / "INDEPENDENT_AUDIT.json"
SMOKE_RESULT_PATH = HERE / "PROTOCOL_SMOKE_RESULT.json"
RUNS = HERE / "runs"
CACHE = RUNS / "cache"
PREDICTIONS = RUNS / "generic_predictions_256.jsonl"
ATTEMPT = HERE / "ATTEMPT.json"
COMPLETION = HERE / "COMPLETION.json"


class RunnerError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(stable_bytes(path).splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunnerError(f"invalid JSONL row {number}") from exc
        if type(value) is not dict:
            raise RunnerError("JSONL row must be object")
        rows.append(value)
    return rows


def descriptor(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"path": path.relative_to(HERE).as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size}
    if rows is not None:
        value["rows"] = rows
    return value


def _verify_desc(value: Any) -> Path:
    if type(value) is not dict or set(value) not in ({"path", "sha256", "size"}, {"path", "rows", "sha256", "size"}):
        raise RunnerError("descriptor schema mismatch")
    path = (HERE / value["path"]).resolve()
    try:
        path.relative_to(HERE.resolve())
    except ValueError as exc:
        raise RunnerError("descriptor escapes experiment") from exc
    if not path.is_file() or path.stat().st_size != value["size"] or sha256_file(path) != value["sha256"]:
        raise RunnerError("descriptor closure mismatch")
    if "rows" in value and len(read_jsonl(path)) != value["rows"]:
        raise RunnerError("descriptor row mismatch")
    return path


def verify_freeze(expected_freeze: str, expected_auth: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if sha256_file(FREEZE_PATH) != expected_freeze or FREEZE_SIDECAR.read_text(encoding="ascii").strip() != expected_freeze:
        raise RunnerError("freeze external pin mismatch")
    freeze = read_json(FREEZE_PATH)
    if (
        freeze.get("schema_version") != "maxim256-hybrid-generic-siliconflow-freeze-v1"
        or freeze.get("state") != "frozen_unexecuted_unscored"
        or freeze.get("rows") != 256
        or freeze.get("model_id") != MODEL_ID
        or freeze.get("provider") != PROVIDER_NAME
        or freeze.get("provider_quantization") != PROVIDER_QUANTIZATION
        or freeze.get("workers") != WORKERS
        or freeze.get("max_attempts") != MAX_ATTEMPTS
        or freeze.get("retryable_kinds") != sorted(RETRYABLE_KINDS)
    ):
        raise RunnerError("freeze semantic mismatch")
    artifacts = freeze.get("artifacts")
    implementation = freeze.get("implementation")
    if type(artifacts) is not dict or type(implementation) is not dict:
        raise RunnerError("freeze closure missing")
    paths = {key: _verify_desc(value) for key, value in artifacts.items()}
    for value in implementation.values():
        _verify_desc(value)
    if paths["queue"] != QUEUE_PATH.resolve() or paths["alignment"] != ALIGNMENT_PATH.resolve() or paths["authorization"] != AUTH_PATH.resolve():
        raise RunnerError("runtime path mismatch")
    if sha256_file(AUTH_PATH) != expected_auth:
        raise RunnerError("authorization external pin mismatch")
    auth = read_json(AUTH_PATH)
    if auth.get("schema_version") != "maxim256-openrouter-authorization-v1" or auth.get("authorized") is not True or auth.get("api_key_storage") != "interactive_memory_only_not_persisted":
        raise RunnerError("authorization closure mismatch")
    queue, alignment = read_jsonl(QUEUE_PATH), read_jsonl(ALIGNMENT_PATH)
    if len(queue) != 256 or len(alignment) != 256:
        raise RunnerError("denominator mismatch")
    for row in queue:
        content_projection(row)
        if any(key in row for key in ("task_id", "controller_id", "content_sha256", "image_sha256")):
            raise RunnerError("identity/hash in runtime content queue")
    ids = [row.get("task_id") for row in alignment]
    if any(type(value) is not str for value in ids) or len(set(ids)) != 256 or any(set(row) != {"schema_version", "task_id"} for row in alignment):
        raise RunnerError("outer alignment mismatch")
    smoke, _ = build_request(fixed_smoke_row())
    if stable_bytes(paths["smoke_request"]) != canonical_json_bytes(smoke):
        raise RunnerError("smoke request mismatch")
    return freeze, queue, alignment


def verify_audit(expected_sha: str, freeze_sha: str, auth_sha: str) -> str:
    if not AUDIT_PATH.is_file() or sha256_file(AUDIT_PATH) != expected_sha:
        raise RunnerError("audit external pin mismatch")
    value = read_json(AUDIT_PATH)
    if value.get("schema_version") != "maxim256-generic-independent-audit-v1" or value.get("status") != "PASS" or value.get("freeze_sha256") != freeze_sha or value.get("authorization_sha256") != auth_sha or value.get("guards") != {"api_called": False, "gold_opened": False, "outcomes_opened": False, "final_opened": False} or type(value.get("checks")) is not dict or any(check is not True for check in value["checks"].values()):
        raise RunnerError("audit contract mismatch")
    parse_utc(value.get("created_utc"))
    return expected_sha


def verify_smoke(expected_sha: str, freeze_sha: str, auth_sha: str, audit_sha: str) -> str:
    if not SMOKE_RESULT_PATH.is_file() or sha256_file(SMOKE_RESULT_PATH) != expected_sha:
        raise RunnerError("smoke external pin mismatch")
    value = read_json(SMOKE_RESULT_PATH)
    wire = value.get("wire")
    usage = wire.get("usage") if type(wire) is dict else None
    routing = wire.get("routing_validation") if type(wire) is dict else None
    checks = routing.get("checks") if type(routing) is dict else None
    expected_checks = {"metadata_present", "requested_model_exact", "strategy_direct", "attempt_one", "one_selected_endpoint", "selected_provider_exact", "selected_model_exact", "pipeline_empty", "attempts_exact_if_present"}
    if (
        value.get("schema_version") != "maxim256-nonstream-smoke-v1"
        or value.get("freeze_sha256") != freeze_sha
        or value.get("authorization_sha256") != auth_sha
        or value.get("independent_audit_sha256") != audit_sha
        or type(value.get("attempt_count")) is not int
        or not 1 <= value["attempt_count"] <= MAX_ATTEMPTS
        or value.get("success") is not True
        or value.get("retryable") is not False
        or value.get("error_kind") is not None
        or value.get("json_contract_valid") is not True
        or value.get("response_content_persisted") is not False
        or value.get("selected_answer_persisted") is not False
        or value.get("gold_opened") is not False
        or value.get("final_opened") is not False
        or type(wire) is not dict
        or set(wire) != {"returned_model", "returned_provider", "finish_reason", "usage", "routing_validation", "cache_status"}
        or wire.get("returned_model") != MODEL_ID
        or wire.get("returned_provider") != PROVIDER_NAME
        or wire.get("finish_reason") != "stop"
        or str(wire.get("cache_status") or "").casefold() == "hit"
        or type(usage) is not dict
        or type(usage.get("prompt_tokens")) is not int
        or type(usage.get("completion_tokens")) is not int
        or usage.get("total_tokens") != usage["prompt_tokens"] + usage["completion_tokens"]
        or type(routing) is not dict
        or routing.get("passed") is not True
        or type(checks) is not dict
        or set(checks) != expected_checks
        or any(check is not True for check in checks.values())
    ):
        raise RunnerError("smoke success closure mismatch")
    attempt_path = HERE / "PROTOCOL_SMOKE_ATTEMPT.json"
    journal = RUNS / "smoke_journal"
    if not attempt_path.is_file() or value.get("smoke_attempt_sha256") != sha256_file(attempt_path) or not journal.is_dir():
        raise RunnerError("smoke attempt/journal closure mismatch")
    expected_names = {f"attempt-{number}.{suffix}.json" for number in range(1, value["attempt_count"] + 1) for suffix in ("intent", "result")}
    if {path.name for path in journal.iterdir() if path.is_file()} != expected_names:
        raise RunnerError("smoke journal file set mismatch")
    rows: list[dict[str, Any]] = []
    for number in range(1, value["attempt_count"] + 1):
        intent_path = journal / f"attempt-{number}.intent.json"
        result_path = journal / f"attempt-{number}.result.json"
        intent, result = read_json(intent_path), read_json(result_path)
        if intent.get("schema_version") != "maxim256-smoke-intent-v1" or intent.get("attempt_number") != number or intent.get("top_attempt_sha256") != value["smoke_attempt_sha256"] or intent.get("replay_after_ambiguous_power_loss") is not False or result.get("schema_version") != "maxim256-smoke-sanitized-attempt-v1" or result.get("attempt_number") != number or result.get("intent_sha256") != sha256_file(intent_path) or result.get("response_content_persisted") is not False or result.get("selected_answer_persisted") is not False:
            raise RunnerError("smoke journal row mismatch")
        rows.append(result)
    if any(row.get("success") is not False or row.get("retryable") is not True or row.get("error_kind") not in RETRYABLE_KINDS for row in rows[:-1]):
        raise RunnerError("smoke retried terminal outcome")
    terminal = rows[-1]
    if any(value[field] != terminal[field] for field in ("success", "retryable", "error_kind", "json_contract_valid", "wire")):
        raise RunnerError("smoke summary/journal mismatch")
    return expected_sha


def dry_run(expected_freeze: str, expected_auth: str) -> dict[str, Any]:
    freeze, queue, alignment = verify_freeze(expected_freeze, expected_auth)
    requests = [build_request(row)[0] for row in queue]
    hashes = [sha256_bytes(canonical_json_bytes(request)) for request in requests]
    return {
        "schema_version": "maxim256-generic-dry-run-v1",
        "freeze_sha256": expected_freeze,
        "rows": 256,
        "alignment_rows": len(alignment),
        "unique_requests": len(set(hashes)),
        "source_text_only": sum(row["source_input_mode"] == "text_only" for row in queue),
        "source_multimodal_degraded_to_ocr_only": sum(row["source_input_mode"] == "multimodal_degraded_to_ocr_only" for row in queue),
        "image_bytes_sent": False,
        "provider_calls_made": 0,
        "gold_opened": False,
        "outcomes_opened": False,
        "freeze_state": freeze["state"],
    }


def _prediction(row: dict[str, Any], align: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    terminal = call["attempts"][-1]
    parsed: dict[str, str] | None = None
    error = terminal.get("error_kind")
    if terminal.get("success") is True:
        try:
            parsed = validate_model_content(terminal["result"]["content"], row["answer_type"])
            error = None
        except Exception as exc:
            error = type(exc).__name__
    return {"schema_version": "maxim256-hybrid-generic-prediction-v1", "task_id": align["task_id"], "final_answer": parsed["answer"] if parsed else "", "option_label": parsed["option_label"] if parsed else "NA", "answer_type": row["answer_type"], "input_mode": "ocr_only", "error": error, "generation": {"gold_access": False, "outcome_access": False, "model": MODEL_ID, "provider": PROVIDER_NAME, "quantization": PROVIDER_QUANTIZATION}}


def _validate_predictions(queue: list[dict[str, Any]], alignment: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stored = read_jsonl(PREDICTIONS)
    if len(stored) != 256:
        raise RunnerError("prediction denominator mismatch")
    for actual, row, align in zip(stored, queue, alignment):
        request, _ = build_request(row)
        call = load_cached_call(request, cache_dir=CACHE)
        if actual != _prediction(row, align, call):
            raise RunnerError("prediction/cache/content/alignment closure mismatch")
    return stored


def _completion_value(freeze_sha: str, auth_sha: str, audit_sha: str, smoke_sha: str, *, cache_hits: int, recovered: bool) -> dict[str, Any]:
    rows = read_jsonl(PREDICTIONS)
    return {"schema_version": "maxim256-generic-completion-v1", "created_utc": utc_now(), "freeze_sha256": freeze_sha, "authorization_sha256": auth_sha, "independent_audit_sha256": audit_sha, "protocol_smoke_sha256": smoke_sha, "attempt_sha256": sha256_file(ATTEMPT), "rows": 256, "prediction_errors": sum(row["error"] is not None for row in rows), "cache_hits": cache_hits, "recovered_after_predictions_write": recovered, "predictions": descriptor(PREDICTIONS, 256), "provider_calls_upper_bound": 256 * MAX_ATTEMPTS, "image_bytes_sent": False, "gold_opened": False, "outcomes_opened": False}


def _validate_completion(value: Any, freeze_sha: str, auth_sha: str, audit_sha: str, smoke_sha: str) -> dict[str, Any]:
    if (
        type(value) is not dict
        or value.get("schema_version") != "maxim256-generic-completion-v1"
        or value.get("freeze_sha256") != freeze_sha
        or value.get("authorization_sha256") != auth_sha
        or value.get("independent_audit_sha256") != audit_sha
        or value.get("protocol_smoke_sha256") != smoke_sha
        or value.get("attempt_sha256") != sha256_file(ATTEMPT)
        or value.get("rows") != 256
        or type(value.get("prediction_errors")) is not int
        or not 0 <= value["prediction_errors"] <= 256
        or type(value.get("cache_hits")) is not int
        or not 0 <= value["cache_hits"] <= 256
        or type(value.get("recovered_after_predictions_write")) is not bool
        or value.get("provider_calls_upper_bound") != 256 * MAX_ATTEMPTS
        or value.get("image_bytes_sent") is not False
        or value.get("gold_opened") is not False
        or value.get("outcomes_opened") is not False
    ):
        raise RunnerError("completion closure mismatch")
    path = _verify_desc(value.get("predictions"))
    if path != PREDICTIONS.resolve():
        raise RunnerError("completion prediction path mismatch")
    parse_utc(value.get("created_utc"))
    return dict(value)


def execute(expected_freeze: str, expected_auth: str, expected_audit: str, expected_smoke: str, *, resume: bool) -> dict[str, Any]:
    freeze, queue, alignment = verify_freeze(expected_freeze, expected_auth)
    audit = verify_audit(expected_audit, expected_freeze, expected_auth)
    smoke = verify_smoke(expected_smoke, expected_freeze, expected_auth, audit)
    if COMPLETION.exists():
        _validate_predictions(queue, alignment)
        return _validate_completion(read_json(COMPLETION), expected_freeze, expected_auth, audit, smoke)
    if ATTEMPT.exists():
        if not resume:
            raise RunnerError("attempt exists; use --resume")
        attempt = read_json(ATTEMPT)
        if (
            attempt.get("schema_version") != "maxim256-generic-attempt-v1"
            or attempt.get("freeze_sha256") != expected_freeze
            or attempt.get("authorization_sha256") != expected_auth
            or attempt.get("independent_audit_sha256") != audit
            or attempt.get("protocol_smoke_sha256") != smoke
            or attempt.get("rows") != 256
            or attempt.get("api_key_storage") != "interactive_memory_only_not_persisted"
            or attempt.get("gold_opened") is not False
            or attempt.get("outcomes_opened") is not False
        ):
            raise RunnerError("attempt resume closure mismatch")
        parse_utc(attempt.get("created_utc"))
    else:
        if resume or PREDICTIONS.exists():
            raise RunnerError("invalid resume state")
        exclusive_json(ATTEMPT, {"schema_version": "maxim256-generic-attempt-v1", "created_utc": utc_now(), "freeze_sha256": expected_freeze, "authorization_sha256": expected_auth, "independent_audit_sha256": audit, "protocol_smoke_sha256": smoke, "rows": 256, "api_key_storage": "interactive_memory_only_not_persisted", "gold_opened": False, "outcomes_opened": False})
    if PREDICTIONS.exists():
        if not resume:
            raise RunnerError("predictions exist without completion; use --resume")
        _validate_predictions(queue, alignment)
        completion = _completion_value(expected_freeze, expected_auth, audit, smoke, cache_hits=256, recovered=True)
        exclusive_json(COMPLETION, completion)
        return _validate_completion(completion, expected_freeze, expected_auth, audit, smoke)
    api_key = prompt_api_key()
    outputs: list[dict[str, Any]] = []
    cache_hits = 0
    for number, (row, align) in enumerate(zip(queue, alignment), 1):
        request, _ = build_request(row)
        call, cached = cached_call(request, api_key=api_key, cache_dir=CACHE)
        cache_hits += int(cached)
        outputs.append(_prediction(row, align, call))
        if number % 10 == 0 or number == 256:
            print(f"[maxim256] {number}/256 cache_hits={cache_hits}", flush=True)
    exclusive_bytes(PREDICTIONS, b"".join(canonical_json_bytes(row) for row in outputs), api_key=api_key)
    _validate_predictions(queue, alignment)
    completion = _completion_value(expected_freeze, expected_auth, audit, smoke, cache_hits=cache_hits, recovered=False)
    exclusive_json(COMPLETION, completion, api_key=api_key)
    return _validate_completion(completion, expected_freeze, expected_auth, audit, smoke)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--expected-independent-audit-sha256")
    parser.add_argument("--expected-protocol-smoke-sha256")
    args = parser.parse_args()
    if args.dry_run:
        result = dry_run(args.expected_freeze_sha256, args.expected_authorization_sha256)
    else:
        if not args.expected_independent_audit_sha256 or not args.expected_protocol_smoke_sha256:
            parser.error("execute requires audit and smoke SHA pins")
        result = execute(args.expected_freeze_sha256, args.expected_authorization_sha256, args.expected_independent_audit_sha256, args.expected_protocol_smoke_sha256, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
