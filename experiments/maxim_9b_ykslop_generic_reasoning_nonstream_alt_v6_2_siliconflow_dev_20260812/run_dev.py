"""Execute frozen public-only 185-row SiliconFlow nonstream candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from generic_candidate import (
    MAX_TOKENS,
    MODEL_ID,
    PROVIDER_NAME,
    PROVIDER_QUANTIZATION,
    PROVIDER_ROUTED_MODEL_ID,
    build_request,
    canonical_json_bytes,
    content_digest,
    fixed_smoke_row,
    map_model_answer,
    theory_projection,
)
from nonstream_protocol import (
    MAX_ATTEMPTS,
    RETRYABLE_KINDS,
    SECRET_PATTERN,
    TIMEOUT_SECONDS,
    WORKERS,
    ProtocolError,
    cached_call,
    exclusive_bytes,
    exclusive_json,
    load_cached_call,
    parse_utc,
    prompt_api_key,
    read_json,
    sha256_bytes,
    sha256_file,
    stable_bytes,
    utc_now,
)


HERE = Path(__file__).resolve().parent
FREEZE_PATH = HERE / "DEV_EXECUTION_FREEZE.json"
FREEZE_SHA_PATH = HERE / "DEV_EXECUTION_FREEZE_SHA256.txt"
QUEUE_PATH = HERE / "frozen" / "queue_public_content_only.jsonl"
PROVIDER_SNAPSHOT_PATH = HERE / "frozen" / "provider_endpoint_snapshot.json"
ZDR_SNAPSHOT_PATH = HERE / "frozen" / "zdr_inventory_snapshot.json"
SMOKE_REQUEST_PATH = HERE / "frozen" / "smoke_request.json"
AUTH_PATH = HERE / "USER_AUTHORIZATION.json"
RUNS_DIR = HERE / "runs"
CACHE_DIR = RUNS_DIR / "stage_cache"
PREDICTIONS_PATH = RUNS_DIR / "predictions_content_only.jsonl"
COMPLETION_PATH = HERE / "DEV_WAVE_COMPLETION.json"
ATTEMPT_PATH = HERE / "ATTEMPT.json"
SMOKE_PATH = HERE / "PROTOCOL_SMOKE_RESULT.json"
SMOKE_ATTEMPT_PATH = HERE / "PROTOCOL_SMOKE_ATTEMPT.json"
SMOKE_JOURNAL_DIR = HERE / "runs" / "smoke_structure_journal"
PROVIDER_SNAPSHOT_SHA256 = "020df7beb073ecdb785e2b96651b9068d911fed8ba72e5895aa32237be824c12"
ZDR_SNAPSHOT_SHA256 = "e2bab064b51183f88535f18c88dc820d71987026c7a757f58c2b2bfa6a9755a2"
DEVELOPMENT_SPLIT_SHA256 = "6fb8474a9e0e71c2afae3c9bef20f22cbc6a5b7ae8928321b2d919afcfd32f9e"
SMOKE_ROUTING_CHECKS = {
    "attempt_one",
    "attempts_exact_if_present",
    "metadata_present",
    "one_selected_endpoint",
    "pipeline_empty",
    "requested_model_exact",
    "selected_model_exact",
    "selected_provider_exact",
    "strategy_direct",
}


class RunnerError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(stable_bytes(path).splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunnerError(f"invalid JSONL row {number}: {path}") from exc
        if type(value) is not dict:
            raise RunnerError(f"non-object JSONL row {number}: {path}")
        rows.append(value)
    return rows


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.relative_to(HERE).as_posix(),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _verify_descriptor(value: Any) -> Path:
    if type(value) is not dict or set(value) not in (
        {"path", "sha256", "size"},
        {"path", "rows", "sha256", "size"},
    ):
        raise RunnerError("artifact descriptor schema mismatch")
    path = (HERE / value["path"]).resolve()
    try:
        path.relative_to(HERE.resolve())
    except ValueError as exc:
        raise RunnerError("artifact path escapes experiment directory") from exc
    if not path.is_file():
        raise RunnerError(f"missing frozen artifact: {path}")
    data = stable_bytes(path)
    if len(data) != value["size"] or sha256_bytes(data) != value["sha256"]:
        raise RunnerError(f"frozen artifact closure mismatch: {path}")
    if "rows" in value and len(_read_jsonl(path)) != value["rows"]:
        raise RunnerError(f"frozen artifact row mismatch: {path}")
    return path


def _validate_provider_snapshot(path: Path) -> None:
    if path.resolve() != PROVIDER_SNAPSHOT_PATH.resolve() or sha256_file(path) != PROVIDER_SNAPSHOT_SHA256:
        raise RunnerError("provider snapshot exact pin mismatch")
    value = read_json(path)
    endpoint = value.get("endpoint")
    supported = endpoint.get("supported_parameters") if type(endpoint) is dict else None
    if (
        value.get("schema_version") != "openrouter-current-siliconflow-endpoint-evidence-v1"
        or type(endpoint) is not dict
        or endpoint.get("provider_name") != PROVIDER_NAME
        or endpoint.get("model_id") != MODEL_ID
        or endpoint.get("name") != f"{PROVIDER_NAME} | {PROVIDER_ROUTED_MODEL_ID}"
        or endpoint.get("quantization") != PROVIDER_QUANTIZATION
        or type(endpoint.get("context_length")) is not int
        or endpoint["context_length"] < MAX_TOKENS
        or type(endpoint.get("max_completion_tokens")) is not int
        or endpoint["max_completion_tokens"] < MAX_TOKENS
        or type(supported) is not list
        or not {"max_tokens", "reasoning", "response_format", "temperature", "top_p"}.issubset(supported)
        or "seed" in supported
    ):
        raise RunnerError("provider snapshot semantic mismatch")


def _validate_queue(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 185:
        raise RunnerError("queue must contain exactly 185 public rows")
    seen: set[str] = set()
    expected_keys = {
        "choices",
        "content_sha256",
        "has_figure",
        "question",
        "schema_version",
        "subject",
        "theory",
        "theory_sha256",
    }
    for row in rows:
        if set(row) != expected_keys or row.get("schema_version") != "generic-public-content-theory-row-v1":
            raise RunnerError("queue row schema mismatch")
        digest = content_digest(row)
        if row.get("content_sha256") != digest or digest in seen:
            raise RunnerError("queue content hash mismatch or duplicate")
        seen.add(digest)
        theory_projection(row)


def verify_freeze(expected_sha256: str, expected_authorization_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(FREEZE_PATH) != expected_sha256:
        raise RunnerError("execution freeze external pin mismatch")
    if FREEZE_SHA_PATH.read_text(encoding="ascii").strip() != expected_sha256:
        raise RunnerError("execution freeze sidecar mismatch")
    freeze = read_json(FREEZE_PATH)
    if (
        freeze.get("schema_version") != "generic-medium-reasoning-nonstream-dev-freeze-v6"
        or freeze.get("state") != "frozen_unexecuted_unscored"
        or freeze.get("row_count") != 185
        or freeze.get("model_id") != MODEL_ID
        or freeze.get("provider") != PROVIDER_NAME
        or freeze.get("provider_quantization") != PROVIDER_QUANTIZATION
        or freeze.get("workers") != WORKERS
        or freeze.get("timeout_seconds") != TIMEOUT_SECONDS
        or freeze.get("max_attempts") != MAX_ATTEMPTS
        or freeze.get("retryable_kinds") != sorted(RETRYABLE_KINDS)
        or freeze.get("development_split_sha256") != DEVELOPMENT_SPLIT_SHA256
        or freeze.get("source_public_rows") != 185
    ):
        raise RunnerError("execution freeze contract mismatch")
    artifacts = freeze.get("artifacts")
    implementation = freeze.get("implementation")
    if type(artifacts) is not dict or set(artifacts) != {"authorization", "provider_snapshot", "queue", "smoke_request", "zdr_snapshot"}:
        raise RunnerError("execution artifact set mismatch")
    if type(implementation) is not dict or set(implementation) != {"candidate", "protocol", "runner", "smoke"}:
        raise RunnerError("execution implementation set mismatch")
    paths = {name: _verify_descriptor(value) for name, value in artifacts.items()}
    code_paths = {name: _verify_descriptor(value) for name, value in implementation.items()}
    if (
        paths["queue"] != QUEUE_PATH.resolve()
        or paths["authorization"] != AUTH_PATH.resolve()
        or paths["smoke_request"] != SMOKE_REQUEST_PATH.resolve()
    ):
        raise RunnerError("execution artifact path mismatch")
    smoke_request, _smoke_aliases = build_request(fixed_smoke_row())
    if stable_bytes(paths["smoke_request"]) != canonical_json_bytes(smoke_request):
        raise RunnerError("frozen smoke request/code mismatch")
    _validate_provider_snapshot(paths["provider_snapshot"])
    if paths["zdr_snapshot"].resolve() != ZDR_SNAPSHOT_PATH.resolve() or sha256_file(paths["zdr_snapshot"]) != ZDR_SNAPSHOT_SHA256:
        raise RunnerError("ZDR inventory exact pin mismatch")
    zdr = read_json(paths["zdr_snapshot"])
    if zdr.get("schema_version") != "openrouter-current-zdr-inventory-evidence-v1" or zdr.get("exact_siliconflow_fp8_present") is not True:
        raise RunnerError("SiliconFlow endpoint is not closed as ZDR")
    if code_paths != {
        "candidate": (HERE / "generic_candidate.py").resolve(),
        "protocol": (HERE / "nonstream_protocol.py").resolve(),
        "runner": Path(__file__).resolve(),
        "smoke": (HERE / "smoke_protocol.py").resolve(),
    }:
        raise RunnerError("execution implementation path mismatch")
    if sha256_file(AUTH_PATH) != expected_authorization_sha256:
        raise RunnerError("authorization external pin mismatch")
    authorization = read_json(AUTH_PATH)
    if (
        set(authorization)
        != {
            "api_key_storage",
            "authorized_openrouter_calls",
            "authorized_scope",
            "created_utc",
            "implementation_sha256",
            "live_gate",
            "model_id",
            "provider",
            "provider_quantization",
            "queue_sha256",
            "schema_version",
            "user_statement",
        }
        or authorization.get("schema_version") != "generic-medium-nonstream-user-authorization-v6"
        or authorization.get("authorized_openrouter_calls") is not True
        or authorization.get("authorized_scope")
        != "185 public DEV calls plus up to three exact-retry fixed non-benchmark protocol smoke attempts"
        or type(authorization.get("user_statement")) is not str
        or not authorization["user_statement"].strip()
        or authorization.get("queue_sha256") != artifacts["queue"]["sha256"]
        or authorization.get("implementation_sha256")
        != {name: descriptor["sha256"] for name, descriptor in freeze["implementation"].items()}
        or authorization.get("model_id") != MODEL_ID
        or authorization.get("provider") != PROVIDER_NAME
        or authorization.get("provider_quantization") != PROVIDER_QUANTIZATION
        or authorization.get("api_key_storage") != "interactive_memory_only_not_persisted"
        or authorization.get("live_gate") != "external independent PASS audit SHA required"
    ):
        raise RunnerError("authorization contract mismatch")
    parse_utc(authorization.get("created_utc"))
    queue = _read_jsonl(paths["queue"])
    _validate_queue(queue)
    return freeze, queue


_AUDIT_CHECKS = {
    "dry_run_185",
    "freeze_closure",
    "journal_cache_closure",
    "no_gold_final_outcomes",
    "provider_snapshot",
    "queue_full_185_ancestry",
    "request_blindness",
    "retry_allowlist",
    "secret_redaction",
    "nonstream_response_closure",
    "zdr_endpoint_inventory",
}


def validate_independent_audit(
    path: Path,
    expected_sha256: str,
    *,
    freeze: dict[str, Any],
    freeze_sha256: str,
    authorization_sha256: str,
) -> str:
    data = stable_bytes(path)
    if sha256_bytes(data) != expected_sha256 or SECRET_PATTERN.search(data.decode("utf-8", errors="strict")):
        raise RunnerError("independent audit external pin or secret scan mismatch")
    value = read_json(path)
    checks = value.get("checks")
    guards = value.get("guards")
    expected_implementation = {
        name: descriptor["sha256"]
        for name, descriptor in freeze["implementation"].items()
    }
    if (
        set(value) != {
            "auditor",
            "authorization_sha256",
            "checks",
            "created_utc",
            "freeze_sha256",
            "guards",
            "implementation_sha256",
            "provider_snapshot_sha256",
            "queue_sha256",
            "schema_version",
            "source_public_sha256",
            "status",
        }
        or value.get("schema_version") != "generic-medium-nonstream-independent-prerun-audit-v6"
        or value.get("status") != "PASS"
        or type(value.get("auditor")) is not str
        or not value["auditor"].strip()
        or type(value.get("created_utc")) is not str
        or value.get("freeze_sha256") != freeze_sha256
        or value.get("authorization_sha256") != authorization_sha256
        or value.get("queue_sha256") != freeze["artifacts"]["queue"]["sha256"]
        or value.get("provider_snapshot_sha256") != PROVIDER_SNAPSHOT_SHA256
        or value.get("source_public_sha256") != freeze.get("source_public_sha256")
        or value.get("implementation_sha256") != expected_implementation
        or type(checks) is not dict
        or set(checks) != _AUDIT_CHECKS
        or any(item is not True for item in checks.values())
        or guards
        != {
            "gold_opened": False,
            "final_opened": False,
            "provider_called": False,
            "outcomes_opened": False,
        }
    ):
        raise RunnerError("independent audit contract mismatch")
    parse_utc(value.get("created_utc"))
    return expected_sha256


def validate_protocol_smoke(
    expected_sha256: str,
    *,
    freeze_sha256: str,
    authorization_sha256: str,
    audit_sha256: str,
) -> str:
    if not SMOKE_PATH.is_file() or sha256_file(SMOKE_PATH) != expected_sha256:
        raise RunnerError("protocol smoke external pin mismatch")
    value = read_json(SMOKE_PATH)
    wire = value.get("wire")
    usage = wire.get("usage") if type(wire) is dict else None
    routing = wire.get("routing_validation") if type(wire) is dict else None
    routing_checks = routing.get("checks") if type(routing) is dict else None
    usage_valid = (
        type(usage) is dict
        and type(usage.get("prompt_tokens")) is int
        and usage["prompt_tokens"] >= 0
        and type(usage.get("completion_tokens")) is int
        and usage["completion_tokens"] >= 0
        and type(usage.get("total_tokens")) is int
        and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        and (
            usage.get("cost") is None
            or (
                type(usage.get("cost")) in (int, float)
                and not isinstance(usage.get("cost"), bool)
                and math.isfinite(float(usage["cost"]))
                and float(usage["cost"]) >= 0
            )
        )
    )
    if (
        set(value)
        != {
            "attempt_count",
            "authorization_sha256",
            "created_utc",
            "error_kind",
            "final_accessed",
            "freeze_sha256",
            "gold_accessed",
            "independent_audit_sha256",
            "json_contract_valid",
            "provider_call_count_upper_bound",
            "response_content_persisted",
            "retryable",
            "schema_version",
            "selected_answer_persisted",
            "smoke_attempt_sha256",
            "success",
            "wire",
        }
        or value.get("schema_version") != "generic-medium-nonstream-protocol-smoke-v6"
        or value.get("freeze_sha256") != freeze_sha256
        or value.get("authorization_sha256") != authorization_sha256
        or value.get("independent_audit_sha256") != audit_sha256
        or not SMOKE_ATTEMPT_PATH.is_file()
        or value.get("smoke_attempt_sha256") != sha256_file(SMOKE_ATTEMPT_PATH)
        or type(value.get("attempt_count")) is not int
        or not 1 <= value["attempt_count"] <= MAX_ATTEMPTS
        or value.get("provider_call_count_upper_bound") != value["attempt_count"]
        or value.get("success") is not True
        or value.get("retryable") is not False
        or value.get("error_kind") is not None
        or value.get("json_contract_valid") is not True
        or value.get("response_content_persisted") is not False
        or value.get("selected_answer_persisted") is not False
        or value.get("gold_accessed") is not False
        or value.get("final_accessed") is not False
        or type(wire) is not dict
        or set(wire)
        != {
            "cache_status",
            "finish_reason",
            "returned_model",
            "returned_provider",
            "routing_validation",
            "usage",
        }
        or wire.get("returned_model") != MODEL_ID
        or wire.get("returned_provider") != PROVIDER_NAME
        or wire.get("finish_reason") != "stop"
        or not usage_valid
        or type(routing) is not dict
        or set(routing) != {"checks", "passed"}
        or routing.get("passed") is not True
        or type(routing_checks) is not dict
        or set(routing_checks) != SMOKE_ROUTING_CHECKS
        or any(check is not True for check in routing_checks.values())
        or str(wire.get("cache_status") or "").casefold() == "hit"
    ):
        raise RunnerError("protocol smoke contract mismatch")
    parse_utc(value.get("created_utc"))
    request, _aliases = build_request(fixed_smoke_row())
    request_sha = sha256_bytes(canonical_json_bytes(request))
    top = read_json(SMOKE_ATTEMPT_PATH)
    if (
        set(top)
        != {
            "authorization_sha256",
            "created_utc",
            "freeze_sha256",
            "independent_audit_sha256",
            "max_attempts",
            "request_sha256",
            "response_content_persisted",
            "schema_version",
            "selected_answer_persisted",
        }
        or top.get("schema_version") != "generic-medium-nonstream-smoke-attempt-v6"
        or top.get("freeze_sha256") != freeze_sha256
        or top.get("authorization_sha256") != authorization_sha256
        or top.get("independent_audit_sha256") != audit_sha256
        or top.get("request_sha256") != request_sha
        or top.get("max_attempts") != MAX_ATTEMPTS
        or top.get("response_content_persisted") is not False
        or top.get("selected_answer_persisted") is not False
    ):
        raise RunnerError("protocol smoke top-level attempt mismatch")
    parse_utc(top.get("created_utc"))
    if not SMOKE_JOURNAL_DIR.is_dir():
        raise RunnerError("protocol smoke structure journal missing")
    expected_names = {
        f"attempt-{number}.{suffix}.json"
        for number in range(1, value["attempt_count"] + 1)
        for suffix in ("intent", "result")
    }
    if {path.name for path in SMOKE_JOURNAL_DIR.iterdir() if path.is_file()} != expected_names:
        raise RunnerError("protocol smoke structure journal file set mismatch")
    sanitized_rows: list[dict[str, Any]] = []
    for number in range(1, value["attempt_count"] + 1):
        intent_path = SMOKE_JOURNAL_DIR / f"attempt-{number}.intent.json"
        result_path = SMOKE_JOURNAL_DIR / f"attempt-{number}.result.json"
        intent = read_json(intent_path)
        sanitized = read_json(result_path)
        if (
            set(intent)
            != {
                "attempt_number",
                "created_utc",
                "replay_after_ambiguous_power_loss",
                "request_sha256",
                "schema_version",
                "top_attempt_sha256",
            }
            or intent.get("schema_version") != "generic-medium-nonstream-smoke-dispatch-intent-v6"
            or intent.get("attempt_number") != number
            or intent.get("top_attempt_sha256") != value["smoke_attempt_sha256"]
            or intent.get("request_sha256") != request_sha
            or intent.get("replay_after_ambiguous_power_loss") is not False
        ):
            raise RunnerError("protocol smoke dispatch intent mismatch")
        parse_utc(intent.get("created_utc"))
        if (
            set(sanitized)
            != {
                "attempt_number",
                "error_kind",
                "intent_sha256",
                "json_contract_valid",
                "recorded_utc",
                "response_content_persisted",
                "retryable",
                "schema_version",
                "selected_answer_persisted",
                "success",
                "wire",
            }
            or sanitized.get("schema_version") != "generic-medium-nonstream-smoke-sanitized-attempt-v6"
            or sanitized.get("attempt_number") != number
            or sanitized.get("intent_sha256") != sha256_file(intent_path)
            or sanitized.get("response_content_persisted") is not False
            or sanitized.get("selected_answer_persisted") is not False
            or type(sanitized.get("success")) is not bool
            or type(sanitized.get("retryable")) is not bool
        ):
            raise RunnerError("protocol smoke sanitized journal mismatch")
        parse_utc(sanitized.get("recorded_utc"))
        if parse_utc(sanitized["recorded_utc"]) < parse_utc(intent["created_utc"]):
            raise RunnerError("protocol smoke result predates dispatch intent")
        sanitized_rows.append(sanitized)
    if any(
        row["success"] is not False
        or row["retryable"] is not True
        or row["error_kind"] not in RETRYABLE_KINDS
        or row["json_contract_valid"] is not False
        or row["wire"] is not None
        for row in sanitized_rows[:-1]
    ):
        raise RunnerError("protocol smoke retried terminal outcome")
    terminal = sanitized_rows[-1]
    if terminal["success"] is True:
        if (
            terminal["retryable"] is not False
            or terminal["error_kind"] is not None
            or terminal["json_contract_valid"] is not True
            or type(terminal["wire"]) is not dict
        ):
            raise RunnerError("protocol smoke terminal success contradiction")
    elif (
        terminal["json_contract_valid"] is not False
        or terminal["wire"] is not None
        or type(terminal["error_kind"]) is not str
        or terminal["retryable"] is not (terminal["error_kind"] in RETRYABLE_KINDS)
    ):
        raise RunnerError("protocol smoke terminal failure contradiction")
    if any(
        value[field] != terminal[field]
        for field in ("error_kind", "json_contract_valid", "retryable", "success", "wire")
    ):
        raise RunnerError("protocol smoke summary/journal mismatch")
    return expected_sha256


def _attempt_value(
    *,
    freeze_sha256: str,
    authorization_sha256: str,
    audit_sha256: str,
    smoke_sha256: str,
    queue_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "generic-medium-nonstream-dev-attempt-v6",
        "created_utc": utc_now(),
        "freeze_sha256": freeze_sha256,
        "authorization_sha256": authorization_sha256,
        "independent_audit_sha256": audit_sha256,
        "protocol_smoke_sha256": smoke_sha256,
        "queue_sha256": queue_sha256,
        "model_id": MODEL_ID,
        "provider": PROVIDER_NAME,
        "provider_quantization": PROVIDER_QUANTIZATION,
        "workers": WORKERS,
        "max_attempts": MAX_ATTEMPTS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "api_key_storage": "interactive_memory_only_not_persisted",
        "gold_accessed": False,
        "final_accessed": False,
        "runtime_opaque_ids": False,
    }


def _validate_attempt(
    value: Any,
    *,
    freeze_sha256: str,
    authorization_sha256: str,
    audit_sha256: str,
    smoke_sha256: str,
    queue_sha256: str,
) -> dict[str, Any]:
    expected = _attempt_value(
        freeze_sha256=freeze_sha256,
        authorization_sha256=authorization_sha256,
        audit_sha256=audit_sha256,
        smoke_sha256=smoke_sha256,
        queue_sha256=queue_sha256,
    )
    if type(value) is not dict or set(value) != set(expected):
        raise RunnerError("ATTEMPT schema mismatch")
    expected_without_time = dict(expected)
    actual_without_time = dict(value)
    expected_without_time.pop("created_utc")
    created = actual_without_time.pop("created_utc", None)
    try:
        parse_utc(created)
    except ProtocolError as exc:
        raise RunnerError("ATTEMPT timestamp mismatch") from exc
    if actual_without_time != expected_without_time:
        raise RunnerError("ATTEMPT closure mismatch")
    return dict(value)


def dry_run(expected_freeze_sha256: str, expected_authorization_sha256: str) -> dict[str, Any]:
    freeze, queue = verify_freeze(expected_freeze_sha256, expected_authorization_sha256)
    request_hashes: set[str] = set()
    theory_rows = 0
    total_prompt_chars = 0
    for row in queue:
        request, aliases = build_request(row)
        if set(aliases) != set("ABCDE") or set(aliases.values()) != set("ABCDE"):
            raise RunnerError("request aliases are not a permutation")
        request_hashes.add(sha256_bytes(canonical_json_bytes(request)))
        theory_rows += int(bool(row["theory"]))
        total_prompt_chars += sum(len(message["content"]) for message in request["messages"])
    if len(request_hashes) != 185:
        raise RunnerError("request hashes are not unique")
    return {
        "schema_version": "generic-medium-nonstream-dry-run-v6",
        "freeze_sha256": expected_freeze_sha256,
        "rows": 185,
        "unique_requests": len(request_hashes),
        "theory_rows": theory_rows,
        "empty_theory_rows": 185 - theory_rows,
        "fixed_model_calls": 185,
        "workers": WORKERS,
        "max_attempts": MAX_ATTEMPTS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "total_prompt_chars": total_prompt_chars,
        "provider_calls_made": 0,
        "gold_accessed": False,
        "final_accessed": False,
        "runtime_opaque_ids": False,
        "freeze_state": freeze["state"],
    }


def _prediction_from_call(
    row: dict[str, Any], request: dict[str, Any], aliases: dict[str, str], call: dict[str, Any]
) -> dict[str, Any]:
    terminal = call["attempts"][-1]
    result = terminal.get("result") if terminal.get("success") is True else None
    prediction: str | None = None
    contract_error: str | None = None
    if type(result) is dict:
        try:
            prediction = map_model_answer(result["content"], aliases)
        except Exception as exc:
            contract_error = f"{type(exc).__name__}:{str(exc)[:500]}"
    return {
        "schema_version": "generic-medium-nonstream-content-prediction-v6",
        "content_sha256": row["content_sha256"],
        "request_sha256": sha256_bytes(canonical_json_bytes(request)),
        "prediction": prediction,
        "terminal_success": call["terminal_success"],
        "attempt_count": call["attempt_count"],
        "terminal_error_kind": terminal.get("error_kind"),
        "model_contract_error": contract_error,
        "gold_access": False,
        "final_access": False,
        "opaque_identifier_access": False,
    }


def _validate_predictions(queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _read_jsonl(PREDICTIONS_PATH)
    if len(rows) != 185:
        raise RunnerError("prediction denominator mismatch")
    for stored, queue_row in zip(rows, queue):
        request, aliases = build_request(queue_row)
        call = load_cached_call(request, cache_dir=CACHE_DIR)
        expected = _prediction_from_call(queue_row, request, aliases, call)
        if stored != expected:
            raise RunnerError("prediction/cache/queue closure mismatch")
    return rows


def _completion_value(
    *,
    freeze_sha256: str,
    authorization_sha256: str,
    audit_sha256: str,
    smoke_sha256: str,
    attempt: dict[str, Any],
    cache_hits: int,
) -> dict[str, Any]:
    return {
        "schema_version": "generic-medium-nonstream-dev-completion-v6",
        "created_utc": utc_now(),
        "started_utc": attempt["created_utc"],
        "freeze_sha256": freeze_sha256,
        "authorization_sha256": authorization_sha256,
        "independent_audit_sha256": audit_sha256,
        "protocol_smoke_sha256": smoke_sha256,
        "attempt_sha256": sha256_file(ATTEMPT_PATH),
        "model_id": MODEL_ID,
        "provider": PROVIDER_NAME,
        "rows": 185,
        "predictions": _artifact(PREDICTIONS_PATH, rows=185),
        "cache_hits": cache_hits,
        "provider_calls_upper_bound": 185 * MAX_ATTEMPTS,
        "gold_accessed": False,
        "final_accessed": False,
        "runtime_opaque_ids": False,
        "api_key_storage": "interactive_memory_only_not_persisted",
    }


def _validate_completion(
    value: Any,
    *,
    freeze_sha256: str,
    authorization_sha256: str,
    audit_sha256: str,
    smoke_sha256: str,
    attempt: dict[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "api_key_storage",
        "attempt_sha256",
        "authorization_sha256",
        "cache_hits",
        "created_utc",
        "final_accessed",
        "freeze_sha256",
        "gold_accessed",
        "independent_audit_sha256",
        "model_id",
        "predictions",
        "provider",
        "provider_calls_upper_bound",
        "protocol_smoke_sha256",
        "rows",
        "runtime_opaque_ids",
        "schema_version",
        "started_utc",
    }:
        raise RunnerError("completion schema mismatch")
    if (
        value.get("schema_version") != "generic-medium-nonstream-dev-completion-v6"
        or value.get("freeze_sha256") != freeze_sha256
        or value.get("authorization_sha256") != authorization_sha256
        or value.get("independent_audit_sha256") != audit_sha256
        or value.get("protocol_smoke_sha256") != smoke_sha256
        or value.get("attempt_sha256") != sha256_file(ATTEMPT_PATH)
        or value.get("started_utc") != attempt["created_utc"]
        or type(value.get("created_utc")) is not str
        or value.get("model_id") != MODEL_ID
        or value.get("provider") != PROVIDER_NAME
        or value.get("rows") != 185
        or type(value.get("cache_hits")) is not int
        or not 0 <= value["cache_hits"] <= 185
        or value.get("provider_calls_upper_bound") != 185 * MAX_ATTEMPTS
        or value.get("gold_accessed") is not False
        or value.get("final_accessed") is not False
        or value.get("runtime_opaque_ids") is not False
        or value.get("api_key_storage") != "interactive_memory_only_not_persisted"
    ):
        raise RunnerError("completion closure mismatch")
    created = parse_utc(value["created_utc"])
    started = parse_utc(attempt["created_utc"])
    if created < started:
        raise RunnerError("completion predates ATTEMPT")
    descriptor_path = _verify_descriptor(value["predictions"])
    if descriptor_path != PREDICTIONS_PATH.resolve():
        raise RunnerError("completion prediction path mismatch")
    return dict(value)


def execute(
    expected_freeze_sha256: str,
    expected_authorization_sha256: str,
    *,
    independent_audit: Path,
    expected_independent_audit_sha256: str,
    expected_protocol_smoke_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    freeze, queue = verify_freeze(expected_freeze_sha256, expected_authorization_sha256)
    audit_sha = validate_independent_audit(
        independent_audit,
        expected_independent_audit_sha256,
        freeze=freeze,
        freeze_sha256=expected_freeze_sha256,
        authorization_sha256=expected_authorization_sha256,
    )
    smoke_sha = validate_protocol_smoke(
        expected_protocol_smoke_sha256,
        freeze_sha256=expected_freeze_sha256,
        authorization_sha256=expected_authorization_sha256,
        audit_sha256=audit_sha,
    )
    queue_sha = freeze["artifacts"]["queue"]["sha256"]
    if ATTEMPT_PATH.exists():
        if not resume and not COMPLETION_PATH.exists():
            raise RunnerError("ATTEMPT exists; use --resume")
        attempt = _validate_attempt(
            read_json(ATTEMPT_PATH),
            freeze_sha256=expected_freeze_sha256,
            authorization_sha256=expected_authorization_sha256,
            audit_sha256=audit_sha,
            smoke_sha256=smoke_sha,
            queue_sha256=queue_sha,
        )
    else:
        if resume or PREDICTIONS_PATH.exists() or COMPLETION_PATH.exists() or (CACHE_DIR.exists() and any(CACHE_DIR.iterdir())):
            raise RunnerError("resume/output/cache exists without ATTEMPT")
        attempt = _attempt_value(
            freeze_sha256=expected_freeze_sha256,
            authorization_sha256=expected_authorization_sha256,
            audit_sha256=audit_sha,
            smoke_sha256=smoke_sha,
            queue_sha256=queue_sha,
        )
        exclusive_json(ATTEMPT_PATH, attempt)

    if COMPLETION_PATH.exists():
        if not PREDICTIONS_PATH.exists():
            raise RunnerError("completion exists without predictions")
        _validate_predictions(queue)
        return _validate_completion(
            read_json(COMPLETION_PATH),
            freeze_sha256=expected_freeze_sha256,
            authorization_sha256=expected_authorization_sha256,
            audit_sha256=audit_sha,
            smoke_sha256=smoke_sha,
            attempt=attempt,
        )
    if PREDICTIONS_PATH.exists():
        if not resume:
            raise RunnerError("predictions exist without completion; use --resume")
        _validate_predictions(queue)
        completion = _completion_value(
            freeze_sha256=expected_freeze_sha256,
            authorization_sha256=expected_authorization_sha256,
            audit_sha256=audit_sha,
            smoke_sha256=smoke_sha,
            attempt=attempt,
            cache_hits=185,
        )
        exclusive_json(COMPLETION_PATH, completion)
        return completion

    api_key = prompt_api_key()
    predictions: list[dict[str, Any]] = []
    cache_hits = 0
    for number, row in enumerate(queue, 1):
        request, aliases = build_request(row)
        call, cached = cached_call(request, api_key=api_key, cache_dir=CACHE_DIR)
        cache_hits += int(cached)
        predictions.append(_prediction_from_call(row, request, aliases, call))
        if number % 10 == 0 or number == len(queue):
            print(f"[medium-nonstream] {number}/{len(queue)} cache_hits={cache_hits}", flush=True)
    exclusive_bytes(
        PREDICTIONS_PATH,
        b"".join(canonical_json_bytes(row) for row in predictions),
        api_key=api_key,
    )
    _validate_predictions(queue)
    completion = _completion_value(
        freeze_sha256=expected_freeze_sha256,
        authorization_sha256=expected_authorization_sha256,
        audit_sha256=audit_sha,
        smoke_sha256=smoke_sha,
        attempt=attempt,
        cache_hits=cache_hits,
    )
    exclusive_json(COMPLETION_PATH, completion, api_key=api_key)
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--independent-audit")
    parser.add_argument("--expected-independent-audit-sha256")
    parser.add_argument("--expected-protocol-smoke-sha256")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run and args.resume:
        raise RunnerError("--resume is valid only with --execute")
    if args.execute and (
        not args.independent_audit
        or not args.expected_independent_audit_sha256
        or not args.expected_protocol_smoke_sha256
    ):
        raise RunnerError("live execution requires pinned independent audit and protocol smoke")
    result = dry_run(args.expected_freeze_sha256, args.expected_authorization_sha256) if args.dry_run else execute(
        args.expected_freeze_sha256,
        args.expected_authorization_sha256,
        independent_audit=Path(args.independent_audit),
        expected_independent_audit_sha256=args.expected_independent_audit_sha256,
        expected_protocol_smoke_sha256=args.expected_protocol_smoke_sha256,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
