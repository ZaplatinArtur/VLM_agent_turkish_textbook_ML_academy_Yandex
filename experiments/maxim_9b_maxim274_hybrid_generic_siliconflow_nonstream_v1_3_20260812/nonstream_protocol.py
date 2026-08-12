"""Strict OpenRouter non-streaming transport with append-only bounded retries.

Only an exact selected-provider overload 429 and typed EOF/timeout failures are
retryable. Provider, model, routing, cache, JSON closure, usage, and metadata are
validated before a response can be successful. The API key is memory-only.
"""

from __future__ import annotations

import datetime as dt
import getpass
import http.client
import json
import math
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

from generic_candidate import (
    MAX_TOKENS,
    MODEL_ID,
    PROVIDER_NAME,
    PROVIDER_QUANTIZATION,
    PROVIDER_ROUTED_MODEL_ID,
    PROVIDER_SLUG,
    REASONING,
    assert_request_blind,
    canonical_json_bytes,
    sha256_bytes,
)


API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (2, 8)
TIMEOUT_SECONDS = 900
WORKERS = 1
SECRET_PATTERN = re.compile(r"sk-or-v1-[A-Za-z0-9_-]{10,}")
RETRYABLE_KINDS = {
    "provider_overloaded_429",
    "transport_eof",
    "transport_timeout",
}
MIDSTREAM_ERROR_TYPES = {
    "eof": "midstream_eof",
    "stream_eof": "midstream_eof",
    "upstream_eof": "midstream_eof",
    "timeout": "midstream_timeout",
    "stream_timeout": "midstream_timeout",
    "upstream_timeout": "midstream_timeout",
}
MAX_SSE_LINE_BYTES = 1024 * 1024
MAX_SSE_TOTAL_BYTES = 16 * 1024 * 1024
MAX_SSE_JSON_EVENTS = 100_000
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


class StreamFailure(ProtocolError):
    def __init__(self, kind: str, detail: str, *, retryable: bool = False) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail
        self.retryable = retryable


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> dt.datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ProtocolError("timestamp must be UTC Z text")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProtocolError("timestamp is not ISO-8601 UTC") from exc
    if parsed.tzinfo != dt.timezone.utc:
        raise ProtocolError("timestamp timezone mismatch")
    return parsed


def stable_bytes(path: Path) -> bytes:
    before = path.stat()
    value = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(value) != before.st_size
    ):
        raise ProtocolError(f"file changed during read: {path}")
    return value


def sha256_file(path: Path) -> str:
    return sha256_bytes(stable_bytes(path))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(stable_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def _assert_secret_absent(data: bytes, api_key: str = "") -> None:
    text = data.decode("utf-8", errors="strict")
    if (api_key and api_key in text) or SECRET_PATTERN.search(text):
        raise ProtocolError("refusing to persist secret-bearing bytes")


def redact_secret(value: str, api_key: str = "") -> str:
    output = value.replace(api_key, "[REDACTED]") if api_key else value
    return SECRET_PATTERN.sub("[REDACTED]", output)


def exclusive_bytes(path: Path, data: bytes, *, api_key: str = "") -> None:
    _assert_secret_absent(data, api_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def exclusive_json(path: Path, value: Any, *, api_key: str = "") -> None:
    exclusive_bytes(path, canonical_json_bytes(value), api_key=api_key)


def prompt_api_key() -> str:
    value = getpass.getpass("OpenRouter API key (hidden, memory-only): ").strip()
    if not value.startswith("sk-or-v1-") or len(value) < 30:
        raise ProtocolError("invalid OpenRouter API key shape")
    return value


def _routing_validation(metadata: Any) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    attempts: Any = None
    pipeline: Any = None
    attempt: Any = None
    if type(metadata) is dict:
        endpoints = metadata.get("endpoints")
        available = endpoints.get("available") if type(endpoints) is dict else None
        if type(available) is list:
            selected = [
                item
                for item in available
                if type(item) is dict and item.get("selected") is True
            ]
        attempts = metadata.get("attempts")
        pipeline = metadata.get("pipeline")
        attempt = metadata.get("attempt")
    checks = {
        "metadata_present": type(metadata) is dict,
        "requested_model_exact": type(metadata) is dict and metadata.get("requested") == MODEL_ID,
        "strategy_direct": type(metadata) is dict and metadata.get("strategy") == "direct",
        "attempt_one": type(attempt) is int and attempt == 1,
        "one_selected_endpoint": len(selected) == 1,
        "selected_provider_exact": len(selected) == 1 and selected[0].get("provider") == PROVIDER_NAME,
        "selected_model_exact": len(selected) == 1 and selected[0].get("model") == PROVIDER_ROUTED_MODEL_ID,
        "pipeline_empty": pipeline in (None, []),
        "attempts_exact_if_present": attempts is None
        or (
            type(attempts) is list
            and len(attempts) == 1
            and type(attempts[0]) is dict
            and attempts[0].get("provider") == PROVIDER_NAME
            and attempts[0].get("model") == PROVIDER_ROUTED_MODEL_ID
            and attempts[0].get("status") == 200
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _decode_line(raw: bytes | str) -> str:
    if type(raw) is bytes:
        try:
            return raw.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            raise StreamFailure("sse_schema", "non-UTF8 SSE line") from exc
    if type(raw) is str:
        return raw.rstrip("\r\n")
    raise StreamFailure("sse_schema", "SSE iterator yielded a non-line")


def _event_payloads(lines: Iterable[bytes | str]) -> tuple[list[str], int, bool]:
    payloads: list[str] = []
    data_lines: list[str] = []
    keepalives = 0
    done = False
    total_bytes = 0

    def flush() -> None:
        nonlocal done
        if not data_lines:
            return
        payload = "\n".join(data_lines)
        data_lines.clear()
        if done:
            raise StreamFailure("sse_schema", "data appeared after [DONE]")
        if payload.strip() == "[DONE]":
            done = True
        else:
            payloads.append(payload)
            if len(payloads) > MAX_SSE_JSON_EVENTS:
                raise StreamFailure("sse_limits", "SSE JSON event limit exceeded")

    for raw in lines:
        raw_size = len(raw) if type(raw) is bytes else len(raw.encode("utf-8")) if type(raw) is str else 0
        if raw_size > MAX_SSE_LINE_BYTES:
            raise StreamFailure("sse_limits", "SSE physical line limit exceeded")
        total_bytes += raw_size
        if total_bytes > MAX_SSE_TOTAL_BYTES:
            raise StreamFailure("sse_limits", "SSE total byte limit exceeded")
        line = _decode_line(raw)
        if done and line != "":
            raise StreamFailure("sse_schema", "non-blank SSE line appeared after [DONE]")
        if line == "":
            flush()
            continue
        if line.startswith(":"):
            keepalives += 1
            continue
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
            continue
        raise StreamFailure("sse_schema", f"unsupported SSE field: {line[:80]}")
    flush()
    return payloads, keepalives, done


def _midstream_failure(error: Any) -> StreamFailure:
    metadata = error.get("metadata") if type(error) is dict else None
    error_type = metadata.get("error_type") if type(metadata) is dict else None
    kind = MIDSTREAM_ERROR_TYPES.get(error_type, "midstream_provider_error")
    retryable = kind in RETRYABLE_KINDS
    detail = f"mid-stream provider error type={error_type!r}"
    return StreamFailure(kind, detail, retryable=retryable)


def _validate_usage(value: Any) -> dict[str, Any]:
    if type(value) is not dict or not value:
        raise StreamFailure("sse_schema", "final usage must be a non-empty object")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        amount = value.get(key)
        if type(amount) is not int or amount < 0:
            raise StreamFailure("sse_schema", f"invalid usage field: {key}")
    if value["total_tokens"] != value["prompt_tokens"] + value["completion_tokens"]:
        raise StreamFailure("sse_schema", "usage token arithmetic mismatch")
    cost = value.get("cost")
    if cost is not None and (
        type(cost) not in (int, float)
        or isinstance(cost, bool)
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise StreamFailure("sse_schema", "invalid usage cost")
    return dict(value)


def parse_sse_lines(
    lines: Iterable[bytes | str], *, response_headers: Mapping[str, str]
) -> dict[str, Any]:
    payloads, keepalives, done = _event_payloads(lines)
    if not payloads:
        kind = "sse_schema" if done else "transport_eof"
        raise StreamFailure(kind, "SSE stream contained no JSON chunks", retryable=not done)

    events: list[dict[str, Any]] = []
    for payload in payloads:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StreamFailure("sse_schema", "SSE data is not JSON") from exc
        if type(event) is not dict:
            raise StreamFailure("sse_schema", "SSE JSON root is not an object")
        if event.get("error") is not None:
            raise _midstream_failure(event.get("error"))
        events.append(event)
    if not done:
        raise StreamFailure("transport_eof", "SSE stream ended before [DONE]", retryable=True)

    content_parts: list[str] = []
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    response_id: str | None = None
    returned_model: str | None = None
    returned_provider: str | None = None
    created: int | None = None
    finish_reasons: list[str] = []

    stop_event_index: int | None = None
    state = "content"
    terminal_variant: str | None = None
    for event_index, event in enumerate(events):
        if event.get("object") != "chat.completion.chunk":
            raise StreamFailure("sse_schema", "unexpected streaming object type")

        for key, expected_type, current_name in (
            ("id", str, "response_id"),
            ("model", str, "returned_model"),
            ("provider", str, "returned_provider"),
            ("created", int, "created"),
        ):
            incoming = event.get(key)
            if incoming is None:
                continue
            if type(incoming) is not expected_type:
                raise StreamFailure("sse_schema", f"invalid {key} type")
            current = {
                "response_id": response_id,
                "returned_model": returned_model,
                "returned_provider": returned_provider,
                "created": created,
            }[current_name]
            if current is not None and current != incoming:
                raise StreamFailure("sse_schema", f"inconsistent {key} across chunks")
            if current_name == "response_id":
                response_id = incoming
            elif current_name == "returned_model":
                returned_model = incoming
            elif current_name == "returned_provider":
                returned_provider = incoming
            else:
                created = incoming

        incoming_usage = event.get("usage")
        incoming_metadata = event.get("openrouter_metadata")
        choices = event.get("choices")
        if choices == []:
            if state != "stopped" or event_index != len(events) - 1:
                raise StreamFailure("sse_schema", "empty terminal event is out of order")
            if incoming_usage is None or incoming_metadata is None:
                raise StreamFailure("sse_schema", "empty terminal event lacks usage or metadata")
            usage = _validate_usage(incoming_usage)
            if type(incoming_metadata) is not dict or not incoming_metadata:
                raise StreamFailure("sse_schema", "openrouter_metadata is not an object")
            metadata = incoming_metadata
            state = "terminal"
            terminal_variant = "separate_empty_choices_usage_metadata"
            continue
        if state != "content":
            raise StreamFailure("sse_schema", "choice event appeared after terminal stop")
        if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
            raise StreamFailure("sse_schema", "stream must contain exactly choice index 0")
        choice = choices[0]
        if choice.get("index") not in (None, 0):
            raise StreamFailure("sse_schema", "unexpected streaming choice index")
        delta = choice.get("delta")
        if type(delta) is not dict:
            raise StreamFailure("sse_schema", "streaming delta is not an object")
        if delta.get("reasoning") not in (None, "") or delta.get("reasoning_details") not in (None, []):
            raise StreamFailure("reasoning_leak", "excluded reasoning appeared in SSE")
        piece = delta.get("content")
        if piece is not None:
            if type(piece) is not str:
                raise StreamFailure("sse_schema", "delta content is not text")
            content_parts.append(piece)
        finish = choice.get("finish_reason")
        native_finish = choice.get("native_finish_reason")
        if finish is None:
            if incoming_usage is not None or incoming_metadata is not None:
                raise StreamFailure("sse_schema", "usage or metadata appeared before terminal stop")
            if native_finish not in (None, ""):
                raise StreamFailure("sse_schema", "native finish appeared before terminal stop")
        else:
            if type(finish) is not str or finish_reasons or finish != "stop":
                raise StreamFailure("sse_schema", "invalid or duplicate finish reason")
            finish_reasons.append(finish)
            stop_event_index = event_index
            if native_finish not in (None, finish):
                raise StreamFailure("sse_schema", "native finish reason mismatch")
            if incoming_usage is None and incoming_metadata is None:
                state = "stopped"
            elif incoming_usage is not None and incoming_metadata is not None:
                if event_index != len(events) - 1:
                    raise StreamFailure("sse_schema", "combined terminal event is not final")
                usage = _validate_usage(incoming_usage)
                if type(incoming_metadata) is not dict or not incoming_metadata:
                    raise StreamFailure("sse_schema", "openrouter_metadata is not an object")
                metadata = incoming_metadata
                state = "terminal"
                terminal_variant = "combined_stop_usage_metadata"
            else:
                raise StreamFailure("sse_schema", "terminal stop must carry both usage and metadata or neither")

    content = "".join(content_parts)
    if finish_reasons != ["stop"]:
        kind = "finish_length" if finish_reasons == ["length"] else "finish_not_stop"
        raise StreamFailure(kind, f"finish reason closure mismatch: {finish_reasons}")
    if state != "terminal" or terminal_variant is None or stop_event_index is None:
        raise StreamFailure("sse_schema", "documented terminal SSE variant is incomplete")
    if not content:
        raise StreamFailure("sse_schema", "successful stream has empty content")
    if type(usage) is not dict:
        raise StreamFailure("sse_schema", "final usage chunk is required")
    if type(metadata) is not dict:
        raise StreamFailure("routing_validation", "final routing metadata is required")
    if response_id is None or returned_model is None or returned_provider is None or created is None:
        raise StreamFailure("sse_schema", "response identity fields are incomplete")
    if returned_model != MODEL_ID or returned_provider != PROVIDER_NAME:
        raise StreamFailure("routing_validation", "returned model/provider mismatch")
    routing = _routing_validation(metadata)
    if routing["passed"] is not True:
        raise StreamFailure("routing_validation", "OpenRouter routing metadata mismatch")
    lowered_headers = {str(key).casefold(): str(value) for key, value in response_headers.items()}
    generation_id = lowered_headers.get("x-generation-id")
    if generation_id != response_id:
        raise StreamFailure("routing_validation", "X-Generation-Id is absent or mismatched")
    cache_status = lowered_headers.get("x-openrouter-cache-status")
    if cache_status is not None and cache_status.strip().casefold() == "hit":
        raise StreamFailure("routing_validation", "provider cache hit is forbidden")
    return {
        "response_id": response_id,
        "generation_id": generation_id,
        "created": created,
        "returned_model": returned_model,
        "returned_provider": returned_provider,
        "provider_slug": PROVIDER_SLUG,
        "quantization": PROVIDER_QUANTIZATION,
        "finish_reason": "stop",
        "content": content,
        "usage": usage,
        "openrouter_metadata": metadata,
        "routing_validation": routing,
        "cache_status": cache_status,
        "sse_json_events": len(payloads),
        "sse_keepalive_comments": keepalives,
        "sse_done_seen": True,
        "terminal_variant": terminal_variant,
    }


def parse_nonstream_response(
    body: bytes, *, response_headers: Mapping[str, str]
) -> dict[str, Any]:
    """Validate one complete OpenRouter JSON response and return a closed result."""

    if len(body) > MAX_RESPONSE_BYTES:
        raise StreamFailure("response_too_large", "JSON response exceeds byte cap")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StreamFailure("response_json_schema", "response is not UTF-8 JSON") from exc
    if type(value) is not dict:
        raise StreamFailure("response_json_schema", "response root is not an object")
    choices = value.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise StreamFailure("response_choice_schema", "exactly one choice is required")
    choice = choices[0]
    if choice.get("index") not in (None, 0):
        raise StreamFailure("response_choice_schema", "choice index is not zero")
    if choice.get("finish_reason") != "stop" or choice.get("native_finish_reason") not in (None, "stop"):
        kind = "finish_length" if choice.get("finish_reason") == "length" else "finish_not_stop"
        raise StreamFailure(kind, "finish_reason closure mismatch")
    message = choice.get("message")
    if type(message) is not dict or type(message.get("content")) is not str or not message["content"]:
        raise StreamFailure("response_message_schema", "non-empty message content is required")
    if message.get("reasoning") not in (None, "") or message.get("reasoning_details") not in (None, []):
        raise StreamFailure("reasoning_leak", "excluded reasoning was returned")
    response_id = value.get("id")
    generation_id = response_headers.get("x-generation-id")
    if type(response_id) is not str or not response_id or generation_id != response_id:
        raise StreamFailure("generation_identity", "X-Generation-Id must equal response id")
    if value.get("model") != MODEL_ID or value.get("provider") != PROVIDER_NAME:
        raise StreamFailure("returned_identity", "returned model/provider mismatch")
    created = value.get("created")
    if type(created) is not int:
        raise StreamFailure("response_json_schema", "created must be an integer")
    usage = _validate_usage(value.get("usage"))
    details = usage.get("completion_tokens_details")
    reasoning_tokens = details.get("reasoning_tokens") if type(details) is dict else None
    # `exclude:true` suppresses reasoning text, not the accounting of tokens
    # consumed by medium reasoning. Accept only a finite non-negative count.
    if reasoning_tokens is not None and (
        type(reasoning_tokens) not in (int, float)
        or isinstance(reasoning_tokens, bool)
        or not math.isfinite(float(reasoning_tokens))
        or float(reasoning_tokens) < 0
    ):
        raise StreamFailure("usage_schema", "invalid reasoning token count")
    metadata = value.get("openrouter_metadata")
    routing = _routing_validation(metadata)
    if routing.get("passed") is not True:
        raise StreamFailure("routing_validation_failed", "strict direct routing metadata failed")
    cache_status = response_headers.get("x-openrouter-cache-status")
    if str(cache_status or "").strip().casefold() == "hit":
        raise StreamFailure("cache_hit", "cached response is forbidden")
    return {
        "response_id": response_id,
        "generation_id": generation_id,
        "created": created,
        "returned_model": value["model"],
        "returned_provider": value["provider"],
        "provider_slug": PROVIDER_SLUG,
        "quantization": PROVIDER_QUANTIZATION,
        "finish_reason": "stop",
        "native_finish_reason": choice.get("native_finish_reason"),
        "content": message["content"],
        "usage": usage,
        "openrouter_metadata": metadata,
        "routing_validation": routing,
        "cache_status": cache_status,
    }


def _exact_provider_overloaded_429(status: int, body: str) -> bool:
    if status != 429:
        return False
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return False
    error = value.get("error") if type(value) is dict else None
    metadata = error.get("metadata") if type(error) is dict else None
    if type(error) is not dict or error.get("code") != 429 or type(metadata) is not dict:
        return False
    if metadata.get("provider_name") != PROVIDER_NAME:
        return False
    return metadata.get("error_type") == "provider_overloaded" or (
        metadata.get("provider_error_code") == "engine_overloaded"
        and metadata.get("error_type") in (None, "provider_overloaded", "engine_overloaded")
    )


def _normalized_http_failure(status: int, body: str) -> tuple[str, str]:
    """Return only allowlisted routing codes; never persist provider/body text."""

    kind = "provider_overloaded_429" if _exact_provider_overloaded_429(status, body) else "http_error"
    safe: dict[str, Any] = {"http_status": status}
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        value = None
    error = value.get("error") if type(value) is dict else None
    metadata = error.get("metadata") if type(error) is dict else None
    if type(error) is dict and type(error.get("code")) is int:
        safe["openrouter_error_code"] = error["code"]
    if type(metadata) is dict:
        for source, target in (
            ("provider_name", "provider_name"),
            ("error_type", "error_type"),
            ("provider_error_code", "provider_error_code"),
        ):
            item = metadata.get(source)
            if type(item) is str and len(item) <= 80 and re.fullmatch(r"[A-Za-z0-9_.:-]+", item):
                safe[target] = item
    return kind, json.dumps(safe, sort_keys=True, separators=(",", ":"))


def _normalized_exception_detail(kind: str, exc: BaseException) -> str:
    """Exception messages may echo URLs, headers, or prompt text; retain type only."""

    return f"{kind}:{type(exc).__name__}"


def _typed_transport_kind(exc: BaseException) -> str | None:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "transport_timeout"
    if isinstance(
        exc,
        (
            EOFError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            ssl.SSLEOFError,
            ConnectionResetError,
            BrokenPipeError,
            ConnectionAbortedError,
        ),
    ):
        return "transport_eof"
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, BaseException):
        return _typed_transport_kind(exc.reason)
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def _failure_envelope(
    *, request_sha256: str, kind: str, detail: str, latency_ms: float
) -> dict[str, Any]:
    return {
        "schema_version": "generic-medium-nonstream-attempt-envelope-v1",
        "request_sha256": request_sha256,
        "requested_model": MODEL_ID,
        "provider_slug": PROVIDER_SLUG,
        "quantization": PROVIDER_QUANTIZATION,
        "success": False,
        "retryable": kind in RETRYABLE_KINDS,
        "error_kind": kind,
        "error_detail": detail[:1000],
        "latency_ms": round(latency_ms, 3),
        "result": None,
    }


def _success_envelope(
    *, request_sha256: str, result: dict[str, Any], latency_ms: float
) -> dict[str, Any]:
    return {
        "schema_version": "generic-medium-nonstream-attempt-envelope-v1",
        "request_sha256": request_sha256,
        "requested_model": MODEL_ID,
        "provider_slug": PROVIDER_SLUG,
        "quantization": PROVIDER_QUANTIZATION,
        "success": True,
        "retryable": False,
        "error_kind": None,
        "error_detail": None,
        "latency_ms": round(latency_ms, 3),
        "result": result,
    }


def validate_request_body(value: Any) -> dict[str, Any]:
    expected_keys = {
        "max_tokens",
        "messages",
        "model",
        "provider",
        "reasoning",
        "response_format",
        "stream",
        "temperature",
        "top_p",
    }
    provider = value.get("provider") if type(value) is dict else None
    response_format = value.get("response_format") if type(value) is dict else None
    json_schema = response_format.get("json_schema") if type(response_format) is dict else None
    schema = json_schema.get("schema") if type(json_schema) is dict else None
    messages = value.get("messages") if type(value) is dict else None
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("model") != MODEL_ID
        or value.get("stream") is not False
        or value.get("reasoning") != REASONING
        or value.get("max_tokens") != MAX_TOKENS
        or value.get("temperature") != 0.0
        or value.get("top_p") != 1.0
        or provider
        != {
            "only": [PROVIDER_SLUG],
            "allow_fallbacks": False,
            "require_parameters": True,
            "quantizations": [PROVIDER_QUANTIZATION],
            "data_collection": "deny",
            "zdr": True,
        }
        or type(messages) is not list
        or len(messages) != 2
        or [item.get("role") for item in messages if type(item) is dict] != ["system", "user"]
        or any(type(item.get("content")) is not str or not item["content"].strip() for item in messages)
        or type(response_format) is not dict
        or response_format.get("type") != "json_schema"
        or type(json_schema) is not dict
        or json_schema.get("strict") is not True
        or json_schema.get("name") != "maxim256_direct_answer_v1"
        or schema
        != {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "minLength": 1, "maxLength": 1000},
                "option_label": {"type": "string", "enum": ["A", "B", "C", "D", "E", "NA"]},
            },
            "required": ["answer", "option_label"],
            "additionalProperties": False,
        }
    ):
        raise ProtocolError("request contract mismatch")
    assert_request_blind(value)
    return dict(value)


_RESULT_KEYS = {
    "cache_status",
    "content",
    "created",
    "finish_reason",
    "generation_id",
    "native_finish_reason",
    "openrouter_metadata",
    "provider_slug",
    "quantization",
    "response_id",
    "returned_model",
    "returned_provider",
    "routing_validation",
    "usage",
}
_ENVELOPE_KEYS = {
    "error_detail",
    "error_kind",
    "latency_ms",
    "provider_slug",
    "quantization",
    "request_sha256",
    "requested_model",
    "result",
    "retryable",
    "schema_version",
    "success",
}


def validate_result(value: Any) -> dict[str, Any]:
    routing = value.get("routing_validation") if type(value) is dict else None
    expected_routing = _routing_validation(value.get("openrouter_metadata")) if type(value) is dict else None
    if (
        type(value) is not dict
        or set(value) != _RESULT_KEYS
        or value.get("response_id") != value.get("generation_id")
        or type(value.get("response_id")) is not str
        or value.get("returned_model") != MODEL_ID
        or value.get("returned_provider") != PROVIDER_NAME
        or value.get("provider_slug") != PROVIDER_SLUG
        or value.get("quantization") != PROVIDER_QUANTIZATION
        or type(value.get("created")) is not int
        or value.get("finish_reason") != "stop"
        or type(value.get("content")) is not str
        or not value["content"]
        or value.get("native_finish_reason") not in (None, "stop")
        or routing != expected_routing
        or type(routing) is not dict
        or routing.get("passed") is not True
        or (value.get("cache_status") is not None and type(value.get("cache_status")) is not str)
        or str(value.get("cache_status") or "").strip().casefold() == "hit"
    ):
        raise ProtocolError("successful SSE result closure mismatch")
    _validate_usage(value.get("usage"))
    return dict(value)


def validate_envelope(value: Any, *, request_sha256: str) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value) != _ENVELOPE_KEYS
        or value.get("schema_version") != "generic-medium-nonstream-attempt-envelope-v1"
        or value.get("request_sha256") != request_sha256
        or value.get("requested_model") != MODEL_ID
        or value.get("provider_slug") != PROVIDER_SLUG
        or value.get("quantization") != PROVIDER_QUANTIZATION
        or type(value.get("success")) is not bool
        or type(value.get("retryable")) is not bool
        or type(value.get("latency_ms")) not in (int, float)
        or isinstance(value.get("latency_ms"), bool)
        or not math.isfinite(float(value["latency_ms"]))
        or float(value["latency_ms"]) < 0
    ):
        raise ProtocolError("attempt envelope schema mismatch")
    if value["success"]:
        if (
            value["retryable"] is not False
            or value.get("error_kind") is not None
            or value.get("error_detail") is not None
            or type(value.get("result")) is not dict
        ):
            raise ProtocolError("successful attempt envelope is contradictory")
        validate_result(value["result"])
    else:
        kind, detail = value.get("error_kind"), value.get("error_detail")
        if (
            type(kind) is not str
            or not kind
            or type(detail) is not str
            or not detail
            or len(detail) > 1000
            or value.get("result") is not None
            or value["retryable"] is not (kind in RETRYABLE_KINDS)
        ):
            raise ProtocolError("failed attempt envelope is contradictory")
    return dict(value)


def call_openrouter_once(
    request_body: dict[str, Any], *, api_key: str, timeout: int = TIMEOUT_SECONDS
) -> dict[str, Any]:
    validate_request_body(request_body)
    request_sha = sha256_bytes(canonical_json_bytes(request_body))
    started = perf_counter()
    try:
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-OpenRouter-Title": "Maxim274 Hybrid V3.1 generic OCR-only SiliconFlow",
                "X-OpenRouter-Metadata": "enabled",
                "X-OpenRouter-Cache": "false",
            },
            method="POST",
        )
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            headers = {key.casefold(): value for key, value in response.headers.items()}
            content_type = headers.get("content-type", "").casefold()
            if not content_type.startswith("application/json"):
                raise StreamFailure("response_json_schema", "response is not application/json")
            body = response.read(MAX_RESPONSE_BYTES + 1)
            result = parse_nonstream_response(body, response_headers=headers)
        envelope = _success_envelope(
            request_sha256=request_sha,
            result=result,
            latency_ms=(perf_counter() - started) * 1000,
        )
        return validate_envelope(envelope, request_sha256=request_sha)
    except urllib.error.HTTPError as exc:
        body = exc.read(65537).decode("utf-8", errors="replace")
        if len(body) > 65536:
            body = ""
        kind, detail = _normalized_http_failure(exc.code, body)
    except StreamFailure as exc:
        kind, detail = exc.kind, exc.detail
    except Exception as exc:
        kind = _typed_transport_kind(exc) or "transport_untyped"
        detail = _normalized_exception_detail(kind, exc)
    envelope = _failure_envelope(
        request_sha256=request_sha,
        kind=kind,
        detail=redact_secret(detail, api_key),
        latency_ms=(perf_counter() - started) * 1000,
    )
    return validate_envelope(envelope, request_sha256=request_sha)


def _attempt_path(cache_dir: Path, request_sha256: str, number: int) -> Path:
    return cache_dir / "attempts" / f"{request_sha256}.attempt-{number}.json"


def _intent_path(cache_dir: Path, request_sha256: str, number: int) -> Path:
    return cache_dir / "intents" / f"{request_sha256}.attempt-{number}.json"


def _attempt_intent(request_sha256: str, attempt_number: int) -> dict[str, Any]:
    return {
        "schema_version": "generic-medium-nonstream-attempt-intent-v1",
        "created_utc": utc_now(),
        "request_sha256": request_sha256,
        "attempt_number": attempt_number,
        "dispatch_policy": "single_dispatch_no_replay_after_ambiguous_power_loss",
    }


def _validate_attempt_intent(
    value: Any, *, request_sha256: str, attempt_number: int
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "attempt_number",
            "created_utc",
            "dispatch_policy",
            "request_sha256",
            "schema_version",
        }
        or value.get("schema_version") != "generic-medium-nonstream-attempt-intent-v1"
        or value.get("request_sha256") != request_sha256
        or value.get("attempt_number") != attempt_number
        or value.get("dispatch_policy")
        != "single_dispatch_no_replay_after_ambiguous_power_loss"
    ):
        raise ProtocolError("attempt intent closure mismatch")
    parse_utc(value.get("created_utc"))
    return dict(value)


def _validate_attempt_journal(
    value: Any,
    *,
    request_sha256: str,
    attempt_number: int,
    intent_sha256: str,
    intent_created_utc: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "attempt_number",
        "envelope",
        "intent_sha256",
        "recorded_utc",
        "request_sha256",
        "schema_version",
    }:
        raise ProtocolError("attempt journal schema mismatch")
    envelope = value.get("envelope")
    if (
        value.get("schema_version") != "generic-medium-nonstream-attempt-journal-v1"
        or value.get("request_sha256") != request_sha256
        or value.get("attempt_number") != attempt_number
        or value.get("intent_sha256") != intent_sha256
        or type(envelope) is not dict
    ):
        raise ProtocolError("attempt journal closure mismatch")
    if parse_utc(value.get("recorded_utc")) < parse_utc(intent_created_utc):
        raise ProtocolError("attempt journal predates dispatch intent")
    return validate_envelope(envelope, request_sha256=request_sha256)


def _numbered_paths(directory: Path, request_sha256: str) -> dict[int, Path]:
    if not directory.exists():
        return {}
    paths: dict[int, Path] = {}
    prefix = f"{request_sha256}.attempt-"
    for path in directory.glob(f"{request_sha256}.attempt-*.json"):
        suffix = path.name[len(prefix) : -len(".json")]
        if not suffix.isdigit():
            raise ProtocolError("malformed attempt journal filename")
        number = int(suffix)
        if number in paths or not 1 <= number <= MAX_ATTEMPTS:
            raise ProtocolError("attempt artifact number mismatch")
        paths[number] = path
    if sorted(paths) != list(range(1, len(paths) + 1)):
        raise ProtocolError("attempt artifacts are not contiguous")
    return paths


def _load_attempt_state(
    cache_dir: Path, request_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    intent_paths = _numbered_paths(cache_dir / "intents", request_sha256)
    journal_paths = _numbered_paths(cache_dir / "attempts", request_sha256)
    if len(journal_paths) > len(intent_paths) or len(intent_paths) > len(journal_paths) + 1:
        raise ProtocolError("intent/journal cardinality mismatch")
    intents = [
        _validate_attempt_intent(
            read_json(intent_paths[number]),
            request_sha256=request_sha256,
            attempt_number=number,
        )
        for number in sorted(intent_paths)
    ]
    attempts = [
        _validate_attempt_journal(
            read_json(journal_paths[number]),
            request_sha256=request_sha256,
            attempt_number=number,
            intent_sha256=sha256_file(intent_paths[number]),
            intent_created_utc=intents[number - 1]["created_utc"],
        )
        for number in sorted(journal_paths)
    ]
    for number in range(1, len(attempts)):
        previous = read_json(journal_paths[number])
        if parse_utc(intents[number]["created_utc"]) < parse_utc(previous["recorded_utc"]):
            raise ProtocolError("next dispatch intent predates previous journal")
    if any(item.get("retryable") is not True for item in attempts[:-1]):
        raise ProtocolError("attempt exists after terminal response")
    orphan_intent = len(intents) == len(attempts) + 1
    if orphan_intent and attempts and attempts[-1].get("retryable") is not True:
        raise ProtocolError("orphan intent exists after terminal response")
    return attempts, intents, orphan_intent


def _load_attempts(cache_dir: Path, request_sha256: str) -> list[dict[str, Any]]:
    attempts, _intents, orphan_intent = _load_attempt_state(cache_dir, request_sha256)
    if orphan_intent:
        raise ProtocolError("unresolved pre-dispatch intent")
    return attempts


def _persist_attempt_journal(
    *,
    cache_dir: Path,
    request_sha256: str,
    attempt_number: int,
    intent: dict[str, Any],
    envelope: dict[str, Any],
    api_key: str,
) -> None:
    intent_path = _intent_path(cache_dir, request_sha256, attempt_number)
    journal = {
        "schema_version": "generic-medium-nonstream-attempt-journal-v1",
        "recorded_utc": utc_now(),
        "request_sha256": request_sha256,
        "attempt_number": attempt_number,
        "intent_sha256": sha256_file(intent_path),
        "envelope": envelope,
    }
    _validate_attempt_journal(
        journal,
        request_sha256=request_sha256,
        attempt_number=attempt_number,
        intent_sha256=journal["intent_sha256"],
        intent_created_utc=intent["created_utc"],
    )
    exclusive_json(_attempt_path(cache_dir, request_sha256, attempt_number), journal, api_key=api_key)


def _final_call(request_sha256: str, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    if not attempts or len(attempts) > MAX_ATTEMPTS:
        raise ProtocolError("invalid final attempt count")
    attempts = [validate_envelope(item, request_sha256=request_sha256) for item in attempts]
    if any(item.get("retryable") is not True for item in attempts[:-1]):
        raise ProtocolError("non-retryable response was retried")
    if len(attempts) < MAX_ATTEMPTS and attempts[-1].get("retryable") is True:
        raise ProtocolError("retry sequence is incomplete")
    value = {
        "schema_version": "generic-medium-nonstream-final-call-v1",
        "request_sha256": request_sha256,
        "attempt_count": len(attempts),
        "attempts": [
            {**attempt, "attempt_number": number}
            for number, attempt in enumerate(attempts, 1)
        ],
        "terminal_success": attempts[-1].get("success") is True,
        "retry_policy": {
            "max_attempts": MAX_ATTEMPTS,
            "delays_seconds": list(RETRY_DELAYS_SECONDS),
            "retryable_kinds": sorted(RETRYABLE_KINDS),
        },
    }
    return validate_final_call(value, request_sha256=request_sha256)


def validate_final_call(value: Any, *, request_sha256: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "attempt_count",
        "attempts",
        "request_sha256",
        "retry_policy",
        "schema_version",
        "terminal_success",
    }:
        raise ProtocolError("final call schema mismatch")
    attempts = value.get("attempts")
    if (
        value.get("schema_version") != "generic-medium-nonstream-final-call-v1"
        or value.get("request_sha256") != request_sha256
        or type(value.get("attempt_count")) is not int
        or type(attempts) is not list
        or not 1 <= len(attempts) <= MAX_ATTEMPTS
        or value["attempt_count"] != len(attempts)
        or value.get("retry_policy")
        != {
            "max_attempts": MAX_ATTEMPTS,
            "delays_seconds": list(RETRY_DELAYS_SECONDS),
            "retryable_kinds": sorted(RETRYABLE_KINDS),
        }
        or type(value.get("terminal_success")) is not bool
    ):
        raise ProtocolError("final call closure mismatch")
    normalized: list[dict[str, Any]] = []
    for number, item in enumerate(attempts, 1):
        if type(item) is not dict or item.get("attempt_number") != number:
            raise ProtocolError("final call attempt order mismatch")
        envelope = dict(item)
        del envelope["attempt_number"]
        normalized.append(validate_envelope(envelope, request_sha256=request_sha256))
    if any(item["retryable"] is not True for item in normalized[:-1]):
        raise ProtocolError("final call retried a terminal attempt")
    if len(normalized) < MAX_ATTEMPTS and normalized[-1]["retryable"] is True:
        raise ProtocolError("final call stopped before retry exhaustion")
    if value["terminal_success"] is not (normalized[-1]["success"] is True):
        raise ProtocolError("final call terminal success mismatch")
    return dict(value)


def call_with_retries(
    request_body: dict[str, Any], *, api_key: str, cache_dir: Path
) -> dict[str, Any]:
    validate_request_body(request_body)
    request_sha = sha256_bytes(canonical_json_bytes(request_body))
    attempts, intents, orphan_intent = _load_attempt_state(cache_dir, request_sha)
    if orphan_intent:
        number = len(intents)
        envelope = _failure_envelope(
            request_sha256=request_sha,
            kind="ambiguous_inflight_after_power_loss",
            detail="pre-dispatch intent exists without immutable response journal; replay forbidden",
            latency_ms=0,
        )
        _persist_attempt_journal(
            cache_dir=cache_dir,
            request_sha256=request_sha,
            attempt_number=number,
            intent=intents[-1],
            envelope=envelope,
            api_key=api_key,
        )
        attempts.append(envelope)
        return _final_call(request_sha, attempts)
    if attempts and (attempts[-1].get("retryable") is not True or len(attempts) == MAX_ATTEMPTS):
        return _final_call(request_sha, attempts)
    if attempts:
        time.sleep(RETRY_DELAYS_SECONDS[len(attempts) - 1])
    for number in range(len(attempts) + 1, MAX_ATTEMPTS + 1):
        intent = _attempt_intent(request_sha, number)
        exclusive_json(_intent_path(cache_dir, request_sha, number), intent, api_key=api_key)
        envelope = call_openrouter_once(request_body, api_key=api_key)
        _persist_attempt_journal(
            cache_dir=cache_dir,
            request_sha256=request_sha,
            attempt_number=number,
            intent=intent,
            envelope=envelope,
            api_key=api_key,
        )
        attempts.append(envelope)
        if envelope.get("retryable") is not True or number == MAX_ATTEMPTS:
            return _final_call(request_sha, attempts)
        time.sleep(RETRY_DELAYS_SECONDS[number - 1])
    raise AssertionError("bounded retry loop did not return")


def cached_call(
    request_body: dict[str, Any], *, api_key: str, cache_dir: Path
) -> tuple[dict[str, Any], bool]:
    validate_request_body(request_body)
    request_sha = sha256_bytes(canonical_json_bytes(request_body))
    path = cache_dir / f"{request_sha}.json"
    if path.is_file():
        return load_cached_call(request_body, cache_dir=cache_dir), True
    value = call_with_retries(request_body, api_key=api_key, cache_dir=cache_dir)
    exclusive_json(path, value, api_key=api_key)
    return value, False


def load_cached_call(request_body: dict[str, Any], *, cache_dir: Path) -> dict[str, Any]:
    """Load a cache only when its exact append-only journals prove provenance."""

    validate_request_body(request_body)
    request_sha = sha256_bytes(canonical_json_bytes(request_body))
    path = cache_dir / f"{request_sha}.json"
    if not path.is_file():
        raise ProtocolError("required stage cache is missing")
    value = validate_final_call(read_json(path), request_sha256=request_sha)
    journal_attempts = _load_attempts(cache_dir, request_sha)
    expected = _final_call(request_sha, journal_attempts)
    if value != expected:
        raise ProtocolError("stage cache does not match append-only attempt journals")
    return value
