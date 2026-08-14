"""Run the frozen 274-row theory-only candidate on local Qwen vLLM endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import protocol


FREEZE = protocol.HERE / "EXECUTION_FREEZE.json"
FREEZE_SHA = protocol.HERE / "EXECUTION_FREEZE_SHA256.txt"
ATTEMPT = protocol.HERE / "ATTEMPT.json"
COMPLETION = protocol.HERE / "COMPLETION.json"
SOLVER = protocol.RUNS / "solver.jsonl"
JOURNAL = protocol.RUNS / "journal"
TIMEOUT_SECONDS = 900


def _verify_artifact(descriptor: Any) -> Path:
    if type(descriptor) is not dict or set(descriptor) not in (
        {"path", "sha256", "size"},
        {"path", "sha256", "size", "rows"},
    ):
        raise protocol.ProtocolError("freeze artifact descriptor mismatch")
    path = (protocol.HERE / descriptor["path"]).resolve()
    try:
        path.relative_to(protocol.HERE.resolve())
    except ValueError as exc:
        raise protocol.ProtocolError("freeze artifact escapes experiment") from exc
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["size"]
        or protocol.sha256_file(path) != descriptor["sha256"]
    ):
        raise protocol.ProtocolError(f"freeze artifact closure mismatch: {path}")
    if "rows" in descriptor and len(protocol.read_jsonl(path)) != descriptor["rows"]:
        raise protocol.ProtocolError("freeze artifact row count mismatch")
    return path


def verify_freeze(expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        not FREEZE.is_file()
        or not FREEZE_SHA.is_file()
        or protocol.sha256_file(FREEZE) != expected_sha256
        or FREEZE_SHA.read_text(encoding="ascii").strip() != expected_sha256
    ):
        raise protocol.ProtocolError("execution freeze external pin mismatch")
    value = protocol.read_json(FREEZE)
    if (
        value.get("schema_version") != "maxim274-theory-only-local-vllm-execution-freeze-v1"
        or value.get("state") != "frozen_unexecuted_unscored"
        or value.get("rows") != protocol.ROWS
        or value.get("model")
        != {"id": protocol.MODEL_ID, "revision": protocol.MODEL_REVISION}
    ):
        raise protocol.ProtocolError("execution freeze semantic mismatch")
    for descriptor in value.get("implementation", {}).values():
        _verify_artifact(descriptor)
    artifacts = value.get("artifacts")
    if type(artifacts) is not dict:
        raise protocol.ProtocolError("execution freeze artifact map missing")
    paths = {key: _verify_artifact(item) for key, item in artifacts.items()}
    queue = protocol.read_jsonl(paths["queue"])
    alignment = protocol.read_jsonl(paths["alignment"])
    corpus = protocol.read_jsonl(paths["strict_theory_corpus"])
    protocol.validate_theory_rows(corpus)
    if len(queue) != protocol.ROWS or len(alignment) != protocol.ROWS:
        raise protocol.ProtocolError("runtime denominator mismatch")
    ids: set[str] = set()
    for index, (row, align) in enumerate(zip(queue, alignment)):
        if type(align) is not dict or set(align) != {"schema_version", "task_id"}:
            raise protocol.ProtocolError("outer alignment schema mismatch")
        task_id = align.get("task_id")
        if type(task_id) is not str or task_id in ids:
            raise protocol.ProtocolError("outer alignment identity mismatch")
        ids.add(task_id)
        if type(row) is not dict or set(row) != {
            "schema_version",
            "content_sha256",
            "public",
            "retrieval",
            "primary_request_sha256",
        }:
            raise protocol.ProtocolError(f"queue row {index} schema mismatch")
        public = row["public"]
        if row["content_sha256"] != protocol.content_sha256(public):
            raise protocol.ProtocolError("queue content hash mismatch")
        retrieval = protocol.validate_retrieval(public, row["retrieval"])
        expected_retrieval = protocol.retrieve_theory(public, corpus)
        if retrieval != expected_retrieval:
            raise protocol.ProtocolError("frozen retrieval does not reproduce")
        request = protocol.build_primary_request(public, retrieval)
        if protocol.sha256_bytes(protocol.canonical_json_bytes(request)) != row["primary_request_sha256"]:
            raise protocol.ProtocolError("primary request does not reproduce")
    serialized = protocol.jsonl_bytes(queue)
    if b"task_id" in serialized or b"controller_id" in serialized:
        raise protocol.ProtocolError("identity leaked into runtime queue")
    return value, queue, alignment


def _url(base_url: str, suffix: str) -> str:
    return base_url.rstrip("/") + suffix


def _get_models(base_url: str, timeout: int) -> list[str]:
    target = _url(base_url, "/models")
    request = urllib.request.Request(target, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.geturl() != target:
            raise protocol.ProtocolError("vLLM models endpoint redirected")
        value = json.loads(response.read().decode("utf-8"))
    return [
        item["id"]
        for item in value.get("data", [])
        if type(item) is dict and type(item.get("id")) is str
    ]


def _post(base_url: str, request_value: Mapping[str, Any], timeout: int) -> dict[str, Any]:
    target = _url(base_url, "/chat/completions")
    request = urllib.request.Request(
        target,
        data=json.dumps(request_value, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.geturl() != target:
                raise protocol.ProtocolError("vLLM completion endpoint redirected")
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise protocol.ProtocolError(f"vLLM HTTP {exc.code}: {body[:500]}") from exc
    latency = time.monotonic() - started
    choices = value.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise protocol.ProtocolError("vLLM returned malformed choices")
    message = choices[0].get("message")
    if type(message) is not dict:
        raise protocol.ProtocolError("vLLM returned malformed message")
    usage = value.get("usage") if type(value.get("usage")) is dict else {}
    return {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "finish_reason": choices[0].get("finish_reason"),
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        },
        "latency_s": round(latency, 6),
        "response_id": value.get("id"),
    }


def _call_trace(stage: str, response: Mapping[str, Any], endpoint_slot: int) -> dict[str, Any]:
    return {
        "stage": stage,
        "endpoint_slot": endpoint_slot,
        "finish_reason": response.get("finish_reason"),
        "latency_s": response.get("latency_s"),
        "input_tokens": response.get("usage", {}).get("input_tokens"),
        "output_tokens": response.get("usage", {}).get("output_tokens"),
        "retry": False,
    }


def _execute_row(
    index: int,
    queue_row: Mapping[str, Any],
    align: Mapping[str, Any],
    base_url: str,
    endpoint_slot: int,
    timeout: int,
) -> dict[str, Any]:
    public = queue_row["public"]
    retrieval = queue_row["retrieval"]
    traces: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0
    total_latency = 0.0
    raw_contents: list[dict[str, str]] = []
    primary_error: str | None = None
    answer: str | None = None
    reasoning = ""
    decision_mode = "v6_2_strict_success"

    try:
        response = _post(base_url, protocol.build_primary_request(public, retrieval), timeout)
        traces.append(_call_trace("v6.2-primary", response, endpoint_slot))
        total_input += int(response["usage"].get("input_tokens") or 0)
        total_output += int(response["usage"].get("output_tokens") or 0)
        total_latency += float(response["latency_s"])
        raw_contents.append({"stage": "v6.2-primary", "content": response["content"]})
        if response.get("finish_reason") != "stop":
            raise protocol.ProtocolError("primary finish_reason is not stop")
        answer = protocol.parse_answer_content(
            response["content"], str(public["answer_type"])
        )["final_answer"]
        reasoning = str(response.get("reasoning_content") or "")
    except Exception as exc:
        primary_error = f"{type(exc).__name__}:{str(exc)[:500]}"

    fallback_candidates: list[dict[str, str]] = []
    fallback_errors: list[str] = []
    if answer is None:
        for variant in protocol.FALLBACK_VARIANTS:
            try:
                response = _post(
                    base_url,
                    protocol.build_fallback_request(public, retrieval, variant),
                    timeout,
                )
                traces.append(_call_trace(f"v5-{variant}", response, endpoint_slot))
                total_input += int(response["usage"].get("input_tokens") or 0)
                total_output += int(response["usage"].get("output_tokens") or 0)
                total_latency += float(response["latency_s"])
                raw_contents.append({"stage": f"v5-{variant}", "content": response["content"]})
                if response.get("finish_reason") != "stop":
                    raise protocol.ProtocolError("fallback finish_reason is not stop")
                fallback_candidates.append(
                    protocol.parse_answer_content(
                        response["content"], str(public["answer_type"]), evidence=True
                    )
                )
            except Exception as exc:
                fallback_errors.append(f"{variant}:{type(exc).__name__}:{str(exc)[:300]}")
        answer, decision_mode = protocol.choose_fallback(
            fallback_candidates, str(public["answer_type"])
        )
        if answer is None and decision_mode == "v5_arbiter_required":
            try:
                response = _post(
                    base_url,
                    protocol.build_arbiter_request(public, retrieval, fallback_candidates),
                    timeout,
                )
                traces.append(_call_trace("v5-blind-arbiter", response, endpoint_slot))
                total_input += int(response["usage"].get("input_tokens") or 0)
                total_output += int(response["usage"].get("output_tokens") or 0)
                total_latency += float(response["latency_s"])
                raw_contents.append({"stage": "v5-blind-arbiter", "content": response["content"]})
                if response.get("finish_reason") != "stop":
                    raise protocol.ProtocolError("arbiter finish_reason is not stop")
                answer = protocol.parse_answer_content(
                    response["content"], str(public["answer_type"])
                )["final_answer"]
                decision_mode = "v5_blind_arbiter"
            except Exception as exc:
                fallback_errors.append(f"arbiter:{type(exc).__name__}:{str(exc)[:300]}")
                decision_mode = "v5_arbiter_failed"
        reasoning = " | ".join(item["evidence"] for item in fallback_candidates)

    error: str | None = None
    if answer is None:
        error = "; ".join(
            item for item in [primary_error, *fallback_errors] if item
        )[:2000] or "all_generic_theory_stages_failed"
    return {
        "task_id": align["task_id"],
        "condition": protocol.CONDITION,
        "model": protocol.MODEL_ID,
        "prompt_version": protocol.PROMPT_VERSION,
        "final_answer": answer or "",
        "solution_steps": "",
        "reasoning": reasoning,
        "forced_answer": False,
        "raw_response": json.dumps(raw_contents, ensure_ascii=False, separators=(",", ":")),
        "generation": {
            "temperature": protocol.TEMPERATURE,
            "top_p": protocol.TOP_P,
            "max_tokens": protocol.PRIMARY_MAX_TOKENS,
            "structured_mode": "response_format",
            "enable_thinking": decision_mode == "v6_2_strict_success",
            "gold_access": False,
            "outcome_access": False,
            "task_id_in_wire": False,
            "call_count": len(traces),
            "retry_calls": 0,
            "decision_mode": decision_mode,
            "primary_error": primary_error,
            "fallback_errors": fallback_errors,
            "retrieval_hits": len(retrieval),
            "call_traces": traces,
        },
        "tool_calls": [],
        "usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "latency_s": round(total_latency, 6),
        },
        "error": error,
    }


def _intent_path(index: int) -> Path:
    return JOURNAL / f"{index:04d}.intent.json"


def _result_path(index: int) -> Path:
    return JOURNAL / f"{index:04d}.result.json"


def _load_or_execute(
    index: int,
    queue_row: Mapping[str, Any],
    align: Mapping[str, Any],
    base_url: str,
    endpoint_slot: int,
    timeout: int,
) -> dict[str, Any]:
    intent_path, result_path = _intent_path(index), _result_path(index)
    if result_path.exists():
        value = protocol.read_json(result_path)
        if value.get("task_id") != align["task_id"]:
            raise protocol.ProtocolError("journal result alignment mismatch")
        return value
    if intent_path.exists():
        # Never replay an ambiguous request after power loss. It stays wrong.
        value = {
            "task_id": align["task_id"],
            "condition": protocol.CONDITION,
            "model": protocol.MODEL_ID,
            "prompt_version": protocol.PROMPT_VERSION,
            "final_answer": "",
            "solution_steps": "",
            "reasoning": "",
            "forced_answer": False,
            "raw_response": "",
            "generation": {
                "gold_access": False,
                "outcome_access": False,
                "task_id_in_wire": False,
                "call_count": 0,
                "retry_calls": 0,
                "decision_mode": "ambiguous_interrupted_attempt_fail_closed",
                "retrieval_hits": len(queue_row["retrieval"]),
                "call_traces": [],
            },
            "tool_calls": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "latency_s": 0.0},
            "error": "ambiguous_interrupted_attempt",
        }
        protocol.exclusive_json(result_path, value)
        return value
    protocol.exclusive_json(
        intent_path,
        {
            "schema_version": "maxim274-theory-only-row-intent-v1",
            "row_index": index,
            "content_sha256": queue_row["content_sha256"],
            "endpoint_slot": endpoint_slot,
            "request_sha256": queue_row["primary_request_sha256"],
            "replay_after_ambiguous_interruption": False,
        },
    )
    value = _execute_row(index, queue_row, align, base_url, endpoint_slot, timeout)
    protocol.exclusive_json(result_path, value)
    return value


def dry_run(expected_freeze_sha256: str, base_urls: Sequence[str]) -> dict[str, Any]:
    freeze, queue, alignment = verify_freeze(expected_freeze_sha256)
    request_hashes: list[str] = []
    for row in queue:
        request = protocol.build_primary_request(row["public"], row["retrieval"])
        request_hashes.append(protocol.sha256_bytes(protocol.canonical_json_bytes(request)))
    return {
        "schema_version": "maxim274-theory-only-dry-run-v1",
        "status": "ready_no_model_calls",
        "freeze_sha256": expected_freeze_sha256,
        "freeze_state": freeze["state"],
        "rows": len(queue),
        "alignment_rows": len(alignment),
        "unique_content_requests": len(set(request_hashes)),
        "endpoints_declared": len(base_urls),
        "image_bytes_sent": False,
        "task_id_in_wire": False,
        "database_or_source_router": False,
        "model_calls": 0,
    }


def execute(
    expected_freeze_sha256: str,
    base_urls: Sequence[str],
    *,
    workers: int,
    timeout: int,
    resume: bool,
) -> dict[str, Any]:
    _freeze, queue, alignment = verify_freeze(expected_freeze_sha256)
    if not base_urls or len(set(base_urls)) != len(base_urls):
        raise protocol.ProtocolError("one or more unique vLLM base URLs are required")
    if not 1 <= workers <= 64 or timeout != TIMEOUT_SECONDS:
        raise protocol.ProtocolError("runtime worker/timeout contract mismatch")
    if SOLVER.exists() or COMPLETION.exists():
        raise protocol.ProtocolError("completed output already exists")
    if ATTEMPT.exists():
        if not resume:
            raise protocol.ProtocolError("attempt exists; use --resume")
        attempt = protocol.read_json(ATTEMPT)
        if (
            attempt.get("schema_version") != "maxim274-theory-only-attempt-v1"
            or attempt.get("freeze_sha256") != expected_freeze_sha256
            or attempt.get("base_urls") != list(base_urls)
        ):
            raise protocol.ProtocolError("resume attempt closure mismatch")
    else:
        if resume:
            raise protocol.ProtocolError("--resume requested without attempt")
        for base_url in base_urls:
            models = _get_models(base_url, timeout=30)
            if protocol.MODEL_ID not in models:
                raise protocol.ProtocolError(
                    f"pinned model is not served by {base_url}: {models}"
                )
        protocol.exclusive_json(
            ATTEMPT,
            {
                "schema_version": "maxim274-theory-only-attempt-v1",
                "freeze_sha256": expected_freeze_sha256,
                "base_urls": list(base_urls),
                "model": protocol.MODEL_ID,
                "workers": workers,
                "timeout_seconds": timeout,
                "rows": protocol.ROWS,
                "gold_access": False,
                "outcome_access": False,
                "non_generic_fallback": False,
            },
        )
    JOURNAL.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any] | None] = [None] * protocol.ROWS
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {}
        for index, (row, align) in enumerate(zip(queue, alignment)):
            slot = index % len(base_urls)
            future = pool.submit(
                _load_or_execute,
                index,
                row,
                align,
                base_urls[slot],
                slot,
                timeout,
            )
            future_map[future] = index
        completed = 0
        for future in concurrent.futures.as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == protocol.ROWS:
                print(f"[maxim274-theory-only] {completed}/{protocol.ROWS}", flush=True)
    final = [row for row in results if row is not None]
    if len(final) != protocol.ROWS or len({row["task_id"] for row in final}) != protocol.ROWS:
        raise protocol.ProtocolError("solver output denominator/alignment mismatch")
    protocol.exclusive_bytes(SOLVER, protocol.jsonl_bytes(final))
    completion = {
        "schema_version": "maxim274-theory-only-completion-v1",
        "freeze_sha256": expected_freeze_sha256,
        "attempt_sha256": protocol.sha256_file(ATTEMPT),
        "solver": protocol.artifact(SOLVER, rows=protocol.ROWS),
        "rows": protocol.ROWS,
        "errors": sum(row["error"] is not None for row in final),
        "decision_modes": dict(Counter(row["generation"]["decision_mode"] for row in final)),
        "model_calls": sum(row["generation"]["call_count"] for row in final),
        "gold_access": False,
        "outcome_access": False,
        "database_or_source_router": False,
        "non_generic_fallback": False,
        "score_executed": False,
    }
    protocol.exclusive_json(COMPLETION, completion)
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument(
        "--base-url",
        action="append",
        default=[],
        help="repeat for multiple local vLLM servers; e.g. http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    base_urls = args.base_url or ["http://127.0.0.1:8000/v1"]
    value = (
        dry_run(args.expected_freeze_sha256, base_urls)
        if args.dry_run
        else execute(
            args.expected_freeze_sha256,
            base_urls,
            workers=args.workers,
            timeout=args.timeout,
            resume=args.resume,
        )
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
