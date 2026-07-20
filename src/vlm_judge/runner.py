from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterable

from .backends import JudgeBackend
from .metrics import deterministic_match
from .parsing import parse_judge_verdict
from .pipeline import request_id
from .prompts import JudgeRequest, build_judge_request
from .schema import EvaluationItem


def _backend_cache_config(backend: JudgeBackend) -> dict[str, Any]:
    config: dict[str, Any] = {
        "backend": str(backend.name),
        "model": str(backend.model),
    }
    for field_name in (
        "endpoint",
        "temperature",
        "max_tokens",
        "seed",
        "use_response_format",
        "enable_thinking",
        "image_mode",
    ):
        value = getattr(backend, field_name, None)
        if value is None or isinstance(value, (str, int, float, bool)):
            config[field_name] = value
    return config


def _cache_config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_cached(path: Path, expected_config: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("backend_config") != expected_config:
            return None
        parse_judge_verdict(json.dumps(value["verdict"], ensure_ascii=False))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value


def _write_cache_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _evaluate_item(
    item: EvaluationItem,
    *,
    backend: JudgeBackend,
    cache_dir: Path | None,
    prompt_version: str,
    max_attempts: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    item.validate()
    deterministic = deterministic_match(
        item.reference_answer,
        item.candidate_answer,
        item.answer_type,
        acceptable_answers=item.acceptable_answers,
    )
    identifier = request_id(item, prompt_version)
    backend_config = _backend_cache_config(backend)
    backend_config_hash = _cache_config_hash(backend_config)
    cache_key = hashlib.sha256(f"{identifier}:{backend_config_hash}".encode("ascii")).hexdigest()
    cache_path = cache_dir / f"{cache_key}.json" if cache_dir else None
    cached = _load_cached(cache_path, backend_config) if cache_path else None
    verdict = cached.get("verdict") if cached else None
    raw_response = cached.get("raw_response") if cached else None
    backend_metadata = cached.get("backend_metadata") if cached else None
    attempts = 0 if cached else None
    judge_error: str | None = None

    if cached is None:
        request = build_judge_request(item)
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            current_request = request
            if attempt > 1:
                if retry_delay_seconds:
                    time.sleep(retry_delay_seconds * (attempt - 1))
                current_request = JudgeRequest(
                    request.system_prompt,
                    request.user_prompt
                    + "\n\nYour previous response was invalid. Return the required JSON object only.",
                    request.image_urls,
                    request.image_labels,
                )
            try:
                response = backend.complete(current_request)
                raw_response = response.text
                backend_metadata = response.metadata
                parsed = parse_judge_verdict(response.text)
                verdict = parsed.to_dict()
                judge_error = None
                break
            except Exception as exc:  # backend and schema errors are preserved in output
                judge_error = f"{type(exc).__name__}: {exc}"
        if verdict is not None and cache_path is not None:
            _write_cache_atomic(
                cache_path,
                {
                    "request_id": identifier,
                    "prompt_version": prompt_version,
                    "backend": backend.name,
                    "model": backend.model,
                    "backend_config": backend_config,
                    "backend_config_hash": backend_config_hash,
                    "raw_response": raw_response,
                    "backend_metadata": backend_metadata,
                    "verdict": verdict,
                },
            )

    return {
        "request_id": identifier,
        "prompt_version": prompt_version,
        "task_id": item.task_id,
        "setup": item.setup,
        "subject": item.subject,
        "grade": item.grade,
        "answer_type": item.answer_type,
        "metadata": item.metadata,
        "deterministic": {
            "applicable": deterministic.applicable,
            "matched": deterministic.matched,
            "method": deterministic.method,
            "normalized_reference": deterministic.normalized_reference,
            "normalized_candidate": deterministic.normalized_candidate,
        },
        "verdict": verdict,
        "judge": {
            "backend": backend.name,
            "model": backend.model,
            "cache_key": cache_key,
            "backend_config": backend_config,
            "backend_config_hash": backend_config_hash,
            "attempts": attempts,
            "cache_hit": cached is not None,
            "error": judge_error,
            "response_metadata": backend_metadata,
        },
    }


def evaluate_items(
    items: Iterable[EvaluationItem],
    backend: JudgeBackend,
    output_path: Path,
    *,
    cache_dir: Path | None = None,
    prompt_version: str = "judge-v2",
    max_attempts: int = 2,
    workers: int = 1,
    retry_delay_seconds: float = 0.0,
) -> int:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must not be negative")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    item_list = list(items)
    run_one = partial(
        _evaluate_item,
        backend=backend,
        cache_dir=cache_dir,
        prompt_version=prompt_version,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    if workers == 1:
        results = map(run_one, item_list)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vlm-judge")
        results = executor.map(run_one, item_list)
    try:
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for result in results:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return len(item_list)
