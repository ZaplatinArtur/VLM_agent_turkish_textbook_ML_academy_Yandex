from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .schema import JudgeVerdict


def validate_judge_completion(
    expected_records: Iterable[dict[str, Any]],
    judge_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Verify that judge output is complete and safe to import into analytics."""

    expected_values = [dict(record) for record in expected_records]
    judge_values = [dict(record) for record in judge_records]
    expected_ids = [str(record.get("task_id") or "").strip() for record in expected_values]
    judge_ids = [str(record.get("task_id") or "").strip() for record in judge_values]
    expected_counts = Counter(expected_ids)
    judge_counts = Counter(judge_ids)
    duplicate_expected = sorted(
        task_id for task_id, count in expected_counts.items() if task_id and count > 1
    )
    duplicate_judge = sorted(
        task_id for task_id, count in judge_counts.items() if task_id and count > 1
    )
    expected_set = {task_id for task_id in expected_ids if task_id}
    judge_set = {task_id for task_id in judge_ids if task_id}
    invalid_verdict_ids: list[str] = []
    judge_error_ids: list[str] = []

    for record, task_id in zip(judge_values, judge_ids):
        judge = record.get("judge") if isinstance(record.get("judge"), dict) else {}
        if judge.get("error"):
            judge_error_ids.append(task_id or "<missing-task-id>")
        verdict = record.get("verdict")
        try:
            if not isinstance(verdict, dict):
                raise ValueError("verdict is missing")
            JudgeVerdict.from_dict(verdict)
        except (KeyError, TypeError, ValueError):
            invalid_verdict_ids.append(task_id or "<missing-task-id>")

    missing_task_ids = sorted(expected_set - judge_set)
    unexpected_task_ids = sorted(judge_set - expected_set)
    missing_expected_ids = expected_ids.count("")
    missing_judge_ids = judge_ids.count("")
    valid = not any(
        (
            missing_expected_ids,
            missing_judge_ids,
            duplicate_expected,
            duplicate_judge,
            missing_task_ids,
            unexpected_task_ids,
            invalid_verdict_ids,
            judge_error_ids,
        )
    )
    return {
        "schema_version": "judge-completion-v1",
        "valid": valid,
        "expected_records": len(expected_values),
        "judge_records": len(judge_values),
        "missing_expected_task_ids": missing_expected_ids,
        "missing_judge_task_ids": missing_judge_ids,
        "duplicate_expected_task_ids": duplicate_expected[:25],
        "duplicate_judge_task_ids": duplicate_judge[:25],
        "missing_task_ids": missing_task_ids[:25],
        "unexpected_task_ids": unexpected_task_ids[:25],
        "invalid_verdict_task_ids": invalid_verdict_ids[:25],
        "judge_error_task_ids": judge_error_ids[:25],
    }


def audit_judge_run(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [dict(record) for record in records]
    valid = 0
    first_attempt_valid = 0
    valid_after_retry = 0
    errors = 0
    unjudgeable = 0
    cache_hits = 0
    request_ids: Counter[str] = Counter()
    attempts: Counter[str] = Counter()
    setups: Counter[str] = Counter()
    configured_models: Counter[str] = Counter()
    served_models: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    prompt_versions: Counter[str] = Counter()
    backend_config_hashes: Counter[str] = Counter()
    model_mismatches: list[dict[str, str]] = []
    error_examples: list[dict[str, Any]] = []
    token_totals = {"input": 0, "output": 0, "total": 0}

    for record in values:
        request_id = str(record.get("request_id") or "")
        if request_id:
            request_ids[request_id] += 1
        setups[str(record.get("setup") or "unknown")] += 1
        prompt_versions[str(record.get("prompt_version") or "unknown")] += 1
        judge = record.get("judge") if isinstance(record.get("judge"), dict) else {}
        configured_model = str(judge.get("model") or "unknown")
        configured_models[configured_model] += 1
        backend_config_hashes[str(judge.get("backend_config_hash") or "unknown")] += 1
        cache_hits += int(bool(judge.get("cache_hit")))
        attempt_value = judge.get("attempts")
        attempts[str(attempt_value if attempt_value is not None else "unknown")] += 1
        metadata = judge.get("response_metadata") if isinstance(judge.get("response_metadata"), dict) else {}
        served_model = str(metadata.get("served_model") or "unknown")
        served_models[served_model] += 1
        finish_reasons[str(metadata.get("finish_reason") or "unknown")] += 1
        if served_model != "unknown" and configured_model != "unknown" and served_model != configured_model:
            if len(model_mismatches) < 25:
                model_mismatches.append(
                    {"request_id": request_id, "configured_model": configured_model, "served_model": served_model}
                )
        usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        total_tokens = usage.get("total_tokens", 0)
        for target, source in (("input", input_tokens), ("output", output_tokens), ("total", total_tokens)):
            if isinstance(source, int) and not isinstance(source, bool):
                token_totals[target] += source

        verdict = record.get("verdict")
        try:
            parsed = JudgeVerdict.from_dict(verdict) if isinstance(verdict, dict) else None
        except (KeyError, TypeError, ValueError) as exc:
            parsed = None
            if len(error_examples) < 25:
                error_examples.append({"request_id": request_id, "error": f"invalid verdict: {exc}"})
        if parsed is not None:
            valid += 1
            if parsed.label == "unjudgeable":
                unjudgeable += 1
            if attempt_value == 1:
                first_attempt_valid += 1
            elif isinstance(attempt_value, int) and attempt_value > 1:
                valid_after_retry += 1
        else:
            errors += 1
            if judge.get("error") and len(error_examples) < 25:
                error_examples.append({"request_id": request_id, "error": str(judge["error"])})

    duplicates = {key: count for key, count in request_ids.items() if count > 1}
    total = len(values)
    return {
        "schema_version": "judge-run-audit-v1",
        "records": total,
        "valid_verdicts": valid,
        "schema_valid_rate_after_retries": valid / total if total else None,
        "valid_on_first_attempt": first_attempt_valid,
        "valid_after_retry": valid_after_retry,
        "failed_records": errors,
        "unjudgeable_records": unjudgeable,
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hits / total if total else None,
        "attempts": dict(sorted(attempts.items())),
        "setups": dict(sorted(setups.items())),
        "prompt_versions": dict(sorted(prompt_versions.items())),
        "backend_config_hashes": dict(sorted(backend_config_hashes.items())),
        "inconsistent_backend_configuration": len(
            [value for value in backend_config_hashes if value != "unknown"]
        ) > 1,
        "configured_models": dict(sorted(configured_models.items())),
        "served_models": dict(sorted(served_models.items())),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "token_totals": token_totals,
        "duplicate_request_ids": len(duplicates),
        "duplicate_request_id_examples": dict(list(sorted(duplicates.items()))[:25]),
        "configured_served_model_mismatches": len(model_mismatches),
        "model_mismatch_examples": model_mismatches,
        "error_examples": error_examples,
    }
