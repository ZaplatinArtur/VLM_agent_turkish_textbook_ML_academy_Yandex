#!/usr/bin/env python3
"""Run the preregistered gold-blind final meta-verifier on its public queue.

The model sees the original question/image before seven deterministically
shuffled anonymous candidates.  It never loads the private identity key.  A
valid, sufficiently supported independent verdict is emitted as a solver row;
any error, abstention, or frozen safety-gate failure copies the exact frozen
subject-router row for that task.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import prepare_maxim_final_meta_verifier_v1 as preparation
    import run_maxim_agent_ideas as core
except ModuleNotFoundError:  # Imported as scripts.run_maxim_final_meta_verifier_v1.
    from scripts import prepare_maxim_final_meta_verifier_v1 as preparation
    from scripts import run_maxim_agent_ideas as core


SCHEMA_VERSION = "maxim-final-meta-verifier-runner-v1"
CONDITION = "maxim_final_gold_blind_meta_verifier_v1"
PROMPT_VERSION = "original-first-anonymous-independent-verification-v1"

SYSTEM_PROMPT = """You are the final independent verifier for Turkish school questions.
You have no answer key, reference answer, scores, judge verdicts, source names, or knowledge
of which candidate is the production default. The original question image is the primary
evidence. Reconstruct and solve the question yourself before using anonymous candidates as
fallible suggestions. Candidate order is random. Never choose by majority, verbosity, style,
or apparent model quality. Check negation, symbols, units, diagram/table values, requested
scope, and option-to-letter mapping. Test every candidate against the image. You may produce
a final answer absent from all candidates when the visible evidence requires it. Abstain when
the image does not permit a sufficiently reliable independent decision. Return exactly the
strict JSON schema."""


class MetaVerifierError(RuntimeError):
    pass


def verdict_schema(candidate_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "question_reconstruction": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1000,
            },
            "decisive_evidence": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 420},
                "minItems": 2,
                "maxItems": 8,
            },
            "candidate_checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string", "enum": list(candidate_ids)},
                        "status": {
                            "type": "string",
                            "enum": ["supported", "refuted", "uncertain"],
                        },
                        "verification": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 420,
                        },
                    },
                    "required": ["candidate_id", "status", "verification"],
                    "additionalProperties": False,
                },
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
            },
            "independent_reasoning": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2400,
            },
            "final_answer": {
                "type": "string",
                "minLength": 1,
                "maxLength": 120,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "answer_format_verified": {"type": "boolean"},
            "abstain": {"type": "boolean"},
        },
        "required": [
            "question_reconstruction",
            "decisive_evidence",
            "candidate_checks",
            "independent_reasoning",
            "final_answer",
            "confidence",
            "answer_format_verified",
            "abstain",
        ],
        "additionalProperties": False,
    }


def _task_prompt(row: Mapping[str, Any]) -> str:
    question = str(row.get("question") or "").strip()
    if question.casefold() in {
        "(soru g\u00f6rselde)",
        "(soru görselde)",
        "(question in image)",
    }:
        question = ""
    return (
        "ORIGINAL QUESTION (primary evidence; solve this before candidates)\n"
        f"Subject: {row.get('subject') or 'unknown'}\n"
        f"Grade: {row.get('grade') if row.get('grade') is not None else 'unknown'}\n"
        f"Answer type: {row.get('answer_type') or 'unknown'}\n"
        f"Additional text: {question or '[the complete question is in the original image]'}\n"
        "Inspect every attached original image now."
    )


def _candidate_prompt(row: Mapping[str, Any]) -> str:
    lines = [
        "ANONYMOUS FALLIBLE CANDIDATES (deterministically shuffled; identities hidden):"
    ]
    for candidate in row.get("candidates") or []:
        evidence = candidate.get("bounded_evidence") or []
        lines.extend(
            [
                "",
                str(candidate["candidate_id"]),
                f"Proposed final answer: {candidate['final_answer']}",
                f"Bounded reasoning: {candidate['bounded_reasoning']}",
                "Bounded evidence: " + " | ".join(str(item) for item in evidence),
            ]
        )
    lines.extend(
        [
            "",
            "Now compare every proposal to your own reconstruction. Return the independently verified final answer, not a candidate ID.",
        ]
    )
    return "\n".join(lines)


def build_messages(
    row: Mapping[str, Any], *, image_root: Path, image_url_root: str
) -> list[dict[str, Any]]:
    # The original task text and original pixels intentionally precede all
    # candidate material in the same user turn.
    content: list[dict[str, Any]] = [{"type": "text", "text": _task_prompt(row)}]
    content.extend(
        core._image_blocks(
            dict(row), image_root=image_root, image_url_root=image_url_root
        )
    )
    content.append({"type": "text", "text": _candidate_prompt(row)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def validate_verdict(
    value: Mapping[str, Any], *, candidate_ids: Sequence[str], answer_type: str
) -> dict[str, Any]:
    schema = verdict_schema(candidate_ids)
    expected = set(schema["required"])
    if set(value) != expected:
        raise MetaVerifierError(
            f"verdict fields mismatch: got {sorted(value)}, expected {sorted(expected)}"
        )
    evidence = value.get("decisive_evidence")
    if not isinstance(evidence, list) or not 2 <= len(evidence) <= 8:
        raise MetaVerifierError("decisive_evidence must contain 2..8 rows")
    if not all(isinstance(item, str) and item.strip() for item in evidence):
        raise MetaVerifierError("decisive_evidence contains an empty/non-string row")
    checks = value.get("candidate_checks")
    if not isinstance(checks, list) or len(checks) != len(candidate_ids):
        raise MetaVerifierError("candidate_checks cardinality mismatch")
    check_ids: list[str] = []
    for check in checks:
        if not isinstance(check, Mapping):
            raise MetaVerifierError("candidate_checks row is not an object")
        if set(check) != {"candidate_id", "status", "verification"}:
            raise MetaVerifierError("candidate_checks fields mismatch")
        check_ids.append(str(check.get("candidate_id")))
        if check.get("status") not in {"supported", "refuted", "uncertain"}:
            raise MetaVerifierError("candidate check has invalid status")
        if not str(check.get("verification") or "").strip():
            raise MetaVerifierError("candidate check has empty verification")
    if sorted(check_ids) != sorted(candidate_ids):
        raise MetaVerifierError("candidate_checks must cover every anonymous ID exactly once")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise MetaVerifierError("confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise MetaVerifierError("confidence outside [0,1]")
    if not isinstance(value.get("answer_format_verified"), bool):
        raise MetaVerifierError("answer_format_verified must be boolean")
    if not isinstance(value.get("abstain"), bool):
        raise MetaVerifierError("abstain must be boolean")
    if not str(value.get("question_reconstruction") or "").strip():
        raise MetaVerifierError("empty question_reconstruction")
    if not str(value.get("independent_reasoning") or "").strip():
        raise MetaVerifierError("empty independent_reasoning")
    answer = str(value.get("final_answer") or "").strip()
    if not answer:
        raise MetaVerifierError("empty final_answer")
    if answer_type.casefold() == "choice":
        answer = answer.upper()
        if not re.fullmatch(r"[A-E]", answer):
            raise MetaVerifierError("choice final_answer must be exactly one letter A-E")
    validated = dict(value)
    validated["final_answer"] = answer
    validated["confidence"] = float(confidence)
    return validated


def validate_queue(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != preparation.EXPECTED_ROWS:
        raise MetaVerifierError(
            f"queue must contain {preparation.EXPECTED_ROWS} rows, got {len(rows)}"
        )
    seen: set[str] = set()
    expected_ids = list(preparation.OPAQUE_IDS)
    for index, row in enumerate(rows):
        if row.get("schema_version") != preparation.QUEUE_SCHEMA_VERSION:
            raise MetaVerifierError(f"queue row {index}: schema mismatch")
        if row.get("queue_index") != index:
            raise MetaVerifierError(f"queue row {index}: queue_index mismatch")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen:
            raise MetaVerifierError(f"queue row {index}: missing/duplicate task_id")
        seen.add(task_id)
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or [
            value.get("candidate_id") if isinstance(value, Mapping) else None
            for value in candidates
        ] != expected_ids:
            raise MetaVerifierError(f"queue row {index}: anonymous candidate IDs mismatch")
        payload = {key: value for key, value in row.items() if key != "request_sha256"}
        preparation.audit_gold_free(payload)
        if row.get("request_sha256") != preparation.stable_sha256(payload):
            raise MetaVerifierError(f"queue row {index}: request SHA mismatch")


def _request_sha(
    row: Mapping[str, Any], *, model: str, max_tokens: int, seed: int
) -> str:
    return preparation.stable_sha256(
        {
            "queue_request_sha256": row["request_sha256"],
            "prompt_version": PROMPT_VERSION,
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "schema_sha256": preparation.stable_sha256(
                verdict_schema(preparation.OPAQUE_IDS)
            ),
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
            "enable_thinking": False,
        }
    )


def _compact_call(call: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if call is None:
        return None
    return {
        "endpoint": call.get("endpoint"),
        "finish_reason": call.get("finish_reason"),
        "attempt": call.get("attempt"),
        "latency_s": call.get("latency_s"),
        "input_tokens": call.get("input_tokens"),
        "output_tokens": call.get("output_tokens"),
        "recovered_partial": bool(call.get("recovered_partial")),
        "parse_error": call.get("parse_error"),
    }


def run_one(
    row: Mapping[str, Any],
    *,
    pool: core.EndpointPool,
    image_root: Path,
    image_url_root: str,
    max_tokens: int,
    base_seed: int,
    semantic_attempts: int,
) -> dict[str, Any]:
    task_seed = base_seed + int(
        hashlib.sha256(str(row["task_id"]).encode("utf-8")).hexdigest()[:8], 16
    ) % 1_000_000
    request_sha = _request_sha(
        row, model=pool.model, max_tokens=max_tokens, seed=task_seed
    )
    failures: list[str] = []
    last_call: Mapping[str, Any] | None = None
    for semantic_attempt in range(1, semantic_attempts + 1):
        messages = build_messages(
            row, image_root=image_root, image_url_root=image_url_root
        )
        if failures:
            messages[-1]["content"].append(
                {
                    "type": "text",
                    "text": (
                        "The previous structured verdict was invalid. Correct only this "
                        f"protocol problem and re-verify independently: {failures[-1]}"
                    ),
                }
            )
        try:
            call = pool.complete(
                messages=messages,
                schema_name="final_meta_verdict_v1",
                schema=verdict_schema(preparation.OPAQUE_IDS),
                max_tokens=max_tokens,
                temperature=0.0,
                seed=task_seed + (semantic_attempt - 1) * 100_003,
                retries=1,
            )
            last_call = call
            verdict = validate_verdict(
                call["parsed"],
                candidate_ids=preparation.OPAQUE_IDS,
                answer_type=str(row.get("answer_type") or "unknown"),
            )
            return {
                "task_id": row["task_id"],
                "queue_request_sha256": row["request_sha256"],
                "verifier_request_sha256": request_sha,
                "prompt_version": PROMPT_VERSION,
                "model": pool.model,
                "verdict": verdict,
                "raw_response": call.get("raw"),
                "call": _compact_call(call),
                "semantic_attempt": semantic_attempt,
                "error": None,
            }
        except (core.CallFailure, MetaVerifierError, KeyError, TypeError, ValueError) as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
    return {
        "task_id": row["task_id"],
        "queue_request_sha256": row["request_sha256"],
        "verifier_request_sha256": request_sha,
        "prompt_version": PROMPT_VERSION,
        "model": pool.model,
        "verdict": None,
        "raw_response": last_call.get("raw") if last_call else None,
        "call": _compact_call(last_call),
        "semantic_attempt": semantic_attempts,
        "error": " | ".join(failures),
    }


def apply_frozen_policy(
    result: Mapping[str, Any], *, min_confidence: float, min_evidence: int
) -> tuple[str, str | None, str]:
    verdict = result.get("verdict")
    if result.get("error") or not isinstance(verdict, Mapping):
        return "router", None, "verifier_error_router_fallback"
    if verdict.get("abstain") is True:
        return "router", None, "explicit_abstention_router_fallback"
    if verdict.get("answer_format_verified") is not True:
        return "router", None, "format_not_verified_router_fallback"
    if float(verdict.get("confidence") or 0.0) < min_confidence:
        return "router", None, "confidence_gate_router_fallback"
    evidence = verdict.get("decisive_evidence")
    if not isinstance(evidence, list) or len(evidence) < min_evidence:
        return "router", None, "evidence_gate_router_fallback"
    return "meta_verifier", str(verdict["final_answer"]), "valid_supported_meta_answer"


def _row_sha(row: Mapping[str, Any]) -> str:
    return preparation.stable_sha256(dict(row))


def compose_solver_row(
    *,
    result: Mapping[str, Any],
    router_row: Mapping[str, Any],
    min_confidence: float,
    min_evidence: int,
    queue_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection, answer, reason = apply_frozen_policy(
        result, min_confidence=min_confidence, min_evidence=min_evidence
    )
    audit = dict(result)
    audit["selection"] = {
        "selected_source": selection,
        "reason": reason,
        "applied_final_answer": answer
        if selection == "meta_verifier"
        else str(router_row.get("final_answer") or ""),
        "router_row_sha256": _row_sha(router_row),
        "gold_access": False,
    }
    if selection == "router":
        # Content-exact fallback: no condition, answer, or provenance field from
        # the frozen subject-router row is mutated.  The separate audit row
        # records why it was selected.
        return copy.deepcopy(dict(router_row)), audit

    verdict = result["verdict"]
    assert isinstance(verdict, Mapping) and answer is not None
    call = result.get("call")
    usage = call if isinstance(call, Mapping) else {}
    solver = {
        "task_id": result["task_id"],
        "condition": CONDITION,
        "model": result["model"],
        "prompt_version": PROMPT_VERSION,
        "final_answer": answer,
        "solution_steps": "\n".join(
            str(item) for item in verdict.get("decisive_evidence") or []
        ),
        "reasoning": str(verdict.get("independent_reasoning") or ""),
        "forced_answer": False,
        "raw_response": result.get("raw_response"),
        "generation": {
            "temperature": 0.0,
            "top_p": 0.95,
            "enable_thinking": False,
            "structured_mode": "strict_response_format",
            "gold_access": False,
            "original_question_and_images_primary": True,
            "candidate_source_identities_seen": False,
            "queue_sha256": queue_sha256,
            "queue_request_sha256": result["queue_request_sha256"],
            "verifier_request_sha256": result["verifier_request_sha256"],
            "confidence": verdict.get("confidence"),
            "answer_format_verified": verdict.get("answer_format_verified"),
            "selection_reason": reason,
        },
        "tool_calls": [],
        "usage": {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "latency_s": float(usage.get("latency_s") or 0.0),
        },
        "error": None,
    }
    return solver, audit


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f".tmp-{os.getpid()}-{threading.get_ident()}"
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(preparation.stable_json(row) + "\n")
    os.replace(temporary, path)


def _load_and_validate_bindings(
    *,
    queue_path: Path,
    preparation_manifest_path: Path,
    profile_path: Path,
    router_solver_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    manifest = preparation.load_json(preparation_manifest_path)
    profile = preparation.load_json(profile_path)
    if not isinstance(manifest, Mapping) or not isinstance(profile, Mapping):
        raise MetaVerifierError("manifest/profile must be JSON objects")
    preparation.validate_profile(profile)
    public = manifest.get("queue_public")
    sources = manifest.get("sources_private")
    if not isinstance(public, Mapping) or not isinstance(sources, Mapping):
        raise MetaVerifierError("preparation manifest is missing queue/source bindings")
    if preparation.sha256_file(queue_path) != public.get("sha256"):
        raise MetaVerifierError("public queue SHA does not match preparation manifest")
    if preparation.sha256_file(profile_path) != manifest.get("profile", {}).get("sha256"):
        raise MetaVerifierError("profile SHA does not match preparation manifest")
    router_binding = sources.get(preparation.ROUTER_SLOT)
    if not isinstance(router_binding, Mapping):
        raise MetaVerifierError("router binding absent from preparation manifest")
    if preparation.sha256_file(router_solver_path) != router_binding.get("sha256"):
        raise MetaVerifierError("router solver SHA does not match preparation manifest")
    queue = preparation.load_jsonl(queue_path)
    validate_queue(queue)
    router_rows = preparation.load_jsonl(router_solver_path)
    router_index = preparation.index_unique(router_rows, "router solver")
    task_ids = [str(row["task_id"]) for row in queue]
    if set(router_index) != set(task_ids):
        raise MetaVerifierError("router solver task set does not match public queue")
    ordered_router = [dict(router_index[task_id]) for task_id in task_ids]
    return queue, ordered_router, manifest, profile


def run_queue(
    *,
    queue_path: Path,
    preparation_manifest_path: Path,
    profile_path: Path,
    router_solver_path: Path,
    verdict_output_path: Path,
    solver_output_path: Path,
    base_urls: Sequence[str],
    model: str,
    image_root: Path,
    image_url_root: str,
    workers: int,
    timeout_s: float,
    resume: bool,
    retry_errors: bool,
) -> dict[str, Any]:
    queue, router_rows, preparation_manifest, profile = _load_and_validate_bindings(
        queue_path=queue_path.resolve(),
        preparation_manifest_path=preparation_manifest_path.resolve(),
        profile_path=profile_path.resolve(),
        router_solver_path=router_solver_path.resolve(),
    )
    generation = profile.get("generation")
    policy = profile.get("selection_policy")
    if not isinstance(generation, Mapping) or not isinstance(policy, Mapping):
        raise MetaVerifierError("profile generation/selection policy is missing")
    if model != generation.get("model"):
        raise MetaVerifierError(
            f"model must match preregistered profile: {generation.get('model')}"
        )
    max_tokens = int(generation["max_tokens"])
    base_seed = int(generation["seed"])
    semantic_attempts = int(generation["semantic_attempts"])
    min_confidence = float(policy["min_confidence"])
    min_evidence = int(policy["min_decisive_evidence"])
    queue_sha = preparation.sha256_file(queue_path)
    task_order = [str(row["task_id"]) for row in queue]
    router_index = {str(row["task_id"]): row for row in router_rows}

    existing: dict[str, dict[str, Any]] = {}
    if verdict_output_path.exists():
        if not resume:
            raise MetaVerifierError(
                f"verdict output exists; use --resume: {verdict_output_path}"
            )
        loaded = preparation.load_jsonl(verdict_output_path)
        existing_index = preparation.index_unique(loaded, "existing verdict output")
        if not set(existing_index).issubset(task_order):
            raise MetaVerifierError("existing verdict output contains unknown tasks")
        for task_id, row in existing_index.items():
            queue_row = queue[task_order.index(task_id)]
            if row.get("queue_request_sha256") != queue_row.get("request_sha256"):
                raise MetaVerifierError(f"{task_id}: existing result queue binding mismatch")
            if retry_errors and row.get("error"):
                continue
            existing[task_id] = dict(row)
    elif solver_output_path.exists():
        raise MetaVerifierError("solver output exists without resumable verdict output")

    pending = [row for row in queue if str(row["task_id"]) not in existing]
    pool = core.EndpointPool(list(base_urls), model=model, timeout_s=timeout_s)
    results: dict[str, dict[str, Any]] = dict(existing)
    write_lock = threading.Lock()

    def flush() -> None:
        selected_solver_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for task_id in task_order:
            result = results.get(task_id)
            if result is None:
                continue
            solver, audited = compose_solver_row(
                result=result,
                router_row=router_index[task_id],
                min_confidence=min_confidence,
                min_evidence=min_evidence,
                queue_sha256=queue_sha,
            )
            selected_solver_rows.append(solver)
            audit_rows.append(audited)
        _write_jsonl_atomic(verdict_output_path, audit_rows)
        _write_jsonl_atomic(solver_output_path, selected_solver_rows)

    if not verdict_output_path.exists():
        verdict_output_path.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_one,
                row,
                pool=pool,
                image_root=image_root,
                image_url_root=image_url_root,
                max_tokens=max_tokens,
                base_seed=base_seed,
                semantic_attempts=semantic_attempts,
            ): str(row["task_id"])
            for row in pending
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # task-level fail-closed record
                result = {
                    "task_id": task_id,
                    "queue_request_sha256": queue[task_order.index(task_id)][
                        "request_sha256"
                    ],
                    "verifier_request_sha256": None,
                    "prompt_version": PROMPT_VERSION,
                    "model": model,
                    "verdict": None,
                    "raw_response": None,
                    "call": None,
                    "semantic_attempt": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with write_lock:
                results[task_id] = result
                flush()
            completed += 1
            print(
                f"[{completed}/{len(pending)}] {task_id} "
                f"error={result.get('error')!r}",
                flush=True,
            )
    if not pending:
        flush()

    final_verdicts = preparation.load_jsonl(verdict_output_path)
    final_solvers = preparation.load_jsonl(solver_output_path)
    error_count = sum(bool(row.get("error")) for row in final_verdicts)
    fallback_count = sum(
        row.get("selection", {}).get("selected_source") == "router"
        for row in final_verdicts
    )
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "gold_blind_final_meta_verification",
        "generation_gold_access": False,
        "private_routing_key_loaded": False,
        "candidate_scores_loaded": False,
        "judge_artifacts_loaded": False,
        "profile": {
            "path": str(profile_path.resolve()),
            "sha256": preparation.sha256_file(profile_path),
        },
        "preparation_manifest": {
            "path": str(preparation_manifest_path.resolve()),
            "sha256": preparation.sha256_file(preparation_manifest_path),
        },
        "queue": {
            "path": str(queue_path.resolve()),
            "sha256": queue_sha,
            "rows": len(queue),
        },
        "router_fallback_solver": {
            "path": str(router_solver_path.resolve()),
            "sha256": preparation.sha256_file(router_solver_path),
            "content_exact_on_fallback": True,
        },
        "backend": {
            "base_urls": list(base_urls),
            "model": model,
            "temperature": 0.0,
            "enable_thinking": False,
            "max_tokens": max_tokens,
            "seed": base_seed,
            "semantic_attempts": semantic_attempts,
            "workers": workers,
        },
        "selection_policy": dict(policy),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "schema_sha256": preparation.stable_sha256(
            verdict_schema(preparation.OPAQUE_IDS)
        ),
        "verdict_output": {
            "path": str(verdict_output_path.resolve()),
            "sha256": preparation.sha256_file(verdict_output_path),
            "rows": len(final_verdicts),
            "errors": error_count,
            "router_fallback_rows": fallback_count,
        },
        "solver_output": {
            "path": str(solver_output_path.resolve()),
            "sha256": preparation.sha256_file(solver_output_path),
            "rows": len(final_solvers),
        },
        "complete": len(final_solvers) == len(queue),
    }
    manifest_path = solver_output_path.with_suffix(
        solver_output_path.suffix + ".manifest.json"
    )
    preparation.write_json(manifest_path, run_manifest)
    return run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--router-solver", type=Path, required=True)
    parser.add_argument("--verdict-output", type=Path, required=True)
    parser.add_argument("--solver-output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-url-root", default="file:///images")
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= 32:
        raise SystemExit("--workers must be in [1,32]")
    manifest = run_queue(
        queue_path=args.queue,
        preparation_manifest_path=args.preparation_manifest,
        profile_path=args.profile,
        router_solver_path=args.router_solver,
        verdict_output_path=args.verdict_output,
        solver_output_path=args.solver_output,
        base_urls=args.base_url,
        model=args.model,
        image_root=args.image_root,
        image_url_root=args.image_url_root,
        workers=args.workers,
        timeout_s=args.timeout_s,
        resume=args.resume,
        retry_errors=args.retry_errors,
    )
    print(preparation.stable_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
