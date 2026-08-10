#!/usr/bin/env python3
"""Run the preregistered paired RAG/no-RAG semantic-support verifier.

Only rows frozen into the public queue contact the model.  Each logical task
has one primary attempt and exactly one error-only retry inside EndpointPool.
Valid negative or low-confidence verdicts are never retried.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import prepare_maxim_paired_rag_norag_semantic_support_v1 as preparation
    import run_maxim_agent_ideas as core
except ModuleNotFoundError:  # Imported as scripts.*
    from scripts import prepare_maxim_paired_rag_norag_semantic_support_v1 as preparation
    from scripts import run_maxim_agent_ideas as core


SCHEMA_VERSION = "maxim-paired-rag-norag-semantic-support-runner-v1"
CONDITION = "paired_rag_norag_semantic_support_on_pinned_structural_context_v1"
PROMPT_VERSION = "rag-support-citations-failclosed-v1"

SYSTEM_PROMPT = """You audit whether a saved RAG answer is actually supported by pinned
textbook context. You have no answer key, reference answer, score, judge verdict, or gold
label. The original question and image are primary evidence. Do not choose RAG merely because
it is longer, agrees with a majority, or sounds plausible. A positive verdict requires the
context to support every decisive step that distinguishes the RAG answer. Copy each citation
quote exactly, character for character, from one supplied chunk and identify its chunk_id.
Report contradictions and unsupported decisive steps explicitly. If evidence is ambiguous or
insufficient, set rag_answer_supported=false. Return exactly the strict JSON schema."""


class RunnerError(RuntimeError):
    pass


VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question_reconstruction": {
            "type": "string",
            "minLength": 1,
            "maxLength": 900,
        },
        "rag_answer_supported": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "contradiction_found": {"type": "boolean"},
        "unsupported_decisive_steps": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
            "maxItems": 8,
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "minLength": 1, "maxLength": 160},
                    "exact_quote": {"type": "string", "minLength": 1, "maxLength": 700},
                    "supports": {"type": "string", "minLength": 1, "maxLength": 500},
                    "decisive": {"type": "boolean"},
                },
                "required": ["chunk_id", "exact_quote", "supports", "decisive"],
                "additionalProperties": False,
            },
            "maxItems": 8,
        },
        "answer_format_verified": {"type": "boolean"},
        "audit_summary": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
    "required": [
        "question_reconstruction",
        "rag_answer_supported",
        "confidence",
        "contradiction_found",
        "unsupported_decisive_steps",
        "citations",
        "answer_format_verified",
        "audit_summary",
    ],
    "additionalProperties": False,
}


def _task_text(row: Mapping[str, Any]) -> str:
    question = str(row.get("question") or "").strip()
    if question.casefold() in {
        "(soru görselde)",
        "(soru gг¶rselde)",
        "(question in image)",
    }:
        question = ""
    return (
        "ORIGINAL QUESTION\n"
        f"Subject: {row.get('subject') or 'unknown'}\n"
        f"Grade: {row.get('grade') if row.get('grade') is not None else 'unknown'}\n"
        f"Answer type: {row.get('answer_type') or 'unknown'}\n"
        f"Additional text: {question or '[the complete question is in the image]'}"
    )


def _candidate_text(row: Mapping[str, Any]) -> str:
    rag = row["rag_candidate"]
    no_rag = row["no_rag_candidate"]
    payload = {
        "RAG_CANDIDATE": rag,
        "NO_RAG_CANDIDATE": no_rag,
    }
    return (
        "SAVED CANDIDATE PAIR\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\nThe task is support auditing: choose no final candidate here. Determine only "
        "whether the pinned context decisively supports the RAG candidate."
    )


def _context_text(row: Mapping[str, Any]) -> str:
    blocks: list[str] = ["PINNED STRUCTURAL TEXTBOOK CONTEXT"]
    for item in row.get("contexts") or []:
        blocks.append(
            "\n".join(
                [
                    f"CHUNK_ID: {item['chunk_id']}",
                    f"DOCUMENT_ID: {item.get('document_id') or ''}",
                    f"PAGE: {item.get('page_number')}",
                    f"TYPE: {item.get('primary_type') or ''}",
                    "TEXT_START",
                    str(item["text"]),
                    "TEXT_END",
                ]
            )
        )
    blocks.append(
        "Return a positive support verdict only if at least one decisive exact quote "
        "supports the RAG answer and no decisive step is unsupported or contradicted."
    )
    return "\n\n".join(blocks)


def build_messages(
    row: Mapping[str, Any], *, image_root: Path, image_url_root: str
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": _task_text(row)}]
    content.extend(
        core._image_blocks(
            dict(row), image_root=image_root, image_url_root=image_url_root
        )
    )
    content.extend(
        [
            {"type": "text", "text": _candidate_text(row)},
            {"type": "text", "text": _context_text(row)},
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def validate_verdict(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = set(VERDICT_SCHEMA["required"])
    if set(value) != expected:
        raise RunnerError(
            f"verdict fields mismatch: got {sorted(value)}, expected {sorted(expected)}"
        )
    for key in ("rag_answer_supported", "contradiction_found", "answer_format_verified"):
        if not isinstance(value.get(key), bool):
            raise RunnerError(f"{key} must be boolean")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise RunnerError("confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise RunnerError("confidence outside [0,1]")
    unsupported = value.get("unsupported_decisive_steps")
    if not isinstance(unsupported, list) or len(unsupported) > 8:
        raise RunnerError("unsupported_decisive_steps must be a list of at most 8")
    if not all(isinstance(item, str) and item.strip() for item in unsupported):
        raise RunnerError("unsupported_decisive_steps contains an empty/non-string item")
    citations = value.get("citations")
    if not isinstance(citations, list) or len(citations) > 8:
        raise RunnerError("citations must be a list of at most 8")
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise RunnerError("citation is not an object")
        if set(citation) != {"chunk_id", "exact_quote", "supports", "decisive"}:
            raise RunnerError("citation fields mismatch")
        if not all(
            isinstance(citation.get(key), str) and str(citation.get(key)).strip()
            for key in ("chunk_id", "exact_quote", "supports")
        ):
            raise RunnerError("citation string field is empty")
        if not isinstance(citation.get("decisive"), bool):
            raise RunnerError("citation decisive must be boolean")
    for key in ("question_reconstruction", "audit_summary"):
        if not isinstance(value.get(key), str) or not str(value.get(key)).strip():
            raise RunnerError(f"{key} must be non-empty")
    result = dict(value)
    result["confidence"] = float(confidence)
    result["unsupported_decisive_steps"] = list(unsupported)
    result["citations"] = [dict(item) for item in citations]
    return result


def validate_queue(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("schema_version") != preparation.QUEUE_SCHEMA_VERSION:
            raise RunnerError(f"queue row {index}: schema mismatch")
        if row.get("queue_index") != index:
            raise RunnerError(f"queue row {index}: queue_index mismatch")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen:
            raise RunnerError(f"queue row {index}: missing/duplicate task_id")
        seen.add(task_id)
        if not isinstance(row.get("contexts"), list) or not row["contexts"]:
            raise RunnerError(f"queue row {index}: no pinned context")
        payload = {key: value for key, value in row.items() if key != "request_sha256"}
        preparation.audit_gold_free(payload)
        if row.get("request_sha256") != preparation.stable_sha256(payload):
            raise RunnerError(f"queue row {index}: request SHA mismatch")


def _write_rows(
    path: Path, queue: Sequence[Mapping[str, Any]], rows: Mapping[str, Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for request in queue:
            task_id = str(request["task_id"])
            if task_id in rows:
                sink.write(preparation.stable_json(rows[task_id]) + "\n")
    os.replace(temporary, path)


def _load_bound_inputs(
    queue_path: Path, manifest_path: Path, profile_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest = preparation.load_json(manifest_path)
    profile = preparation.load_json(profile_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != preparation.SCHEMA_VERSION:
        raise RunnerError("preparation manifest schema mismatch")
    if not isinstance(profile, dict) or profile.get("schema_version") != preparation.PROFILE_SCHEMA_VERSION:
        raise RunnerError("profile schema mismatch")
    if preparation.sha256_file(queue_path) != manifest["queue"]["sha256"]:
        raise RunnerError("queue hash differs from preparation manifest")
    if preparation.sha256_file(profile_path) != manifest["profile"]["sha256"]:
        raise RunnerError("profile hash differs from preparation manifest")
    rows = preparation.load_jsonl(queue_path)
    if len(rows) != int(manifest["queue"]["rows"]):
        raise RunnerError("queue row count differs from preparation manifest")
    validate_queue(rows)
    return rows, manifest, profile


def run_one(
    row: Mapping[str, Any],
    *,
    pool: core.EndpointPool,
    image_root: Path,
    image_url_root: str,
    max_tokens: int,
    base_seed: int,
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    parsed: dict[str, Any] | None = None
    call_trace: dict[str, Any] | None = None
    error: str | None = None
    try:
        call = pool.complete(
            messages=build_messages(
                row, image_root=image_root, image_url_root=image_url_root
            ),
            schema_name="paired_rag_norag_semantic_support_verdict_v1",
            schema=VERDICT_SCHEMA,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=base_seed + int(row["queue_index"]),
            retries=1,
        )
        parsed = validate_verdict(call["parsed"])
        call_trace = {
            "endpoint": call.get("endpoint"),
            "finish_reason": call.get("finish_reason"),
            "attempt": call.get("attempt"),
            "latency_s": call.get("latency_s"),
            "input_tokens": call.get("input_tokens"),
            "output_tokens": call.get("output_tokens"),
            "recovered_partial": bool(call.get("recovered_partial")),
            "parse_error": call.get("parse_error"),
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "queue_index": int(row["queue_index"]),
        "request_sha256": row["request_sha256"],
        "condition": CONDITION,
        "prompt_version": PROMPT_VERSION,
        "parsed": parsed,
        "call": call_trace,
        "error": error,
        "gold_access": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-url-root", default="file:///images")
    parser.add_argument("--base-url", action="append", default=[])
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.workers <= 32:
        raise SystemExit("--workers must be in [1,32]")
    queue_path = args.queue.resolve()
    manifest_path = args.preparation_manifest.resolve()
    profile_path = args.profile.resolve()
    output_path = args.output.resolve()
    if output_path in {queue_path, manifest_path, profile_path}:
        raise SystemExit("--output must not overwrite an input")

    queue, manifest, profile = _load_bound_inputs(
        queue_path, manifest_path, profile_path
    )
    expected_model = str(profile["model"]["name"])
    if args.model != expected_model:
        raise SystemExit(f"model must be exactly {expected_model!r}")
    generation = profile["generation"]
    if generation.get("retry_policy") != "one_error_only_retry":
        raise SystemExit("profile retry policy mismatch")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "queue_rows": len(queue),
                    "queue_sha256": manifest["queue"]["sha256"],
                    "profile_sha256": manifest["profile"]["sha256"],
                    "model": expected_model,
                    "max_attempts_per_task": 2,
                    "gold_access": False,
                    "endpoint_contacted": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not args.base_url:
        raise SystemExit("at least one --base-url is required")

    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        if not args.resume:
            raise SystemExit("output exists; pass --resume to run only missing tasks")
        for row in preparation.load_jsonl(output_path):
            task_id = str(row.get("task_id") or "")
            if task_id:
                existing[task_id] = row
    pending = [row for row in queue if str(row["task_id"]) not in existing]
    pool = core.EndpointPool(args.base_url, model=args.model, timeout_s=args.timeout_s)
    output_rows = dict(existing)
    lock = threading.Lock()

    def execute(row: Mapping[str, Any]) -> dict[str, Any]:
        return run_one(
            row,
            pool=pool,
            image_root=args.image_root.resolve(),
            image_url_root=args.image_url_root,
            max_tokens=int(generation["max_tokens"]),
            base_seed=int(generation["seed"]),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(execute, row): row for row in pending}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            request = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": str(request["task_id"]),
                    "queue_index": int(request["queue_index"]),
                    "request_sha256": request["request_sha256"],
                    "condition": CONDITION,
                    "prompt_version": PROMPT_VERSION,
                    "parsed": None,
                    "call": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "gold_access": False,
                }
            with lock:
                output_rows[str(result["task_id"])] = result
                _write_rows(output_path, queue, output_rows)
            completed += 1
            print(
                f"[{completed}/{len(pending)}] {result['task_id']} "
                f"attempt={(result.get('call') or {}).get('attempt')} "
                f"error={result.get('error')!r}",
                flush=True,
            )

    _write_rows(output_path, queue, output_rows)
    errors = sum(bool(row.get("error")) for row in output_rows.values())
    print(
        json.dumps(
            {
                "rows": len(output_rows),
                "expected_rows": len(queue),
                "errors": errors,
                "output": str(output_path),
                "output_sha256": preparation.sha256_file(output_path),
                "gold_access": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if len(output_rows) == len(queue) and errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
