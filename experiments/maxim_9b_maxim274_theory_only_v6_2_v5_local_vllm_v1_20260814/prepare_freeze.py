"""Audit aggregate corpus coverage and freeze the public Maxim-274 queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import protocol


QUEUE_OUT = protocol.FROZEN / "queue_content_theory_274.jsonl"
ALIGNMENT_OUT = protocol.FROZEN / "outer_alignment_274.jsonl"
CORPUS_OUT = protocol.FROZEN / "strict_theory_corpus.jsonl"
COVERAGE_OUT = protocol.FROZEN / "coverage_aggregate.json"
FREEZE_OUT = protocol.HERE / "EXECUTION_FREEZE.json"
FREEZE_SHA_OUT = protocol.HERE / "EXECUTION_FREEZE_SHA256.txt"
FUTURE_OUTPUTS = (
    protocol.HERE / "ATTEMPT.json",
    protocol.HERE / "COMPLETION.json",
    protocol.RUNS / "solver.jsonl",
)


def _source_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if protocol.sha256_file(protocol.SOURCE_QUEUE) != protocol.SOURCE_QUEUE_SHA256:
        raise protocol.ProtocolError("frozen Maxim-274 public queue hash mismatch")
    if protocol.sha256_file(protocol.SOURCE_THEORY) != protocol.SOURCE_THEORY_SHA256:
        raise protocol.ProtocolError("audited strict theory corpus hash mismatch")
    source = protocol.read_jsonl(protocol.SOURCE_QUEUE)
    theory = protocol.read_jsonl(protocol.SOURCE_THEORY)
    protocol.validate_source_public_rows(source)
    protocol.validate_theory_rows(theory)
    return source, theory


def derive() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    source, theory = _source_inputs()
    public = [protocol.public_content_projection(row) for row in source]
    alignment = [
        {"schema_version": "maxim274-outer-alignment-v1", "task_id": row["controller_id"]}
        for row in source
    ]
    queue: list[dict[str, Any]] = []
    for row in public:
        retrieval = protocol.retrieve_theory(row, theory)
        request = protocol.build_primary_request(row, retrieval)
        queue.append(
            {
                "schema_version": "maxim274-theory-only-queue-row-v1",
                "content_sha256": protocol.content_sha256(row),
                "public": row,
                "retrieval": retrieval,
                "primary_request_sha256": protocol.sha256_bytes(
                    protocol.canonical_json_bytes(request)
                ),
            }
        )
    if len(queue) != protocol.ROWS or len(alignment) != protocol.ROWS:
        raise protocol.ProtocolError("derived denominator mismatch")
    if len({row["task_id"] for row in alignment}) != protocol.ROWS:
        raise protocol.ProtocolError("outer alignment is not unique")
    serialized_queue = protocol.jsonl_bytes(queue)
    if b"task_id" in serialized_queue or b"controller_id" in serialized_queue:
        raise protocol.ProtocolError("controller identity leaked into content queue")
    coverage = protocol.coverage_aggregate(public, theory)
    return queue, alignment, theory, coverage


def print_coverage() -> dict[str, Any]:
    queue, _alignment, theory, coverage = derive()
    if coverage["benchmark_rows"] != len(queue) or coverage["strict_theory_chunks"] != len(theory):
        raise protocol.ProtocolError("coverage denominator mismatch")
    # Windows DataSphere launch shells can inherit a legacy code page.  Keep
    # terminal output ASCII-safe while the frozen JSON artifact remains UTF-8.
    print(json.dumps(coverage, ensure_ascii=True, indent=2, sort_keys=True))
    return coverage


def freeze() -> dict[str, Any]:
    for path in (*FUTURE_OUTPUTS, FREEZE_OUT, FREEZE_SHA_OUT):
        if path.exists():
            raise protocol.ProtocolError(f"refusing to overwrite existing artifact: {path}")
    queue, alignment, theory, coverage = derive()
    protocol.exclusive_bytes(QUEUE_OUT, protocol.jsonl_bytes(queue))
    protocol.exclusive_bytes(ALIGNMENT_OUT, protocol.jsonl_bytes(alignment))
    # Exact byte copy of the independently audited V6 corpus.
    protocol.exclusive_bytes(CORPUS_OUT, protocol.stable_bytes(protocol.SOURCE_THEORY))
    protocol.exclusive_json(COVERAGE_OUT, coverage)

    implementation_names = (
        "protocol.py",
        "prepare_freeze.py",
        "run_candidate.py",
        "dry_run.py",
        "test_protocol.py",
    )
    implementation = {
        name.replace(".", "_"): protocol.artifact(protocol.HERE / name)
        for name in implementation_names
    }
    static = {
        "audit_template": protocol.artifact(protocol.HERE / "INDEPENDENT_AUDIT_TEMPLATE.json"),
        "scoring_contract": protocol.artifact(protocol.HERE / "SCORING_CONTRACT_TEMPLATE.json"),
        "readme": protocol.artifact(protocol.HERE / "README.md"),
    }
    value = {
        "schema_version": "maxim274-theory-only-local-vllm-execution-freeze-v1",
        "state": "frozen_unexecuted_unscored",
        "created_date": "2026-08-14",
        "rows": protocol.ROWS,
        "model": {"id": protocol.MODEL_ID, "revision": protocol.MODEL_REVISION},
        "lineage": {
            "primary": "YKS generic V6.2 medium reasoning adapted to OCR/all answer types",
            "failure_only_fallback": "YKS generic V5 compact derive/falsify/crosscheck plus blind arbiter",
            "retrieval_normalization": "exact dependency-light V6 mojibake+NFKC+Turkish-I BM25",
        },
        "ablation": {
            "only_theory_search": True,
            "ocr_only": True,
            "task_or_example_database": False,
            "source_or_noid_router": False,
            "task_id_route_seed_wire": False,
            "image_bytes_sent": False,
            "fallback_to_base_or_non_generic": False,
            "errors_remain_wrong_in_denominator": True,
        },
        "runtime": {
            "transport": "local OpenAI-compatible vLLM",
            "temperature": protocol.TEMPERATURE,
            "top_p": protocol.TOP_P,
            "primary_max_tokens": protocol.PRIMARY_MAX_TOKENS,
            "primary_thinking": True,
            "fallback_max_tokens": protocol.FALLBACK_MAX_TOKENS,
            "fallback_thinking": False,
            "automatic_transport_retries": 0,
            "append_only_resumable_journal": True,
        },
        "source_inputs": {
            "maxim274_public_queue": {
                "path": protocol.SOURCE_QUEUE.relative_to(protocol.REPO_ROOT).as_posix(),
                "rows": protocol.ROWS,
                "sha256": protocol.SOURCE_QUEUE_SHA256,
            },
            "audited_strict_theory_corpus": {
                "path": protocol.SOURCE_THEORY.relative_to(protocol.REPO_ROOT).as_posix(),
                "rows": 75,
                "sha256": protocol.SOURCE_THEORY_SHA256,
            },
        },
        "artifacts": {
            "queue": protocol.artifact(QUEUE_OUT, rows=len(queue)),
            "alignment": protocol.artifact(ALIGNMENT_OUT, rows=len(alignment)),
            "strict_theory_corpus": protocol.artifact(CORPUS_OUT, rows=len(theory)),
            "coverage_aggregate": protocol.artifact(COVERAGE_OUT),
            **static,
        },
        "implementation": implementation,
        "coverage_aggregate": coverage,
        "score_contract": {
            "phase": "post-run-only",
            "deterministic_rows": 177,
            "image_judge_rows": 97,
            "image_judge": "fresh or exact-solver-bound reusable artifact",
            "standard_scorer_sha256": protocol.STANDARD_SCORER_SHA256,
        },
        "planned_outputs_absent": [path.relative_to(protocol.HERE).as_posix() for path in FUTURE_OUTPUTS],
        "gold_or_outcomes_opened": False,
        "model_calls": 0,
    }
    protocol.exclusive_json(FREEZE_OUT, value)
    freeze_sha = protocol.sha256_file(FREEZE_OUT)
    protocol.exclusive_bytes(FREEZE_SHA_OUT, (freeze_sha + "\n").encode("ascii"))
    return {"status": "frozen_unexecuted_unscored", "freeze_sha256": freeze_sha, "coverage": coverage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--coverage-only", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    value = print_coverage() if args.coverage_only else freeze()
    if args.freeze:
        print(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
