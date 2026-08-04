"""Compose a complete solver using the preregistered frozen-Router fallback.

The composer is deterministic and opens benchmark rows only to recover the
canonical task order.  A candidate row is retained unless it has a non-null
``error`` or an empty ``final_answer``.  Persistent failures are replaced with
the answer-bearing fields of the exact frozen subject-router row and both
source rows are hash-bound in provenance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


FROZEN_BENCHMARK_SHA256 = "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
FROZEN_ROUTER_SHA256 = "34da8ef69619a8ba1f184cdfd1e6dcaf0fbdbdd1bfc50c711244a68f7d26a574"
POLICY_SHA256 = "6f23cd2280adcaa8b9c6214ebf74f9a9f341f28bebb38bd0579682445e5e0627"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def index_rows(rows: list[dict[str, Any]], label: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    index: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row_number, row in enumerate(rows, 1):
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError(f"{label} row {row_number}: empty task_id")
        if task_id in index:
            raise ValueError(f"{label}: duplicate task_id {task_id}")
        index[task_id] = row
        order.append(task_id)
    return index, order


def _assert_gold_blind(row: dict[str, Any], label: str) -> None:
    generation = row.get("generation")
    if not isinstance(generation, dict) or generation.get("gold_access") is not False:
        raise ValueError(f"{label}: generation.gold_access must be false")


def compose(
    *, benchmark_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]],
    router_rows: list[dict[str, Any]], condition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, benchmark_order = index_rows(benchmark_rows, "benchmark")
    candidate, candidate_order = index_rows(candidate_rows, "candidate")
    router, router_order = index_rows(router_rows, "router")
    benchmark_ids = set(benchmark_order)
    if set(candidate_order) != benchmark_ids:
        raise ValueError("candidate task-id set differs from benchmark")
    if set(router_order) != benchmark_ids:
        raise ValueError("router task-id set differs from benchmark")
    output: list[dict[str, Any]] = []
    fallback_ids: list[str] = []
    for task_id in benchmark_order:
        failed = candidate[task_id]
        default = router[task_id]
        _assert_gold_blind(failed, f"candidate {task_id}")
        _assert_gold_blind(default, f"router {task_id}")
        failure = bool(failed.get("error")) or not str(failed.get("final_answer") or "").strip()
        chosen = default if failure else failed
        if not str(chosen.get("final_answer") or "").strip():
            raise ValueError(f"{task_id}: chosen source has empty final_answer")
        row = copy.deepcopy(chosen)
        chosen_generation = copy.deepcopy(row.get("generation") or {})
        chosen_generation["gold_access"] = False
        chosen_generation["failclosed_composition"] = {
            "schema_version": "maxim-failclosed-composition-v1",
            "policy_sha256": POLICY_SHA256,
            "chosen_source": "frozen_subject_router" if failure else "candidate",
            "candidate_condition": failed.get("condition"),
            "candidate_row_sha256": row_sha256(failed),
            "router_row_sha256": row_sha256(default),
            "failure_text": str(failed.get("error") or "empty final_answer") if failure else None,
        }
        row["condition"] = condition
        row["prompt_version"] = condition
        row["generation"] = chosen_generation
        row["error"] = None
        output.append(row)
        if failure:
            fallback_ids.append(task_id)
    stats = {
        "rows": len(output),
        "candidate_rows_retained": len(output) - len(fallback_ids),
        "router_fallback_rows": len(fallback_ids),
        "router_fallback_task_ids": fallback_ids,
    }
    return output, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--skip-frozen-sha-check", action="store_true")
    args = parser.parse_args(argv)
    if not args.condition.strip():
        raise SystemExit("--condition must be non-empty")
    if not args.skip_frozen_sha_check:
        actual_benchmark = sha256_file(args.benchmark)
        actual_router = sha256_file(args.router)
        if actual_benchmark != FROZEN_BENCHMARK_SHA256:
            raise SystemExit(f"benchmark SHA mismatch: {actual_benchmark}")
        if actual_router != FROZEN_ROUTER_SHA256:
            raise SystemExit(f"router SHA mismatch: {actual_router}")
    rows, stats = compose(
        benchmark_rows=load_jsonl(args.benchmark),
        candidate_rows=load_jsonl(args.candidate),
        router_rows=load_jsonl(args.router),
        condition=args.condition,
    )
    write_jsonl(args.output, rows)
    manifest = {
        "schema_version": "maxim-failclosed-composition-manifest-v1",
        "condition": args.condition,
        "policy_sha256": POLICY_SHA256,
        "sources": {
            "benchmark": {"path": str(args.benchmark.resolve()), "sha256": sha256_file(args.benchmark)},
            "candidate": {"path": str(args.candidate.resolve()), "sha256": sha256_file(args.candidate)},
            "router": {"path": str(args.router.resolve()), "sha256": sha256_file(args.router)},
        },
        "stats": stats,
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
        "gold_access": False,
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
