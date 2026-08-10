#!/usr/bin/env python3
"""Build strict normalized legacy milestones and the seven-stage 9B wrapper."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL = "Qwen/Qwen3.5-9B"
BENCHMARK_SHA256 = "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
EMPTY_UNION_SHA256 = hashlib.sha256(b"").hexdigest()


class BuildError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise BuildError(f"{path}: expected object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise BuildError(f"{path}: expected JSON objects")
    return rows


def index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise BuildError(f"{label}: missing/duplicate task_id")
        result[task_id] = row
    return result


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def role_artifact(role: str, path: Path) -> dict[str, str]:
    return {"role": role, **artifact(path)}


def metric(rows: int, correct: int) -> dict[str, Any]:
    return {"rows": rows, "correct": correct, "accuracy": correct / rows}


def score_metrics(score: dict[str, Any]) -> dict[str, Any]:
    overall = score["overall"]
    by_source = score["by_source"]
    by_subject = score["by_subject"]
    total = int(overall["n"])
    correct = int(overall["new_correct"])
    math_rows = int(by_subject["Math"]["n"])
    math_correct = int(by_subject["Math"]["new_correct"])
    return {
        **metric(total, correct),
        "slices": {
            "deterministic": metric(
                int(by_source["deterministic"]["n"]),
                int(by_source["deterministic"]["new_correct"]),
            ),
            "image": metric(
                int(by_source["image_judge"]["n"]),
                int(by_source["image_judge"]["new_correct"]),
            ),
            "math": metric(math_rows, math_correct),
            "non_math": metric(total - math_rows, correct - math_correct),
        },
    }


def page_metrics(
    benchmark: dict[str, dict[str, Any]], judge: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, bool]]:
    correct = {
        task_id: bool((row.get("verdict") or {}).get("strict_correct"))
        for task_id, row in judge.items()
    }
    deterministic = {
        task_id
        for task_id, row in judge.items()
        if (row.get("metadata") or {}).get("score_source") == "exact"
    }
    math = {task_id for task_id, row in benchmark.items() if row.get("subject") == "Math"}
    total_correct = sum(correct.values())
    det_correct = sum(correct[task_id] for task_id in deterministic)
    math_correct = sum(correct[task_id] for task_id in math)
    return (
        {
            **metric(len(correct), total_correct),
            "slices": {
                "deterministic": metric(len(deterministic), det_correct),
                "image": metric(len(correct) - len(deterministic), total_correct - det_correct),
                "math": metric(len(math), math_correct),
                "non_math": metric(len(correct) - len(math), total_correct - math_correct),
            },
        },
        correct,
    )


def outcome_map(score: dict[str, Any]) -> dict[str, bool]:
    rows = score.get("task_outcomes")
    if not isinstance(rows, list):
        raise BuildError("score has no task_outcomes")
    return {str(row["task_id"]): bool(row["new_correct"]) for row in rows}


def comparison(
    milestone_id: str,
    solver_sha256: str,
    before: dict[str, bool],
    after: dict[str, bool],
) -> dict[str, Any]:
    if set(before) != set(after):
        raise BuildError("comparison task sets differ")
    fixes = sum(not before[key] and after[key] for key in before)
    regressions = sum(before[key] and not after[key] for key in before)
    return {
        "baseline_milestone_id": milestone_id,
        "baseline_solver_sha256": solver_sha256,
        "fixes": fixes,
        "regressions": regressions,
        "unchanged": len(before) - fixes - regressions,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))


def normalized_solver(
    source: Path, output: Path, benchmark_ids: set[str]
) -> tuple[Path, dict[str, dict[str, Any]]]:
    rows = read_jsonl(source)
    indexed = index(rows, str(source))
    if set(indexed) != benchmark_ids or any(row.get("model") != MODEL for row in rows):
        raise BuildError(f"{source}: not an exact 274-row Qwen3.5-9B solver")
    normalized = [{**row, "final_origin": "model_anchor"} for row in rows]
    write_jsonl(output, normalized)
    return output, index(normalized, str(output))


def hybrid_judge(
    *,
    benchmark_ids: set[str],
    score: dict[str, Any],
    image_judge_path: Path,
    output: Path,
) -> Path:
    image = index(read_jsonl(image_judge_path), str(image_judge_path))
    outcomes = {str(row["task_id"]): row for row in score["task_outcomes"]}
    rows: list[dict[str, Any]] = []
    for task_id in sorted(benchmark_ids):
        if outcomes[task_id]["score_source"] == "image_judge":
            rows.append(image[task_id])
        else:
            rows.append(
                {
                    "task_id": task_id,
                    "score_source": "deterministic",
                    "authority": "score_maxim_full274.task_outcomes",
                }
            )
    write_jsonl(output, rows)
    return output


def build_legacy(args: argparse.Namespace) -> list[dict[str, Any]]:
    benchmark_path = args.benchmark.resolve()
    if sha256_file(benchmark_path) != BENCHMARK_SHA256:
        raise BuildError("benchmark SHA mismatch")
    benchmark = index(read_jsonl(benchmark_path), "benchmark")
    ids = set(benchmark)
    root = args.output_dir.resolve() / "legacy"

    page_source = args.page_solver.resolve()
    no_tools_source = args.no_tools_solver.resolve()
    active_source = args.active_solver.resolve()
    page_dir, no_dir, active_dir = root / "page_rag", root / "no_tools", root / "active_crop"
    page_solver, _ = normalized_solver(page_source, page_dir / "solver.jsonl", ids)
    no_solver, _ = normalized_solver(no_tools_source, no_dir / "solver.jsonl", ids)
    active_solver, _ = normalized_solver(active_source, active_dir / "solver.jsonl", ids)

    page_judge_source = args.page_judge.resolve()
    page_judge_rows = index(read_jsonl(page_judge_source), "page judge")
    if set(page_judge_rows) != ids:
        raise BuildError("Page RAG judge task set differs")
    page_judge = page_dir / "judge.jsonl"
    write_jsonl(page_judge, read_jsonl(page_judge_source))
    no_score_source = read_json(args.no_tools_score.resolve())
    active_score_source = read_json(args.active_score.resolve())
    no_judge = hybrid_judge(
        benchmark_ids=ids,
        score=no_score_source,
        image_judge_path=args.no_tools_image_judge.resolve(),
        output=no_dir / "judge.jsonl",
    )
    active_judge = hybrid_judge(
        benchmark_ids=ids,
        score=active_score_source,
        image_judge_path=args.active_image_judge.resolve(),
        output=active_dir / "judge.jsonl",
    )

    page_metric, page_outcomes = page_metrics(benchmark, page_judge_rows)
    no_outcomes = outcome_map(no_score_source)
    active_outcomes = outcome_map(active_score_source)
    page_comparisons: list[dict[str, Any]] = []
    no_comparisons = [
        comparison("page_rag_9b", sha256_file(page_solver), page_outcomes, no_outcomes)
    ]
    active_comparisons = [
        comparison("no_tools_9b", sha256_file(no_solver), no_outcomes, active_outcomes)
    ]

    empty_union = {
        "sha256": EMPTY_UNION_SHA256,
        "size": 0,
        "replacements": 0,
        "confirmations": 0,
        "stage_counts": {},
    }
    closure = {
        "expected_model": MODEL,
        "checked_rows": len(ids),
        "matching_rows": len(ids),
        "foreign_models": [],
    }
    origins = {"model_anchor": len(ids), "deterministic_source_replacement": 0, "unknown": 0}
    evaluator = {
        "semantics": "frozen deterministic matcher + pinned Qwen3.5-9B strict image judge",
        "deterministic_rows": 177,
        "image_rows": 97,
        "source_certified_image_rows": 0,
        "model_judged_image_rows": 97,
        "judge_model": MODEL,
    }

    specs = [
        {
            "milestone_id": "page_rag_9b",
            "pipeline": "page_rag",
            "status": "historical_output_control",
            "bound": None,
            "caveats": [
                "historical control output; no immutable pre-score binding claim is made"
            ],
            "solver": page_solver,
            "raw_solver": page_source,
            "judge": page_judge,
            "metrics": page_metric,
            "comparisons": page_comparisons,
            "provenance": [
                role_artifact("historical_baseline_lock", args.page_manifest.resolve())
            ],
            "absence": "Page RAG has no official-source certificate layer.",
            "dir": page_dir,
        },
        {
            "milestone_id": "no_tools_9b",
            "pipeline": "model_only",
            "status": "matched_judge_replay_partial_generation_provenance",
            "bound": None,
            "caveats": [
                "exact raw 9B output is rescored with the matched judge; an immutable original generation manifest is unavailable"
            ],
            "solver": no_solver,
            "raw_solver": no_tools_source,
            "judge": no_judge,
            "metrics": score_metrics(no_score_source),
            "comparisons": no_comparisons,
            "provenance": [
                role_artifact("matched_image_judge_manifest", args.no_tools_judge_manifest.resolve())
            ],
            "absence": "No-tools has no retrieval or official-source certificate layer.",
            "dir": no_dir,
        },
        {
            "milestone_id": "query_active_crop_v2_9b",
            "pipeline": "query_active_crop_v2",
            "status": "preregistered_gold_blind",
            "bound": True,
            "caveats": [
                "development benchmark replay; this is the preregistered 9B ActiveCrop anchor selected before the source rebase"
            ],
            "solver": active_solver,
            "raw_solver": active_source,
            "judge": active_judge,
            "metrics": score_metrics(active_score_source),
            "comparisons": active_comparisons,
            "provenance": [
                role_artifact("preregistered_profile", args.active_preregistration.resolve()),
                role_artifact("composition_manifest", args.active_composition_manifest.resolve()),
            ],
            "absence": "ActiveCrop changes visual input selection but has no official-source certificate layer.",
            "dir": active_dir,
        },
    ]
    descriptors: list[dict[str, Any]] = []
    for spec in specs:
        score_path = spec["dir"] / "score.json"
        score_projection = {
            "schema_version": "vlm-9b-milestone-score-v2",
            "milestone_id": spec["milestone_id"],
            "model": MODEL,
            "pipeline": spec["pipeline"],
            "benchmark_sha256": BENCHMARK_SHA256,
            "solver_sha256": sha256_file(spec["solver"]),
            "judge_sha256": sha256_file(spec["judge"]),
            "certificate_sha256s": [],
            "metrics": spec["metrics"],
            "source_union": empty_union,
            "comparisons": spec["comparisons"],
            "evaluator": evaluator,
            "final_origin_counts": origins,
        }
        write_json(score_path, score_projection)
        aggregate = {
            "schema_version": "vlm-9b-milestone-aggregate-v2",
            "milestone_id": spec["milestone_id"],
            "model": MODEL,
            "pipeline": spec["pipeline"],
            "provenance_status": spec["status"],
            "bound_before_score": spec["bound"],
            "caveats": spec["caveats"],
            "provenance_manifests": spec["provenance"],
            "artifacts": {
                "solver": artifact(spec["solver"]),
                "raw_solver": artifact(spec["raw_solver"]),
                "score": artifact(score_path),
                "judge": artifact(spec["judge"]),
                "certificates": [],
            },
            "certificate_absence_reason": spec["absence"],
            "benchmark_sha256": BENCHMARK_SHA256,
            "metrics": spec["metrics"],
            "model_closure": closure,
            "source_union": empty_union,
            "comparisons": spec["comparisons"],
            "evaluator": evaluator,
            "final_origin_counts": origins,
        }
        aggregate_path = spec["dir"] / "aggregate.json"
        write_json(aggregate_path, aggregate)
        descriptors.append(
            {
                "milestone_id": spec["milestone_id"],
                "adapter": "normalized_v2",
                "aggregate": artifact(aggregate_path),
            }
        )
    return descriptors


def build_wrapper(args: argparse.Namespace, legacy: list[dict[str, Any]]) -> Path:
    native_specs = [
        ("source_v1_rebase_9b", args.source_v1_aggregate),
        ("source_v3_rebase_9b", args.source_v3_aggregate),
        ("source_v6_rebase_9b", args.source_v6_aggregate),
        ("source_v7_rebase_9b", args.source_v7_aggregate),
    ]
    native = []
    for milestone_id, raw_path in native_specs:
        if raw_path is None:
            raise BuildError("all four native source aggregates are required")
        path = raw_path.resolve()
        native.append(
            {
                "milestone_id": milestone_id,
                "adapter": "maxim_9b_source_replay_aggregate_v1",
                "aggregate": artifact(path),
            }
        )
    wrapper = {
        "schema_version": "vlm-9b-milestone-comparison-v2",
        "model": MODEL,
        "benchmark": artifact(args.benchmark.resolve()),
        "milestones": [*legacy, *native],
    }
    output = args.output_dir.resolve() / "comparison.json"
    write_json(output, wrapper)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--page-solver", type=Path, required=True)
    parser.add_argument("--page-judge", type=Path, required=True)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--no-tools-solver", type=Path, required=True)
    parser.add_argument("--no-tools-score", type=Path, required=True)
    parser.add_argument("--no-tools-image-judge", type=Path, required=True)
    parser.add_argument("--no-tools-judge-manifest", type=Path, required=True)
    parser.add_argument("--active-solver", type=Path, required=True)
    parser.add_argument("--active-score", type=Path, required=True)
    parser.add_argument("--active-image-judge", type=Path, required=True)
    parser.add_argument("--active-preregistration", type=Path, required=True)
    parser.add_argument("--active-composition-manifest", type=Path, required=True)
    parser.add_argument("--source-v1-aggregate", type=Path)
    parser.add_argument("--source-v3-aggregate", type=Path)
    parser.add_argument("--source-v6-aggregate", type=Path)
    parser.add_argument("--source-v7-aggregate", type=Path)
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        legacy = build_legacy(args)
        output = None if args.legacy_only else build_wrapper(args, legacy)
    except (BuildError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        json.dumps(
            {"legacy_milestones": legacy, "comparison": str(output) if output else None},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
