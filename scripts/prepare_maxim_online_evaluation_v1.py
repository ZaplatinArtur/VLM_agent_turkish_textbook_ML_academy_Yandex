#!/usr/bin/env python3
"""Prepare matched judge-v2 evaluation artifacts for one online solver run.

This is a strictly post-generation utility.  It accepts a completed solver
JSONL and frozen evaluation assets, then:

* rebuilds the canonical 97-row image-judge input by changing only the
  template's ``candidate_answer`` and ``setup`` fields;
* reuses a frozen page-RAG judge-v2 verdict only when the UTF-8 bytes of the
  candidate text serialized from both solver rows are identical;
* emits a fresh judge queue for every changed candidate;
* audits the 177 deterministic rows and reports score bounds before the fresh
  queue is judged;
* records byte-level provenance and the exact arguments needed by the existing
  merge and scoring utilities.

Gold/reference data is read only here, after generation has finished.  The
solver artifact is rejected if it contains top-level gold fields or if any row
does not explicitly declare ``generation.gold_access=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import merge_maxim_judge_v2_results as judge_merger
import score_maxim_full274 as scorer
from prepare_canonical_judge97_input import candidate_text


SCHEMA_VERSION = "maxim-online-postgeneration-evaluation-v1"
DEFAULT_EXPECTED_ROWS = 274
DEFAULT_EXPECTED_DETERMINISTIC = 177
DEFAULT_EXPECTED_IMAGE_JUDGE = 97

# Pins for the local frozen common-benchmark assets verified on 2026-08-03.
FROZEN_BENCHMARK_SHA256 = scorer.FROZEN_BENCHMARK_SHA256
FROZEN_BASELINE_JUDGE_SHA256 = scorer.FROZEN_BASELINE_JUDGE_SHA256
FROZEN_BASELINE_SOLVER_SHA256 = (
    "62bc952c3802308bc0fbf8d8dc1f82ec523a3ab1e3264bae87a5f8828021d75d"
)
FROZEN_IMAGE_TEMPLATE_SHA256 = (
    "41f35172092f67bc14368d7312cb91ad50256166dc42d0f630ed5e7a9965aa46"
)

CANONICAL_TEMPLATE_KEYS = frozenset(
    {
        "task_id",
        "candidate_answer",
        "subject",
        "grade",
        "answer_type",
        "setup",
        "question_text",
        "question_image_url",
        "reference_answer",
        "reference_image_url",
        "acceptable_answers",
        "metadata",
    }
)

OUTPUT_NAMES = {
    "image97_input": "image97_input.jsonl",
    "fresh_input": "fresh_judge_v2_input.jsonl",
    "reusable_judge": "reusable_judge_v2.jsonl",
    "deterministic": "deterministic_partial.json",
    "manifest": "evaluation_manifest.json",
    "checksums": "evaluation.sha256",
}


class PreparationError(ValueError):
    """Raised when a frozen input or solver artifact violates the protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PreparationError(f"{label}: file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreparationError(
                    f"{label}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise PreparationError(f"{label}:{line_number}: row is not an object")
            rows.append(row)
    if not rows:
        raise PreparationError(f"{label}: no JSONL rows")
    return rows


def index_rows(
    rows: Iterable[dict[str, Any]], label: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for position, row in enumerate(rows, 1):
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            raise PreparationError(f"{label}:{position}: missing task_id")
        if task_id in indexed:
            raise PreparationError(f"{label}: duplicate task_id {task_id}")
        indexed[task_id] = row
        order.append(task_id)
    return indexed, order


def _assert_hash(path: Path, expected: str | None, label: str) -> str:
    actual = sha256_file(path)
    if expected is None:
        return actual
    normalized = expected.strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise PreparationError(f"{label}: expected SHA256 is not 64 hex characters")
    if actual != normalized:
        raise PreparationError(
            f"{label}: SHA256 mismatch; expected={normalized}, actual={actual}, "
            f"path={path}"
        )
    return actual


def _assert_exact_ids(
    actual_order: Sequence[str], expected_order: Sequence[str], label: str
) -> None:
    if list(actual_order) == list(expected_order):
        return
    actual = set(actual_order)
    expected = set(expected_order)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise PreparationError(
            f"{label}: task-ID mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )
    raise PreparationError(f"{label}: task order differs from the frozen benchmark")


def _assert_same_ids(
    actual_order: Sequence[str], expected_order: Sequence[str], label: str
) -> None:
    actual = set(actual_order)
    expected = set(expected_order)
    if len(actual_order) == len(expected_order) and actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise PreparationError(
        f"{label}: task-ID mismatch; missing={missing[:10]}, extra={extra[:10]}"
    )


def _validate_online_solver(rows: dict[str, dict[str, Any]]) -> None:
    try:
        scorer._validate_no_gold(rows)
    except scorer.ScoringError as exc:
        raise PreparationError(str(exc)) from exc
    missing_declaration: list[str] = []
    for task_id, row in rows.items():
        generation = row.get("generation")
        if not isinstance(generation, dict) or generation.get("gold_access") is not False:
            missing_declaration.append(task_id)
    if missing_declaration:
        raise PreparationError(
            "solver results must explicitly declare generation.gold_access=false "
            f"on every row; invalid task IDs={missing_declaration[:20]}"
        )


def _validate_template(row: dict[str, Any], task_id: str) -> None:
    keys = set(row)
    if keys != CANONICAL_TEMPLATE_KEYS:
        raise PreparationError(
            f"image template {task_id}: canonical key set changed; "
            f"missing={sorted(CANONICAL_TEMPLATE_KEYS - keys)}, "
            f"extra={sorted(keys - CANONICAL_TEMPLATE_KEYS)}"
        )
    if str(row.get("task_id") or "") != task_id:
        raise PreparationError(f"image template {task_id}: internal task_id changed")
    if not str(row.get("question_image_url") or "").strip():
        raise PreparationError(f"image template {task_id}: missing question_image_url")
    if not str(row.get("reference_image_url") or "").strip():
        raise PreparationError(f"image template {task_id}: missing reference_image_url")
    if row.get("reference_answer") not in (None, ""):
        raise PreparationError(
            f"image template {task_id}: expected an image reference, found text gold"
        )


def canonical_input(
    template: dict[str, Any], solver_row: dict[str, Any], setup: str
) -> dict[str, Any]:
    """Copy a frozen template, mutating only judge candidate and setup."""

    row = dict(template)
    row["candidate_answer"] = candidate_text(solver_row)
    row["setup"] = setup
    return row


def candidate_bytes(row: dict[str, Any]) -> bytes:
    """Return the exact bytes compared by the verdict-reuse policy."""

    return candidate_text(row).encode("utf-8")


def _tag_reused_judge(
    row: dict[str, Any], *, baseline_solver_sha256: str
) -> dict[str, Any]:
    copied = dict(row)
    copied["post_generation_reuse"] = {
        "verdict_reused": True,
        "source": "frozen_page_rag_judge_v2",
        "policy": "candidate_text_utf8_bytes_equal",
        "baseline_solver_sha256": baseline_solver_sha256,
    }
    return copied


def deterministic_audit(
    *,
    task_order: Sequence[str],
    benchmark: dict[str, dict[str, Any]],
    solver: dict[str, dict[str, Any]],
    baseline_judge: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    for task_id in task_order:
        if scorer._baseline_source(baseline_judge[task_id], task_id) != "deterministic":
            continue
        task = benchmark[task_id]
        result = solver[task_id]
        matched, method = scorer.deterministic_match(
            str(result.get("final_answer") or ""),
            str(task.get("reference_answer") or ""),
            str(task.get("answer_type") or "short_text"),
        )
        if matched is None:
            raise PreparationError(
                f"deterministic partition unexpectedly needs judge: {task_id}"
            )
        eligible = not scorer._solver_error(result) and not scorer._missing_answer(result)
        outcomes.append(
            {
                "task_id": task_id,
                "subject": str(task.get("subject") or "<missing>"),
                "correct": bool(matched) and eligible,
                "method": method,
            }
        )
    subjects: dict[str, dict[str, int]] = {}
    for subject in sorted({row["subject"] for row in outcomes}):
        group = [row for row in outcomes if row["subject"] == subject]
        subjects[subject] = {
            "correct": sum(int(row["correct"]) for row in group),
            "n": len(group),
        }
    correct = sum(int(row["correct"]) for row in outcomes)
    return {
        "correct": correct,
        "n": len(outcomes),
        "accuracy": round(correct / len(outcomes), 6) if outcomes else None,
        "by_subject": subjects,
        "task_outcomes": outcomes,
    }


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write_many(items: Sequence[tuple[Path, bytes]]) -> None:
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for destination, data in items:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=destination.name + ".",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_paths.append((Path(temporary.name), destination))
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in temporary_paths:
            if temporary.exists():
                temporary.unlink()


def _source(path: Path, digest: str, rows: int) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": digest, "rows": rows}


def prepare_from_paths(
    *,
    benchmark_path: Path,
    solver_path: Path,
    baseline_solver_path: Path,
    baseline_judge_path: Path,
    image_template_path: Path,
    output_dir: Path,
    setup: str,
    mode: str | None = None,
    label: str | None = None,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
    expected_deterministic: int = DEFAULT_EXPECTED_DETERMINISTIC,
    expected_image_judge: int = DEFAULT_EXPECTED_IMAGE_JUDGE,
    expected_benchmark_sha256: str | None = FROZEN_BENCHMARK_SHA256,
    expected_baseline_solver_sha256: str | None = FROZEN_BASELINE_SOLVER_SHA256,
    expected_baseline_judge_sha256: str | None = FROZEN_BASELINE_JUDGE_SHA256,
    expected_image_template_sha256: str | None = FROZEN_IMAGE_TEMPLATE_SHA256,
) -> dict[str, Any]:
    if not setup.strip():
        raise PreparationError("setup must be non-empty")
    if mode is not None and not mode.strip():
        raise PreparationError("mode must be non-empty when provided")

    input_paths = {
        path.resolve()
        for path in (
            benchmark_path,
            solver_path,
            baseline_solver_path,
            baseline_judge_path,
            image_template_path,
        )
    }
    outputs = {name: output_dir / filename for name, filename in OUTPUT_NAMES.items()}
    if len({path.resolve() for path in outputs.values()}) != len(outputs):
        raise PreparationError("internal error: output paths are not distinct")
    for path in outputs.values():
        if path.resolve() in input_paths:
            raise PreparationError(f"output aliases an input: {path}")
        if path.exists():
            raise PreparationError(f"refusing to overwrite existing output: {path}")

    benchmark_hash = _assert_hash(
        benchmark_path, expected_benchmark_sha256, "benchmark"
    )
    baseline_solver_hash = _assert_hash(
        baseline_solver_path, expected_baseline_solver_sha256, "baseline solver"
    )
    baseline_judge_hash = _assert_hash(
        baseline_judge_path, expected_baseline_judge_sha256, "baseline judge"
    )
    template_hash = _assert_hash(
        image_template_path, expected_image_template_sha256, "image template"
    )
    solver_hash = sha256_file(solver_path)

    benchmark_rows = read_jsonl(benchmark_path, "benchmark")
    solver_rows = read_jsonl(solver_path, "solver results")
    baseline_solver_rows = read_jsonl(baseline_solver_path, "baseline solver")
    baseline_judge_rows = read_jsonl(baseline_judge_path, "baseline judge")
    template_rows = read_jsonl(image_template_path, "image template")

    benchmark, task_order = index_rows(benchmark_rows, "benchmark")
    solver, solver_order = index_rows(solver_rows, "solver results")
    baseline_solver, baseline_solver_order = index_rows(
        baseline_solver_rows, "baseline solver"
    )
    baseline_judge, baseline_judge_order = index_rows(
        baseline_judge_rows, "baseline judge"
    )
    templates, template_order = index_rows(template_rows, "image template")

    if len(task_order) != expected_rows:
        raise PreparationError(
            f"expected {expected_rows} benchmark tasks, found {len(task_order)}"
        )
    _assert_exact_ids(solver_order, task_order, "solver results")
    # The frozen page-RAG solver was generated in batched execution order, not
    # benchmark order.  Membership is authoritative; lookup is by task_id.
    _assert_same_ids(baseline_solver_order, task_order, "baseline solver")
    _assert_same_ids(baseline_judge_order, task_order, "baseline judge")
    _validate_online_solver(solver)
    try:
        scorer._validate_no_gold(baseline_solver)
    except scorer.ScoringError as exc:
        raise PreparationError(str(exc)) from exc

    image_ids: list[str] = []
    deterministic_ids: list[str] = []
    for task_id in task_order:
        source = scorer._baseline_source(baseline_judge[task_id], task_id)
        (image_ids if source == "image_judge" else deterministic_ids).append(task_id)
    if len(deterministic_ids) != expected_deterministic:
        raise PreparationError(
            f"expected {expected_deterministic} deterministic tasks, "
            f"found {len(deterministic_ids)}"
        )
    if len(image_ids) != expected_image_judge:
        raise PreparationError(
            f"expected {expected_image_judge} image tasks, found {len(image_ids)}"
        )
    _assert_exact_ids(template_order, image_ids, "image template")

    for task_id in image_ids:
        _validate_template(templates[task_id], task_id)
        try:
            judge_merger.validate_judge_row(
                baseline_judge[task_id],
                label="baseline judge-v2",
                task_id=task_id,
                expected_prompt_version="judge-v2",
            )
        except judge_merger.MergeError as exc:
            raise PreparationError(str(exc)) from exc

    deterministic = deterministic_audit(
        task_order=task_order,
        benchmark=benchmark,
        solver=solver,
        baseline_judge=baseline_judge,
    )

    full_inputs: list[dict[str, Any]] = []
    fresh_inputs: list[dict[str, Any]] = []
    reusable_rows: list[dict[str, Any]] = []
    unchanged_ids: list[str] = []
    changed_ids: list[str] = []
    reused_strict_correct = 0
    fresh_scorable = 0

    for task_id in image_ids:
        result = solver[task_id]
        full_input = canonical_input(templates[task_id], result, setup.strip())
        full_inputs.append(full_input)
        if candidate_bytes(result) == candidate_bytes(baseline_solver[task_id]):
            unchanged_ids.append(task_id)
            tagged = _tag_reused_judge(
                baseline_judge[task_id],
                baseline_solver_sha256=baseline_solver_hash,
            )
            reusable_rows.append(tagged)
            eligible = not scorer._solver_error(result) and not scorer._missing_answer(result)
            if eligible and scorer._strict_correct(tagged, "reusable judge", task_id):
                reused_strict_correct += 1
        else:
            changed_ids.append(task_id)
            fresh_inputs.append(full_input)
            if not scorer._solver_error(result) and not scorer._missing_answer(result):
                fresh_scorable += 1

    deterministic_data = _json_bytes(deterministic)
    image_data = _jsonl_bytes(full_inputs)
    fresh_data = _jsonl_bytes(fresh_inputs)
    reusable_data = _jsonl_bytes(reusable_rows)
    artifact_data = {
        "image97_input": image_data,
        "fresh_input": fresh_data,
        "reusable_judge": reusable_data,
        "deterministic": deterministic_data,
    }
    artifact_hashes = {name: sha256_bytes(data) for name, data in artifact_data.items()}

    lower_correct = deterministic["correct"] + reused_strict_correct
    upper_correct = lower_correct + fresh_scorable
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode.strip() if mode is not None else (label or setup.strip()),
        "setup": setup.strip(),
        "label": label or setup.strip(),
        "stage": "post-generation evaluation preparation",
        "generation_gold_access": False,
        "evaluation_gold_access": True,
        "reuse_policy": {
            "name": "candidate_text_utf8_bytes_equal",
            "adapter": "prepare_canonical_judge97_input.candidate_text",
            "requires": [
                "identical task_id",
                "identical UTF-8 bytes after frozen candidate serialization",
                "valid error-free baseline judge-v2 row",
            ],
            "normalization_after_serialization": False,
            "unchanged_rows": len(unchanged_ids),
            "changed_rows": len(changed_ids),
            "unchanged_task_ids": unchanged_ids,
            "changed_task_ids": changed_ids,
        },
        "sources": {
            "benchmark": _source(benchmark_path, benchmark_hash, len(benchmark_rows)),
            "solver": _source(solver_path, solver_hash, len(solver_rows)),
            "baseline_solver": _source(
                baseline_solver_path, baseline_solver_hash, len(baseline_solver_rows)
            ),
            "baseline_judge": _source(
                baseline_judge_path, baseline_judge_hash, len(baseline_judge_rows)
            ),
            "image_template": _source(
                image_template_path, template_hash, len(template_rows)
            ),
        },
        "artifacts": {
            name: {
                "path": str(outputs[name].resolve()),
                "sha256": artifact_hashes[name],
                "rows": (
                    len(full_inputs)
                    if name == "image97_input"
                    else len(fresh_inputs)
                    if name == "fresh_input"
                    else len(reusable_rows)
                    if name == "reusable_judge"
                    else deterministic["n"]
                ),
            }
            for name in artifact_data
        },
        "deterministic": {
            "correct": deterministic["correct"],
            "n": deterministic["n"],
        },
        "current_score_bounds_before_fresh_judge": {
            "lower_correct": lower_correct,
            "upper_correct": upper_correct,
            "n": len(task_order),
            "lower_accuracy": round(lower_correct / len(task_order), 6),
            "upper_accuracy": round(upper_correct / len(task_order), 6),
            "reused_judge_strict_correct_after_solver_failure_override": reused_strict_correct,
            "fresh_scorable_rows": fresh_scorable,
            "note": "bounds only; fresh rows have not been assigned verdicts",
        },
        "compatibility": {
            "merge_script": "scripts/merge_maxim_judge_v2_results.py",
            "merge_arguments": {
                "image_template": str(outputs["image97_input"].resolve()),
                "reusable_judge": str(outputs["reusable_judge"].resolve()),
                "fresh_judge": str((output_dir / "fresh_judge_v2_result.jsonl").resolve()),
                "out_jsonl": str((output_dir / "matched_image97_judge.jsonl").resolve()),
                "out_manifest": str(
                    (output_dir / "matched_image97_judge_manifest.json").resolve()
                ),
                "out_sha256": str(
                    (output_dir / "matched_image97_judge.sha256").resolve()
                ),
            },
            "score_script": "scripts/score_maxim_full274.py",
            "score_arguments": {
                "benchmark": str(benchmark_path.resolve()),
                "solver_results": str(solver_path.resolve()),
                "image_judge": str(
                    (output_dir / "matched_image97_judge.jsonl").resolve()
                ),
                "baseline_judge": str(baseline_judge_path.resolve()),
                "out_json": str((output_dir / "score.json").resolve()),
                "out_md": str((output_dir / "score.md").resolve()),
                "out_sha256": str((output_dir / "score.sha256").resolve()),
            },
        },
    }
    manifest_data = _json_bytes(manifest)
    checksum_lines = [
        f"{artifact_hashes[name]}  {outputs[name].name}"
        for name in ("image97_input", "fresh_input", "reusable_judge", "deterministic")
    ]
    checksum_lines.append(f"{sha256_bytes(manifest_data)}  {outputs['manifest'].name}")
    checksum_data = ("\n".join(checksum_lines) + "\n").encode("utf-8")

    _atomic_write_many(
        [
            (outputs["image97_input"], image_data),
            (outputs["fresh_input"], fresh_data),
            (outputs["reusable_judge"], reusable_data),
            (outputs["deterministic"], deterministic_data),
            (outputs["manifest"], manifest_data),
            (outputs["checksums"], checksum_data),
        ]
    )
    return {
        "manifest": str(outputs["manifest"].resolve()),
        "manifest_sha256": sha256_file(outputs["manifest"]),
        "solver_sha256": solver_hash,
        "deterministic_correct": deterministic["correct"],
        "deterministic_rows": deterministic["n"],
        "reusable_judge_rows": len(reusable_rows),
        "fresh_judge_rows": len(fresh_inputs),
        "lower_correct": lower_correct,
        "upper_correct": upper_correct,
        "n": len(task_order),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--solver-results", type=Path, required=True)
    parser.add_argument("--baseline-solver", type=Path, required=True)
    parser.add_argument("--baseline-judge", type=Path, required=True)
    parser.add_argument("--image-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--setup", required=True)
    parser.add_argument("--mode")
    parser.add_argument("--label")
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    parser.add_argument(
        "--expected-deterministic",
        type=int,
        default=DEFAULT_EXPECTED_DETERMINISTIC,
    )
    parser.add_argument(
        "--expected-image-judge",
        type=int,
        default=DEFAULT_EXPECTED_IMAGE_JUDGE,
    )
    parser.add_argument(
        "--expected-benchmark-sha256", default=FROZEN_BENCHMARK_SHA256
    )
    parser.add_argument(
        "--expected-baseline-solver-sha256", default=FROZEN_BASELINE_SOLVER_SHA256
    )
    parser.add_argument(
        "--expected-baseline-judge-sha256", default=FROZEN_BASELINE_JUDGE_SHA256
    )
    parser.add_argument(
        "--expected-image-template-sha256", default=FROZEN_IMAGE_TEMPLATE_SHA256
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = prepare_from_paths(
            benchmark_path=args.benchmark,
            solver_path=args.solver_results,
            baseline_solver_path=args.baseline_solver,
            baseline_judge_path=args.baseline_judge,
            image_template_path=args.image_template,
            output_dir=args.output_dir,
            setup=args.setup,
            mode=args.mode,
            label=args.label,
            expected_rows=args.expected_rows,
            expected_deterministic=args.expected_deterministic,
            expected_image_judge=args.expected_image_judge,
            expected_benchmark_sha256=args.expected_benchmark_sha256,
            expected_baseline_solver_sha256=args.expected_baseline_solver_sha256,
            expected_baseline_judge_sha256=args.expected_baseline_judge_sha256,
            expected_image_template_sha256=args.expected_image_template_sha256,
        )
    except (OSError, PreparationError) as exc:
        print(f"PREPARATION ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
