"""Fail-closed evaluation of frozen opaque Math12 source bindings.

This evaluator measures only whether an opaque input was bound to the expected
Math12 activity.  It never reads an answer, correctness label, manual score, or
solution text as evaluation truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .official_ogm import canonical_json_bytes, canonical_json_sha256, sha256_file


RUN_SCHEMA = "math12-opaque-source-batch-run-v1"
RESULT_SCHEMA = "math12-opaque-source-batch-result-v1"
SEAL_SCHEMA = "holdout80-opaque-resolver-input-seal-v1"
EVALUATION_SCHEMA = "math12-opaque-source-binding-evaluation-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_INPUT_ID = re.compile(r"^input-[0-9a-f]{20}$")
_TASK_ID = re.compile(r"^h80-math12-a([0-9]{3})$")


class Math12BindingEvaluationError(ValueError):
    """The frozen run or private source-address map is incomplete or invalid."""


def _strict_json(text: str, *, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise Math12BindingEvaluationError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise Math12BindingEvaluationError(f"non-finite JSON value in {label}: {value}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise Math12BindingEvaluationError(f"malformed JSON in {label}") from exc


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = _strict_json(path.read_text(encoding="utf-8-sig"), label=label)
    except (OSError, UnicodeError) as exc:
        raise Math12BindingEvaluationError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise Math12BindingEvaluationError(f"{label} must be one JSON object")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Math12BindingEvaluationError(f"cannot read {label}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise Math12BindingEvaluationError(f"{label} is empty or contains blank lines")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        value = _strict_json(line, label=f"{label} line {index}")
        if not isinstance(value, dict):
            raise Math12BindingEvaluationError(f"{label} line {index} is not an object")
        rows.append(value)
    return rows


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.name != "run_manifest.json":
            artifacts[path.relative_to(run_dir).as_posix()] = sha256_file(path)
    return artifacts


def _validate_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = run_dir.resolve(strict=False)
    if not run_dir.is_dir() or ".tmp-" in run_dir.name:
        raise Math12BindingEvaluationError("run directory is missing or is a temporary partial output")
    manifest = _load_json(run_dir / "run_manifest.json", label="run manifest")
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise Math12BindingEvaluationError("unexpected run manifest schema")
    if manifest.get("component_scope") != "source_resolution_only":
        raise Math12BindingEvaluationError("run scope is not source resolution only")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise Math12BindingEvaluationError("run manifest has no artifact pins")
    if any(not isinstance(key, str) or _HEX64.fullmatch(str(value)) is None for key, value in artifacts.items()):
        raise Math12BindingEvaluationError("malformed run artifact pins")
    observed = _artifact_hashes(run_dir)
    if observed != artifacts:
        raise Math12BindingEvaluationError("run artifacts differ from the frozen manifest")
    if canonical_json_sha256(artifacts) != manifest.get("artifacts_projection_sha256"):
        raise Math12BindingEvaluationError("run artifact projection mismatch")
    results = _load_jsonl(run_dir / "results.jsonl", label="run results")
    if len(results) != manifest.get("input_count"):
        raise Math12BindingEvaluationError("run input count mismatch")
    return manifest, results


def _load_expected_map(
    input_seal_path: Path,
    private_map_path: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    seal = _load_json(input_seal_path, label="opaque input seal")
    if seal.get("schema_version") != SEAL_SCHEMA or seal.get("family_partition") != "math12":
        raise Math12BindingEvaluationError("unexpected opaque input seal")
    expected_map_sha = seal.get("private_task_map_sha256")
    if _HEX64.fullmatch(str(expected_map_sha)) is None or sha256_file(private_map_path) != expected_map_sha:
        raise Math12BindingEvaluationError("private source-address map SHA mismatch")
    rows = _load_jsonl(private_map_path, label="private source-address map")
    expected: dict[str, int] = {}
    for row in rows:
        if set(row) != {"input_id", "task_id"}:
            raise Math12BindingEvaluationError("private map contains unexpected fields")
        input_id = str(row["input_id"])
        match = _TASK_ID.fullmatch(str(row["task_id"]))
        if _INPUT_ID.fullmatch(input_id) is None or match is None:
            raise Math12BindingEvaluationError("private map contains malformed identifiers")
        if input_id in expected:
            raise Math12BindingEvaluationError("private map contains duplicate input IDs")
        expected[input_id] = int(match.group(1))
    if len(expected) != seal.get("count"):
        raise Math12BindingEvaluationError("private map count differs from its seal")
    return seal, expected


def evaluate_math12_bindings(
    *,
    run_dir: Path,
    input_seal_path: Path,
    private_map_path: Path,
) -> dict[str, Any]:
    """Evaluate source-address transfer with abstention counted as incorrect."""

    manifest, results = _validate_run(run_dir)
    seal, expected = _load_expected_map(input_seal_path, private_map_path)
    if manifest.get("input_jsonl_sha256") != seal.get("public_inputs_sha256"):
        raise Math12BindingEvaluationError("run inputs differ from the sealed opaque inputs")

    observed_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("schema_version") != RESULT_SCHEMA:
            raise Math12BindingEvaluationError("unexpected result schema")
        if result.get("component_scope") != "source_resolution_only":
            raise Math12BindingEvaluationError("result scope is not source resolution only")
        input_id = str(result.get("input_id") or "")
        if input_id in observed_ids or input_id not in expected:
            raise Math12BindingEvaluationError("duplicate or unknown result input ID")
        observed_ids.add(input_id)
        aggregate = result.get("aggregate")
        if not isinstance(aggregate, dict):
            raise Math12BindingEvaluationError("result aggregate is missing")
        accepted = aggregate.get("accepted")
        selected = aggregate.get("selected_activity_number")
        if not isinstance(accepted, bool):
            raise Math12BindingEvaluationError("aggregate accepted flag is malformed")
        if accepted:
            if not isinstance(selected, int) or selected < 1 or selected > 95:
                raise Math12BindingEvaluationError("accepted aggregate lacks a valid activity")
        elif selected is not None:
            raise Math12BindingEvaluationError("abstained aggregate exposes an activity")
        expected_activity = expected[input_id]
        correct = bool(accepted and selected == expected_activity)
        rows.append(
            {
                "input_id": input_id,
                "expected_activity": expected_activity,
                "accepted": accepted,
                "predicted_activity": selected,
                "correct": correct,
                "reason": str(aggregate.get("reason") or ""),
            }
        )
    if observed_ids != set(expected):
        raise Math12BindingEvaluationError("run is missing sealed input IDs")

    rows.sort(key=lambda item: item["input_id"])
    total = len(rows)
    accepted_count = sum(row["accepted"] for row in rows)
    correct_count = sum(row["correct"] for row in rows)
    accepted_correct = sum(row["correct"] for row in rows if row["accepted"])
    return {
        "schema_version": EVALUATION_SCHEMA,
        "scope": "source_activity_binding_only_not_qa_accuracy",
        "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "input_seal_sha256": sha256_file(input_seal_path),
        "private_map_sha256": sha256_file(private_map_path),
        "input_jsonl_sha256": manifest["input_jsonl_sha256"],
        "total": total,
        "accepted": accepted_count,
        "abstained": total - accepted_count,
        "correct": correct_count,
        "incorrect": total - correct_count,
        "coverage": accepted_count / total if total else None,
        "source_binding_accuracy": correct_count / total if total else None,
        "conditional_precision": accepted_correct / accepted_count if accepted_count else None,
        "rows": rows,
        "evaluation_projection_sha256": canonical_json_sha256(rows),
    }


def write_evaluation(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Math12BindingEvaluationError("evaluation output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")

