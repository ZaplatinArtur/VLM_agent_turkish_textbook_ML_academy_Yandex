from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .ingest import read_records


REQUIRED_SETUPS = ("no_tools", "web_search", "textbook_retrieval")
_TASK_CONTRACT_FIELDS = (
    "subject",
    "grade",
    "answer_type",
    "question_text",
    "question_image_url",
    "reference_answer",
    "reference_image_url",
    "acceptable_answers",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _Issues:
    def __init__(self, *, maximum_examples: int = 25) -> None:
        self.counts: Counter[tuple[str, str]] = Counter()
        self.examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.maximum_examples = maximum_examples

    def add(self, severity: str, code: str, **detail: Any) -> None:
        key = (severity, code)
        self.counts[key] += 1
        if len(self.examples[key]) < self.maximum_examples:
            self.examples[key].append(detail)

    def count(self, severity: str) -> int:
        return sum(value for (level, _), value in self.counts.items() if level == severity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": {
                severity: {
                    code: self.counts[(severity, code)]
                    for level, code in sorted(self.counts)
                    if level == severity
                }
                for severity in ("error", "warning")
            },
            "examples": {
                severity: {
                    code: self.examples[(severity, code)]
                    for level, code in sorted(self.examples)
                    if level == severity
                }
                for severity in ("error", "warning")
            },
        }


def _metadata_value(metadata: dict[str, Any], key: str) -> Any:
    if metadata.get(key) not in (None, ""):
        return metadata[key]
    agent_run = metadata.get("agent_run")
    if isinstance(agent_run, dict):
        return agent_run.get(key)
    return None


def _same_contract_value(left: Any, right: Any) -> bool:
    if left is None and right in (None, ""):
        return True
    if right is None and left in (None, ""):
        return True
    if isinstance(left, list) or isinstance(right, list):
        return list(left or []) == list(right or [])
    return str(left) == str(right)


def parse_run_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"run must be SETUP=PATH, got {value!r}")
        setup, path_value = value.split("=", 1)
        setup = setup.strip()
        path_value = path_value.strip()
        if not setup or not path_value:
            raise ValueError(f"run must be SETUP=PATH, got {value!r}")
        if setup in result:
            raise ValueError(f"duplicate run specification for setup {setup!r}")
        result[setup] = Path(path_value)
    return result


def validate_experiment_runs(
    benchmark_path: Path,
    run_paths: dict[str, Path],
    *,
    required_setups: tuple[str, ...] = REQUIRED_SETUPS,
    strict_metadata: bool = False,
) -> dict[str, Any]:
    issues = _Issues()
    benchmark_records = read_records(benchmark_path)
    benchmark: dict[str, dict[str, Any]] = {}
    for record in benchmark_records:
        task_id = str(record.get("task_id") or "").strip()
        if not task_id:
            issues.add("error", "benchmark_missing_task_id")
            continue
        if task_id in benchmark:
            issues.add("error", "benchmark_duplicate_task_id", task_id=task_id)
        benchmark[task_id] = record

    for setup in required_setups:
        if setup not in run_paths:
            issues.add("error", "missing_required_setup", setup=setup)
    for setup in run_paths:
        if setup not in required_setups:
            issues.add("warning", "unexpected_setup", setup=setup)

    per_setup: dict[str, Any] = {}
    task_sets: dict[str, set[str]] = {}
    run_fingerprints: dict[str, str] = {}
    for expected_setup, path in sorted(run_paths.items()):
        if not path.is_file():
            issues.add("error", "run_file_missing", setup=expected_setup, path=str(path))
            per_setup[expected_setup] = {"path": str(path), "records": 0, "file_missing": True}
            task_sets[expected_setup] = set()
            continue
        records = read_records(path)
        run_fingerprints[expected_setup] = _file_sha256(path)
        seen: set[str] = set()
        failures = 0
        metadata_missing = Counter()
        metadata_values: dict[str, set[str]] = defaultdict(set)
        drift_counts = Counter()
        for row_number, record in enumerate(records, start=1):
            task_id = str(record.get("task_id") or "").strip()
            if not task_id:
                issues.add("error", "run_missing_task_id", setup=expected_setup, row=row_number)
                continue
            if task_id in seen:
                issues.add("error", "duplicate_task_id_in_run", setup=expected_setup, task_id=task_id)
                continue
            seen.add(task_id)
            task = benchmark.get(task_id)
            if task is None:
                issues.add("error", "unknown_task_id", setup=expected_setup, task_id=task_id)
                continue
            declared_setup = str(record.get("setup") or "").strip()
            if declared_setup != expected_setup:
                issues.add(
                    "error",
                    "setup_mismatch",
                    expected=expected_setup,
                    declared=declared_setup or None,
                    task_id=task_id,
                )
            if "candidate_answer" not in record:
                issues.add("error", "candidate_answer_missing", setup=expected_setup, task_id=task_id)
            else:
                candidate = str(record.get("candidate_answer") or "")
                failed = not candidate.strip() or candidate.strip() in {
                    "[EMPTY_RESPONSE]",
                    "[TIMEOUT]",
                    "[AGENT_ERROR]",
                }
                failures += int(failed)
                if not candidate.strip():
                    issues.add(
                        "error",
                        "empty_failure_not_normalized",
                        setup=expected_setup,
                        task_id=task_id,
                    )
            for field in _TASK_CONTRACT_FIELDS:
                if field not in record:
                    issues.add(
                        "error",
                        "benchmark_field_missing_from_run",
                        setup=expected_setup,
                        task_id=task_id,
                        field=field,
                    )
                    drift_counts[f"missing:{field}"] += 1
                elif not _same_contract_value(record.get(field), task.get(field)):
                    issues.add(
                        "error",
                        "benchmark_field_drift",
                        setup=expected_setup,
                        task_id=task_id,
                        field=field,
                    )
                    drift_counts[field] += 1

            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            for key in ("run_id", "agent_model", "agent_prompt_version", "seed", "latency_ms"):
                metadata_value = _metadata_value(metadata, key)
                if metadata_value in (None, ""):
                    metadata_missing[key] += 1
                    issues.add(
                        "error" if strict_metadata else "warning",
                        "run_metadata_missing",
                        setup=expected_setup,
                        task_id=task_id,
                        field=key,
                    )
                elif key != "latency_ms":
                    metadata_values[key].add(
                        json.dumps(metadata_value, ensure_ascii=False, sort_keys=True, default=str)
                    )
            if expected_setup == "textbook_retrieval":
                if _metadata_value(metadata, "retrieved_chunk_ids") in (None, "", []):
                    metadata_missing["retrieved_chunk_ids"] += 1
                    issues.add(
                        "error" if strict_metadata else "warning",
                        "retrieval_trace_missing",
                        setup=expected_setup,
                        task_id=task_id,
                    )
                retrieval_config_hash = _metadata_value(metadata, "retrieval_config_hash")
                if retrieval_config_hash in (None, ""):
                    metadata_missing["retrieval_config_hash"] += 1
                    issues.add(
                        "error" if strict_metadata else "warning",
                        "retrieval_config_missing",
                        setup=expected_setup,
                        task_id=task_id,
                    )
                else:
                    metadata_values["retrieval_config_hash"].add(str(retrieval_config_hash))

        for key, values in metadata_values.items():
            if len(values) > 1:
                issues.add(
                    "error",
                    "inconsistent_run_metadata",
                    setup=expected_setup,
                    field=key,
                    distinct_values=len(values),
                    examples=sorted(values)[:5],
                )

        missing = sorted(set(benchmark) - seen)
        for task_id in missing:
            issues.add("error", "task_missing_from_run", setup=expected_setup, task_id=task_id)
        task_sets[expected_setup] = seen & set(benchmark)
        per_setup[expected_setup] = {
            "path": str(path),
            "sha256": run_fingerprints[expected_setup],
            "records": len(records),
            "unique_known_tasks": len(task_sets[expected_setup]),
            "missing_benchmark_tasks": len(missing),
            "recorded_agent_failures": failures,
            "metadata_missing": dict(metadata_missing),
            "metadata_distinct_values": {key: len(values) for key, values in metadata_values.items()},
            "contract_drift": dict(drift_counts),
        }

    required_sets = [task_sets.get(setup, set()) for setup in required_setups]
    complete_grid = set.intersection(*required_sets) if required_sets else set()
    union = set.union(*required_sets) if required_sets else set()
    issue_data = issues.to_dict()
    errors = issues.count("error")
    warnings = issues.count("warning")
    return {
        "ready_for_experiment": errors == 0,
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": _file_sha256(benchmark_path),
            "records": len(benchmark_records),
            "unique_task_ids": len(benchmark),
        },
        "required_setups": list(required_setups),
        "provided_setups": sorted(run_paths),
        "strict_metadata": strict_metadata,
        "per_setup": per_setup,
        "grid": {
            "benchmark_tasks": len(benchmark),
            "task_union": len(union),
            "complete_task_setup_grid": len(complete_grid),
            "complete_grid_rate": len(complete_grid) / len(benchmark) if benchmark else None,
        },
        "error_count": errors,
        "warning_count": warnings,
        "issues": issue_data,
    }
