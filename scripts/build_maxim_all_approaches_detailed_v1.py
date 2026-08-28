#!/usr/bin/env python3
"""Build a detailed, rerunnable report from the strict full274 ledger.

This script deliberately does not discover score files.  The only score JSON
files it opens are artifacts whose branch status is ``final`` and whose exact
path and SHA-256 are accepted by the strict ``RESULTS.json`` ledger.  Pending,
non-final, and conflict entries are copied into separate sections without
opening any score path they may mention.  In particular, rejected candidate
scores are never read.

For every accepted final, task-level verdicts are used to recompute overall,
Math (139), and non-Math (135) accuracy, plus transitions against the accepted
page-RAG replay and frozen subject-router references.  Operational statistics
are recomputed from task usage where available.  Explicit or exact-output-bound
manifests are summarized for additional calls/errors/fallback information.
Experiment artifacts are read-only; only REPORT.json and REPORT.md are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "maxim-all-approaches-detailed-v1"
LEDGER_SCHEMA_VERSION = "maxim-full274-results-ledger-v1"
EXPECTED_ROWS = 274
EXPECTED_MATH_ROWS = 139
EXPECTED_NON_MATH_ROWS = 135
DEFAULT_PAGE_BASELINE_ID = "answer_canonicalization"
DEFAULT_ROUTER_ID = "subject_router"
_SHA256_CHARS = frozenset("0123456789abcdef")
_DISCOVERABLE_BOUND_MANIFEST_NAMES = (
    "failclosed_manifest.json",
    "solver.run_manifest.json",
    "composition_manifest.json",
    "manifest.json",
)


class DetailedReportError(ValueError):
    """Raised when the accepted ledger or a final artifact is inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetailedReportError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise DetailedReportError(f"JSON root is not an object: {path}")
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value.lower()) <= _SHA256_CHARS
    )


def _inside_repo(path: Path, repo_root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise DetailedReportError(f"{label} escapes repository root: {resolved}") from exc
    return resolved


def _resolve(raw: str, repo_root: Path, relative_to: Path | None = None) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return _inside_repo(path, repo_root, "artifact path")
    if relative_to is not None:
        local = (relative_to / path).resolve()
        if local.exists():
            return _inside_repo(local, repo_root, "artifact path")
    return _inside_repo(repo_root / path, repo_root, "artifact path")


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _verified_descriptor(
    descriptor: Mapping[str, Any],
    repo_root: Path,
    *,
    label: str,
    relative_to: Path | None = None,
) -> tuple[Path, str]:
    raw_path = descriptor.get("path")
    expected_sha = descriptor.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise DetailedReportError(f"{label}.path is missing")
    if not _valid_sha256(expected_sha):
        raise DetailedReportError(f"{label}.sha256 is invalid")
    path = _resolve(raw_path, repo_root, relative_to)
    if not path.is_file():
        raise DetailedReportError(f"{label} is missing: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != str(expected_sha).lower():
        raise DetailedReportError(
            f"{label} SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    return path, actual_sha


def _as_nonnegative_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return int(value) if isinstance(value, int) else number


def _round(value: float) -> float:
    return round(value, 9)


def _read_final_outcomes(
    branch: Mapping[str, Any], repo_root: Path, expected_rows: int
) -> tuple[Path, str, Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    report_descriptor = branch.get("report")
    if not isinstance(report_descriptor, Mapping):
        raise DetailedReportError(f"final branch {branch.get('id')!r} has no report")
    path, digest = _verified_descriptor(
        report_descriptor, repo_root, label=f"final branch {branch.get('id')!r} report"
    )
    report = read_object(path)
    outcomes = report.get("task_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != expected_rows:
        raise DetailedReportError(
            f"final branch {branch.get('id')!r} must have {expected_rows} task outcomes"
        )

    by_id: dict[str, Mapping[str, Any]] = {}
    correct = 0
    for index, row in enumerate(outcomes):
        if not isinstance(row, Mapping):
            raise DetailedReportError(f"task_outcomes[{index}] is not an object in {path}")
        task_id = row.get("task_id")
        verdict = row.get("new_correct")
        subject = row.get("subject")
        if not isinstance(task_id, str) or not task_id or task_id in by_id:
            raise DetailedReportError(f"invalid/duplicate task_id at row {index} in {path}")
        if not isinstance(verdict, bool):
            raise DetailedReportError(f"new_correct is not boolean for {task_id} in {path}")
        if not isinstance(subject, str) or not subject:
            raise DetailedReportError(f"subject is missing for {task_id} in {path}")
        by_id[task_id] = row
        correct += int(verdict)

    ledger_correct = branch.get("correct")
    ledger_denominator = branch.get("denominator")
    if ledger_correct != correct or ledger_denominator != expected_rows:
        raise DetailedReportError(
            f"recomputed {correct}/{expected_rows} disagrees with ledger for {branch.get('id')!r}"
        )
    ledger_accuracy = branch.get("accuracy")
    if (
        isinstance(ledger_accuracy, bool)
        or not isinstance(ledger_accuracy, (int, float))
        or abs(float(ledger_accuracy) - correct / expected_rows) > 1e-9
    ):
        raise DetailedReportError(
            f"recomputed accuracy disagrees with ledger for {branch.get('id')!r}"
        )
    return path, digest, report, by_id


def _segments(subject_by_id: Mapping[str, str]) -> dict[str, set[str]]:
    overall = set(subject_by_id)
    math_ids = {task_id for task_id, subject in subject_by_id.items() if subject == "Math"}
    non_math_ids = overall - math_ids
    if len(overall) != EXPECTED_ROWS:
        raise DetailedReportError(f"reference task set must contain {EXPECTED_ROWS} rows")
    if len(math_ids) != EXPECTED_MATH_ROWS or len(non_math_ids) != EXPECTED_NON_MATH_ROWS:
        raise DetailedReportError(
            "reference split must be Math139/nonMath135, got "
            f"Math{len(math_ids)}/nonMath{len(non_math_ids)}"
        )
    return {"overall": overall, "math": math_ids, "non_math": non_math_ids}


def _metric(verdicts: Mapping[str, bool], ids: set[str]) -> dict[str, Any]:
    correct = sum(int(verdicts[task_id]) for task_id in ids)
    n = len(ids)
    return {"n": n, "correct": correct, "accuracy": _round(correct / n)}


def _metrics(verdicts: Mapping[str, bool], segments: Mapping[str, set[str]]) -> dict[str, Any]:
    return {name: _metric(verdicts, ids) for name, ids in segments.items()}


def _transition(
    candidate: Mapping[str, bool], reference: Mapping[str, bool], ids: set[str]
) -> dict[str, Any]:
    fixed_ids = sorted(
        task_id for task_id in ids if candidate[task_id] and not reference[task_id]
    )
    regressed_ids = sorted(
        task_id for task_id in ids if not candidate[task_id] and reference[task_id]
    )
    both_correct = sum(int(candidate[t] and reference[t]) for t in ids)
    both_wrong = sum(int(not candidate[t] and not reference[t]) for t in ids)
    candidate_correct = sum(int(candidate[t]) for t in ids)
    reference_correct = sum(int(reference[t]) for t in ids)
    n = len(ids)
    return {
        "n": n,
        "candidate_correct": candidate_correct,
        "reference_correct": reference_correct,
        "fixed": len(fixed_ids),
        "regressed": len(regressed_ids),
        "net_correct": candidate_correct - reference_correct,
        "delta_accuracy": _round((candidate_correct - reference_correct) / n),
        "delta_pp": _round(100.0 * (candidate_correct - reference_correct) / n),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "fixed_task_ids": fixed_ids,
        "regressed_task_ids": regressed_ids,
    }


def _comparisons(
    candidate: Mapping[str, bool],
    reference: Mapping[str, bool],
    segments: Mapping[str, set[str]],
) -> dict[str, Any]:
    return {
        name: _transition(candidate, reference, ids) for name, ids in segments.items()
    }


def _nearest_rank_p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _usage_stats(outcomes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    calls: list[int] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    latencies: list[float] = []
    solver_errors = 0
    missing_answers = 0
    forced_answers = 0
    verifier_errors = 0

    for row in outcomes:
        usage = row.get("usage")
        if isinstance(usage, Mapping):
            call_count = _as_nonnegative_number(usage.get("call_count"))
            input_count = _as_nonnegative_number(usage.get("input_tokens"))
            output_count = _as_nonnegative_number(usage.get("output_tokens"))
            latency = _as_nonnegative_number(usage.get("latency_s"))
            if isinstance(call_count, int):
                calls.append(call_count)
            if isinstance(input_count, int):
                input_tokens.append(input_count)
            if isinstance(output_count, int):
                output_tokens.append(output_count)
            if latency is not None:
                latencies.append(float(latency))
        solver_errors += int(bool(row.get("solver_error")))
        missing_answers += int(row.get("missing_final_answer") is True)
        forced_answers += int(row.get("forced_answer") is True)
        verifier_errors += int(bool(row.get("verifier_error")))

    latency_summary: dict[str, Any] = {
        "reported_rows": len(latencies),
        "total_s": None,
        "mean_s": None,
        "median_s": None,
        "p95_nearest_rank_s": None,
        "max_s": None,
    }
    if latencies:
        latency_summary.update(
            {
                "total_s": _round(sum(latencies)),
                "mean_s": _round(statistics.fmean(latencies)),
                "median_s": _round(statistics.median(latencies)),
                "p95_nearest_rank_s": _round(_nearest_rank_p95(latencies)),
                "max_s": _round(max(latencies)),
            }
        )

    input_total = sum(input_tokens) if input_tokens else None
    output_total = sum(output_tokens) if output_tokens else None
    combined_total = (
        input_total + output_total
        if input_total is not None and output_total is not None
        else None
    )
    return {
        "model_calls": {
            "reported_rows": len(calls),
            "total": sum(calls) if calls else None,
            "mean_per_reported_row": _round(statistics.fmean(calls)) if calls else None,
        },
        "tokens": {
            "input_reported_rows": len(input_tokens),
            "input_total": input_total,
            "output_reported_rows": len(output_tokens),
            "output_total": output_total,
            "combined_total": combined_total,
        },
        "latency": latency_summary,
        "errors": {
            "solver_error_count": solver_errors,
            "missing_final_answer_count": missing_answers,
            "forced_answer_count": forced_answers,
            "verifier_error_count": verifier_errors,
        },
    }


def _iter_manifest_descriptors(
    value: Any, key_path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        raw_path = value.get("path")
        digest = value.get("sha256")
        path_hint = str(raw_path).lower() if isinstance(raw_path, str) else ""
        key_hint = ".".join(key_path).lower()
        if (
            isinstance(raw_path, str)
            and _valid_sha256(digest)
            and ("manifest" in key_hint or "manifest" in Path(path_hint).name)
        ):
            yield key_path, value
        for key, nested in value.items():
            yield from _iter_manifest_descriptors(nested, key_path + (str(key),))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _iter_manifest_descriptors(nested, key_path + (str(index),))


def _find_solver_descriptor(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = (
        ("provenance", "solver_results"),
        ("sources", "selected_solver"),
        ("sources", "solver_results"),
    )
    for path in candidates:
        current: Any = report
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if (
            isinstance(current, Mapping)
            and isinstance(current.get("path"), str)
            and _valid_sha256(current.get("sha256"))
        ):
            return current
    return None


def _descriptor_sha_bound_under_output(value: Any, solver_sha: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    for key in ("output", "solver", "selected_solver"):
        descriptor = value.get(key)
        if isinstance(descriptor, Mapping) and descriptor.get("sha256") == solver_sha:
            return True
    return False


def _flatten_numeric_signals(
    value: Any, key_path: tuple[str, ...] = ()
) -> dict[str, float | int]:
    signals: dict[str, float | int] = {}
    interesting = {
        "call_count_total",
        "model_call_count",
        "model_calls",
        "request_count",
        "input_tokens_total",
        "output_tokens_total",
        "combined_tokens_total",
        "total_tokens",
        "latency_s_total",
        "total_latency_s",
        "error_count",
        "errors",
        "verifier_errors",
        "solver_error_count",
        "router_fallback_rows",
        "fallback_rows",
        "fallback_count",
        "forced_fallback_count",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = key_path + (str(key),)
            number = _as_nonnegative_number(nested)
            if str(key).lower() in interesting and number is not None:
                signals[".".join(path)] = number
            else:
                signals.update(_flatten_numeric_signals(nested, path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            signals.update(_flatten_numeric_signals(nested, key_path + (str(index),)))
    return signals


def _manifest_summary(
    path: Path, digest: str, payload: Mapping[str, Any], repo_root: Path, source: str
) -> dict[str, Any]:
    signals = _flatten_numeric_signals(payload)
    # This manifest schema has one verifier request per output row.
    if (
        payload.get("schema_version") == "maxim-blind-disagreement-verifier-v1"
        and payload.get("stage") == "blind_model_verifier"
        and isinstance(payload.get("output"), Mapping)
    ):
        rows = _as_nonnegative_number(payload["output"].get("rows"))
        if isinstance(rows, int):
            signals["derived.model_calls_from_output_rows"] = rows
    return {
        "path": _relative(path, repo_root),
        "sha256": digest,
        "schema_version": payload.get("schema_version"),
        "source": source,
        "numeric_signals": dict(sorted(signals.items())),
    }


def _collect_manifests(
    branch: Mapping[str, Any],
    report: Mapping[str, Any],
    report_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    descriptors: list[tuple[str, Mapping[str, Any], Path | None]] = []
    ledger_manifest = branch.get("finalization_manifest")
    if isinstance(ledger_manifest, Mapping):
        descriptors.append(("ledger.finalization_manifest", ledger_manifest, None))
    for key_path, descriptor in _iter_manifest_descriptors(report):
        descriptors.append(("accepted_report." + ".".join(key_path), descriptor, report_path.parent))

    solver_descriptor = _find_solver_descriptor(report)
    solver_path: Path | None = None
    solver_sha: str | None = None
    if solver_descriptor is not None:
        solver_path, solver_sha = _verified_descriptor(
            solver_descriptor,
            repo_root,
            label=f"accepted solver for {branch.get('id')!r}",
            relative_to=report_path.parent,
        )
        for name in _DISCOVERABLE_BOUND_MANIFEST_NAMES:
            candidate = solver_path.parent / name
            if candidate.is_file():
                payload = read_object(candidate)
                if _descriptor_sha_bound_under_output(payload, solver_sha):
                    descriptors.append(("exact_solver_output_binding", {
                        "path": str(candidate),
                        "sha256": sha256_file(candidate),
                    }, None))

    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, descriptor, relative_to in descriptors:
        path, digest = _verified_descriptor(
            descriptor,
            repo_root,
            label=f"manifest for {branch.get('id')!r}",
            relative_to=relative_to,
        )
        relative = _relative(path, repo_root)
        if relative in seen:
            continue
        seen.add(relative)
        payload = read_object(path)
        summaries.append(_manifest_summary(path, digest, payload, repo_root, source))
    return sorted(summaries, key=lambda item: item["path"])


def _normalized_manifest_counts(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "model_calls": [],
        "errors": [],
        "fallbacks": [],
        "input_tokens": [],
        "output_tokens": [],
        "total_tokens": [],
        "latency_s": [],
    }
    for manifest in manifests:
        signals = manifest.get("numeric_signals")
        if not isinstance(signals, Mapping):
            continue
        for path, value in signals.items():
            key = str(path).split(".")[-1].lower()
            record = {"path": manifest.get("path"), "field": path, "value": value}
            if key in {"call_count_total", "model_call_count", "model_calls", "request_count", "model_calls_from_output_rows"}:
                buckets["model_calls"].append(record)
            elif key in {"error_count", "errors", "verifier_errors", "solver_error_count"}:
                buckets["errors"].append(record)
            elif key in {"router_fallback_rows", "fallback_rows", "fallback_count", "forced_fallback_count"}:
                buckets["fallbacks"].append(record)
            elif key == "input_tokens_total":
                buckets["input_tokens"].append(record)
            elif key == "output_tokens_total":
                buckets["output_tokens"].append(record)
            elif key in {"combined_tokens_total", "total_tokens"}:
                buckets["total_tokens"].append(record)
            elif key in {"latency_s_total", "total_latency_s"}:
                buckets["latency_s"].append(record)
    return {key: values for key, values in buckets.items() if values}


def _report_supplemental(report: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    operational = report.get("operational")
    if isinstance(operational, Mapping):
        output["accepted_score_operational"] = operational
    operations = report.get("operations")
    if isinstance(operations, Mapping):
        output["accepted_score_operations"] = operations
    return output


def _manifest_fallback_count(manifest_counts: Mapping[str, Any]) -> int | None:
    records = manifest_counts.get("fallbacks")
    if not isinstance(records, list) or not records:
        return None
    values = {record.get("value") for record in records if isinstance(record, Mapping)}
    integer_values = {value for value in values if isinstance(value, int)}
    return next(iter(integer_values)) if len(integer_values) == 1 else None


def _manifest_error_count(manifest_counts: Mapping[str, Any]) -> int | None:
    records = manifest_counts.get("errors")
    if not isinstance(records, list) or not records:
        return None
    values = [
        record.get("value") for record in records
        if isinstance(record, Mapping) and isinstance(record.get("value"), int)
    ]
    return max(values) if values else None


def _manifest_model_calls(manifest_counts: Mapping[str, Any]) -> int | None:
    records = manifest_counts.get("model_calls")
    if not isinstance(records, list) or not records:
        return None
    values = [
        record.get("value") for record in records
        if isinstance(record, Mapping) and isinstance(record.get("value"), int)
    ]
    return max(values) if values else None


def _manifest_numeric_max(
    manifest_counts: Mapping[str, Any], bucket: str
) -> float | int | None:
    records = manifest_counts.get(bucket)
    if not isinstance(records, list) or not records:
        return None
    values = [
        record.get("value")
        for record in records
        if isinstance(record, Mapping)
        and _as_nonnegative_number(record.get("value")) is not None
    ]
    return max(values) if values else None


def _format_score(metric: Mapping[str, Any]) -> str:
    return f"{metric['correct']}/{metric['n']} ({100.0 * metric['accuracy']:.3f}%)"


def _format_delta(transition: Mapping[str, Any]) -> str:
    sign = "+" if transition["net_correct"] >= 0 else ""
    return (
        f"{sign}{transition['net_correct']} "
        f"(fix {transition['fixed']} / reg {transition['regressed']})"
    )


def _format_cell(value: Any) -> str:
    """Экранирует разделитель столбцов: внутри f-строки бэкслеш запрещён до 3.12."""
    return str(value).replace("|", "\\|")


def _format_optional_int(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}"


def _format_optional_seconds(value: Any) -> str:
    return "—" if value is None else f"{float(value):,.1f}s"


def render_markdown(report: Mapping[str, Any]) -> str:
    benchmark = report["benchmark"]
    judge = report["judge"]
    summary = report["summary"]
    references = report["references"]
    lines = [
        "# Maxim: detailed frozen full274 report",
        "",
        f"Benchmark: `{benchmark['sha256']}` ({benchmark['rows']} tasks; Math 139 / non-Math 135).  ",
        f"Judge lineage: `{judge['lineage']}`.  ",
        f"Strict ledger: `{report['source_ledger']['path']}` (`{report['source_ledger']['sha256']}`).",
        "",
        "Only ledger-accepted `status=final` score artifacts are opened and ranked. "
        "Rejected candidates and non-final/pending scores are not read.",
        "",
        "## Reference anchors",
        "",
        f"- Page baseline (`{references['page_baseline']['id']}`): "
        f"{_format_score(references['page_baseline']['metrics']['overall'])}; "
        f"Math {_format_score(references['page_baseline']['metrics']['math'])}; "
        f"non-Math {_format_score(references['page_baseline']['metrics']['non_math'])}.",
        f"- Frozen Router (`{references['frozen_router']['id']}`): "
        f"{_format_score(references['frozen_router']['metrics']['overall'])}; "
        f"Math {_format_score(references['frozen_router']['metrics']['math'])}; "
        f"non-Math {_format_score(references['frozen_router']['metrics']['non_math'])}.",
        "",
        "## Final accepted results",
        "",
        f"Final branches: **{summary['final']}**. Best: **{summary['best_final']['label']}**, "
        f"{summary['best_final']['correct']}/274 ({100.0 * summary['best_final']['accuracy']:.3f}%).",
        "",
        "| Rank | Branch | Overall | Math139 | nonMath135 | vs page | vs Router | Calls | Tokens | Latency | Errors | Fallbacks |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(report["final_results"], 1):
        usage = item["operational"]["task_usage"]
        normalized = item["operational"]["normalized"]
        calls = usage["model_calls"]["total"]
        if calls is None:
            calls = normalized.get("manifest_model_calls")
        tokens = usage["tokens"]["combined_total"]
        if tokens is None:
            tokens = normalized.get("manifest_total_tokens")
        latency = usage["latency"]["total_s"]
        if latency is None:
            latency = normalized.get("manifest_latency_s")
        errors = max(
            usage["errors"]["solver_error_count"],
            usage["errors"]["verifier_error_count"],
            normalized.get("manifest_error_count") or 0,
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    _format_cell(item["label"]),
                    _format_score(item["metrics"]["overall"]),
                    _format_score(item["metrics"]["math"]),
                    _format_score(item["metrics"]["non_math"]),
                    _format_delta(item["comparisons"]["vs_page_baseline"]["overall"]),
                    _format_delta(item["comparisons"]["vs_frozen_router"]["overall"]),
                    _format_optional_int(calls),
                    _format_optional_int(tokens),
                    _format_optional_seconds(latency),
                    str(errors),
                    _format_optional_int(normalized.get("fallback_count")),
                ]
            )
            + " |"
        )

    lines.extend(["", "### Per-branch details", ""])
    for item in report["final_results"]:
        page = item["comparisons"]["vs_page_baseline"]
        router = item["comparisons"]["vs_frozen_router"]
        lines.extend(
            [
                f"#### {item['label']} (`{item['id']}`)",
                "",
                f"Accepted: `{item['accepted_report']['path']}` (`{item['accepted_report']['sha256']}`).",
                "",
                f"- Overall: {_format_score(item['metrics']['overall'])}; "
                f"vs page {_format_delta(page['overall'])}; "
                f"vs Router {_format_delta(router['overall'])}.",
                f"- Math139: {_format_score(item['metrics']['math'])}; "
                f"vs page {_format_delta(page['math'])}; "
                f"vs Router {_format_delta(router['math'])}.",
                f"- nonMath135: {_format_score(item['metrics']['non_math'])}; "
                f"vs page {_format_delta(page['non_math'])}; "
                f"vs Router {_format_delta(router['non_math'])}.",
                f"- Bound/explicit manifests summarized: {len(item['operational']['manifests'])}.",
                "",
            ]
        )

    lines.extend(["## Non-final (excluded from ranking)", ""])
    if report["non_final"]:
        lines.extend(["| Branch | Reason | Ledger artifact (not opened) |", "|---|---|---|"])
        for item in report["non_final"]:
            artifact = item.get("report", {}).get("path")
            if artifact is None:
                artifact = item.get("supersession_attestation", {}).get("path", "—")
            lines.append(
                f"| {_format_cell(item['label'])} | "
                f"{_format_cell(item.get('reason', 'non-final'))} | `{artifact}` |"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Pending (no accepted final)", ""])
    if report["pending"]:
        lines.extend(["| Branch | Reason |", "|---|---|"])
        for item in report["pending"]:
            lines.append(
                f"| {_format_cell(item['label'])} | "
                f"{_format_cell(item.get('reason', 'pending'))} |"
            )
    else:
        lines.append("None.")

    if report["conflict"]:
        lines.extend(["", "## Conflicts (excluded from ranking)", ""])
        for item in report["conflict"]:
            lines.append(f"- `{item['id']}`: {item.get('reason', 'ledger conflict')}")
    lines.extend(
        [
            "",
            "> Operational totals describe solver artifacts as reported/recomputed from accepted "
            "task outcomes and exact-output-bound manifests; unavailable fields remain `—` and "
            "are never inferred from unrelated files.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    ledger_path: Path,
    repo_root: Path,
    *,
    page_baseline_id: str = DEFAULT_PAGE_BASELINE_ID,
    router_id: str = DEFAULT_ROUTER_ID,
) -> dict[str, Any]:
    ledger_path = _inside_repo(ledger_path, repo_root, "ledger path")
    ledger = read_object(ledger_path)
    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise DetailedReportError(
            f"ledger schema_version must be {LEDGER_SCHEMA_VERSION!r}"
        )
    benchmark = ledger.get("benchmark")
    judge = ledger.get("judge")
    branches = ledger.get("branches")
    if not isinstance(benchmark, Mapping) or benchmark.get("rows") != EXPECTED_ROWS:
        raise DetailedReportError(f"ledger benchmark must be frozen {EXPECTED_ROWS}")
    if not _valid_sha256(benchmark.get("sha256")):
        raise DetailedReportError("ledger benchmark SHA256 is invalid")
    if not isinstance(judge, Mapping) or not isinstance(judge.get("lineage"), str):
        raise DetailedReportError("ledger judge lineage is missing")
    if not isinstance(branches, list):
        raise DetailedReportError("ledger.branches must be an array")

    final_branches: list[Mapping[str, Any]] = []
    separated: dict[str, list[dict[str, Any]]] = {
        "pending": [],
        "non_final": [],
        "conflict": [],
    }
    ids: set[str] = set()
    for index, branch in enumerate(branches):
        if not isinstance(branch, Mapping):
            raise DetailedReportError(f"ledger branch {index} is not an object")
        branch_id = branch.get("id")
        status = branch.get("status")
        if not isinstance(branch_id, str) or not branch_id or branch_id in ids:
            raise DetailedReportError(f"invalid/duplicate ledger branch id at {index}")
        ids.add(branch_id)
        if status == "final":
            final_branches.append(branch)
        elif status in separated:
            # Do not dereference report/rejected candidate paths for these entries.
            copied = {
                key: branch[key]
                for key in (
                    "id",
                    "label",
                    "status",
                    "reason",
                    "result_kind",
                    "terminal_state",
                    "source_call_count",
                    "validity",
                    "report",
                    "supersession_attestation",
                )
                if key in branch
            }
            copied["artifact_access"] = "not_opened"
            copied["rejected_candidates_count"] = (
                len(branch.get("rejected_candidates", []))
                if isinstance(branch.get("rejected_candidates"), list)
                else None
            )
            separated[str(status)].append(copied)
        else:
            raise DetailedReportError(f"unsupported ledger status {status!r} for {branch_id!r}")

    branch_by_id = {str(branch["id"]): branch for branch in final_branches}
    for required in (page_baseline_id, router_id):
        if required not in branch_by_id:
            raise DetailedReportError(f"reference branch {required!r} is not final in ledger")

    loaded: dict[str, tuple[Path, str, Mapping[str, Any], dict[str, Mapping[str, Any]]]] = {}
    for branch in final_branches:
        loaded[str(branch["id"])] = _read_final_outcomes(branch, repo_root, EXPECTED_ROWS)

    page_rows = loaded[page_baseline_id][3]
    router_rows = loaded[router_id][3]
    subject_by_id = {task_id: str(row["subject"]) for task_id, row in page_rows.items()}
    segments = _segments(subject_by_id)
    task_ids = set(subject_by_id)
    page_verdicts = {task_id: bool(row["new_correct"]) for task_id, row in page_rows.items()}
    router_verdicts = {task_id: bool(row["new_correct"]) for task_id, row in router_rows.items()}
    if set(router_rows) != task_ids:
        raise DetailedReportError("Router task-id set differs from page baseline")
    for task_id, row in router_rows.items():
        if row["subject"] != subject_by_id[task_id]:
            raise DetailedReportError(f"Router subject mismatch for {task_id}")

    final_results: list[dict[str, Any]] = []
    for branch in final_branches:
        branch_id = str(branch["id"])
        report_path, digest, score, rows = loaded[branch_id]
        if set(rows) != task_ids:
            raise DetailedReportError(f"task-id set mismatch for {branch_id!r}")
        for task_id, row in rows.items():
            if row["subject"] != subject_by_id[task_id]:
                raise DetailedReportError(f"subject mismatch for {branch_id!r}/{task_id}")
            baseline_correct = row.get("baseline_correct")
            if isinstance(baseline_correct, bool) and baseline_correct != page_verdicts[task_id]:
                raise DetailedReportError(
                    f"embedded baseline verdict mismatch for {branch_id!r}/{task_id}"
                )

        verdicts = {task_id: bool(row["new_correct"]) for task_id, row in rows.items()}
        manifests = _collect_manifests(branch, score, report_path, repo_root)
        manifest_counts = _normalized_manifest_counts(manifests)
        usage = _usage_stats(rows.values())
        operational = {
            "task_usage": usage,
            "normalized": {
                "manifest_model_calls": _manifest_model_calls(manifest_counts),
                "manifest_error_count": _manifest_error_count(manifest_counts),
                "fallback_count": _manifest_fallback_count(manifest_counts),
                "manifest_input_tokens": _manifest_numeric_max(
                    manifest_counts, "input_tokens"
                ),
                "manifest_output_tokens": _manifest_numeric_max(
                    manifest_counts, "output_tokens"
                ),
                "manifest_total_tokens": _manifest_numeric_max(
                    manifest_counts, "total_tokens"
                ),
                "manifest_latency_s": _manifest_numeric_max(
                    manifest_counts, "latency_s"
                ),
            },
            "manifest_counts": manifest_counts,
            "manifests": manifests,
            "supplemental_reported": _report_supplemental(score),
        }
        final_results.append(
            {
                "id": branch_id,
                "label": branch.get("label", branch_id),
                "status": "final",
                "accepted_report": {
                    "path": _relative(report_path, repo_root),
                    "sha256": digest,
                },
                "metrics": _metrics(verdicts, segments),
                "comparisons": {
                    "vs_page_baseline": _comparisons(verdicts, page_verdicts, segments),
                    "vs_frozen_router": _comparisons(verdicts, router_verdicts, segments),
                },
                "operational": operational,
            }
        )

    final_results.sort(
        key=lambda item: (-item["metrics"]["overall"]["correct"], str(item["id"]))
    )
    best = final_results[0] if final_results else None
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_ledger": {
            "path": _relative(ledger_path, repo_root),
            "sha256": sha256_file(ledger_path),
            "schema_version": ledger.get("schema_version"),
        },
        "benchmark": dict(benchmark),
        "judge": dict(judge),
        "policy": {
            "score_access": "only status=final report.path accepted by strict ledger",
            "accepted_report_hashes_reverified": True,
            "rejected_scores_opened": False,
            "non_final_scores_opened": False,
            "pending_scores_opened": False,
            "math_split": "subject == 'Math'",
            "reference_page_baseline_id": page_baseline_id,
            "reference_router_id": router_id,
        },
        "references": {
            "page_baseline": {
                "id": page_baseline_id,
                "metrics": _metrics(page_verdicts, segments),
                "accepted_report": {
                    "path": _relative(loaded[page_baseline_id][0], repo_root),
                    "sha256": loaded[page_baseline_id][1],
                },
            },
            "frozen_router": {
                "id": router_id,
                "metrics": _metrics(router_verdicts, segments),
                "accepted_report": {
                    "path": _relative(loaded[router_id][0], repo_root),
                    "sha256": loaded[router_id][1],
                },
            },
        },
        "summary": {
            "branches": len(branches),
            "final": len(final_results),
            "non_final": len(separated["non_final"]),
            "pending": len(separated["pending"]),
            "conflict": len(separated["conflict"]),
            "best_final": (
                {
                    "id": best["id"],
                    "label": best["label"],
                    **best["metrics"]["overall"],
                }
                if best is not None
                else None
            ),
        },
        "final_results": final_results,
        "non_final": separated["non_final"],
        "pending": separated["pending"],
        "conflict": separated["conflict"],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("reports/maxim_full274_results_ledger_v1_20260803/RESULTS.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/maxim_all_approaches_detailed_v1_20260803"),
    )
    parser.add_argument("--page-baseline-id", default=DEFAULT_PAGE_BASELINE_ID)
    parser.add_argument("--router-id", default=DEFAULT_ROUTER_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    ledger = args.ledger if args.ledger.is_absolute() else repo_root / args.ledger
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_dir = _inside_repo(output_dir, repo_root, "output directory")
    report = build_report(
        ledger,
        repo_root,
        page_baseline_id=args.page_baseline_id,
        router_id=args.router_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "REPORT.json"
    markdown_path = output_dir / "REPORT.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "report_json": _relative(json_path, repo_root),
        "report_md": _relative(markdown_path, repo_root),
        "final": report["summary"]["final"],
        "pending": report["summary"]["pending"],
        "non_final": report["summary"]["non_final"],
        "best_final": report["summary"]["best_final"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
