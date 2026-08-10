#!/usr/bin/env python3
"""Audit the single preregistered error-only retry of final meta-v3.

The audit is deliberately gold-blind.  It proves that rows which were not
errors in the initial 274-row run stayed content-exact, that every final solver
row is the deterministic output of the frozen v3 policy, and that both source
manifests bind the supplied queue/profile/router/verifier/solver artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import run_maxim_final_meta_verifier_v3 as v3
except ModuleNotFoundError:  # pragma: no cover - package-style invocation
    from scripts import run_maxim_final_meta_verifier_v3 as v3


SCHEMA_VERSION = "maxim-final-meta-verifier-v3-error-retry-audit-v1"
EXPECTED_ROWS = 274


class AuditError(ValueError):
    """Raised when retry provenance or frozen-policy reconstruction fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be a JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AuditError(f"{label} line {line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid {label}: {path}: {exc}") from exc
    return rows


def index_rows(
    rows: Sequence[Mapping[str, Any]], label: str, expected_order: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    task_ids = [str(row.get("task_id") or "") for row in rows]
    if len(rows) != EXPECTED_ROWS:
        raise AuditError(f"{label}: expected {EXPECTED_ROWS} rows, got {len(rows)}")
    if any(not task_id for task_id in task_ids):
        raise AuditError(f"{label}: empty task_id")
    if len(set(task_ids)) != len(task_ids):
        raise AuditError(f"{label}: duplicate task_id")
    if list(task_ids) != list(expected_order):
        raise AuditError(f"{label}: task order differs from frozen queue")
    return {task_id: row for task_id, row in zip(task_ids, rows)}


def validate_run_manifest(
    manifest: Mapping[str, Any],
    *,
    queue: Path,
    profile: Path,
    preparation_manifest: Path,
    router: Path,
    verifier: Path,
    solver: Path,
    label: str,
) -> None:
    if manifest.get("schema_version") != v3.SCHEMA_VERSION:
        raise AuditError(f"{label}: schema mismatch")
    if manifest.get("complete") is not True:
        raise AuditError(f"{label}: run is not complete")
    if manifest.get("generation_gold_access") is not False:
        raise AuditError(f"{label}: generation is not gold-blind")
    if manifest.get("candidate_scores_loaded") is not False:
        raise AuditError(f"{label}: candidate scores were loaded")
    if manifest.get("judge_artifacts_loaded") is not False:
        raise AuditError(f"{label}: judge artifacts were loaded")
    bindings = {
        "queue": queue,
        "profile": profile,
        "preparation_manifest": preparation_manifest,
        "router_fallback_solver": router,
        "verdict_output": verifier,
        "solver_output": solver,
    }
    for key, path in bindings.items():
        binding = manifest.get(key)
        if not isinstance(binding, Mapping):
            raise AuditError(f"{label}: missing {key} binding")
        if binding.get("sha256") != sha256_file(path):
            raise AuditError(f"{label}: {key} SHA mismatch")
    if int(manifest["queue"].get("rows") or -1) != EXPECTED_ROWS:
        raise AuditError(f"{label}: queue row count mismatch")
    if int(manifest["verdict_output"].get("rows") or -1) != EXPECTED_ROWS:
        raise AuditError(f"{label}: verifier row count mismatch")
    if int(manifest["solver_output"].get("rows") or -1) != EXPECTED_ROWS:
        raise AuditError(f"{label}: solver row count mismatch")


def validate_retry_log(path: Path, expected_task_ids: set[str]) -> list[str]:
    """Prove that the one runner invocation processed every initial error row."""

    pattern = re.compile(r"^\[(\d+)/(\d+)\] (\S+) error=")
    completed: list[tuple[int, int, str]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.match(line)
            if match:
                completed.append(
                    (int(match.group(1)), int(match.group(2)), match.group(3))
                )
    except OSError as exc:
        raise AuditError(f"cannot read retry log: {path}: {exc}") from exc
    expected_count = len(expected_task_ids)
    if len(completed) != expected_count:
        raise AuditError(
            f"retry log: expected {expected_count} completion rows, got {len(completed)}"
        )
    if [item[0] for item in completed] != list(range(1, expected_count + 1)):
        raise AuditError("retry log: completion counters are not one exact batch")
    if any(item[1] != expected_count for item in completed):
        raise AuditError("retry log: pending total differs from initial error count")
    completed_ids = [item[2] for item in completed]
    if len(set(completed_ids)) != expected_count:
        raise AuditError("retry log: duplicate completion task")
    if set(completed_ids) != expected_task_ids:
        raise AuditError("retry log: completed task set differs from initial errors")
    return completed_ids


def source(path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def audit(args: argparse.Namespace) -> dict[str, Any]:
    profile = read_json(args.profile, "v3 profile")
    v3.preparation.validate_profile(profile)
    queue = read_jsonl(args.queue, "public queue")
    v3.validate_queue(queue)
    task_order = [str(row["task_id"]) for row in queue]
    if len(task_order) != EXPECTED_ROWS:
        raise AuditError(f"queue: expected {EXPECTED_ROWS}, got {len(task_order)}")

    router_rows = read_jsonl(args.router, "Router solver")
    initial_verifier = read_jsonl(args.initial_verifier, "initial verifier")
    initial_solver = read_jsonl(args.initial_solver, "initial solver")
    final_verifier = read_jsonl(args.final_verifier, "final verifier")
    final_solver = read_jsonl(args.final_solver, "final solver")
    router = index_rows(router_rows, "Router solver", task_order)
    initial_v = index_rows(initial_verifier, "initial verifier", task_order)
    initial_s = index_rows(initial_solver, "initial solver", task_order)
    final_v = index_rows(final_verifier, "final verifier", task_order)
    final_s = index_rows(final_solver, "final solver", task_order)

    initial_manifest = read_json(args.initial_run_manifest, "initial run manifest")
    final_manifest = read_json(args.final_run_manifest, "final run manifest")
    validate_run_manifest(
        initial_manifest,
        queue=args.queue,
        profile=args.profile,
        preparation_manifest=args.preparation_manifest,
        router=args.router,
        verifier=args.initial_verifier,
        solver=args.initial_solver,
        label="initial run manifest",
    )
    validate_run_manifest(
        final_manifest,
        queue=args.queue,
        profile=args.profile,
        preparation_manifest=args.preparation_manifest,
        router=args.router,
        verifier=args.final_verifier,
        solver=args.final_solver,
        label="final run manifest",
    )
    immutable_manifest_keys = (
        "schema_version",
        "stage",
        "generation_gold_access",
        "private_routing_key_loaded",
        "candidate_scores_loaded",
        "judge_artifacts_loaded",
        "profile",
        "preparation_manifest",
        "queue",
        "router_fallback_solver",
        "backend",
        "selection_policy",
        "prompt_version",
        "prompt_sha256",
        "schema_sha256",
    )
    for key in immutable_manifest_keys:
        if initial_manifest.get(key) != final_manifest.get(key):
            raise AuditError(f"run manifest frozen field changed across retry: {key}")

    policy = profile.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise AuditError("profile selection_policy missing")
    min_confidence = float(policy["min_confidence"])
    min_evidence = int(policy["min_decisive_evidence"])
    queue_sha = sha256_file(args.queue)

    initial_errors = {
        task_id for task_id, row in initial_v.items() if bool(row.get("error"))
    }
    if not initial_errors:
        raise AuditError("initial run has no errors, so an error-only retry is invalid")
    retry_completion_order = validate_retry_log(args.retry_log, initial_errors)

    changed_verifier: set[str] = set()
    changed_solver: set[str] = set()
    final_errors: set[str] = set()
    selection_counts: Counter[str] = Counter()
    for queue_row in queue:
        task_id = str(queue_row["task_id"])
        if initial_v[task_id] != final_v[task_id]:
            changed_verifier.add(task_id)
        if initial_s[task_id] != final_s[task_id]:
            changed_solver.add(task_id)
        if task_id not in initial_errors:
            if initial_v[task_id] != final_v[task_id]:
                raise AuditError(f"{task_id}: non-error verifier row changed")
            if initial_s[task_id] != final_s[task_id]:
                raise AuditError(f"{task_id}: non-error solver row changed")
        if final_v[task_id].get("queue_request_sha256") != queue_row.get(
            "request_sha256"
        ):
            raise AuditError(f"{task_id}: final queue binding mismatch")
        expected_solver, expected_audit = v3.compose_solver_row(
            result=final_v[task_id],
            router_row=router[task_id],
            min_confidence=min_confidence,
            min_evidence=min_evidence,
            queue_sha256=queue_sha,
        )
        if dict(final_s[task_id]) != expected_solver:
            raise AuditError(f"{task_id}: final solver is not frozen-policy output")
        if dict(final_v[task_id]) != expected_audit:
            raise AuditError(f"{task_id}: final verifier audit selection mismatch")
        selected = str((final_v[task_id].get("selection") or {}).get("selected_source"))
        selection_counts[selected] += 1
        if final_v[task_id].get("error"):
            final_errors.add(task_id)
            if dict(final_s[task_id]) != dict(router[task_id]):
                raise AuditError(f"{task_id}: final error did not copy exact Router")

    if not changed_verifier.issubset(initial_errors):
        raise AuditError("retry changed verifier rows outside the initial error set")
    if not changed_solver.issubset(initial_errors):
        raise AuditError("retry changed solver rows outside the initial error set")
    for manifest, verifier_rows, label in (
        (initial_manifest, initial_verifier, "initial"),
        (final_manifest, final_verifier, "final"),
    ):
        actual_errors = sum(bool(row.get("error")) for row in verifier_rows)
        actual_fallbacks = sum(
            (row.get("selection") or {}).get("selected_source") == "router"
            for row in verifier_rows
        )
        verdict_binding = manifest.get("verdict_output") or {}
        if int(verdict_binding.get("errors") or 0) != actual_errors:
            raise AuditError(f"{label} manifest error count mismatch")
        if int(verdict_binding.get("router_fallback_rows") or 0) != actual_fallbacks:
            raise AuditError(f"{label} manifest fallback count mismatch")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "gold_access": False,
        "score_or_judge_inputs_loaded": False,
        "task_specific_retry_selection": False,
        "retry_selection_rule": "all and only rows with nonempty initial error",
        "retry_batches": 1,
        "retry_completion_log": "PASS",
        "retry_completion_order": retry_completion_order,
        "rows": EXPECTED_ROWS,
        "initial_error_rows": len(initial_errors),
        "final_error_rows": len(final_errors),
        "changed_verifier_rows": len(changed_verifier),
        "changed_solver_rows": len(changed_solver),
        "nonerror_rows_content_exact": EXPECTED_ROWS - len(initial_errors),
        "final_selection_counts": dict(sorted(selection_counts.items())),
        "initial_error_task_ids": sorted(initial_errors),
        "final_error_task_ids": sorted(final_errors),
        "changed_verifier_task_ids": sorted(changed_verifier),
        "changed_solver_task_ids": sorted(changed_solver),
        "frozen_policy_reconstruction": "PASS",
        "manifest_bindings": "PASS",
        "sources": {
            "queue": source(args.queue, len(queue)),
            "profile": source(args.profile),
            "preparation_manifest": source(args.preparation_manifest),
            "router": source(args.router, len(router_rows)),
            "initial_verifier": source(args.initial_verifier, len(initial_verifier)),
            "initial_solver": source(args.initial_solver, len(initial_solver)),
            "initial_run_manifest": source(args.initial_run_manifest),
            "final_verifier": source(args.final_verifier, len(final_verifier)),
            "final_solver": source(args.final_solver, len(final_solver)),
            "final_run_manifest": source(args.final_run_manifest),
            "retry_log": source(args.retry_log),
            "runner": source(Path(v3.__file__).resolve()),
            "auditor": source(Path(__file__).resolve()),
        },
    }


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(data)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--initial-verifier", type=Path, required=True)
    parser.add_argument("--initial-solver", type=Path, required=True)
    parser.add_argument("--initial-run-manifest", type=Path, required=True)
    parser.add_argument("--final-verifier", type=Path, required=True)
    parser.add_argument("--final-solver", type=Path, required=True)
    parser.add_argument("--final-run-manifest", type=Path, required=True)
    parser.add_argument("--retry-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit(args)
        write_json_atomic(args.output, report)
    except (OSError, AuditError, KeyError, TypeError, ValueError) as exc:
        print(f"META V3 RETRY AUDIT ERROR: {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
