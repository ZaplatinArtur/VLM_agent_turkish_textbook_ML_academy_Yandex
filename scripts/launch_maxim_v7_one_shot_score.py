#!/usr/bin/env python3
"""Audit and optionally consume the frozen V7 one-shot scoring launch.

The attempt marker is created atomically before the scorer starts.  A failed
scorer process therefore still consumes the preregistered attempt and cannot
silently become an outcome-guided retry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FREEZE_SCHEMA = "maxim-v7-one-shot-evaluation-freeze-v1"
FINAL_JUDGE_SCHEMA = "maxim-fill-blank-page-activity-image-judge-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _repo_path(repo_root: Path, raw: str, label: str) -> Path:
    _require(isinstance(raw, str) and raw, f"{label} must be non-empty")
    relative = Path(raw)
    _require(not relative.is_absolute(), f"{label} must be repository-relative")
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository root") from exc
    return resolved


def _pinned(repo_root: Path, ref: dict[str, Any], label: str) -> Path:
    _require(isinstance(ref, dict), f"{label} must be an object")
    _require(set(ref) == {"path", "sha256", *({"rows"} if "rows" in ref else set())}, f"unexpected fields in {label}")
    path = _repo_path(repo_root, ref.get("path"), f"{label}.path")
    expected = ref.get("sha256")
    _require(isinstance(expected, str) and len(expected) == 64, f"bad {label}.sha256")
    _require(path.is_file(), f"missing pinned artifact: {label}")
    _require(_sha256(path) == expected, f"SHA-256 mismatch for {label}")
    if "rows" in ref:
        with path.open("rb") as handle:
            rows = sum(1 for raw in handle if raw.strip())
        _require(rows == ref["rows"], f"row-count mismatch for {label}")
    return path


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _atomic_json_create(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor: int | None = None
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link gives create-if-absent semantics on both Windows and POSIX.
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ValueError("the one-shot score attempt was already consumed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def audit(freeze_path: Path, repo_root: Path, *, check_remote: bool) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    freeze_path = freeze_path.resolve()
    freeze = _load_json(freeze_path)
    _require(freeze.get("schema_version") == FREEZE_SCHEMA, "unexpected freeze schema")
    policy = freeze.get("policy")
    _require(isinstance(policy, dict), "missing policy")
    _require(policy.get("gold_access") is False, "gold access must remain false")
    _require(policy.get("score_attempts") == 0, "score_attempts must be zero")
    _require(policy.get("same_wave_retuning_allowed") is False, "same-wave retuning enabled")
    _require(policy.get("must_not_exist_before_launch") is True, "output absence gate disabled")

    branch = freeze.get("git", {}).get("branch")
    source_commit = freeze.get("git", {}).get("source_freeze_commit")
    _require(branch == "feature/maxim-benchmark-results", "unexpected branch")
    _require(isinstance(source_commit, str) and len(source_commit) == 40, "bad source commit")
    current_branch = _git(repo_root, "branch", "--show-current")
    _require(current_branch == branch, "current branch differs from freeze")
    head = _git(repo_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, head],
        cwd=repo_root,
        check=False,
    )
    _require(ancestor.returncode == 0, "source freeze is not an ancestor of HEAD")
    if check_remote:
        remote_line = _git(repo_root, "ls-remote", "origin", f"refs/heads/{branch}")
        _require(bool(remote_line), "remote branch is missing")
        remote_head = remote_line.split()[0]
        _require(remote_head == head, "local and remote HEAD differ")

    _pinned(repo_root, freeze["launch_guard"], "launch guard")
    scorer = _pinned(repo_root, freeze["scorer"], "scorer")
    benchmark = _pinned(repo_root, freeze["benchmark"], "benchmark")
    baseline = _pinned(repo_root, freeze["baseline_judge"], "baseline judge")
    solver = _pinned(repo_root, freeze["final_solver"], "final solver")
    final_judge = _pinned(repo_root, freeze["final_image_judge"], "final image judge")
    final_manifest_path = _pinned(
        repo_root, freeze["final_image_judge_manifest"], "final image judge manifest"
    )
    final_manifest = _load_json(final_manifest_path)
    _require(final_manifest.get("schema_version") == FINAL_JUDGE_SCHEMA, "bad final judge schema")
    _require(final_manifest.get("output", {}).get("sha256") == freeze["final_image_judge"]["sha256"], "final manifest does not bind judge")
    _require(final_manifest.get("output", {}).get("rows") == 97, "final manifest does not bind 97 rows")

    for label in (
        "main_image_judge",
        "main_image_judge_manifest",
        "history_profile",
        "history_resolver_manifest",
        "history_composition_manifest",
        "composite_source_audit",
    ):
        _pinned(repo_root, freeze[label], label.replace("_", " "))

    launch = freeze.get("launch")
    _require(isinstance(launch, dict), "missing launch")
    expected_args = [
        "--benchmark", freeze["benchmark"]["path"],
        "--solver-results", freeze["final_solver"]["path"],
        "--image-judge", freeze["final_image_judge"]["path"],
        "--baseline-judge", freeze["baseline_judge"]["path"],
        "--out-json", launch["outputs"]["json"],
        "--out-md", launch["outputs"]["markdown"],
        "--out-sha256", launch["outputs"]["sha256"],
        "--label", launch["label"],
        "--expected-rows", "274",
        "--expected-deterministic", "177",
        "--expected-image-judge", "97",
        "--expected-benchmark-sha256", freeze["benchmark"]["sha256"],
        "--expected-baseline-judge-sha256", freeze["baseline_judge"]["sha256"],
    ]
    _require(launch.get("arguments") == expected_args, "launch arguments differ from freeze")
    outputs = [
        _repo_path(repo_root, launch["outputs"][key], f"score output {key}")
        for key in ("json", "markdown", "sha256", "attempt_marker")
    ]
    _require(len(set(outputs)) == 4, "score outputs collide")
    for output in outputs:
        _require(not output.exists(), f"score output already exists: {output.name}")
    pinned_inputs = {scorer, benchmark, baseline, solver, final_judge, final_manifest_path, freeze_path}
    _require(not any(output in pinned_inputs for output in outputs), "score output overwrites input")

    return {
        "schema_version": "maxim-v7-one-shot-launch-audit-v1",
        "status": "ready",
        "freeze_sha256": _sha256(freeze_path),
        "head": head,
        "remote_checked": check_remote,
        "solver_sha256": freeze["final_solver"]["sha256"],
        "image_judge_sha256": freeze["final_image_judge"]["sha256"],
        "score_outputs_absent": True,
        "score_attempts": 0,
    }


def execute(freeze_path: Path, repo_root: Path) -> int:
    ready = audit(freeze_path, repo_root, check_remote=True)
    freeze = _load_json(freeze_path)
    marker = _repo_path(
        repo_root,
        freeze["launch"]["outputs"]["attempt_marker"],
        "attempt marker",
    )
    _atomic_json_create(
        marker,
        {
            "schema_version": "maxim-v7-score-attempt-v1",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": ready["freeze_sha256"],
            "head": ready["head"],
            "attempt": 1,
        },
    )
    scorer = _repo_path(repo_root, freeze["scorer"]["path"], "scorer")
    command = [sys.executable, str(scorer), *freeze["launch"]["arguments"]]
    completed = subprocess.run(command, cwd=repo_root, check=False)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check-remote", action="store_true")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(execute(args.freeze, args.repo_root))
    result = audit(args.freeze, args.repo_root, check_remote=args.check_remote)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
