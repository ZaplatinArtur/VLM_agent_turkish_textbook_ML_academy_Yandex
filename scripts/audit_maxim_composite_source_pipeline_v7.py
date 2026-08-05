#!/usr/bin/env python3
"""Fail-closed audit for the frozen V7 source-only composite pipeline.

This audit deliberately runs before either V7 image-judge artifact exists.  It
checks hashes and lineage only; it never reads benchmark answers, judge
verdicts, score reports, or candidate outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "maxim-composite-source-pipeline-v7"
AUDIT_SCHEMA = "maxim-composite-source-pipeline-v7-audit-v1"
EXPECTED_CANDIDATES = [
    "val_0042",
    "val_0043",
    "val_0044",
    "val_0046",
    "val_0149",
    "val_0150",
    "val_0178",
    "val_0196",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _jsonl_rows(path: Path) -> int:
    # Row-count attestation deliberately avoids decoding candidate answers or
    # judge verdicts.  Content integrity is already pinned by SHA-256.
    with path.open("rb") as handle:
        return sum(1 for raw in handle if raw.strip())


def _repo_path(repo_root: Path, raw: str, label: str) -> Path:
    _require(isinstance(raw, str) and raw, f"{label}.path must be non-empty")
    relative = Path(raw)
    _require(not relative.is_absolute(), f"{label}.path must be repository-relative")
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label}.path escapes repository root") from exc
    return resolved


def _pinned(
    repo_root: Path,
    ref: dict[str, Any],
    label: str,
    seen_inputs: set[Path],
) -> tuple[Path, str]:
    _require(isinstance(ref, dict), f"{label} must be an object")
    _require(set(ref) == {"path", "sha256"}, f"unexpected fields in {label}")
    path = _repo_path(repo_root, ref["path"], label)
    expected = ref["sha256"]
    _require(
        isinstance(expected, str) and len(expected) == 64,
        f"{label}.sha256 must be a full SHA-256",
    )
    _require(path.is_file(), f"missing pinned artifact: {label}")
    actual = _sha256(path)
    _require(actual == expected, f"SHA-256 mismatch for {label}")
    seen_inputs.add(path)
    return path, actual


def _manifest_ref(
    manifest: dict[str, Any],
    keys: tuple[str, ...],
    expected_sha256: str,
    label: str,
) -> None:
    value: Any = manifest
    for key in keys:
        _require(isinstance(value, dict) and key in value, f"missing {label}.{key}")
        value = value[key]
    _require(isinstance(value, dict), f"{label} reference must be an object")
    _require(value.get("sha256") == expected_sha256, f"lineage mismatch for {label}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def audit(profile_path: Path, output_path: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    profile_path = profile_path.resolve()
    output_path = output_path.resolve()
    _require(profile_path.is_file(), "composite profile is missing")
    _require(profile_path != output_path, "audit output may not overwrite the profile")
    profile = _json(profile_path)
    _require(profile.get("schema_version") == SCHEMA, "unexpected composite schema")
    _require(profile.get("expected_rows") == 274, "expected_rows must stay frozen at 274")
    _require(
        profile.get("expected_image_rows") == 97,
        "expected_image_rows must stay frozen at 97",
    )
    preregistration = profile.get("preregistration")
    _require(isinstance(preregistration, dict), "missing preregistration")
    _require(
        preregistration.get("candidate_task_ids") == EXPECTED_CANDIDATES,
        "candidate set or order changed after preregistration",
    )

    policy = profile.get("policy")
    _require(isinstance(policy, dict), "missing policy")
    required_false = (
        "gold_access",
        "benchmark_candidate_or_outcome_access",
        "outcome_guided_candidate_removal_allowed",
        "same_wave_retuning_after_score_allowed",
    )
    for key in required_false:
        _require(policy.get(key) is False, f"policy.{key} must be false")
    required_true = (
        "task_ids_used_for_alignment_only",
        "all_candidates_frozen_before_v6_outcome_access",
        "image_judge_outputs_must_not_exist_at_source_freeze",
        "keep_v6_if_v7_does_not_improve",
    )
    for key in required_true:
        _require(policy.get(key) is True, f"policy.{key} must be true")
    _require(policy.get("score_attempts_before_launch") == 0, "score already attempted")

    seen_inputs: set[Path] = {profile_path}
    prereg_path, prereg_sha = _pinned(
        repo_root,
        {"path": preregistration["path"], "sha256": preregistration["sha256"]},
        "preregistration",
        seen_inputs,
    )

    v6 = profile.get("v6_anchor")
    main = profile.get("main_stage")
    history = profile.get("history_stage")
    _require(isinstance(v6, dict), "missing v6_anchor")
    _require(isinstance(main, dict), "missing main_stage")
    _require(isinstance(history, dict), "missing history_stage")

    v6_comp_path, v6_comp_sha = _pinned(
        repo_root, v6["composition_manifest"], "v6 composition manifest", seen_inputs
    )
    v6_solver_path, v6_solver_sha = _pinned(
        repo_root, v6["solver"], "v6 solver", seen_inputs
    )
    v6_judge_manifest_path, v6_judge_manifest_sha = _pinned(
        repo_root, v6["image_judge_manifest"], "v6 image judge manifest", seen_inputs
    )
    v6_judge_path, v6_judge_sha = _pinned(
        repo_root, v6["image_judge"], "v6 image judge", seen_inputs
    )

    main_profile_path, main_profile_sha = _pinned(
        repo_root, main["profile"], "main profile", seen_inputs
    )
    main_resolver_path, main_resolver_sha = _pinned(
        repo_root, main["resolver_manifest"], "main resolver manifest", seen_inputs
    )
    main_comp_path, main_comp_sha = _pinned(
        repo_root, main["composition_manifest"], "main composition manifest", seen_inputs
    )
    main_solver_path, main_solver_sha = _pinned(
        repo_root, main["solver"], "main solver", seen_inputs
    )
    _pinned(repo_root, main["image_judge_builder"], "main judge builder", seen_inputs)

    history_profile_path, history_profile_sha = _pinned(
        repo_root, history["profile"], "history profile", seen_inputs
    )
    history_resolver_path, history_resolver_sha = _pinned(
        repo_root, history["resolver_manifest"], "history resolver manifest", seen_inputs
    )
    history_comp_path, history_comp_sha = _pinned(
        repo_root, history["composition_manifest"], "history composition manifest", seen_inputs
    )
    history_solver_path, history_solver_sha = _pinned(
        repo_root, history["solver"], "history solver", seen_inputs
    )
    _pinned(
        repo_root, history["image_judge_builder"], "history judge builder", seen_inputs
    )

    implementation = profile.get("implementation", {})
    _require(isinstance(implementation, dict), "implementation must be an object")
    for label, ref in implementation.items():
        _pinned(repo_root, ref, f"implementation.{label}", seen_inputs)

    _require(_jsonl_rows(v6_solver_path) == 274, "v6 solver row count changed")
    _require(_jsonl_rows(v6_judge_path) == 97, "v6 image judge row count changed")
    _require(_jsonl_rows(main_solver_path) == 274, "main solver row count changed")
    _require(_jsonl_rows(history_solver_path) == 274, "history solver row count changed")

    v6_comp = _json(v6_comp_path)
    v6_judge_manifest = _json(v6_judge_manifest_path)
    main_profile = _json(main_profile_path)
    main_resolver = _json(main_resolver_path)
    main_comp = _json(main_comp_path)
    history_profile = _json(history_profile_path)
    history_resolver = _json(history_resolver_path)
    history_comp = _json(history_comp_path)

    _require(v6_comp.get("rows") == 274, "v6 composition is not 274 rows")
    _require(v6_judge_manifest.get("output", {}).get("rows") == 97, "v6 judge is not 97 rows")
    _require(main_profile.get("anchor", {}).get("sha256") == v6_solver_sha, "main anchor is not V6")
    _require(main_resolver.get("rows") == 274, "main resolver is not 274 rows")
    _require(main_resolver.get("gold_access") is False, "main resolver declares gold access")
    _require(
        main_resolver.get("benchmark_candidate_or_outcome_access") is False,
        "main resolver declares outcome access",
    )
    _manifest_ref(main_comp, ("profile",), main_profile_sha, "main composition profile")
    _manifest_ref(
        main_comp, ("resolver_manifest",), main_resolver_sha, "main composition resolver"
    )
    _manifest_ref(main_comp, ("anchor",), v6_solver_sha, "main composition anchor")
    _manifest_ref(main_comp, ("output", "solver"), main_solver_sha, "main solver output")
    _require(main_comp.get("rows") == 274, "main composition is not 274 rows")
    _require(main_comp.get("overrides") == 1, "unexpected main override count")
    _require(main_comp.get("gold_access") is False, "main composition declares gold access")
    _require(main_comp.get("score_or_outcome_access") is False, "main composition declares outcome access")

    _require(
        history_profile.get("anchor", {}).get("sha256") == main_solver_sha,
        "history anchor is not the frozen main solver",
    )
    _require(history_resolver.get("rows") == 274, "history resolver is not 274 rows")
    _require(history_resolver.get("accepted_certificates") == 1, "unexpected history certificate count")
    _require(history_resolver.get("gold_access") is False, "history resolver declares gold access")
    _require(
        history_resolver.get("benchmark_candidate_or_outcome_access") is False,
        "history resolver declares outcome access",
    )
    _manifest_ref(history_comp, ("profile",), history_profile_sha, "history composition profile")
    _manifest_ref(
        history_comp,
        ("resolver_manifest",),
        history_resolver_sha,
        "history composition resolver",
    )
    _manifest_ref(history_comp, ("anchor",), main_solver_sha, "history composition anchor")
    _manifest_ref(
        history_comp, ("artifacts", "solver"), history_solver_sha, "history solver output"
    )
    _require(history_comp.get("rows") == 274, "history composition is not 274 rows")
    _require(history_comp.get("source_overrides") == 1, "unexpected history override count")
    _require(history_comp.get("gold_access") is False, "history composition declares gold access")
    _require(
        history_comp.get("benchmark_candidate_or_outcome_access") is False,
        "history composition declares outcome access",
    )

    fixed_outputs: list[Path] = []
    for stage_name, stage in (("main", main), ("history", history)):
        judge = _repo_path(repo_root, stage["image_judge_output"], f"{stage_name} judge output")
        manifest = _repo_path(
            repo_root,
            stage["image_judge_manifest_output"],
            f"{stage_name} judge manifest output",
        )
        _require(judge != manifest, f"{stage_name} judge and manifest outputs collide")
        _require(judge not in seen_inputs, f"{stage_name} judge output overwrites an input")
        _require(manifest not in seen_inputs, f"{stage_name} manifest output overwrites an input")
        _require(not judge.exists(), f"{stage_name} judge output already exists")
        _require(not manifest.exists(), f"{stage_name} judge manifest already exists")
        fixed_outputs.extend((judge, manifest))
    _require(len(set(fixed_outputs)) == len(fixed_outputs), "fixed judge outputs collide")
    _require(output_path not in seen_inputs, "audit output overwrites a pinned input")
    _require(output_path not in fixed_outputs, "audit output collides with a judge output")

    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "pass",
        "profile": {
            "path": profile_path.relative_to(repo_root).as_posix(),
            "sha256": _sha256(profile_path),
        },
        "preregistration_sha256": prereg_sha,
        "v6": {
            "composition_manifest_sha256": v6_comp_sha,
            "solver_sha256": v6_solver_sha,
            "image_judge_manifest_sha256": v6_judge_manifest_sha,
            "image_judge_sha256": v6_judge_sha,
        },
        "main": {
            "profile_sha256": main_profile_sha,
            "resolver_manifest_sha256": main_resolver_sha,
            "composition_manifest_sha256": main_comp_sha,
            "solver_sha256": main_solver_sha,
            "overrides": 1,
        },
        "history": {
            "profile_sha256": history_profile_sha,
            "resolver_manifest_sha256": history_resolver_sha,
            "composition_manifest_sha256": history_comp_sha,
            "solver_sha256": history_solver_sha,
            "overrides": 1,
        },
        "candidate_task_ids": EXPECTED_CANDIDATES,
        "fixed_outputs_absent": True,
        "gold_access": False,
        "benchmark_candidate_or_outcome_access": False,
        "score_attempts": 0,
    }
    _atomic_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    result = audit(args.profile, args.output, args.repo_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
