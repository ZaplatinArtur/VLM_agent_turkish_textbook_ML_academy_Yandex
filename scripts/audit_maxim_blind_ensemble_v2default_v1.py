"""Independently re-run and audit a frozen blind ensemble composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import compose_maxim_blind_ensemble_v2default_v1 as frozen
except ModuleNotFoundError:
    from scripts import compose_maxim_blind_ensemble_v2default_v1 as frozen


FORBIDDEN_EVALUATION_KEYS = {
    "reference",
    "reference_answer",
    "reference_solution",
    "gold",
    "gold_answer",
    "qrels",
    "judge",
    "judge_result",
    "score",
    "correct",
    "task_outcomes",
    "oracle",
}


def assert_no_evaluation_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_EVALUATION_KEYS:
                raise frozen.CompositionError(f"forbidden evaluation key at {path}.{key}")
            assert_no_evaluation_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_evaluation_keys(item, f"{path}[{index}]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, required=True)
    parser.add_argument("--composer", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--v2-solver", type=Path, required=True)
    parser.add_argument("--v2-audit", type=Path, required=True)
    parser.add_argument("--v2-verifier", type=Path, required=True)
    parser.add_argument("--v3-solver", type=Path, required=True)
    parser.add_argument("--v3-audit", type=Path, required=True)
    parser.add_argument("--v3-verifier", type=Path, required=True)
    parser.add_argument("--active-solver", type=Path, required=True)
    args = parser.parse_args(argv)

    profile_path = args.branch_dir / "profile.json"
    prereg_path = args.branch_dir / "preregistered_manifest.json"
    run_dir = args.branch_dir / "run"
    solver_path = run_dir / "solver.jsonl"
    audit_path = run_dir / "selection_audit.jsonl"
    manifest_path = run_dir / "composition_manifest.json"
    profile = frozen.load_json(profile_path)
    prereg = frozen.load_json(prereg_path)
    manifest = frozen.load_json(manifest_path)
    frozen._validate_profile(profile)
    for label, path, binding in (
        ("profile", profile_path, prereg.get("profile")),
        ("composer", args.composer, prereg.get("composer")),
        ("tests", args.tests, prereg.get("tests")),
    ):
        if not isinstance(binding, Mapping):
            raise frozen.CompositionError(f"preregistration {label} binding missing")
        if frozen.sha256_file(path) != binding.get("sha256"):
            raise frozen.CompositionError(f"preregistered {label} SHA mismatch")
    if prereg.get("source_row_values_read_before_freeze") is not False:
        raise frozen.CompositionError("invalid source-row freeze declaration")
    if prereg.get("source_score_or_judge_seen") is not False:
        raise frozen.CompositionError("invalid score/judge freeze declaration")

    source_paths = {
        "meta_v21_solver": args.v2_solver,
        "meta_v21_audit": args.v2_audit,
        "meta_v2_verifier": args.v2_verifier,
        "meta_v31_solver": args.v3_solver,
        "meta_v31_audit": args.v3_audit,
        "meta_v3_verifier": args.v3_verifier,
        "active_crop_v2_solver": args.active_solver,
    }
    frozen._assert_source_hashes(profile, source_paths)
    source_rows = {
        "v2_rows": frozen.load_jsonl(args.v2_solver),
        "v2_audits": frozen.load_jsonl(args.v2_audit),
        "v2_verifiers": frozen.load_jsonl(args.v2_verifier),
        "v3_rows": frozen.load_jsonl(args.v3_solver),
        "v3_audits": frozen.load_jsonl(args.v3_audit),
        "v3_verifiers": frozen.load_jsonl(args.v3_verifier),
        "active_rows": frozen.load_jsonl(args.active_solver),
    }
    expected_solver, expected_audit, expected_counts = frozen.compose(
        profile=profile, **source_rows
    )
    actual_solver = frozen.load_jsonl(solver_path)
    actual_audit = frozen.load_jsonl(audit_path)
    if actual_solver != expected_solver:
        raise frozen.CompositionError("solver differs from independent recomposition")
    if actual_audit != expected_audit:
        raise frozen.CompositionError("selection audit differs from recomposition")
    if len(actual_solver) != frozen.EXPECTED_ROWS or len(actual_audit) != frozen.EXPECTED_ROWS:
        raise frozen.CompositionError("full274 row-count audit failed")
    if manifest.get("decision_counts") != expected_counts:
        raise frozen.CompositionError("manifest decision counts mismatch")
    if manifest.get("artifacts", {}).get("solver", {}).get("sha256") != frozen.sha256_file(
        solver_path
    ):
        raise frozen.CompositionError("manifest solver SHA mismatch")
    if manifest.get("artifacts", {}).get("audit", {}).get("sha256") != frozen.sha256_file(
        audit_path
    ):
        raise frozen.CompositionError("manifest audit SHA mismatch")
    assert_no_evaluation_keys(actual_solver)
    assert_no_evaluation_keys(actual_audit)
    overrides = sum(
        count
        for decision, count in expected_counts.items()
        if decision != "default_exact_meta_v21"
    )
    result = {
        "schema_version": "maxim-blind-ensemble-v2-default-independent-audit-v1",
        "status": "pass",
        "condition": profile.get("condition"),
        "policy_id": profile.get("policy_id"),
        "rows": len(actual_solver),
        "override_rows": overrides,
        "default_rows": len(actual_solver) - overrides,
        "source_hashes_valid": True,
        "preregistered_profile_code_and_tests_hashes_valid": True,
        "exact_recomposition_match": True,
        "task_order_and_uniqueness_valid": True,
        "output_and_audit_row_hashes_valid": True,
        "forbidden_evaluation_keys_absent": True,
        "gold_access": False,
        "score_or_judge_access": False,
        "solver_sha256": frozen.sha256_file(solver_path),
        "selection_audit_sha256": frozen.sha256_file(audit_path),
    }
    result_path = run_dir / "independent_audit.json"
    frozen.write_json(result_path, result)
    (run_dir / "independent_audit.sha256").write_text(
        frozen.sha256_file(result_path) + "  independent_audit.json\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
