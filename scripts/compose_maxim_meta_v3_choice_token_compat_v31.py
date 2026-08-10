#!/usr/bin/env python3
"""Compose the preregistered gold-blind meta-v3 choice-token compatibility v3.1.

The source meta-v3 artifacts remain immutable.  A choice row that fell back to
the frozen Router may be recovered only when both frozen semantic attempts
failed with the identical strict A-E validator error, the saved last response
is exact nonpartial JSON whose final answer is one ASCII digit, and every other
frozen v3 validator and policy gate passes.  Transport, length, repetition,
parse, mixed, confidence, evidence, and abstention failures remain exact Router
rows.  No task ID, subject, label, reference, score, or judge result is used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import compose_maxim_meta_v2_choice_token_compat_v21 as audited_base
    import run_maxim_final_meta_verifier_v3 as v3
except ModuleNotFoundError:
    from scripts import compose_maxim_meta_v2_choice_token_compat_v21 as audited_base
    from scripts import run_maxim_final_meta_verifier_v3 as v3


REPO = Path(__file__).resolve().parents[1]
DEFAULT_COMPAT_PROFILE = (
    REPO
    / "reports"
    / "maxim_final_meta_verifier_v3_choice_token_compat_v31_20260803"
    / "profile.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO
    / "reports"
    / "maxim_final_meta_verifier_v3_choice_token_compat_v31_20260803"
    / "run"
)

SCHEMA_VERSION = "maxim-final-meta-choice-token-compat-composer-v31"
AUDIT_SCHEMA_VERSION = "maxim-final-meta-choice-token-compat-audit-v31"
MANIFEST_SCHEMA_VERSION = "maxim-final-meta-choice-token-compat-manifest-v31"
CONDITION = "maxim_final_meta_verifier_v3_choice_token_compat_v31"
PROMPT_VERSION = "original-first-anonymous-12way-verification-v3+choice-token-compat-v31"
EXPECTED_ROWS = 274
EXPECTED_PROFILE_SHA256 = "cc780f3e9c47b729ea706135b46cb768f379aab5e14f04cb576294e1aa3aa43b"
EXPECTED_QUEUE_SHA256 = "641334d304482aa1e6235ced739647d4e99759f6394766d884696f22038f202e"
EXPECTED_PREPARATION_MANIFEST_SHA256 = (
    "2cdfa3e3f476df8f4aad0dc78a5a73f87abe6d416dcff8746abbc8e16c6d4b84"
)
EXPECTED_V3_PROFILE_SHA256 = (
    "12fcd5d00fc7b638a2902bd909409c8fe8ec67e4db93feba7dbce5365a6ad687"
)
EXPECTED_ROUTER_SHA256 = (
    "34da8ef69619a8ba1f184cdfd1e6dcaf0fbdbdd1bfc50c711244a68f7d26a574"
)
AUDITED_BASE_SHA256 = (
    "b24a8c95e64536df66fe04067f4a9a076e284cd31fb854d5cbc3cee0e6b11224"
)
V3_RUNNER_SHA256 = (
    "3bb59907b348fc133e21e9aa862d90aee26385e0e3de3b64c9086fe9184932dc"
)
V3_PREPARATION_SHA256 = (
    "08eb031c661b4238ddbf585d1e6c31e7ec47d9aad8470956d3223ca43dabe86d"
)
STRICT_REJECTION = audited_base.STRICT_REJECTION
SEMANTIC_ATTEMPTS = 2
MIN_CONFIDENCE = 0.70
MIN_EVIDENCE = 2
CompatibilityError = audited_base.CompatibilityError


def _configure_audited_base() -> None:
    """Bind the already-tested generic implementation to frozen meta-v3."""

    audited_base.v2 = v3
    audited_base.CONDITION = CONDITION
    audited_base.PROMPT_VERSION = PROMPT_VERSION
    audited_base.SCHEMA_VERSION = SCHEMA_VERSION
    audited_base.AUDIT_SCHEMA_VERSION = AUDIT_SCHEMA_VERSION
    audited_base.MANIFEST_SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION
    audited_base.EXPECTED_ROWS = EXPECTED_ROWS
    audited_base.EXPECTED_PROFILE_SHA256 = EXPECTED_PROFILE_SHA256
    audited_base.EXPECTED_QUEUE_SHA256 = EXPECTED_QUEUE_SHA256
    audited_base.EXPECTED_PREPARATION_MANIFEST_SHA256 = (
        EXPECTED_PREPARATION_MANIFEST_SHA256
    )
    audited_base.EXPECTED_V2_PROFILE_SHA256 = EXPECTED_V3_PROFILE_SHA256
    audited_base.EXPECTED_ROUTER_SHA256 = EXPECTED_ROUTER_SHA256
    audited_base.SEMANTIC_ATTEMPTS = SEMANTIC_ATTEMPTS
    audited_base.MIN_CONFIDENCE = MIN_CONFIDENCE
    audited_base.MIN_EVIDENCE = MIN_EVIDENCE


_configure_audited_base()


def validate_compat_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != "maxim-final-meta-choice-token-compat-profile-v31":
        raise CompatibilityError("compat profile schema mismatch")
    if profile.get("condition") != CONDITION:
        raise CompatibilityError("compat profile condition mismatch")
    if profile.get("gold_access") is not False:
        raise CompatibilityError("compat profile is not gold-blind")
    if profile.get("score_or_judge_inputs_allowed") is not False:
        raise CompatibilityError("compat profile permits score/judge inputs")
    if profile.get("selection_uses_task_id") is not False:
        raise CompatibilityError("compat profile permits task-specific selection")
    if profile.get("selection_uses_subject") is not False:
        raise CompatibilityError("compat profile permits subject-specific selection")
    binding = profile.get("frozen_v3_bindings")
    if not isinstance(binding, Mapping):
        raise CompatibilityError("compat profile v3 bindings missing")
    if binding.get("strict_rejection") != STRICT_REJECTION:
        raise CompatibilityError("strict rejection binding mismatch")
    if binding.get("semantic_attempts") != SEMANTIC_ATTEMPTS:
        raise CompatibilityError("semantic-attempt binding mismatch")
    if float(binding.get("min_confidence") or -1) != MIN_CONFIDENCE:
        raise CompatibilityError("confidence binding mismatch")
    if int(binding.get("min_decisive_evidence") or -1) != MIN_EVIDENCE:
        raise CompatibilityError("evidence binding mismatch")
    if list(binding.get("candidate_ids") or []) != list(v3.preparation.OPAQUE_IDS):
        raise CompatibilityError("candidate ID binding mismatch")


def assess_choice_token_compatibility(
    *, verifier_row: Mapping[str, Any], queue_row: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    return audited_base.assess_numeric_compatibility(
        verifier_row=verifier_row,
        queue_row=queue_row,
    )


def compose_rows(
    *,
    queue: Sequence[Mapping[str, Any]],
    verifier: Sequence[Mapping[str, Any]],
    v3_solver: Sequence[Mapping[str, Any]],
    router: Sequence[Mapping[str, Any]],
    queue_sha256: str,
    compat_profile_sha256: str,
    expected_rows: int = EXPECTED_ROWS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    solver, audit, base_counts = audited_base.compose_rows(
        queue=queue,
        verifier=verifier,
        v2_solver=v3_solver,
        router=router,
        queue_sha256=queue_sha256,
        compat_profile_sha256=compat_profile_sha256,
        expected_rows=expected_rows,
    )
    for row in solver:
        generation = row.get("generation")
        if not isinstance(generation, dict):
            continue
        if generation.get("source_v2_solver_row_sha256"):
            generation["source_v3_solver_row_sha256"] = generation.pop(
                "source_v2_solver_row_sha256"
            )
            generation["structured_mode"] = (
                "strict_response_format_choice_token_compat_v31"
            )
            generation["selection_reason"] = (
                "valid_supported_meta_answer_choice_token_compat_v31"
            )
    for row in audit:
        if row.get("source_v2_solver_row_sha256"):
            row["source_v3_solver_row_sha256"] = row.pop(
                "source_v2_solver_row_sha256"
            )
        if row.get("decision") == "unchanged_v2_content_exact":
            row["decision"] = "unchanged_v3_content_exact"
    decisions = {
        (
            "unchanged_v3_content_exact"
            if key == "unchanged_v2_content_exact"
            else key
        ): value
        for key, value in base_counts.items()
    }
    return solver, audit, decisions


def _validate_source_manifest(
    *,
    run_manifest: Mapping[str, Any],
    queue_path: Path,
    verifier_path: Path,
    solver_path: Path,
    router_path: Path,
    v3_profile_path: Path,
    preparation_manifest_path: Path,
) -> None:
    if run_manifest.get("complete") is not True:
        raise CompatibilityError("v3 run manifest is not complete")
    if run_manifest.get("generation_gold_access") is not False:
        raise CompatibilityError("v3 run manifest is not gold-blind")
    bindings = [
        ("queue", queue_path),
        ("verdict_output", verifier_path),
        ("solver_output", solver_path),
        ("router_fallback_solver", router_path),
        ("profile", v3_profile_path),
        ("preparation_manifest", preparation_manifest_path),
    ]
    for key, path in bindings:
        value = run_manifest.get(key)
        if not isinstance(value, Mapping):
            raise CompatibilityError(f"v3 run manifest {key} binding missing")
        if value.get("sha256") != audited_base._sha256_file(path):
            raise CompatibilityError(f"v3 run manifest {key} SHA binding mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--v3-solver", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--v3-run-manifest", type=Path, required=True)
    parser.add_argument("--v3-profile", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--compat-profile", type=Path, default=DEFAULT_COMPAT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    audited_base._assert_file_sha(args.queue, EXPECTED_QUEUE_SHA256, "queue")
    audited_base._assert_file_sha(
        args.preparation_manifest,
        EXPECTED_PREPARATION_MANIFEST_SHA256,
        "v3 preparation manifest",
    )
    audited_base._assert_file_sha(
        args.v3_profile, EXPECTED_V3_PROFILE_SHA256, "v3 profile"
    )
    audited_base._assert_file_sha(args.router, EXPECTED_ROUTER_SHA256, "Router")
    audited_base._assert_file_sha(
        args.compat_profile, EXPECTED_PROFILE_SHA256, "compat profile"
    )
    audited_base._assert_file_sha(
        Path(audited_base.__file__).resolve(), AUDITED_BASE_SHA256, "audited base"
    )
    audited_base._assert_file_sha(
        Path(v3.__file__).resolve(), V3_RUNNER_SHA256, "v3 runner"
    )
    audited_base._assert_file_sha(
        Path(v3.preparation.__file__).resolve(),
        V3_PREPARATION_SHA256,
        "v3 preparation",
    )

    compat_profile = audited_base._read_json(args.compat_profile, "compat profile")
    validate_compat_profile(compat_profile)
    v3_profile = audited_base._read_json(args.v3_profile, "v3 profile")
    v3.preparation.validate_profile(v3_profile)
    queue = audited_base._read_jsonl(args.queue, "queue")
    v3.validate_queue(queue)
    source_manifest = audited_base._read_json(args.v3_run_manifest, "v3 run manifest")
    _validate_source_manifest(
        run_manifest=source_manifest,
        queue_path=args.queue,
        verifier_path=args.verifier,
        solver_path=args.v3_solver,
        router_path=args.router,
        v3_profile_path=args.v3_profile,
        preparation_manifest_path=args.preparation_manifest,
    )
    verifier = audited_base._read_jsonl(args.verifier, "verifier")
    v3_solver = audited_base._read_jsonl(args.v3_solver, "v3 solver")
    router = audited_base._read_jsonl(args.router, "Router")
    solver, audit, decisions = compose_rows(
        queue=queue,
        verifier=verifier,
        v3_solver=v3_solver,
        router=router,
        queue_sha256=EXPECTED_QUEUE_SHA256,
        compat_profile_sha256=EXPECTED_PROFILE_SHA256,
    )

    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        solver_path = temporary / "solver.jsonl"
        audit_path = temporary / "compatibility_audit.jsonl"
        manifest_path = temporary / "composition_manifest.json"
        audited_base._write_jsonl(solver_path, solver)
        audited_base._write_jsonl(audit_path, audit)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "condition": CONDITION,
            "gold_access": False,
            "score_or_judge_inputs_loaded": False,
            "task_id_or_subject_used_for_selection": False,
            "profile": audited_base._source(args.compat_profile),
            "sources": {
                "queue": audited_base._source(args.queue, len(queue)),
                "verifier": audited_base._source(args.verifier, len(verifier)),
                "v3_solver": audited_base._source(args.v3_solver, len(v3_solver)),
                "router": audited_base._source(args.router, len(router)),
                "v3_run_manifest": audited_base._source(args.v3_run_manifest),
                "v3_profile": audited_base._source(args.v3_profile),
                "v3_preparation_manifest": audited_base._source(
                    args.preparation_manifest
                ),
            },
            "code": {
                "composer": audited_base._source(Path(__file__).resolve()),
                "audited_base": audited_base._source(
                    Path(audited_base.__file__).resolve()
                ),
                "v3_runner": audited_base._source(Path(v3.__file__).resolve()),
                "v3_preparation": audited_base._source(
                    Path(v3.preparation.__file__).resolve()
                ),
            },
            "outputs": {
                "solver": audited_base._source(solver_path, len(solver)),
                "compatibility_audit": audited_base._source(audit_path, len(audit)),
            },
            "decision_counts": decisions,
            "recursive_gold_free_audit": "PASS",
        }
        audited_base._write_json(manifest_path, manifest)
        files = [solver_path, audit_path, manifest_path]
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(
                f"{audited_base._sha256_file(path)}  {path.name}\n" for path in files
            ),
            encoding="ascii",
            newline="\n",
        )
        os.replace(temporary, output)
    except Exception:
        if temporary.exists() and temporary.parent.resolve() == output.parent.resolve():
            shutil.rmtree(temporary)
        raise
    print(
        json.dumps(
            {
                "rows": len(solver),
                "decision_counts": decisions,
                "solver_sha256": audited_base._sha256_file(output / "solver.jsonl"),
                "audit_sha256": audited_base._sha256_file(
                    output / "compatibility_audit.jsonl"
                ),
                "manifest_sha256": audited_base._sha256_file(
                    output / "composition_manifest.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
