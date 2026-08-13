"""Post-inference exact Hybrid V3.1 composition bound to this candidate completion."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any

from nonstream_protocol import exclusive_bytes, exclusive_json, read_json, sha256_file
from run_candidate import ALIGNMENT_PATH, AUDIT_PATH, COMPLETION, FREEZE_PATH, HERE, PREDICTIONS, read_jsonl, stable_bytes

HYBRID_DIR = HERE.parent / "maxim_9b_strict_noid_db_generic_hybrid_v3_1_20260812"
HYBRID_FREEZE = HYBRID_DIR / "HYBRID_RULE_FREEZE.json"
HYBRID_AUDIT = HYBRID_DIR / "INDEPENDENT_AUDIT.json"
HYBRID_IMPLEMENTATION = HYBRID_DIR / "strict_hybrid.py"
DECISIONS = HYBRID_DIR / "runs" / "maxim274" / "route_decisions.jsonl"
COMPOSED = HERE / "runs" / "hybrid_solver_274.jsonl"
COMPOSE_COMPLETION = HERE / "HYBRID_COMPOSE_COMPLETION.json"

HYBRID_FREEZE_SHA = "c904f1ea7151513cb83757cc80e21e8dd1cdbd8c7eb4fbf47a40ee40e35ac177"
HYBRID_AUDIT_SHA = "412809bf7dd33b25e582f425ac04eded14ed9ac919f6f0ec59ae781be3301128"
HYBRID_IMPLEMENTATION_SHA = "074d8baba2d081c71174501f418df97eff20344d7e9299434b67bf9f0aa4d4ce"
DECISIONS_SHA = "0f6a31ec8d862ba67f61a8d90e96a0bd1e2a77d071a51ff1a36c512eeaea974c"


class ComposeError(RuntimeError):
    pass


def _load_pinned_hybrid() -> Any:
    spec = importlib.util.spec_from_file_location("pinned_hybrid_v3_1", HYBRID_IMPLEMENTATION)
    if spec is None or spec.loader is None:
        raise ComposeError("cannot load pinned Hybrid V3.1 implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_exact_composition(
    execution_sha: str,
    audit_sha: str,
    completion_sha: str,
    expected_composed_sha: str,
) -> str:
    """Independently recompute the 274-row composition and compare exact bytes."""
    if sha256_file(FREEZE_PATH) != execution_sha or sha256_file(AUDIT_PATH) != audit_sha or sha256_file(COMPLETION) != completion_sha:
        raise ComposeError("candidate freeze/audit/completion external pin mismatch")
    if any(sha256_file(path) != digest for path, digest in ((HYBRID_FREEZE, HYBRID_FREEZE_SHA), (HYBRID_AUDIT, HYBRID_AUDIT_SHA), (HYBRID_IMPLEMENTATION, HYBRID_IMPLEMENTATION_SHA), (DECISIONS, DECISIONS_SHA))):
        raise ComposeError("Hybrid V3.1 ancestry pin mismatch")
    completion = read_json(COMPLETION)
    if completion.get("freeze_sha256") != execution_sha or completion.get("rows") != 256:
        raise ComposeError("candidate completion mismatch")
    if sha256_file(PREDICTIONS) != completion.get("predictions", {}).get("sha256"):
        raise ComposeError("candidate predictions closure mismatch")
    alignment = read_jsonl(ALIGNMENT_PATH)
    predictions = read_jsonl(PREDICTIONS)
    if len(alignment) != 256 or len(predictions) != 256 or [row.get("task_id") for row in predictions] != [row.get("task_id") for row in alignment]:
        raise ComposeError("prediction/alignment mismatch")
    module = _load_pinned_hybrid()
    with tempfile.TemporaryDirectory(prefix="maxim_v1_3_recompose_") as raw:
        output = Path(raw) / "solver.jsonl"
        result = module.compose(DECISIONS, PREDICTIONS, output, HYBRID_FREEZE_SHA, "qwen35_9b_frozen_candidate", FREEZE_PATH, execution_sha, AUDIT_PATH, audit_sha)
        actual_sha = sha256_file(output)
        if result.get("rows") != 274 or result.get("output_sha256") != actual_sha or actual_sha != expected_composed_sha:
            raise ComposeError("independent exact composition mismatch")
        if not COMPOSED.is_file() or sha256_file(COMPOSED) != actual_sha or stable_bytes(COMPOSED) != stable_bytes(output):
            raise ComposeError("persisted solver differs from exact recomposition")
    return actual_sha


def compose(execution_sha: str, audit_sha: str, completion_sha: str) -> dict[str, Any]:
    if COMPOSED.exists() or COMPOSE_COMPLETION.exists():
        raise ComposeError("composition output exists")
    if sha256_file(FREEZE_PATH) != execution_sha or sha256_file(AUDIT_PATH) != audit_sha or sha256_file(COMPLETION) != completion_sha:
        raise ComposeError("candidate freeze/audit/completion external pin mismatch")
    if any(sha256_file(path) != digest for path, digest in ((HYBRID_FREEZE, HYBRID_FREEZE_SHA), (HYBRID_AUDIT, HYBRID_AUDIT_SHA), (HYBRID_IMPLEMENTATION, HYBRID_IMPLEMENTATION_SHA), (DECISIONS, DECISIONS_SHA))):
        raise ComposeError("Hybrid V3.1 ancestry pin mismatch")
    freeze, audit, completion = read_json(FREEZE_PATH), read_json(AUDIT_PATH), read_json(COMPLETION)
    if freeze.get("state") != "frozen_unexecuted_unscored" or audit.get("status") != "PASS" or audit.get("freeze_sha256") != execution_sha or completion.get("freeze_sha256") != execution_sha or completion.get("rows") != 256 or completion.get("gold_opened") is not False or completion.get("outcomes_opened") is not False:
        raise ComposeError("candidate closure mismatch")
    if sha256_file(PREDICTIONS) != completion.get("predictions", {}).get("sha256") or len(read_jsonl(PREDICTIONS)) != 256:
        raise ComposeError("candidate predictions closure mismatch")
    alignment = read_jsonl(ALIGNMENT_PATH)
    predictions = read_jsonl(PREDICTIONS)
    expected_ids = [row["task_id"] for row in alignment]
    if [row.get("task_id") for row in predictions] != expected_ids:
        raise ComposeError("prediction/alignment order mismatch")
    module = _load_pinned_hybrid()
    result = module.compose(DECISIONS, PREDICTIONS, COMPOSED, HYBRID_FREEZE_SHA, "qwen35_9b_frozen_candidate", FREEZE_PATH, execution_sha, AUDIT_PATH, audit_sha)
    if result.get("rows") != 274 or result.get("output_sha256") != sha256_file(COMPOSED):
        raise ComposeError("composition result mismatch")
    rows = read_jsonl(COMPOSED)
    if len(rows) != 274 or len({row.get("task_id") for row in rows}) != 274:
        raise ComposeError("composed solver denominator mismatch")
    if any(row.get("generation", {}).get("identity_used_by_branch_selector") is not False or row.get("generation", {}).get("hybrid_freeze_sha256") != HYBRID_FREEZE_SHA for row in rows):
        raise ComposeError("composed selector provenance mismatch")
    value = {"schema_version": "maxim256-hybrid-compose-completion-v1", "candidate_execution_freeze_sha256": execution_sha, "candidate_independent_audit_sha256": audit_sha, "candidate_completion_sha256": completion_sha, "candidate_predictions_sha256": sha256_file(PREDICTIONS), "candidate_outer_alignment_sha256": sha256_file(ALIGNMENT_PATH), "hybrid_v3_1_freeze_sha256": HYBRID_FREEZE_SHA, "hybrid_v3_1_audit_sha256": HYBRID_AUDIT_SHA, "route_decisions_sha256": DECISIONS_SHA, "rows": 274, "generic_rows": 256, "certified_rows": 18, "composed_solver_sha256": sha256_file(COMPOSED), "gold_opened": False, "outcomes_opened": False}
    exclusive_json(COMPOSE_COMPLETION, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-execution-freeze-sha256", required=True)
    parser.add_argument("--expected-independent-audit-sha256", required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(compose(args.expected_execution_freeze_sha256, args.expected_independent_audit_sha256, args.expected_completion_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
