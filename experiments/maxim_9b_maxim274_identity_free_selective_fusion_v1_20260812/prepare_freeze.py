"""Freeze the selective-fusion rule without opening future generic outputs."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "selective_fusion.py"
TEST_PATH = HERE / "test_selective_fusion.py"
README = HERE / "README.md"
AUDIT_TEMPLATE = HERE / "INDEPENDENT_AUDIT_TEMPLATE.json"
FREEZE = HERE / "FUSION_RULE_FREEZE.json"
SIDECAR = HERE / "FUSION_RULE_FREEZE_SHA256.txt"
INVALID_V1 = HERE / "INVALID_V1_FREEZE.json"
INVALID_V1_SIDECAR = HERE / "INVALID_V1_FREEZE_SHA256.txt"
INVALID_V1_SHA = "2e6fea7188b1e501ec2eb2dc56ffff3488bc946d177c2cb5504355d52c844f50"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("selective_fusion_freeze_builder", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    if path.read_bytes() != data:
        raise RuntimeError("post-write mismatch")


def main() -> None:
    module = load_module()
    if any(path.exists() for path in (FREEZE, SIDECAR, module.COMPLETION, module.PREDICTIONS, module.OUTPUT, module.MANIFEST)):
        raise RuntimeError("freeze or future atomic output already exists")
    if stable_sha(INVALID_V1) != INVALID_V1_SHA or INVALID_V1_SIDECAR.read_text(encoding="ascii").strip() != INVALID_V1_SHA:
        raise RuntimeError("invalid V1 lineage pin mismatch")
    pins = {
        module.CANDIDATE / "EXECUTION_FREEZE.json": module.CANDIDATE_FREEZE_SHA,
        module.CANDIDATE / "INDEPENDENT_AUDIT.json": module.CANDIDATE_AUDIT_SHA,
        module.BASE251: module.BASE251_SHA,
        module.PUBLIC_QUEUE: module.PUBLIC_QUEUE_SHA,
        module.GENERIC_QUEUE: module.GENERIC_QUEUE_SHA,
        module.ALIGNMENT: module.ALIGNMENT_SHA,
        module.DECISIONS: module.DECISIONS_SHA,
        module.NORMALIZATION: module.NORMALIZATION_SHA,
    }
    if any(stable_sha(path) != expected for path, expected in pins.items()):
        raise RuntimeError("ancestry pin mismatch")
    public = module.read_jsonl(module.PUBLIC_QUEUE)
    base = module.read_jsonl(module.BASE251)
    decisions = module.read_jsonl(module.DECISIONS)
    alignment = module.read_jsonl(module.ALIGNMENT)
    generic_queue = module.read_jsonl(module.GENERIC_QUEUE)
    # Placeholder rows are used only to validate frozen identity/order closure;
    # no future prediction bytes are opened.
    placeholders = [{"task_id": row["task_id"]} for row in alignment]
    base_ids, base_by_id, _, answer_type_by_id = module._alignment_closure(public, base, decisions, alignment, generic_queue, placeholders)
    normalizer = module._load_normalizer()
    generic_ids = set(answer_type_by_id)
    census: dict[str, int] = {"generic_rows": 0, "certified_rows": 0, "baseline_parseable_generic_rows": 0, "baseline_unparseable_generic_rows": 0}
    by_type: dict[str, dict[str, int]] = {}
    for task_id in base_ids:
        if task_id not in generic_ids:
            census["certified_rows"] += 1
            continue
        census["generic_rows"] += 1
        answer_type = answer_type_by_id[task_id]
        bucket = by_type.setdefault(answer_type, {"parseable": 0, "unparseable": 0})
        key = "parseable" if module.baseline_parseable(base_by_id[task_id].get("final_answer"), answer_type, normalizer) else "unparseable"
        bucket[key] += 1
        census[f"baseline_{key}_generic_rows"] += 1
    if census != {"generic_rows": 256, "certified_rows": 18, "baseline_parseable_generic_rows": 246, "baseline_unparseable_generic_rows": 10}:
        raise RuntimeError(f"unexpected precompletion census: {census}")

    artifacts = {
        "candidate_execution_freeze": module.descriptor(module.CANDIDATE / "EXECUTION_FREEZE.json"),
        "candidate_independent_audit": module.descriptor(module.CANDIDATE / "INDEPENDENT_AUDIT.json"),
        "base251_solver": module.descriptor(module.BASE251, 274),
        "public_queue": module.descriptor(module.PUBLIC_QUEUE, 274),
        "generic_queue": module.descriptor(module.GENERIC_QUEUE, 256),
        "outer_alignment": module.descriptor(module.ALIGNMENT, 256),
        "hybrid_decisions": module.descriptor(module.DECISIONS, 274),
        "normalization": module.descriptor(module.NORMALIZATION),
        "invalid_v1_freeze": module.descriptor(INVALID_V1),
        "invalid_v1_sidecar": module.descriptor(INVALID_V1_SIDECAR),
    }
    implementation = {
        "selective_fusion": module.descriptor(MODULE_PATH),
        "tests": module.descriptor(TEST_PATH),
        "prepare_freeze": module.descriptor(Path(__file__).resolve()),
        "README": module.descriptor(README),
        "audit_template": module.descriptor(AUDIT_TEMPLATE),
    }
    freeze = {
        "schema_version": "maxim274-identity-free-selective-fusion-freeze-v1.1",
        "state": "frozen_before_generic_atomic_completion",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "rule": {
            "certified_noid_rows": "preserve base251 answer",
            "generic_invalid": "preserve base251 answer",
            "generic_valid_and_baseline_parseable": "preserve base251 answer regardless of agreement",
            "generic_valid_and_baseline_unparseable": "select structurally valid generic answer",
            "parseability_is_syntax_only": True,
            "semantic_correctness_claimed": False,
        },
        "selector_inputs_exact": ["answer_type", "baseline_answer", "generic_projection_without_identity"],
        "selector_forbidden": ["task_id", "controller_id", "row_index", "filename", "image_hash", "content_hash", "subject", "route", "gold", "outcome", "prior_verdict", "partial_run_state"],
        "precompletion_census": {**census, "by_answer_type": by_type},
        "risk_contract": {
            "malformed_or_failed_generic_regression_protected": True,
            "parseable_baseline_rows_exposed_to_generic_switch": 0,
            "maximum_rows_exposed_to_valid_but_wrong_generic": census["baseline_unparseable_generic_rows"],
            "guaranteed_accuracy_improvement": False,
            "guaranteed_nonregression_vs_251": False,
            "reason": "baseline syntax invalidity is not proof that the valid generic answer is semantically correct",
        },
        "chronology": {
            "candidate_atomic_completion_absent": True,
            "candidate_atomic_predictions_absent": True,
            "partial_generic_cache_opened": False,
            "gold_opened": False,
            "outcomes_opened": False,
            "score_opened": False,
        },
        "supersession": {
            "invalid_v1_freeze_sha256": INVALID_V1_SHA,
            "invalid_v1_preserved_unmodified": True,
            "invalid_v1_runtime_outputs_absent": True,
            "reason": "V1 narrative said 9 exposed rows although its census contained 10, and its selector received a generic mapping that retained task_id; V1.1 derives the count from census and removes identity before arbitration",
        },
        "artifacts": artifacts,
        "implementation": implementation,
        "tests": {"command": "python -B -m unittest test_selective_fusion.py", "passed_immediately_before_freeze": True},
    }
    payload = canonical(freeze)
    write_exclusive(FREEZE, payload)
    digest = hashlib.sha256(payload).hexdigest()
    write_exclusive(SIDECAR, (digest + "\n").encode("ascii"))
    print(json.dumps({"freeze_sha256": digest, "precompletion_census": census}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
