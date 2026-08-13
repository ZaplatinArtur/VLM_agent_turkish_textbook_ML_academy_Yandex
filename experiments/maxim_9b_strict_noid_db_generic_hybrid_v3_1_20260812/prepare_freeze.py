"""Freeze strict audited no-ID branch + transport-neutral generic contract."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from strict_hybrid import (
    BASE240, HERE, NOID_AUDIT, NOID_FREEZE, NOID_IMPLEMENTATION, NOID_SOURCE_DB,
    canonical_bytes, read_jsonl, sha256,
)


FREEZE = HERE / "HYBRID_RULE_FREEZE.json"
SIDECAR = HERE / "HYBRID_RULE_FREEZE_SHA256.txt"
CONTRACT = HERE / "GENERIC_OUTPUT_CONTRACT.json"
V2_FREEZE = HERE.parent / "maxim_9b_content_only_db_generic_hybrid_v2_20260812/HYBRID_RULE_FREEZE.json"
BAD_V3_FREEZE = HERE.parent / "maxim_9b_strict_noid_db_generic_hybrid_v3_20260812/HYBRID_RULE_FREEZE.json"
MAXIM_QUEUE = HERE.parent / "maxim_9b_maxim274_generic_content_adapter_v1_20260812/frozen/maxim274_public_runtime_queue.jsonl"
YKS_QUEUE = HERE.parent / "maxim_9b_ykslop_generic_reasoning_sse_alt_v3_dev_20260812/frozen/queue_public_content_only.jsonl"


def descriptor(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": Path(os.path.relpath(path.resolve(), HERE)).as_posix(),
        "sha256": sha256(path),
        "size": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = rows
    return value


def main() -> None:
    if FREEZE.exists() or SIDECAR.exists():
        raise RuntimeError("freeze already exists; refusing overwrite")
    noid_freeze = json.loads(NOID_FREEZE.read_text(encoding="utf-8"))
    noid_audit = json.loads(NOID_AUDIT.read_text(encoding="utf-8"))
    if noid_audit.get("status") != "PASS" or noid_audit.get("freeze_sha256") != sha256(NOID_FREEZE):
        raise RuntimeError("certified no-ID branch lacks exact PASS audit")
    scripts = ("strict_hybrid.py", "run_hybrid.py", "prepare_freeze.py", "score_branch_deltas.py", "test_strict_hybrid.py")
    freeze = {
        "schema_version": "strict-noid-db-generic-hybrid-rule-freeze-v3.1",
        "state": "frozen_unexecuted_unscored",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "selector_inputs_exact": ["ocr_text", "answer_type", "input_mode"],
        "selector_forbidden": [
            "task_id", "controller_id", "benchmark_id", "row_order", "filename",
            "image_hash", "content_hash", "subject", "route", "base249_answer",
            "gold", "outcome", "prior_prediction",
        ],
        "outer_alignment": {
            "runtime_id_read_only_after_selector_returns": True,
            "runtime_id_role": "output_order_and_prediction_join_only",
        },
        "certified_branch": {
            "policy": "audited maxim-noid-content-source-freeze-v1 imported without modification",
            "noid_freeze_sha256": sha256(NOID_FREEZE),
            "noid_projection_sha256": noid_freeze["freeze_projection_sha256"],
            "noid_independent_audit_sha256": sha256(NOID_AUDIT),
            "candidate_census": noid_freeze["candidate_census"],
            "fail_closed": True,
            "source_records": 17,
            "answer_bindings": 16,
            "deterministic_tools": 2,
            "known_postfreeze_evidence": "base240 control 240->251, +11 fixes, 0 regressions; development benchmark, not unseen generalization",
        },
        "generic_branch": {
            "model_exact": "Qwen/Qwen3.5-9B",
            "transport_neutral_output_contract_sha256": sha256(CONTRACT),
            "external_candidate_freeze_and_independent_pass_audit_required": True,
            "provider_transport_quantization_not_selector_inputs": True,
            "silent_fallback_forbidden": True,
            "hosted_bitwise_determinism_claimed": False,
        },
        "maxim_policy": {
            "strict_control_anchor": "base240",
            "base249_official16": "diagnostic_only_never_runtime_fallback",
            "old_plus9_must_be_recovered_by_audited_visible_content_policy": True,
        },
        "artifacts": {
            "noid_freeze": descriptor(NOID_FREEZE),
            "noid_source_db": descriptor(NOID_SOURCE_DB),
            "noid_implementation": descriptor(NOID_IMPLEMENTATION),
            "noid_independent_audit": descriptor(NOID_AUDIT),
            "strict_maxim_anchor_base240": descriptor(BASE240, len(read_jsonl(BASE240))),
            "generic_output_contract": descriptor(CONTRACT),
            "maxim274_public_queue": descriptor(MAXIM_QUEUE, len(read_jsonl(MAXIM_QUEUE))),
            "ykslop_dev185_public_queue": descriptor(YKS_QUEUE, len(read_jsonl(YKS_QUEUE))),
            "superseded_broad_matcher_v2_freeze": descriptor(V2_FREEZE),
            "superseded_preflight_v3_freeze": descriptor(BAD_V3_FREEZE),
        },
        "implementation": {name: descriptor(HERE / name) for name in scripts},
        "evaluation_sets": {"maxim274": 274, "ykslop_dev185": 185},
        "guards": {
            "api_called": False,
            "gold_opened": False,
            "outcomes_opened": False,
            "sealed_final80_opened": False,
            "coverage_outputs_present_at_freeze": False,
            "hybrid_outputs_present_at_freeze": False,
        },
        "supersession": {
            "supersedes_broad_matcher_v2_sha256": sha256(V2_FREEZE),
            "supersedes_failed_preflight_v3_sha256": sha256(BAD_V3_FREEZE),
            "reason": "headline hybrid consumes the independently audited 18-action no-ID branch that proved +11/0 over base240; broad 64-row matcher remains diagnostic and unscored; preliminary V3 was never authorized because one pre-freeze test had a false-positive metadata substring assertion",
        },
        "report_contract": ["score", "certified_coverage", "abstention", "fixes", "regressions"],
    }
    FREEZE.write_bytes(canonical_bytes(freeze))
    digest = sha256(FREEZE)
    SIDECAR.write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"freeze_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
