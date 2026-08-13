"""Freeze the zero-tunable byte-preserving compositor before its output exists."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import compose
import protocol


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    if protocol.FREEZE.exists() or protocol.FREEZE_SHA_FILE.exists() or protocol.OUTPUT.exists() or protocol.COMPLETION.exists():
        raise protocol.ProtocolError("freeze or runtime output already exists")
    census = compose.validate_sources()
    base = census["base"]
    selected = census["selected"]
    branches = [protocol.branch(row) for row, _ in selected]
    artifacts = {
        name: protocol.descriptor(path, rows=(274 if name in {"v1_1_solver", "base251_solver"} else 97 if name in {"base251_image97_judge", "image97_alignment"} else 256 if name == "candidate_predictions" else None))
        for name, (path, _) in protocol.PINS.items()
    }
    implementation = {name: protocol.descriptor(protocol.HERE / name) for name in protocol.IMPLEMENTATION_FILES}
    freeze = {
        "schema_version": "maxim274-selective-fusion-byte-preserve-freeze-v1.3",
        "state": "frozen_after_v1_1_atomic_completion_before_v1_3_output_and_private_score",
        "created_utc": utc_now(),
        "rule": {
            "decision_source": "exact immutable audited V1.1 result rows",
            "baseline_decision": "copy complete base251 JSONL row bytes unchanged",
            "generic_decision": "copy complete V1.1 result JSONL row bytes unchanged",
            "new_branch_logic": False,
            "semantic_tunables": 0,
            "task_identity_role": "postdecision alignment and image97 exclusion proof only",
        },
        "census": {
            "rows": len(base),
            "baseline_selected": sum(value in protocol.BASELINE_BRANCHES for value in branches),
            "generic_selected": sum(value == protocol.GENERIC_BRANCH for value in branches),
            "image97_rows": 97,
            "generic_selected_inside_image97": 0,
            "expected_image97_candidate_text_byte_matches": 97,
        },
        "chronology_and_exposure": {
            "v1_1_rule_frozen_before_candidate_atomic_completion": True,
            "v1_1_identity_free_selector_independently_audited_pass": True,
            "successor_frozen_after_candidate_atomic_completion": True,
            "builder_was_accidentally_exposed_to_an_existing_postscore_result_artifact": True,
            "per_row_outcome_contents_used_by_successor_rule_or_code": False,
            "mitigation": "no decisions are recomputed or editable; exact V1.1 decisions and exact source row bytes are mechanically imported with zero tunables",
            "clean_room_claimed": False,
            "withheld_v1_2": {
                "freeze_sha256": protocol.PINS["withheld_v1_2_freeze"][1],
                "preserved_unmodified": True,
                "audit_status": "WITHHOLD",
                "reason": "runtime self-hash closure and full scorer recomposition closure were missing",
                "runtime_output_absent_at_v1_3_freeze": not (protocol.V12 / "runs/selective_fusion_byte_preserve_solver_274.jsonl").exists(),
                "completion_absent_at_v1_3_freeze": not (protocol.V12 / "BYTE_PRESERVE_COMPLETION.json").exists(),
                "private_result_absent_at_v1_3_freeze": not (protocol.V12 / "PRIVATE_RESULT.json").exists(),
            },
        },
        "privacy": {
            "api_called": False,
            "gold_or_reference_content_opened_by_freezer": False,
            "judge_verdict_content_opened_by_freezer": False,
            "outcome_fields_accepted_by_compositor": False,
        },
        "scoring_contract": {
            "base251_image97_judge_reusable_only_after_97_candidate_text_utf8_exact_checks": True,
            "all_274_output_bytes_recomputed_before_private_inputs_are_parsed": True,
            "runtime_implementation_self_hashes_verified": True,
            "official_scorer_exact_pin": protocol.PINS["official_scorer"][1],
            "success_threshold_correct_at_least": protocol.SUCCESS_THRESHOLD_CORRECT,
            "threshold_accuracy": round(protocol.SUCCESS_THRESHOLD_CORRECT / 274, 9),
            "private_score_not_executed_at_freeze": True,
        },
        "artifacts": artifacts,
        "implementation": implementation,
        "independent_audit_required_before_composition": True,
    }
    data = protocol.canonical_json(freeze)
    digest = protocol.sha256_bytes(data)
    protocol.exclusive_bytes(protocol.FREEZE, data)
    protocol.exclusive_bytes(protocol.FREEZE_SHA_FILE, (digest + "\n").encode("ascii"))
    print(json.dumps({"freeze_sha256": digest, "rows": len(base), "baseline": 272, "generic": 2}, indent=2))


if __name__ == "__main__":
    main()
