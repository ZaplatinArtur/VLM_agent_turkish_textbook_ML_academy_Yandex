from __future__ import annotations

import copy

from scripts import compose_maxim_blind_ensemble_v2default_v1 as ensemble


THRESHOLDS = {
    "v3_min_confidence": 0.90,
    "v3_min_decisive_evidence": 2,
    "active_min_verifier_confidence": 0.90,
    "active_min_locator_confidence": 0.80,
    "active_min_region_confidence": 0.70,
    "active_min_visible_facts": 2,
    "active_min_verification_checks": 2,
}


def profile(policy_id: str) -> dict:
    return {
        "schema_version": ensemble.PROFILE_SCHEMA,
        "condition": f"synthetic_{policy_id}",
        "policy_id": policy_id,
        "frozen_before_source_row_values_read": True,
        "gold_access": False,
        "score_or_judge_inputs_allowed": False,
        "selection_uses_task_id": False,
        "default_source": "meta_v21",
        "thresholds": copy.deepcopy(THRESHOLDS),
    }


def source_rows(count: int = ensemble.EXPECTED_ROWS) -> dict[str, list[dict]]:
    v2_rows: list[dict] = []
    v2_audits: list[dict] = []
    v2_verifiers: list[dict] = []
    v3_rows: list[dict] = []
    v3_audits: list[dict] = []
    v3_verifiers: list[dict] = []
    active_rows: list[dict] = []
    for index in range(count):
        task_id = f"synthetic_{index:04d}"
        v2 = {"task_id": task_id, "final_answer": "A", "error": None}
        v3 = {"task_id": task_id, "final_answer": "B", "error": None}
        active = {
            "task_id": task_id,
            "final_answer": "B",
            "forced_answer": False,
            "error": None,
            "generation": {
                "gold_access": False,
                "active_crop_failclosed_composition": {
                    "selected_source": "active_crop",
                    "gate_passed": True,
                    "failed_clauses": [],
                    "candidate_row_sha256": "1" * 64,
                    "fallback_row_sha256": "2" * 64,
                },
                "selection_evidence": {
                    "baseline_supported": False,
                    "confidence": 0.95,
                    "all_required_evidence_visible": True,
                    "original_crop_consistent": True,
                    "answer_format_verified": True,
                    "visible_facts": ["fact one", "fact two"],
                    "verification_checks": ["check one", "check two"],
                },
                "locator": {
                    "overall_confidence": 0.9,
                    "used_regions": [{"confidence": 0.8}],
                },
            },
        }
        v2_rows.append(v2)
        v3_rows.append(v3)
        active_rows.append(active)
        v2_audits.append(
            {
                "task_id": task_id,
                "output_row_sha256": ensemble.stable_sha256(v2),
                "decision": "unchanged_v2_content_exact",
                "gold_access": False,
                "task_id_or_subject_used_for_selection": False,
            }
        )
        v2_verifiers.append(
            {
                "task_id": task_id,
                "error": None,
                "selection": {
                    "gold_access": False,
                    "selected_source": "router",
                    "reason": "confidence_gate_router_fallback",
                    "applied_final_answer": "A",
                },
            }
        )
        v3_audits.append(
            {
                "task_id": task_id,
                "output_row_sha256": ensemble.stable_sha256(v3),
                "decision": "unchanged_v3_content_exact",
                "gold_access": False,
                "task_id_or_subject_used_for_selection": False,
            }
        )
        v3_verifiers.append(
            {
                "task_id": task_id,
                "error": None,
                "selection": {
                    "gold_access": False,
                    "selected_source": "meta_verifier",
                    "reason": "valid_supported_meta_answer",
                    "applied_final_answer": "B",
                },
                "verdict": {
                    "abstain": False,
                    "answer_format_verified": True,
                    "confidence": 0.95,
                    "decisive_evidence": ["evidence one", "evidence two"],
                    "final_answer": "B",
                },
                "call": {
                    "parse_error": None,
                    "recovered_partial": False,
                    "finish_reason": "stop",
                },
            }
        )
    return {
        "v2_rows": v2_rows,
        "v2_audits": v2_audits,
        "v2_verifiers": v2_verifiers,
        "v3_rows": v3_rows,
        "v3_audits": v3_audits,
        "v3_verifiers": v3_verifiers,
        "active_rows": active_rows,
    }


def run(policy_id: str, sources: dict[str, list[dict]]):
    return ensemble.compose(profile=profile(policy_id), **sources)


def test_triple_agreement_overrides_and_copies_v31_exactly() -> None:
    sources = source_rows()
    output, audit, counts = run("triple_agreement_v3_active", sources)
    assert output == sources["v3_rows"]
    assert counts == {"override_v3_active_agree_against_v21": 274}
    assert all(row["selected_source"] == "meta_v31" for row in audit)


def test_triple_agreement_disagreement_defaults_exact_v21() -> None:
    sources = source_rows()
    for row in sources["active_rows"]:
        row["final_answer"] = "C"
    output, _, counts = run("triple_agreement_v3_active", sources)
    assert output == sources["v2_rows"]
    assert counts == {"default_exact_meta_v21": 274}


def test_v31_repairs_only_generic_v21_failclosed() -> None:
    sources = source_rows()
    output, _, counts = run("v3_repairs_v2_failclosed", sources)
    assert output == sources["v3_rows"]
    assert counts == {"override_v31_repairs_v21_failclosed": 274}
    sources["v2_verifiers"][0]["selection"]["selected_source"] = "meta_verifier"
    output, _, counts = run("v3_repairs_v2_failclosed", sources)
    assert output[0] == sources["v2_rows"][0]
    assert counts["default_exact_meta_v21"] == 1


def test_active_repairs_only_generic_v21_failclosed() -> None:
    sources = source_rows()
    output, _, counts = run("active_repairs_v2_failclosed", sources)
    assert output == sources["active_rows"]
    assert counts == {"override_active_repairs_v21_failclosed": 274}


def test_any_missing_optional_gate_fails_closed_to_exact_v21() -> None:
    sources = source_rows()
    del sources["active_rows"][0]["generation"]["selection_evidence"]["confidence"]
    output, audit, counts = run("triple_agreement_v3_active", sources)
    assert output[0] == sources["v2_rows"][0]
    assert counts["default_exact_meta_v21"] == 1
    assert "active_verifier_confidence_below_0_90" in audit[0][
        "failed_generic_gates"
    ]["active_crop_v2"]


def test_task_id_never_changes_decision_when_metadata_is_identical() -> None:
    sources = source_rows()
    _, audit, _ = run("v3_repairs_v2_failclosed", sources)
    assert {row["decision"] for row in audit} == {
        "override_v31_repairs_v21_failclosed"
    }
    assert all(row["task_id_or_subject_used_for_selection"] is False for row in audit)
