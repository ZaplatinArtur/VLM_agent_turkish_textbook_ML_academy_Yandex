from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import tempfile

import selector as s


def expect_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except s.SelectorError:
        return
    raise AssertionError(f"{function.__name__} did not fail closed")


def test_normalization_and_strict_salvage() -> None:
    assert s.normalize_answer("  a).  ") == "A"
    assert s.normalize_answer("  Çok   iyi  ") == "çok iyi"
    assert s.normalize_answer(7) is None
    assert s.candidate_answer({"final_answer": '{"answer":{"result":"b)"}}'}) == "B"
    assert s.candidate_answer({"final_answer": '```json\n{"final_answer": 12}\n```'}) == "12"
    assert s.candidate_answer({"final_answer": '{"answer":"A","extra":"B"}'}) is None
    assert s.candidate_answer({"final_answer": '["A"]'}) is None
    assert s.candidate_answer({"final_answer": '{"answer":"A","answer":"B"}'}) is None
    assert s.candidate_answer({"final_answer": '{broken json'}) is None
    assert s.candidate_answer(
        {
            "final_answer": '{"answer":"A","extra":"B"}',
            "raw_response": '{"result":{"output":"C"}}',
        }
    ) == "C"
    assert s.candidate_answer({"final_answer": "ordinary free text"}) == "ordinary free text"


def test_model_closure() -> None:
    row = {
        "task_id": "synthetic",
        "model": s.MODEL,
        "final_answer": "A",
        "generation": {"model": s.MODEL},
    }
    assert s.validate_solver_row(row, role="test") is row
    foreign = dict(row)
    foreign["generation"] = {"model": "Qwen/Qwen3.5-27B"}
    expect_error(s.validate_solver_row, foreign, role="test")
    missing = {"task_id": "synthetic", "final_answer": "A"}
    expect_error(s.validate_solver_row, missing, role="test")


def synthetic_calibration(extra_agreements: bool = False):
    labels = {}
    candidates = {role: {} for role in s.ROLE_ORDER}
    families = ["owner_a", "owner_b", "owner_c", "owner_d", "owner_e"]
    for family_index, family in enumerate(families):
        for item in range(10):
            task_id = f"base_{family_index}_{item}"
            target = "A"
            anchor = "B" if item < 4 else "A"
            answers = {
                "active_crop_v2": anchor,
                "no_tools_v1": "A",
                "native_thinking_v4": "B" if anchor == "A" else anchor,
                "native_thinking_v5": "A",
                "parallel8_v1": "A",
                "parallel8_reasoning_first_v2": "A",
            }
            labels[task_id] = {
                "owner_stage": family,
                "answer": target,
                "answer_sha256": hashlib.sha256(target.encode()).hexdigest(),
            }
            for role, answer in answers.items():
                candidates[role][task_id] = {"final_answer": answer}
        if extra_agreements:
            for item in range(10):
                task_id = f"agreement_{family_index}_{item}"
                target = "A"
                labels[task_id] = {
                    "owner_stage": family,
                    "answer": target,
                    "answer_sha256": hashlib.sha256(target.encode()).hexdigest(),
                }
                for role in s.ROLE_ORDER:
                    candidates[role][task_id] = {"final_answer": "A"}
    return labels, candidates


def test_conditional_calibration_and_group_cap() -> None:
    labels, candidates = synthetic_calibration(extra_agreements=False)
    report, effective = s.calibrate(labels, candidates)
    labels_more, candidates_more = synthetic_calibration(extra_agreements=True)
    report_more, effective_more = s.calibrate(labels_more, candidates_more)

    roles = report["roles"]
    assert roles["active_crop_v2"]["base_weight_after_gate"] == 0.0
    assert roles["no_tools_v1"]["global_fixes"] == 20
    assert roles["no_tools_v1"]["global_regressions"] == 0
    assert roles["no_tools_v1"]["precision_gate_passed"] is True
    assert roles["native_thinking_v4"]["global_fixes"] == 0
    assert roles["native_thinking_v4"]["global_regressions"] == 30
    assert roles["native_thinking_v4"]["precision_gate_passed"] is False
    assert effective["native_thinking_v4"] == 0.0

    # Rows where candidate and anchor agree are deliberately zero override
    # evidence and therefore cannot alter the conditional reliability weight.
    assert (
        roles["no_tools_v1"]["global_override_logit_weight"]
        == report_more["roles"]["no_tools_v1"]["global_override_logit_weight"]
    )
    assert effective["no_tools_v1"] == effective_more["no_tools_v1"]

    parallel = report["correlation_groups"]["parallel8_pair"]
    assert abs(parallel["effective_sum"] - parallel["group_cap"]) < 1e-12
    assert parallel["effective_sum"] <= parallel["uncapped_sum"]
    assert all(fold["gate_passed"] for fold in roles["no_tools_v1"]["leave_one_owner_stage_out"])


def one_task_candidates(anchor: str, **answers: str):
    rows = {}
    for role in s.ROLE_ORDER:
        value = answers.get(role, anchor)
        rows[role] = {"t": {"final_answer": value}}
    return rows


def weights(**values: float):
    result = {role: 0.0 for role in s.ROLE_ORDER}
    result.update(values)
    return result


def test_selection_gates_and_invalid_anchor() -> None:
    candidates = one_task_candidates(
        "A", no_tools_v1="B", native_thinking_v4="B"
    )
    role, detail = s.choose_uncovered(
        "t", candidates, weights(no_tools_v1=0.13, native_thinking_v4=0.13)
    )
    assert role == "active_crop_v2"
    assert detail["reason"] == "margin_over_anchor_below_preregistered_threshold"
    assert detail["anchor_valid"] is True

    role, detail = s.choose_uncovered(
        "t", candidates, weights(no_tools_v1=0.5, native_thinking_v4=0.4)
    )
    assert role == "no_tools_v1"
    assert detail["reason"] == "source_calibrated_replacement"
    assert detail["support_groups"] == 2

    correlated = one_task_candidates(
        "A", native_thinking_v4="B", native_thinking_v5="B"
    )
    role, detail = s.choose_uncovered(
        "t", correlated, weights(native_thinking_v4=0.5, native_thinking_v5=0.5)
    )
    assert role == "active_crop_v2"
    assert detail["reason"] == "insufficient_independent_candidate_groups"

    malformed = one_task_candidates(
        '{"answer":"A","extra":"B"}',
        no_tools_v1="C",
        native_thinking_v4="C",
    )
    role, detail = s.choose_uncovered(
        "t", malformed, weights(no_tools_v1=0.5, native_thinking_v4=0.4)
    )
    assert role == "no_tools_v1"
    assert detail["reason"] == "invalid_anchor_replaced_by_crossfitted_multigroup_consensus"
    assert detail["anchor_valid"] is False

    role, detail = s.choose_uncovered(
        "t", malformed, weights(no_tools_v1=0.5)
    )
    assert role == "active_crop_v2"
    assert detail["reason"] == "invalid_anchor_insufficient_crossfitted_groups_fail_closed_anchor_bytes"
    assert detail["anchor_valid"] is False

    role, detail = s.choose_uncovered("t", malformed, weights())
    assert role == "active_crop_v2"
    assert detail["reason"] == "no_calibrated_weight_fail_closed"
    assert detail["anchor_valid"] is False


def test_atomic_write_refuses_overwrite() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artifact.json"
        s.write_bytes_atomic(path, b"first\n")
        assert path.read_bytes() == b"first\n"
        expect_error(s.write_bytes_atomic, path, b"second\n")
        assert path.read_bytes() == b"first\n"


def test_exact_pinned_authorities_without_evaluation() -> None:
    experiment_root = Path(__file__).resolve().parent
    root = s.repo_root(experiment_root)
    profile = s.verify_preregistration(experiment_root)
    assert profile["evaluation_status"].startswith("must_remain_not_evaluated")
    labels, source_order, source_rows, source_raw, authority = s.load_source_authority(root)
    order, candidates, raw = s.load_candidates(root)
    assert len(labels) == 156
    assert len(order) == len(source_order) == 274
    assert len(set(order) - set(labels)) == 118
    assert set(labels).issubset(source_rows)
    assert authority["source_union_projection_sha256"] == s.SOURCE["union_sha256"]
    assert all(set(candidate) == set(order) for candidate in candidates.values())
    assert all(set(candidate) == set(order) for candidate in raw.values())
    assert all(task_id in source_raw for task_id in labels)
    routes, input_authority = s.load_input_authority(
        root, expected_order=order, source_ids=set(labels)
    )
    assert Counter(routes.values()) == Counter({"deterministic": 177, "image_judge": 97})
    assert input_authority["route_use"] == "safety_veto_only_not_quality_feature"
    assert input_authority["package"]["sha256"] == s.INPUT_AUTHORITY["package"]["sha256"]

    # Calibration is source-label-only.  Its artifact intentionally exposes no
    # benchmark accuracy or score field, and ActiveCrop cannot earn override
    # mass merely from source-confirmation agreements.
    calibration, effective = s.calibrate(labels, candidates)
    assert "accuracy" not in json.dumps(calibration).casefold()
    assert "benchmark_score" not in json.dumps(calibration).casefold()
    assert effective["active_crop_v2"] == 0.0
    for group in calibration["correlation_groups"].values():
        assert group["effective_sum"] <= group["group_cap"] + 1e-12


def main() -> int:
    tests = [
        test_normalization_and_strict_salvage,
        test_model_closure,
        test_conditional_calibration_and_group_cap,
        test_selection_gates_and_invalid_anchor,
        test_atomic_write_refuses_overwrite,
        test_exact_pinned_authorities_without_evaluation,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} selector test groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
