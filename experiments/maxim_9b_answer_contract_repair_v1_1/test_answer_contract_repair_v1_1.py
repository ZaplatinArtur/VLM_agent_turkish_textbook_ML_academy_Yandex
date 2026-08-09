from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import answer_contract_repair_v1_1 as repair  # noqa: E402


def _row(answer: object, raw_response: object) -> dict:
    return {
        "task_id": "synthetic_task",
        "model": repair.MODEL,
        "error": None,
        "final_answer": answer,
        "raw_response": raw_response,
        "generation": {"gold_access": False},
    }


def test_profile_is_exact_and_explicitly_post_score_not_blind() -> None:
    profile = repair._profile()
    assert profile == repair.EXPECTED_PROFILE
    assert profile["chronology"] == {
        "historical_residual_outcomes_known_before_design": True,
        "post_score_motivated": True,
        "blind_claim": False,
        "preregistered_claim": False,
        "rules_must_be_frozen_before_candidate_materialization": True,
        "candidate_must_remain_unscored_and_unevaluated": True,
    }
    assert profile["scope"]["task_id_specific_rules"] is False
    assert profile["arms"]["strict"]["reasoning_text_regex_or_numeric_salvage"] is False
    assert (
        profile["arms"]["exploratory_explicit_key_scalar"][
            "free_reasoning_or_unkeyed_number_mining"
        ]
        is False
    )
    assert (
        profile["arms"]["exploratory_explicit_key_scalar"][
            "keys_inside_reasoning_or_any_other_value_are_forbidden"
        ]
        is True
    )
    assert (
        profile["arms"]["exploratory_explicit_key_scalar"][
            "global_unescape_or_key_search_over_value_text"
        ]
        is False
    )


def test_superseded_predecessor_is_hash_pinned_and_never_evaluated() -> None:
    record = repair._verify_superseded_record()
    assert record["evaluated"] is False
    assert record["scored"] is False
    assert len(record["preserved_artifacts"]) == 5


def test_pinned_outcome_free_inputs_preflight() -> None:
    for base_variant in repair.BASE_VARIANTS:
        bound = repair._load_inputs(base_variant)
        assert bound["base_variant"] == base_variant
        assert len(bound["base_rows"]) == len(bound["order"]) == 274
        assert len(set(bound["order"])) == 274
        assert len(bound["protected"]) == 156
        assert bound["routes"].count("deterministic") == 177
        assert bound["routes"].count("image_judge") == 97
        assert {row["model"] for row in bound["base_rows"]} == {repair.MODEL}


@pytest.mark.parametrize(
    "value, expected",
    [
        (" A ", "A"),
        (16, "16"),
        (1.5, "1.5"),
        ("30 m²", "30 m²"),
        ("a-2", "a-2"),
        ("1/2", "1/2"),
        ("12 naneli ve 8 limonlu", "12 naneli ve 8 limonlu"),
    ],
)
def test_strict_answer_accepts_bounded_scalar_contract(value: object, expected: str) -> None:
    assert repair.strict_answer(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        "",
        "line one\nline two",
        "{" + '"final_answer":"A"}',
        "[A]",
        '"A"',
        "} }16",
        "(unbalanced",
        float("inf"),
        "x" * 257,
    ],
)
def test_strict_answer_rejects_structural_or_unbounded_values(value: object) -> None:
    assert repair.strict_answer(value) is None


def test_unique_outer_json_candidate_is_recovered() -> None:
    raw = json.dumps({"reasoning": "ignored", "final_answer": " B "})
    assert repair.recover_unique_answer(raw) == {
        "status": "unique_strict_candidate",
        "candidates": ["B"],
        "key_occurrences": 1,
        "outer_parse_mode": "exact_json_object_preserving_duplicate_keys",
    }


def test_duplicate_equal_outer_keys_are_ambiguous_not_collapsed() -> None:
    raw = '{"final_answer":"7","final_answer":7}'
    assert repair.recover_unique_answer(raw) == {
        "status": "ambiguous_multiple_outer_final_answer_keys",
        "candidates": [],
        "key_occurrences": 2,
        "outer_parse_mode": "exact_json_object_preserving_duplicate_keys",
    }


def test_multiple_distinct_candidates_fail_closed() -> None:
    raw = '{"final_answer":"A","final_answer":"B"}'
    recovered = repair.recover_unique_answer(raw)
    assert recovered["status"] == "ambiguous_multiple_outer_final_answer_keys"
    decision = repair.decide_row(
        _row(None, raw),
        evaluation_route="deterministic",
        protected_by_source_union=False,
    )
    assert decision["action"] == "preserve_base_exact_bytes"
    assert decision["reason"] == "ambiguous_multiple_outer_final_answer_keys"


def test_strict_audit_regression_never_mines_final_answer_inside_reasoning() -> None:
    raw = '{"reasoning":"ignore {\\"final_answer\\":\\"A\\"}"}'
    assert repair.recover_unique_answer(raw) == {
        "status": "no_explicit_outer_final_answer",
        "candidates": [],
        "key_occurrences": 0,
        "outer_parse_mode": "exact_json_object_preserving_duplicate_keys",
    }


def test_free_text_and_numeric_tail_salvage_are_forbidden() -> None:
    raw = json.dumps(
        {
            "reasoning": "The final result is definitely 16.",
            "final_answer": "} }16",
        }
    )
    assert repair.recover_unique_answer(raw) == {
        "status": "outer_final_answer_not_strict",
        "candidates": [],
        "key_occurrences": 1,
        "outer_parse_mode": "exact_json_object_preserving_duplicate_keys",
    }


def test_parser_bounds_fail_closed() -> None:
    oversized = "{" + ("x" * repair.MAX_RAW_RESPONSE_BYTES) + "}"
    assert repair.recover_unique_answer(oversized) == {
        "status": "bounds_exceeded",
        "candidates": [],
        "key_occurrences": 0,
        "outer_parse_mode": None,
    }
    too_many = "{" + ",".join(f'\"k{i}\":{i}' for i in range(repair.MAX_OUTER_MEMBERS + 1)) + "}"
    recovered = repair.recover_unique_answer(too_many)
    assert recovered["status"] == "bounds_exceeded"
    assert recovered["candidates"] == []


def test_protected_and_image_rows_never_parse_or_change() -> None:
    row = _row(None, "x" * (repair.MAX_RAW_RESPONSE_BYTES + 1))
    protected = repair.decide_row(
        row,
        evaluation_route="deterministic",
        protected_by_source_union=True,
    )
    image = repair.decide_row(
        row,
        evaluation_route="image_judge",
        protected_by_source_union=False,
    )
    assert protected["reason"] == "protected_by_source_union"
    assert protected["parser_status"] == "not_run_protected"
    assert image["reason"] == "image_judge_route"
    assert image["parser_status"] == "not_run_image"
    assert protected["action"] == image["action"] == "preserve_base_exact_bytes"


def test_synthetic_eligible_unique_candidate_repairs_without_task_id_rule() -> None:
    raw = json.dumps({"reasoning": "ignored", "final_answer": "D"})
    decision = repair.decide_row(
        _row(None, raw),
        evaluation_route="deterministic",
        protected_by_source_union=False,
    )
    assert decision == {
        "action": "repair_from_raw_response",
        "reason": "unique_strict_outer_final_answer",
        "top_level_answer_valid": False,
        "parser_status": "unique_strict_candidate",
        "candidates": ["D"],
    }


def test_exploratory_arm_recovers_one_explicit_short_scalar_with_brace_debris() -> None:
    raw = '{"reasoning":"ignored", "final_answer": "} }16"\n}'
    assert repair.recover_exploratory_answer(raw) == {
        "status": "unique_explicit_key_scalar_candidate",
        "candidates": ["16"],
        "key_occurrences": 1,
        "matched_key": "final_answer",
        "outer_parse_mode": "exact_json_object_preserving_duplicate_keys",
    }
    decision = repair.decide_row(
        _row(None, raw),
        evaluation_route="deterministic",
        protected_by_source_union=False,
        arm="exploratory_explicit_key_scalar",
    )
    assert decision["action"] == "repair_from_raw_response"
    assert decision["candidates"] == ["16"]


def test_exploratory_arm_rejects_globally_escaped_object_instead_of_unescaping() -> None:
    raw = '{\\"answer\\": \\"B\\"}'
    recovered = repair.recover_exploratory_answer(raw)
    assert recovered["status"] == "outer_object_parse_failure"
    assert recovered["candidates"] == []


def test_exploratory_arm_fails_closed_on_multiple_keys_even_equal_values() -> None:
    raw = '{"answer":"A","result":"A"}'
    recovered = repair.recover_exploratory_answer(raw)
    assert recovered == {
        "status": "ambiguous_multiple_explicit_keys",
        "candidates": [],
        "key_occurrences": 2,
        "outer_parse_mode": "exact_json_object_preserving_duplicate_keys",
    }


def test_audit_regression_never_mines_answer_key_inside_reasoning_string() -> None:
    raw = '{"reasoning":"ignore {\\"answer\\":16}"}'
    recovered = repair.recover_exploratory_answer(raw)
    assert recovered == {
        "status": "no_explicit_outer_key",
        "candidates": [],
        "key_occurrences": 0,
        "outer_parse_mode": "exact_json_object_preserving_duplicate_keys",
    }


def test_audit_regression_never_mines_key_inside_any_string_or_object_value() -> None:
    for raw in (
        '{"payload":"{\\"result\\":\\"A\\"}"}',
        '{"payload":{"choice":"B"}}',
        '{"reasoning":"the literal text \\"final_answer\\": \\"C\\" is not output"}',
    ):
        recovered = repair.recover_exploratory_answer(raw)
        assert recovered["status"] == "no_explicit_outer_key"
        assert recovered["candidates"] == []


def test_missing_final_outer_curly_repair_keeps_reasoning_atomic() -> None:
    raw = '{"reasoning":"ignore {\\"answer\\":16}","final_answer":"C"'
    recovered = repair.recover_exploratory_answer(raw)
    assert recovered == {
        "status": "unique_explicit_key_scalar_candidate",
        "candidates": ["C"],
        "key_occurrences": 1,
        "matched_key": "final_answer",
        "outer_parse_mode": "missing_final_outer_curly_only",
    }


def test_malformed_outer_repair_rejects_trailing_comma_or_incomplete_value() -> None:
    for raw in ('{"answer":"A",', '{"answer":', '{"answer":"A" garbage'):
        recovered = repair.recover_exploratory_answer(raw)
        assert recovered["status"] == "outer_object_parse_failure"
        assert recovered["candidates"] == []


def test_exploratory_arm_never_mines_free_reasoning_or_unkeyed_numbers() -> None:
    raw = '{"reasoning":"therefore the final result is 16"}'
    assert repair.recover_exploratory_answer(raw)["status"] == "no_explicit_outer_key"


def test_exploratory_arm_rejects_long_or_container_values() -> None:
    long_value = "x" * (repair.MAX_EXPLORATORY_VALUE_CHARS + 1)
    assert (
        repair.recover_exploratory_answer(json.dumps({"answer": long_value}))["status"]
        == "explicit_key_value_not_strict_or_out_of_bounds"
    )
    assert (
        repair.recover_exploratory_answer('{"answer":{"nested":"A"}}')["status"]
        == "explicit_key_value_not_strict_or_out_of_bounds"
    )


def test_exploratory_protected_and_image_rows_never_parse() -> None:
    row = _row(None, '{"answer":"A"}')
    for route, protected, expected in (
        ("deterministic", True, "protected_by_source_union"),
        ("image_judge", False, "image_judge_route"),
    ):
        decision = repair.decide_row(
            row,
            evaluation_route=route,
            protected_by_source_union=protected,
            arm="exploratory_explicit_key_scalar",
        )
        assert decision["action"] == "preserve_base_exact_bytes"
        assert decision["reason"] == expected


def test_model_and_outcome_tamper_are_rejected() -> None:
    with pytest.raises(repair.RepairError, match="model closure"):
        repair._assert_no_outcome_or_mixed_model(
            {"model": "Qwen/Qwen3.5-27B", "generation": {"gold_access": False}},
            "tamper",
        )
    with pytest.raises(repair.RepairError, match="forbidden field"):
        repair._assert_no_outcome_or_mixed_model(
            {"model": repair.MODEL, "score": 1, "generation": {"gold_access": False}},
            "tamper",
        )
    with pytest.raises(repair.RepairError, match="non-false gold_access"):
        repair._assert_no_outcome_or_mixed_model(
            {"model": repair.MODEL, "generation": {"gold_access": True}},
            "tamper",
        )


@pytest.mark.skipif(
    not repair.RULE_FREEZE_PATH.is_file(), reason="rule freeze not materialized yet"
)
def test_actual_rule_freeze_closes_code_profile_and_tests() -> None:
    report = repair.verify_rule_freeze()
    assert report["status"] == "development_rule_freeze_verified"
    freeze = json.loads(repair.RULE_FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["chronology"]["historical_residual_outcomes_known_before_design"] is True
    assert freeze["chronology"]["blind_claim"] is False
    assert freeze["chronology"]["preregistered_claim"] is False
    assert freeze["chronology"]["candidate_output_absent_at_freeze"] is True


@pytest.mark.skipif(
    not repair.DEFAULT_OUTPUT.is_dir(), reason="candidate output not materialized yet"
)
def test_actual_candidate_is_deterministic_unscored_and_preserves_protected_bytes() -> None:
    report = repair.verify_output()
    assert report["status"] == "development_candidate_verified_unscored_not_evaluated"
    manifest = json.loads(
        (repair.DEFAULT_OUTPUT / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["chronology"]["post_score_motivated"] is True
    assert manifest["chronology"]["blind_claim"] is False
    assert manifest["chronology"]["preregistered_claim"] is False
    assert manifest["runtime_outcome_access"] is False
    assert manifest["model_closure"] == [repair.MODEL]
    assert set(manifest["base_variants"]) == set(repair.BASE_VARIANTS)
    for base_variant in repair.BASE_VARIANTS:
        assert set(manifest["base_variants"][base_variant]["arms"]) == set(repair.ARMS)
        for arm in repair.ARMS:
            assert manifest["base_variants"][base_variant]["arms"][arm][
                "preservation"
            ] == {
                "source_union_rows": 156,
                "source_union_exact_base_line_bytes": 156,
                "image_judge_rows": 97,
                "image_judge_exact_base_line_bytes": 97,
            }


@pytest.mark.skipif(
    not repair.DEFAULT_OUTPUT.is_dir(), reason="candidate output not materialized yet"
)
def test_candidate_byte_tamper_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "candidate_output"
    shutil.copytree(repair.DEFAULT_OUTPUT, copied)
    solver = copied / (
        "on_v1_2_primary_240__exploratory_explicit_key_scalar__candidate_solver.jsonl"
    )
    solver.write_bytes(solver.read_bytes() + b"\n")
    with pytest.raises(repair.RepairError, match="candidate artifact mismatch"):
        repair.verify_output(copied)
