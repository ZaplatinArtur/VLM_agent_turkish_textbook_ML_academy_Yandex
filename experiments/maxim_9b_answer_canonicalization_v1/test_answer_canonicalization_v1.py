from __future__ import annotations

import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import answer_canonicalization_v1 as canon  # noqa: E402


def _row(answer: object, raw_response: str = "opaque raw response") -> dict:
    return {
        "task_id": "synthetic",
        "model": canon.MODEL,
        "final_answer": answer,
        "raw_response": raw_response,
        "generation": {"gold_access": False},
    }


def _observable(answer_type: str, question: str) -> dict:
    return {
        "schema_version": canon.PROJECTION_SCHEMA,
        "row_index": 0,
        "task_id": "synthetic",
        "question": question,
        "answer_type": answer_type,
        "subject": "Math",
    }


def test_profile_is_exact_post_score_and_not_blind_or_preregistered() -> None:
    profile = canon._profile()
    assert profile == canon._expected_profile()
    chronology = profile["chronology"]
    assert chronology["historical_240_score_and_task_outcomes_known_before_design"] is True
    assert chronology["post_score_motivated"] is True
    assert chronology["blind_claim"] is False
    assert chronology["preregistered_claim"] is False
    assert profile["scope"]["task_id_lists_or_rules"] is False
    assert profile["scope"]["scorer_or_gold_edits"] is False


def test_observable_projection_is_exact_allowlist_and_outcome_free() -> None:
    rows, raw = canon._load_projection()
    assert len(rows) == len(raw) == 274
    assert len({row["task_id"] for row in rows}) == 274
    expected_keys = {"schema_version", "row_index", "task_id", "question", "answer_type", "subject"}
    for index, row in enumerate(rows):
        assert set(row) == expected_keys
        assert row["row_index"] == index
        canon._assert_runtime_clean(row, f"projection {index}")
    manifest = json.loads(canon.PROJECTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["source_benchmark_sha256"] == canon.BENCHMARK["sha256"]
    assert manifest["runtime_outcome_access"] is False
    assert manifest["projection"]["sha256"] == canon.OBSERVABLE_PROJECTION_SHA256


def test_authorities_close_274_order_route_source_and_9b_model() -> None:
    bound = canon._load_authorities()
    assert len(bound["base_rows"]) == len(bound["order"]) == 274
    assert len(set(bound["order"])) == 274
    assert Counter(bound["routes"]) == {"deterministic": 177, "image_judge": 97}
    assert len(bound["protected"]) == 156
    assert {row["model"] for row in bound["base_rows"]} == {canon.MODEL}


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Ａ", "A"),
        ("ｂ", "B"),
        ("Ⓒ", "C"),
        ("Ⓓ", "D"),
        ("Ｅ", "E"),
    ],
)
def test_primary_nfkc_choice_maps_only_whole_single_choice_codepoint(
    value: str, expected: str
) -> None:
    assert canon.normalize_choice_nfkc(value, "choice") == expected


@pytest.mark.parametrize("value", ["A", "a", "AB", " Α ", "", "D."])
def test_primary_nfkc_choice_fails_closed_without_compatibility_mapping(value: str) -> None:
    assert canon.normalize_choice_nfkc(value, "choice") is None


@pytest.mark.parametrize(
    "value, expected",
    [("Α", "A"), ("В", "B"), ("с", "C"), ("Ε", "E")],
)
def test_exploratory_curated_choice_map_is_single_codepoint_and_frozen(
    value: str, expected: str
) -> None:
    assert canon.normalize_choice_curated(value, "choice") == expected


def test_choice_normalizers_require_observable_choice_contract() -> None:
    assert canon.normalize_choice_nfkc("Ａ", "short_text") is None
    assert canon.normalize_choice_curated("А", "numeric") is None
    assert "D" not in set(canon.CURATED_CHOICE_CONFUSABLES.values())


@pytest.mark.parametrize(
    "answer, question, expected",
    [
        ("1/2", "Sonucu yüzde olarak yazınız.", "50%"),
        ("1/8", "Express the result as a percentage.", "12.5%"),
        ("3/4", "Tuliskan dalam bentuk persen.", "75%"),
        ("50%", "Sonucu kesir olarak yazınız.", "1/2"),
        ("12,5%", "Write the answer as a fraction.", "1/8"),
    ],
)
def test_fraction_percent_conversion_requires_explicit_observable_output_contract(
    answer: str, question: str, expected: str
) -> None:
    assert canon.normalize_fraction_percent(answer, "short_text", question) == expected


def test_fraction_percent_fails_closed_on_image_placeholder_bare_symbol_or_ambiguous_contract() -> None:
    assert canon.normalize_fraction_percent("1/2", "short_text", "(soru görselde)") is None
    assert canon.normalize_fraction_percent("1/2", "short_text", "Oran % ile ilgilidir") is None
    assert canon.normalize_fraction_percent(
        "1/2", "short_text", "Write as a percentage and as a fraction"
    ) is None
    assert canon.normalize_fraction_percent("1/3", "short_text", "As a percentage") is None
    assert canon.normalize_fraction_percent("1/2", "numeric", "As a percentage") is None


def test_invalid_or_json_like_top_level_answer_is_left_to_separate_successor() -> None:
    raw = '{"final_answer":"C"}'
    row = _row('{"reasoning":"truncated"', raw)
    for arm in canon.ARMS:
        result = canon.decide_row(
            row,
            _observable("choice", "Choose A, B, C, D or E"),
            evaluation_route="deterministic",
            protected_by_source_union=False,
            arm=arm,
        )
        assert result == {
            "action": "preserve_base_exact_bytes",
            "reason": "invalid_or_absent_answer_owned_by_separate_explicit_json_successor",
        }
        assert row["raw_response"] == raw


def test_protected_and_image_rows_always_preserve_before_any_canonicalizer() -> None:
    row = _row("Ａ")
    observable = _observable("choice", "Choose one option")
    protected = canon.decide_row(
        row,
        observable,
        evaluation_route="deterministic",
        protected_by_source_union=True,
        arm="choice_nfkc_only",
    )
    image = canon.decide_row(
        row,
        observable,
        evaluation_route="image_judge",
        protected_by_source_union=False,
        arm="choice_nfkc_only",
    )
    assert protected["reason"] == "protected_by_source_union"
    assert image["reason"] == "image_judge_route"


def test_no_task_id_literal_rule_and_no_raw_response_parser() -> None:
    source = inspect.getsource(canon)
    assert "val_" not in source
    assert "raw_response" not in inspect.getsource(canon.decide_row)
    assert canon._expected_profile()["division_of_responsibility"][
        "this_experiment_reads_or_parses_raw_response"
    ] is False


def test_actual_frozen_rules_have_zero_safe_candidates_before_any_score() -> None:
    synthetic_rule_sha = "0" * 64
    counts = {
        arm: canon._plan_candidate(synthetic_rule_sha, arm)["counts"]["canonicalized_rows"]
        for arm in canon.ARMS
    }
    assert counts == {
        "choice_nfkc_only": 0,
        "choice_curated_confusable_exploratory": 0,
        "fraction_percent_explicit_question_contract_exploratory": 0,
    }


def test_runtime_outcome_or_model_tamper_fails_closed() -> None:
    with pytest.raises(canon.CanonicalizationError, match="forbidden runtime field"):
        canon._assert_runtime_clean({"score": 1}, "tamper")
    with pytest.raises(canon.CanonicalizationError, match="model closure"):
        canon._assert_runtime_clean({"model": "Qwen/Qwen3.5-27B"}, "tamper")
    with pytest.raises(canon.CanonicalizationError, match="gold_access"):
        canon._assert_runtime_clean({"gold_access": True}, "tamper")


@pytest.mark.skipif(not canon.RULE_FREEZE_PATH.is_file(), reason="rule freeze not written yet")
def test_materialized_rule_freeze_verifies_and_predates_candidates() -> None:
    report = canon.verify_rule_freeze()
    assert report["status"] == "development_rule_freeze_verified"
    freeze = json.loads(canon.RULE_FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["chronology"]["historical_240_score_and_task_outcomes_known_before_design"]
    assert freeze["chronology"]["candidate_output_absent_at_freeze"]
    assert freeze["chronology"]["new_arm_scores_absent_at_freeze"]


@pytest.mark.skipif(not canon.DEFAULT_OUTPUT.is_dir(), reason="candidates not built yet")
def test_materialized_candidates_verify_unscored_with_exact_protected_bytes() -> None:
    report = canon.verify_output()
    assert report["status"] == "candidate_outputs_verified_unscored_not_evaluated"
    assert report["counts"] == {arm: 0 for arm in canon.ARMS}
    assert set(report["solver_sha256"].values()) == {canon.INPUTS["base_solver"]["sha256"]}
