from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import selector_v1_2 as selector


ROOT = Path(__file__).resolve().parent
STRUCTURAL_FIXTURE_SHA = "f" * 64


def _upstream() -> dict[str, str]:
    return {**selector.base.UPSTREAM_SHA256, selector.STRUCTURAL_ROLE: STRUCTURAL_FIXTURE_SHA}


def _generation(role: str, available: bool = True) -> dict:
    return {
        "finish_reason": "stop" if available else "missing",
        "error": None,
        "forced_answer": False,
        "input_tokens": 100,
        "output_tokens": 20,
        "call_count": 1,
        "temperature": 0.0,
        "seed": 7,
        "prompt_version": f"{role}_profile",
        "upstream_artifact_sha256": _upstream()[role],
        "new_arm_gold_reference_correctness_outcome_or_judge_access": False,
        "source_access": False,
    }


def _candidate(answer: str | None, role: str) -> dict:
    return {
        "available": answer is not None,
        "model": selector.MODEL,
        "final_answer": answer,
        "generation": _generation(role, answer is not None),
    }


def _batch(role: str, final: str | None, raw: list[str | None] | None = None) -> dict:
    votes = raw if raw is not None else [final] * 8
    return {
        "final": _candidate(final, role),
        "raw_votes": [_candidate(answer, role) for answer in votes],
    }


def _row(
    *,
    anchor: str | None = "A",
    structural: str | None = "B",
    v4: str | None = "B",
    v5: str | None = "B",
    p1: str | None = "B",
    p2: str | None = "B",
    p1_raw: list[str | None] | None = None,
    p2_raw: list[str | None] | None = None,
) -> dict:
    return {
        "schema_version": selector.POOL_ROW_SCHEMA,
        "row_index": 0,
        "anchor": _candidate(anchor, "active_crop_v2"),
        "structural": _candidate(structural, selector.STRUCTURAL_ROLE),
        "routers": {
            "v4": _candidate(v4, "native_thinking_math_router_v4"),
            "v5": _candidate(v5, "native_thinking_math_router_v5"),
        },
        "parallel_batches": {
            "parallel8_v1": _batch("parallel8_v1", p1, p1_raw),
            "parallel8_reasoning_first_v2": _batch(
                "parallel8_reasoning_first_v2", p2, p2_raw
            ),
        },
    }


def _select(row: dict, *, route: str = "deterministic", protected: bool = False) -> dict:
    return selector.select_bound_row(
        row,
        task_id="authoritative-task",
        route=route,
        protected=protected,
        upstream=_upstream(),
    )


def test_profile_is_honest_and_locked_before_v1_2_evaluation() -> None:
    profile = selector.load_profile(ROOT / "profile_v1_2.json", require_ready=True)
    assert profile["chronology"] == selector.CHRONOLOGY
    assert profile["algorithm"] == selector.ALGORITHM
    assert profile["status"] == (
        "preregistered_after_historical_outcomes_known_before_v1_2_evaluation"
    )
    assert profile["authority_pins"]["status"] == "locked_before_v1_2_evaluation"
    assert "declared_source_access" in profile["group_provenance"]["structural_group"]


def test_primary_requires_unanimity_of_structural_native_and_parallel_groups() -> None:
    output = _select(_row())
    assert output["structural_challenger"] == "B"
    assert output["native_group_answer"] == "B"
    assert output["parallel_group_answer"] == "B"
    assert output["primary"]["action"] == "propose_challenger"
    assert output["primary"]["selected_answer"] == "B"
    assert output["primary"]["reason"] == "all_three_preregistered_groups_agree"


def test_native_disagreement_blocks_primary_but_exploratory_accepts_one_native() -> None:
    output = _select(_row(v4="B", v5="C"))
    assert output["native_group_answer"] is None
    assert output["primary"]["action"] == "preserve_anchor"
    assert output["primary"]["reason"] == "native_group_has_no_v4_v5_consensus"
    assert output["exploratory"]["action"] == "propose_challenger"
    assert output["exploratory"]["selected_answer"] == "B"


def test_parallel_final_disagreement_blocks_both_arms() -> None:
    output = _select(_row(p1="B", p2="C"))
    assert output["parallel_group_answer"] is None
    assert output["primary"]["action"] == "preserve_anchor"
    assert output["exploratory"]["action"] == "preserve_anchor"
    assert output["exploratory"]["reason"] == "parallel_group_has_no_final_consensus"


def test_exploratory_requires_one_native_member_on_structural_parallel_challenger() -> None:
    output = _select(_row(v4="C", v5="D"))
    assert output["primary"]["action"] == "preserve_anchor"
    assert output["exploratory"]["action"] == "preserve_anchor"
    assert output["exploratory"]["reason"] == (
        "neither_native_member_supports_structural_challenger"
    )


def test_structural_must_differ_from_anchor() -> None:
    output = _select(_row(structural="A", v4="A", v5="A", p1="A", p2="A"))
    assert output["primary"]["action"] == "preserve_anchor"
    assert output["exploratory"]["action"] == "preserve_anchor"
    assert output["primary"]["reason"] == "structural_agrees_with_anchor_no_change"


@pytest.mark.parametrize(
    ("route", "protected", "reason"),
    [
        ("image_judge", False, "image_judge_route_preserved"),
        ("deterministic", True, "protected_by_source_union"),
    ],
)
def test_authoritative_safety_gates_preserve_anchor(
    route: str, protected: bool, reason: str
) -> None:
    output = _select(_row(), route=route, protected=protected)
    assert output["primary"]["action"] == "preserve_anchor"
    assert output["exploratory"]["action"] == "preserve_anchor"
    assert output["primary"]["reason"] == reason


def test_raw_parallel_votes_are_not_an_algorithm_feature() -> None:
    first = _select(_row(p1_raw=["A"] * 8, p2_raw=["C"] * 8))
    second = _select(_row(p1_raw=["D"] * 8, p2_raw=["E"] * 8))
    assert first == second
    assert first["primary"]["action"] == "propose_challenger"


@pytest.mark.parametrize("foreign_key", ["task_id", "opaque_id", "evaluation_route"])
def test_pool_cannot_supply_identity_or_route(foreign_key: str) -> None:
    row = _row()
    row[foreign_key] = "attacker-controlled"
    with pytest.raises(selector.SelectorV12Error, match="unknown"):
        selector.validate_pool_row(row, 0, _upstream())


@pytest.mark.parametrize(
    "foreign_key",
    ["gold_answer", "reference_answer", "correctness", "outcome", "judge_verdict", "score"],
)
def test_candidate_payload_rejects_outcome_fields(foreign_key: str) -> None:
    row = _row()
    row["structural"][foreign_key] = "forbidden"
    with pytest.raises(selector.SelectorV12Error):
        selector.validate_pool_row(row, 0, _upstream())


def test_structural_profile_mismatch_fails_closed() -> None:
    row = _row()
    row["structural"]["generation"]["upstream_artifact_sha256"] = "0" * 64
    with pytest.raises(selector.SelectorV12Error, match="upstream pin mismatch"):
        selector.validate_pool_row(row, 0, _upstream())


def test_only_structural_role_may_declare_source_access() -> None:
    row = _row()
    row["structural"]["generation"]["source_access"] = True
    selector.validate_pool_row(row, 0, _upstream())
    row["routers"]["v4"]["generation"]["source_access"] = True
    with pytest.raises(selector.SelectorV12Error, match="non-structural source access"):
        selector.validate_pool_row(row, 0, _upstream())


def test_v1_1_frozen_files_remain_unchanged() -> None:
    expected = {
        "profile_v1_1.json": "260b7998e196e78db121ea0783747ae3eaa6736262200ba38d4c56ccbd8fef46",
        "selector_v1_1.py": "56816ee66d5e197e91d61a7f2de2dd2ca5af75254e8a58344d47d5aaca50b468",
        "test_selector_v1_1.py": "8177db4d77ebf1c5429ad5dc33cdd26477c74d3df8592464615daab842fa8a49",
        "PREREGISTERED_FREEZE.json": "858ef54c3bb558bdd31f8d0ead605bf3a3bcdb8816f97c0d0d93f86b1eaf4193",
    }
    for filename, digest in expected.items():
        assert selector._sha256(ROOT / filename) == digest


def test_actual_v1_2_package_strictly_loads_without_generating_arm_outputs() -> None:
    profile = selector.load_profile(ROOT / "profile_v1_2.json", require_ready=True)
    rows, ids, routes, protected, pool_sha = selector.load_input_package(
        ROOT / selector.INPUT_PACKAGE_RELATIVE_PATH,
        profile,
    )
    assert len(rows) == len(ids) == len(routes) == 274
    assert len(set(ids)) == 274
    assert routes.count("deterministic") == 177
    assert routes.count("image_judge") == 97
    assert len(protected) == 156
    assert sum(row["structural"]["generation"]["source_access"] for row in rows) == 256
    assert pool_sha == profile["authority_pins"]["candidate_pool_sha256"]


def test_v1_2_freeze_rejects_code_profile_or_test_tamper(tmp_path: Path) -> None:
    report = selector.verify_preregistered_freeze(ROOT)
    assert report["status"] == "selector_v1_2_freeze_verified"
    for filename in (
        "profile_v1_2.json",
        "selector_v1_2.py",
        "test_selector_v1_2.py",
        "PREREGISTERED_FREEZE_v1_2.json",
    ):
        (tmp_path / filename).write_bytes((ROOT / filename).read_bytes())
    (tmp_path / "selector_v1_2.py").write_bytes(
        (ROOT / "selector_v1_2.py").read_bytes() + b"\n"
    )
    with pytest.raises(selector.SelectorV12Error, match="code pin mismatch"):
        selector.verify_preregistered_freeze(tmp_path)
