from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

import compositor_v1_2 as compositor
import selector_v1_2 as selector


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "compositor_output_v1_2"
V1_1_FROZEN_SHA256 = {
    "profile_v1_1.json": "260b7998e196e78db121ea0783747ae3eaa6736262200ba38d4c56ccbd8fef46",
    "selector_v1_1.py": "56816ee66d5e197e91d61a7f2de2dd2ca5af75254e8a58344d47d5aaca50b468",
    "test_selector_v1_1.py": "8177db4d77ebf1c5429ad5dc33cdd26477c74d3df8592464615daab842fa8a49",
    "PREREGISTERED_FREEZE.json": "858ef54c3bb558bdd31f8d0ead605bf3a3bcdb8816f97c0d0d93f86b1eaf4193",
}


def _jsonl_raw(path: Path) -> list[tuple[dict, bytes]]:
    result: list[tuple[dict, bytes]] = []
    for raw in path.read_bytes().splitlines(keepends=True):
        if raw.strip():
            result.append((json.loads(raw.decode("utf-8")), raw))
    return result


def test_profile_is_exact_and_discloses_historical_context_and_structural_source_access() -> None:
    profile = compositor.load_profile(ROOT)
    assert profile == compositor.EXPECTED_PROFILE
    assert profile["chronology"][
        "historical_benchmark_aggregate_score_and_prior_task_outcomes_were_known"
    ] is True
    assert profile["chronology"][
        "compositor_runtime_does_not_read_gold_reference_correctness_outcomes_or_judge_verdicts"
    ] is True
    assert "source_access" in profile["rules"]["structural_source_access_disclosure"]


def test_v1_1_selector_freeze_is_untouched() -> None:
    for name, digest in V1_1_FROZEN_SHA256.items():
        assert compositor._sha256(ROOT / name) == digest


def test_bound_inputs_have_exact_identity_route_membership_and_model_closure() -> None:
    bound = compositor.load_bound_inputs(ROOT)
    assert len(bound["ordered_ids"]) == len(set(bound["ordered_ids"])) == 274
    assert Counter(bound["routes"]) == {"deterministic": 177, "image_judge": 97}
    assert len(bound["protected_ids"]) == 156
    assert bound["protected_ids"].issubset(set(bound["ordered_ids"]))
    assert all(row.value["model"] == compositor.MODEL for row in bound["base_rows"])
    assert all(row.value["model"] == compositor.MODEL for row in bound["structural_rows"])


def test_selector_proposals_never_change_source_union_or_image_rows() -> None:
    bound = compositor.load_bound_inputs(ROOT)
    counts = {"primary": 0, "exploratory": 0}
    for index, proposal_raw in enumerate(bound["proposals"]):
        proposal = proposal_raw.value
        protected = proposal["task_id"] in bound["protected_ids"]
        image = bound["routes"][index] == "image_judge"
        for arm in counts:
            if proposal[arm]["action"] == "propose_challenger":
                counts[arm] += 1
                assert protected is False
                assert image is False
                assert proposal[arm]["selected_answer"] == proposal["structural_challenger"]
    assert counts == {"primary": 2, "exploratory": 3}


@pytest.mark.parametrize("arm,expected_replacements", [("primary", 2), ("exploratory", 3)])
def test_in_memory_composition_preserves_every_nonproposal_exactly(
    arm: str, expected_replacements: int
) -> None:
    bound = compositor.load_bound_inputs(ROOT)
    lines, decisions, counts = compositor._compose_arm(
        arm=arm,
        bound=bound,
        selector_manifest_sha256=compositor.SELECTOR_MANIFEST_SHA256,
    )
    assert len(lines) == len(decisions) == 274
    assert counts == Counter(
        {
            "base_passthrough_exact_bytes": 274 - expected_replacements,
            "bound_structural_selector_replacement": expected_replacements,
        }
    )
    for index, raw in enumerate(lines):
        base_raw = bound["base_rows"][index].raw_line
        proposal = bound["proposals"][index].value[arm]
        task_id = bound["ordered_ids"][index]
        if proposal["action"] == "preserve_anchor":
            assert raw == base_raw
        else:
            assert raw != base_raw
            row = json.loads(raw.decode("utf-8"))
            assert row["task_id"] == task_id
            assert row["model"] == compositor.MODEL
            assert row["final_answer"] == proposal["selected_answer"]
            assert row["final_answer"] == bound["structural_rows"][index].value["final_answer"]
            provenance = row["generation"]["baseline_selector_composition_v1_2"]
            assert provenance["arm"] == arm
            assert provenance["selector_freeze_sha256"] == compositor.SELECTOR_FREEZE_SHA256
            assert isinstance(provenance["structural_source_access"], bool)
            assert provenance["runtime_outcome_access"] is False
        if task_id in bound["protected_ids"] or bound["routes"][index] == "image_judge":
            assert raw == base_raw


def test_compositor_rejects_route_relabel_and_structural_answer_mismatch() -> None:
    bound = compositor.load_bound_inputs(ROOT)
    changed_index = next(
        index
        for index, row in enumerate(bound["proposals"])
        if row.value["primary"]["action"] == "propose_challenger"
    )
    proposal = copy.deepcopy(bound["proposals"][changed_index].value)
    proposal["authoritative_evaluation_route"] = "image_judge"
    with pytest.raises(compositor.CompositorV12Error, match="route differs"):
        compositor._validate_proposal_row(
            proposal,
            index=changed_index,
            task_id=bound["ordered_ids"][changed_index],
            route=bound["routes"][changed_index],
            protected=False,
            normalized_row=bound["normalized_rows"][changed_index],
        )

    proposal = copy.deepcopy(bound["proposals"][changed_index].value)
    proposal["primary"]["selected_answer"] = "E"
    with pytest.raises(compositor.CompositorV12Error, match="Structural challenger"):
        compositor._validate_proposal_row(
            proposal,
            index=changed_index,
            task_id=bound["ordered_ids"][changed_index],
            route=bound["routes"][changed_index],
            protected=False,
            normalized_row=bound["normalized_rows"][changed_index],
        )


def test_compositor_rejects_swapped_incoherent_or_gold_exposed_structural_row() -> None:
    bound = compositor.load_bound_inputs(ROOT)
    changed_index = next(
        index
        for index, row in enumerate(bound["proposals"])
        if row.value["primary"]["action"] == "propose_challenger"
    )
    selected = bound["proposals"][changed_index].value["primary"]["selected_answer"]
    wrong_task = copy.deepcopy(bound["structural_rows"][changed_index].value)
    wrong_task["task_id"] = "relabelled-task"
    with pytest.raises(compositor.CompositorV12Error, match="task identity"):
        compositor._validate_structural_payload(
            wrong_task, bound["ordered_ids"][changed_index], selected
        )

    wrong_answer = copy.deepcopy(bound["structural_rows"][changed_index].value)
    wrong_answer["final_answer"] = "E"
    with pytest.raises(compositor.CompositorV12Error, match="selector challenger"):
        compositor._validate_structural_payload(
            wrong_answer, bound["ordered_ids"][changed_index], selected
        )

    gold_exposed = copy.deepcopy(bound["structural_rows"][changed_index].value)
    gold_exposed["generation"]["gold_access"] = True
    with pytest.raises((compositor.CompositorV12Error, selector.base.SelectorError)):
        compositor._validate_structural_payload(
            gold_exposed, bound["ordered_ids"][changed_index], selected
        )


def test_preregistered_compositor_freeze_rejects_tamper(tmp_path: Path) -> None:
    report = compositor.verify_preregistered_freeze(ROOT)
    assert report["status"] == "compositor_v1_2_preregistered_freeze_verified"
    for filename in (
        "compositor_profile_v1_2.json",
        "compositor_v1_2.py",
        "test_compositor_v1_2.py",
        "selector_v1_2.py",
        "compositor_v1_1.py",
        "COMPOSITOR_PREREGISTERED_FREEZE_v1_2.json",
    ):
        (tmp_path / filename).write_bytes((ROOT / filename).read_bytes())
    (tmp_path / "compositor_profile_v1_2.json").write_bytes(
        (ROOT / "compositor_profile_v1_2.json").read_bytes() + b"\n"
    )
    with pytest.raises(compositor.CompositorV12Error, match="profile SHA mismatch"):
        compositor.verify_preregistered_freeze(tmp_path)


@pytest.mark.skipif(not OUTPUT.is_dir(), reason="output exists only after compositor freeze")
def test_actual_output_manifest_and_freeze_are_unscored_and_hash_bound() -> None:
    manifest_path = OUTPUT / "composition_manifest_v1_2.json"
    freeze_path = OUTPUT / "COMPOSITION_OUTPUT_FREEZE_v1_2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == compositor.MANIFEST_SCHEMA
    assert manifest["status"] == "composited_frozen_before_evaluation"
    assert manifest["runtime_outcome_access"] is False
    assert manifest["model_closure"] == [compositor.MODEL]
    assert freeze["schema_version"] == compositor.OUTPUT_FREEZE_SCHEMA
    assert freeze["status"] == "output_frozen_unscored_not_evaluated"
    assert freeze["runtime_outcome_access"] is False
    assert freeze["composition_manifest"]["sha256"] == compositor._sha256(manifest_path)
    for arm in ("primary", "exploratory"):
        for artifact in ("solver", "decisions"):
            descriptor = manifest["artifacts"][arm][artifact]
            path = OUTPUT / descriptor["path"]
            assert compositor._sha256(path) == descriptor["sha256"]
            assert freeze["artifacts"][arm][artifact] == descriptor


@pytest.mark.skipif(not OUTPUT.is_dir(), reason="output exists only after compositor freeze")
@pytest.mark.parametrize("arm,expected_replacements", [("primary", 2), ("exploratory", 3)])
def test_actual_solver_preserves_source_image_and_all_nonproposals_as_exact_bytes(
    arm: str, expected_replacements: int
) -> None:
    bound = compositor.load_bound_inputs(ROOT)
    output_rows = _jsonl_raw(OUTPUT / f"{arm}_solver.jsonl")
    decisions = _jsonl_raw(OUTPUT / f"{arm}_decisions.jsonl")
    assert len(output_rows) == len(decisions) == 274
    changed = 0
    for index, ((row, raw), (decision, _)) in enumerate(zip(output_rows, decisions)):
        task_id = bound["ordered_ids"][index]
        base_raw = bound["base_rows"][index].raw_line
        proposal = bound["proposals"][index].value[arm]
        assert row["task_id"] == task_id
        assert row["model"] == compositor.MODEL
        assert decision["task_id"] == task_id
        assert decision["row_index"] == index
        selector.base._assert_no_excluded_runtime_fields(row, f"actual {arm} row {index}")
        if proposal["action"] == "preserve_anchor":
            assert raw == base_raw
            assert decision["composition_action"] == "base_passthrough_exact_bytes"
        else:
            changed += 1
            assert raw != base_raw
            assert row["final_answer"] == proposal["selected_answer"]
            assert row["final_answer"] == bound["structural_rows"][index].value["final_answer"]
            provenance = row["generation"]["baseline_selector_composition_v1_2"]
            assert provenance["selector_output_manifest_sha256"] == compositor.SELECTOR_MANIFEST_SHA256
            assert decision["composition_action"] == "bound_structural_selector_replacement"
        if task_id in bound["protected_ids"] or bound["routes"][index] == "image_judge":
            assert raw == base_raw
    assert changed == expected_replacements
