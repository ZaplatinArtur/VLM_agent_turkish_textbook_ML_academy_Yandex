from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import selector


ROOT = Path(__file__).resolve().parent


def _observable(*, route: str = "deterministic", answer_type: str = "choice") -> dict:
    return {
        "answer_type": answer_type,
        "option_count": 5 if answer_type == "choice" else None,
        "evaluation_route": route,
        "subject": "Math",
        "grade": "12",
        "question_char_count": 120,
        "ocr_char_count": 110,
        "image_count": 1,
        "image_widths": [1200],
        "image_heights": [1600],
    }


def _generation(
    upstream_role: str,
    *,
    finish_reason: str = "stop",
    error: str | None = None,
) -> dict:
    return {
        "finish_reason": finish_reason,
        "error": error,
        "forced_answer": False,
        "input_tokens": 100,
        "output_tokens": 20,
        "call_count": 1,
        "temperature": 0.0,
        "seed": 7,
        "prompt_version": f"{upstream_role}_profile",
        "upstream_artifact_sha256": selector.UPSTREAM_SHA256[upstream_role],
        "gold_access": False,
        "score_or_outcome_access": False,
        "known_error_memory_access": False,
        "source_access": False,
    }


def _candidate(answer: str | None, upstream_role: str) -> dict:
    return {
        "available": answer is not None,
        "model": selector.MODEL,
        "final_answer": answer,
        "generation": _generation(
            upstream_role,
            finish_reason="stop" if answer is not None else "missing",
        ),
    }


def _batch(
    role: str,
    *,
    final: str | None,
    votes: list[str | None],
) -> dict:
    assert len(votes) == 8
    return {
        "final": _candidate(final, role),
        "raw_votes": [_candidate(answer, role) for answer in votes],
    }


def _row(
    *,
    opaque_id: str = "opaque-001",
    anchor: str | None = "A",
    v4: str | None = "B",
    v5: str | None = "B",
    p8_final: str | None = "B",
    p8_reasoning_final: str | None = "B",
    p8_votes: list[str | None] | None = None,
    p8_reasoning_votes: list[str | None] | None = None,
    route: str = "deterministic",
    answer_type: str = "choice",
) -> dict:
    return {
        "schema_version": selector.POOL_ROW_SCHEMA,
        "opaque_id": opaque_id,
        "observable": _observable(route=route, answer_type=answer_type),
        "anchor": _candidate(anchor, "active_crop_v2"),
        "routers": {
            "v4": _candidate(v4, "native_thinking_math_router_v4"),
            "v5": _candidate(v5, "native_thinking_math_router_v5"),
        },
        "parallel_batches": {
            "parallel8_v1": _batch(
                "parallel8_v1",
                final=p8_final,
                votes=p8_votes or ["B"] * 8,
            ),
            "parallel8_reasoning_first_v2": _batch(
                "parallel8_reasoning_first_v2",
                final=p8_reasoning_final,
                votes=p8_reasoning_votes or ["B"] * 8,
            ),
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_upstream_pins_to_fixtures(monkeypatch, root: Path) -> dict[str, dict[str, str]]:
    descriptors: dict[str, dict[str, str]] = {}
    for role in selector.UPSTREAM_SHA256:
        path = root / "upstream" / f"{role}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{role}\n", encoding="utf-8")
        digest = _sha(path)
        monkeypatch.setitem(selector.UPSTREAM_SHA256, role, digest)
        descriptors[role] = {
            "path": f"upstream/{role}.jsonl",
            "sha256": digest,
        }
    return descriptors


def _pool_manifest(
    root: Path,
    rows: list[dict],
    upstream: dict[str, dict[str, str]],
    *,
    freeze_sha: str,
    profile_sha: str,
) -> Path:
    pool = root / "candidate_pool.jsonl"
    pool.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest = root / "pool_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": selector.POOL_MANIFEST_SCHEMA,
            "pool_id": "fixture_pool_v1",
            "created_after_selector_freeze": True,
            "created_before_evaluation": True,
            "rows": len(rows),
            "model_closure": [selector.MODEL],
            "selector_freeze_sha256": freeze_sha,
            "selector_profile_sha256": profile_sha,
            "source_union_authority": {
                "aggregate_sha256": selector.SOURCE_AGGREGATE_SHA256,
                "source_union_sha256": selector.SOURCE_UNION_SHA256,
                "source_union_size": selector.SOURCE_UNION_SIZE,
            },
            "upstream_artifacts": upstream,
            "candidate_pool": {
                "path": "candidate_pool.jsonl",
                "sha256": _sha(pool),
            },
            "access_attestations": {
                "gold_access": False,
                "reference_answer_access": False,
                "benchmark_score_access": False,
                "benchmark_correctness_access": False,
                "benchmark_outcome_access": False,
                "judge_access": False,
                "known_error_memory_access": False,
                "official_or_web_source_access_for_candidate_generation": False,
            },
        },
    )
    return manifest


def test_profile_is_exact_and_records_correlation_caveat() -> None:
    assert selector.load_profile(ROOT / "profile.json") == selector.EXPECTED_PROFILE
    provenance = selector.EXPECTED_PROFILE["candidate_provenance"]
    assert provenance["independence_claim"] == "not_four_fully_independent_systems"
    assert selector.EXPECTED_PROFILE["algorithm"]["primary_arm"]["raw_vote_threshold"] == 13


def test_primary_requires_at_least_13_of_16_raw_votes() -> None:
    thirteen = _row(
        p8_votes=["B"] * 8,
        p8_reasoning_votes=["B"] * 5 + ["C"] * 3,
    )
    twelve = _row(
        p8_votes=["B"] * 8,
        p8_reasoning_votes=["B"] * 4 + ["C"] * 4,
    )

    accepted = selector.select_row(thirteen, frozenset())
    rejected = selector.select_row(twelve, frozenset())

    assert accepted["primary"]["action"] == "propose_challenger"
    assert accepted["primary"]["selected_answer"] == "B"
    assert accepted["primary"]["raw_parallel_support"] == 13
    assert rejected["primary"]["action"] == "preserve_anchor"
    assert rejected["primary"]["selected_answer"] == "A"
    assert rejected["primary"]["raw_parallel_support"] == 12


def test_primary_router_disagreement_or_invalid_answer_preserves_anchor() -> None:
    disagreement = selector.select_row(_row(v5="C"), frozenset())
    invalid = selector.select_row(_row(v5="not-a-choice"), frozenset())
    missing = selector.select_row(_row(v4=None), frozenset())

    assert disagreement["primary"]["reason"] == "router_disagreement"
    assert invalid["primary"]["reason"] == "router_answer_missing_or_invalid"
    assert missing["primary"]["reason"] == "router_answer_missing_or_invalid"
    assert {value["primary"]["selected_answer"] for value in (disagreement, invalid, missing)} == {"A"}


def test_secondary_is_separately_preregistered_four_final_unanimity_arm() -> None:
    accepted = selector.select_row(_row(p8_final="B", p8_reasoning_final="B"), frozenset())
    rejected = selector.select_row(_row(p8_final="B", p8_reasoning_final="C"), frozenset())

    assert accepted["secondary"]["action"] == "propose_challenger"
    assert accepted["secondary"]["selected_answer"] == "B"
    assert rejected["secondary"]["action"] == "preserve_anchor"
    assert rejected["secondary"]["reason"] == "four_final_answers_not_unanimous"


def test_source_union_is_only_a_safety_veto_and_never_selects_challenger() -> None:
    row = _row(opaque_id="protected-opaque")
    output = selector.select_row(row, frozenset({"protected-opaque"}))

    assert output["protected_by_source_union"] is True
    assert output["primary"]["selected_answer"] == row["anchor"]["final_answer"]
    assert output["secondary"]["selected_answer"] == row["anchor"]["final_answer"]
    assert output["primary"]["reason"] == "protected_by_pinned_source_union"


def test_image_judge_route_is_always_byte_preserved() -> None:
    row = _row(route="image_judge")
    output = selector.select_row(row, frozenset())

    assert output["primary"]["action"] == "preserve_anchor"
    assert output["secondary"]["action"] == "preserve_anchor"
    assert output["primary"]["selected_answer"] == row["anchor"]["final_answer"]
    assert output["primary"]["reason"] == "image_judge_route_is_byte_preserved"


def test_opaque_id_rename_cannot_change_quality_decision_outside_protected_scope() -> None:
    first = selector.select_row(_row(opaque_id="opaque-A"), frozenset())
    second = selector.select_row(_row(opaque_id="opaque-renamed"), frozenset())
    first.pop("opaque_id")
    second.pop("opaque_id")
    assert first == second


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda row: row["anchor"].update({"correctness": True}),
            "unknown=\\['correctness'\\]",
        ),
        (
            lambda row: row.update({"task_id_overrides": ["opaque-001"]}),
            "unknown=\\['task_id_overrides'\\]",
        ),
        (
            lambda row: row["routers"]["v4"]["generation"].update({"gold_access": True}),
            "gold_access",
        ),
        (
            lambda row: row["routers"]["v4"].update({"model": "Qwen/Qwen3.5-27B"}),
            "pure Qwen3.5-9B",
        ),
        (
            lambda row: row["routers"]["v4"].update({"tool_calls": []}),
            "unknown=\\['tool_calls'\\]",
        ),
    ],
)
def test_forbidden_foreign_or_task_specific_signals_fail_closed(mutate, message: str) -> None:
    row = _row()
    mutate(row)
    with pytest.raises(selector.SelectorError, match=message):
        selector.select_row(row, frozenset())


def test_upstream_profile_mismatch_fails_closed_before_any_patch() -> None:
    row = _row()
    row["routers"]["v4"]["generation"]["upstream_artifact_sha256"] = "0" * 64
    with pytest.raises(selector.SelectorError, match="upstream profile mismatch"):
        selector.select_row(row, frozenset())


def test_source_union_projection_is_schema_size_and_sha_bound(monkeypatch) -> None:
    projection = [
        {
            "task_id": "opaque-A",
            "answer_sha256": "1" * 64,
            "owner_stage": "source_v1",
        },
        {
            "task_id": "opaque-B",
            "answer_sha256": "2" * 64,
            "owner_stage": "source_v3",
        },
    ]
    digest = hashlib.sha256(selector._canonical_json(projection, newline=False)).hexdigest()
    monkeypatch.setattr(selector, "SOURCE_UNION_SIZE", 2)
    monkeypatch.setattr(selector, "SOURCE_UNION_SHA256", digest)
    aggregate = {
        "schema_version": selector.SOURCE_AGGREGATE_SCHEMA,
        "source_union": {
            "size": 2,
            "sha256": digest,
            "latest_stage_owner_projection": projection,
            "answer_conflicts": 0,
        },
    }
    assert selector._validate_source_union_object(aggregate) == frozenset(
        {"opaque-A", "opaque-B"}
    )

    aggregate["source_union"]["latest_stage_owner_projection"][0]["task_id"] = "tampered"
    with pytest.raises(selector.SelectorError, match="projection bytes"):
        selector._validate_source_union_object(aggregate)


def test_candidate_pool_is_freeze_profile_upstream_and_evaluation_bound(
    tmp_path: Path, monkeypatch
) -> None:
    freeze_sha = "a" * 64
    profile_sha = "b" * 64
    upstream = _patch_upstream_pins_to_fixtures(monkeypatch, tmp_path)
    manifest = _pool_manifest(
        tmp_path,
        [_row()],
        upstream,
        freeze_sha=freeze_sha,
        profile_sha=profile_sha,
    )
    loaded, rows, pool_sha = selector.load_candidate_pool(
        manifest,
        freeze_sha256=freeze_sha,
        profile_sha256=profile_sha,
    )
    assert loaded["created_before_evaluation"] is True
    assert len(rows) == 1
    assert pool_sha == _sha(tmp_path / "candidate_pool.jsonl")

    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["access_attestations"]["benchmark_outcome_access"] = True
    _write_json(manifest, value)
    with pytest.raises(selector.SelectorError, match="benchmark_outcome_access"):
        selector.load_candidate_pool(
            manifest,
            freeze_sha256=freeze_sha,
            profile_sha256=profile_sha,
        )


def test_pool_descriptors_reject_traversal_hash_tamper_and_wrong_freeze(
    tmp_path: Path, monkeypatch
) -> None:
    freeze_sha = "a" * 64
    profile_sha = "b" * 64
    upstream = _patch_upstream_pins_to_fixtures(monkeypatch, tmp_path)
    manifest = _pool_manifest(
        tmp_path,
        [_row()],
        upstream,
        freeze_sha=freeze_sha,
        profile_sha=profile_sha,
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["candidate_pool"]["path"] = "../candidate_pool.jsonl"
    _write_json(manifest, value)
    with pytest.raises(selector.SelectorError, match="traversal-free"):
        selector.load_candidate_pool(
            manifest,
            freeze_sha256=freeze_sha,
            profile_sha256=profile_sha,
        )

    manifest = _pool_manifest(
        tmp_path,
        [_row()],
        upstream,
        freeze_sha=freeze_sha,
        profile_sha=profile_sha,
    )
    (tmp_path / "upstream" / "active_crop_v2.jsonl").write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(selector.SelectorError, match="SHA-256 mismatch"):
        selector.load_candidate_pool(
            manifest,
            freeze_sha256=freeze_sha,
            profile_sha256=profile_sha,
        )

    with pytest.raises(selector.SelectorError, match="freeze pin mismatch"):
        selector.load_candidate_pool(
            manifest,
            freeze_sha256="c" * 64,
            profile_sha256=profile_sha,
        )


def test_preregistered_freeze_rejects_profile_code_or_test_tamper(tmp_path: Path) -> None:
    report = selector.verify_preregistered_freeze(ROOT)
    assert report["status"] == "preregistered_freeze_verified"

    for filename in (
        "profile.json",
        "selector.py",
        "test_selector.py",
        "PREREGISTERED_FREEZE.json",
    ):
        (tmp_path / filename).write_bytes((ROOT / filename).read_bytes())
    (tmp_path / "profile.json").write_bytes((ROOT / "profile.json").read_bytes() + b"\n")
    with pytest.raises(selector.SelectorError, match="SHA-256 mismatch"):
        selector.verify_preregistered_freeze(tmp_path)


def test_real_protected_source_union_matches_preregistered_authority_after_freeze() -> None:
    selector.verify_preregistered_freeze(ROOT)
    opaque_ids, authority = selector.load_protected_source_union(ROOT)
    assert len(opaque_ids) == 156
    assert authority["aggregate_sha256"] == selector.SOURCE_AGGREGATE_SHA256
    assert authority["source_union_sha256"] == selector.SOURCE_UNION_SHA256


def test_output_writer_records_proposals_without_any_quality_outcome(tmp_path: Path) -> None:
    outputs = [
        selector.select_row(_row(), frozenset()),
        selector.select_row(_row(opaque_id="opaque-002", route="image_judge"), frozenset()),
    ]
    report = selector._write_outputs(
        outputs=outputs,
        output_dir=tmp_path / "output",
        freeze_report={
            "freeze_sha256": "a" * 64,
            "artifacts": {
                "profile": {"sha256": "b" * 64},
                "selector_code": {"sha256": "c" * 64},
                "tests": {"sha256": "d" * 64},
            },
        },
        pool_sha256="e" * 64,
        source_authority={
            "aggregate_sha256": selector.SOURCE_AGGREGATE_SHA256,
            "source_union_sha256": selector.SOURCE_UNION_SHA256,
        },
    )
    manifest = json.loads(
        (tmp_path / "output" / "selector_manifest.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "selection_proposals_frozen_without_evaluation"
    assert manifest["artifact_kind"] == "gold_blind_patch_proposals_not_a_scored_solver"
    assert manifest["image_judge_preservation"]["primary_changes"] == 0
    assert "accuracy" not in manifest
    assert "correctness" not in manifest
    assert "overall" not in manifest
