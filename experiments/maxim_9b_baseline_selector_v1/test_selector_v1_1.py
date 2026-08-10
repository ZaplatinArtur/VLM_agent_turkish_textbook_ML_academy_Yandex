from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

import selector_v1_1 as selector


ROOT = Path(__file__).resolve().parent


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(selector._canonical_json(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generation(role: str, *, available: bool = True) -> dict:
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
        "upstream_artifact_sha256": selector.UPSTREAM_SHA256[role],
        "new_arm_gold_reference_correctness_outcome_or_judge_access": False,
        "source_access": False,
    }


def _candidate(answer: str | None, role: str) -> dict:
    return {
        "available": answer is not None,
        "model": selector.MODEL,
        "final_answer": answer,
        "generation": _generation(role, available=answer is not None),
    }


def _batch(role: str, final: str | None, votes: list[str | None]) -> dict:
    assert len(votes) == 8
    return {
        "final": _candidate(final, role),
        "raw_votes": [_candidate(answer, role) for answer in votes],
    }


def _row(
    index: int,
    *,
    anchor: str | None = "A",
    v4: str | None = "B",
    v5: str | None = "B",
    p8_final: str | None = "B",
    p8_reasoning_final: str | None = "B",
    p8_votes: list[str | None] | None = None,
    p8_reasoning_votes: list[str | None] | None = None,
) -> dict:
    return {
        "schema_version": selector.POOL_ROW_SCHEMA,
        "row_index": index,
        "anchor": _candidate(anchor, "active_crop_v2"),
        "routers": {
            "v4": _candidate(v4, "native_thinking_math_router_v4"),
            "v5": _candidate(v5, "native_thinking_math_router_v5"),
        },
        "parallel_batches": {
            "parallel8_v1": _batch(
                "parallel8_v1", p8_final, p8_votes or ["B"] * 8
            ),
            "parallel8_reasoning_first_v2": _batch(
                "parallel8_reasoning_first_v2",
                p8_reasoning_final,
                p8_reasoning_votes or ["B"] * 8,
            ),
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(selector._canonical_json(row) for row in rows))


def _build_package(tmp_path: Path, monkeypatch) -> tuple[Path, dict, list[dict]]:
    monkeypatch.setattr(selector, "ROW_COUNT", 3)
    monkeypatch.setattr(selector, "SOURCE_UNION_SIZE", 1)
    task_ids = ["task-A", "task-B", "task-C"]
    upstream_descriptors: dict[str, dict[str, str]] = {}
    for role in selector.UPSTREAM_ROLES:
        path = tmp_path / "upstream" / f"{role}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"outcome-free fixture {role}\n", encoding="utf-8")
        digest = _sha(path)
        monkeypatch.setitem(selector.UPSTREAM_SHA256, role, digest)
        upstream_descriptors[role] = {
            "path": f"upstream/{role}.jsonl",
            "sha256": digest,
        }

    rows = [
        _row(0, anchor="A"),
        _row(1, anchor="C", v4="D", v5="D", p8_final="D", p8_reasoning_final="D", p8_votes=["D"] * 8, p8_reasoning_votes=["D"] * 8),
        _row(2, anchor="E", v4=None, v5=None, p8_final=None, p8_reasoning_final=None, p8_votes=[None] * 8, p8_reasoning_votes=[None] * 8),
    ]
    pool_path = tmp_path / "candidate_pool_v1_1.jsonl"
    _write_jsonl(pool_path, rows)

    order_path = tmp_path / "benchmark_order_v1_1.json"
    _write_json(
        order_path,
        {
            "schema_version": selector.ORDER_SCHEMA,
            "benchmark_sha256": selector.BENCHMARK_SHA256,
            "rows": task_ids,
        },
    )
    order_sha = _sha(order_path)

    route_path = tmp_path / "evaluator_route_map_v1_1.json"
    _write_json(
        route_path,
        {
            "schema_version": selector.ROUTE_SCHEMA,
            "benchmark_sha256": selector.BENCHMARK_SHA256,
            "benchmark_order_sha256": order_sha,
            "derivation": "question_image_format_metadata_only_no_gold_score_correctness_outcome_or_judge",
            "rows": [
                {
                    "row_index": index,
                    "task_id": task_id,
                    "evaluation_route": "deterministic" if index < 2 else "image_judge",
                }
                for index, task_id in enumerate(task_ids)
            ],
        },
    )

    membership_path = tmp_path / "source_union_membership_v1_1.json"
    _write_json(
        membership_path,
        {
            "schema_version": selector.MEMBERSHIP_SCHEMA,
            "authority": {
                "aggregate_sha256": selector.SOURCE_AGGREGATE_SHA256,
                "source_union_projection_sha256": selector.SOURCE_UNION_SHA256,
                "source_union_size": 1,
            },
            "derivation": "projection_task_id_only_no_answer_score_correctness_outcome_or_judge",
            "task_ids": ["task-A"],
        },
    )

    bindings_path = tmp_path / "row_bindings_v1_1.json"
    _write_json(
        bindings_path,
        {
            "schema_version": selector.COMBINED_BINDING_SCHEMA,
            "benchmark_sha256": selector.BENCHMARK_SHA256,
            "benchmark_order_sha256": order_sha,
            "projection_contract": "sha256_of_utf8_json_sort_keys_compact_role_projection",
            "upstream_artifact_sha256": dict(selector.UPSTREAM_SHA256),
            "rows": [
                {
                    "row_index": index,
                    "task_id": task_ids[index],
                    "role_projection_sha256": {
                        role: hashlib.sha256(
                            selector._canonical_json(
                                selector._role_projection(row, role), newline=False
                            )
                        ).hexdigest()
                        for role in selector.UPSTREAM_ROLES
                    },
                }
                for index, row in enumerate(rows)
            ],
        },
    )

    package_path = tmp_path / "input_package_v1_1.json"
    package = {
        "schema_version": selector.PACKAGE_SCHEMA,
        "rows": 3,
        "benchmark_sha256": selector.BENCHMARK_SHA256,
        "created_before_new_arm_evaluation": True,
        "runtime_outcome_access": False,
        "benchmark_order": {
            "path": order_path.name,
            "sha256": order_sha,
        },
        "route_map": {
            "path": route_path.name,
            "sha256": _sha(route_path),
        },
        "source_union_membership": {
            "path": membership_path.name,
            "sha256": _sha(membership_path),
        },
        "upstream_artifacts": upstream_descriptors,
        "row_bindings": {
            "path": bindings_path.name,
            "sha256": _sha(bindings_path),
        },
        "candidate_pool": {
            "path": pool_path.name,
            "sha256": _sha(pool_path),
        },
    }
    _write_json(package_path, package)
    pins = {
        "status": "locked_before_new_arm_evaluation",
        "input_package_manifest_sha256": _sha(package_path),
        "benchmark_order_sha256": order_sha,
        "route_map_sha256": _sha(route_path),
        "source_union_membership_sha256": _sha(membership_path),
        "candidate_pool_sha256": _sha(pool_path),
        "row_bindings_sha256": _sha(bindings_path),
    }
    return package_path, pins, rows


def _repin_package(package_path: Path, pins: dict, **artifact_updates: Path) -> dict:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    key_map = {
        "route_map": "route_map_sha256",
        "source_union_membership": "source_union_membership_sha256",
        "candidate_pool": "candidate_pool_sha256",
        "row_bindings": "row_bindings_sha256",
    }
    for descriptor_key, artifact_path in artifact_updates.items():
        digest = _sha(artifact_path)
        package[descriptor_key]["sha256"] = digest
        pins[key_map[descriptor_key]] = digest
    _write_json(package_path, package)
    pins["input_package_manifest_sha256"] = _sha(package_path)
    return pins


def test_profile_discloses_historical_score_and_prior_outcomes() -> None:
    profile = selector.load_profile(ROOT / "profile_v1_1.json", require_ready=True)
    assert profile["chronology"] == selector.CHRONOLOGY_CONTRACT
    assert profile["chronology"][
        "historical_benchmark_aggregate_score_and_prior_task_outcomes_were_known_before_freeze"
    ] is True
    assert profile["status"] == (
        "preregistered_after_historical_outcomes_known_before_new_arm_evaluation"
    )
    assert profile["authority_pins"]["status"] == "locked_before_new_arm_evaluation"


def test_superseded_v1_freeze_is_explicitly_unexecuted() -> None:
    record = json.loads((ROOT / "SUPERSEDED_FREEZE_v1.json").read_text(encoding="utf-8"))
    assert record["status"] == "superseded_not_executed"
    assert record["selection_was_run"] is False
    assert record["evaluation_was_run"] is False


def test_primary_and_secondary_rules_are_fixed_without_outcome_input() -> None:
    row = _row(
        0,
        p8_votes=["B"] * 8,
        p8_reasoning_votes=["B"] * 5 + ["C"] * 3,
    )
    output = selector.select_bound_row(
        row,
        task_id="authoritative-task",
        authoritative_route="deterministic",
        protected_source_union=frozenset(),
    )
    assert output["primary"]["action"] == "propose_challenger"
    assert output["primary"]["raw_parallel_support"] == 13
    assert output["secondary"]["action"] == "propose_challenger"


def test_authoritative_route_and_source_union_veto_pool_candidates() -> None:
    row = _row(0)
    image = selector.select_bound_row(
        row,
        task_id="task-A",
        authoritative_route="image_judge",
        protected_source_union=frozenset(),
    )
    protected = selector.select_bound_row(
        row,
        task_id="task-A",
        authoritative_route="deterministic",
        protected_source_union=frozenset({"task-A"}),
    )
    assert image["primary"]["action"] == "preserve_anchor"
    assert image["primary"]["reason"] == "authoritative_image_judge_route_byte_preserved"
    assert protected["primary"]["action"] == "preserve_anchor"
    assert protected["primary"]["reason"] == "protected_by_pinned_source_union"


@pytest.mark.parametrize("foreign_key", ["opaque_id", "task_id", "evaluation_route"])
def test_pool_cannot_supply_identity_or_route(foreign_key: str) -> None:
    row = _row(0)
    row[foreign_key] = "attacker-controlled"
    with pytest.raises(selector.SelectorError, match="unknown"):
        selector.validate_pool_row(row, 0)


@pytest.mark.parametrize(
    "foreign_key",
    ["gold_answer", "reference_answer", "correctness", "outcome", "judge_verdict", "score"],
)
def test_runtime_candidate_payload_rejects_quality_outcome_fields(foreign_key: str) -> None:
    row = _row(0)
    row["anchor"][foreign_key] = "forbidden"
    with pytest.raises(selector.SelectorError):
        selector.validate_pool_row(row, 0)


def test_locked_package_loads_by_authoritative_order_route_and_bindings(
    tmp_path: Path, monkeypatch
) -> None:
    package_path, pins, _ = _build_package(tmp_path, monkeypatch)
    rows, ordered_ids, routes, protected, pool_sha = selector.load_input_package(package_path, pins)
    assert len(rows) == 3
    assert ordered_ids == ("task-A", "task-B", "task-C")
    assert routes == ("deterministic", "deterministic", "image_judge")
    assert protected == frozenset({"task-A"})
    assert pool_sha == pins["candidate_pool_sha256"]


def test_route_relabel_fails_against_preregistered_route_and_package_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    package_path, pins, _ = _build_package(tmp_path, monkeypatch)
    route_path = tmp_path / "evaluator_route_map_v1_1.json"
    route = json.loads(route_path.read_text(encoding="utf-8"))
    route["rows"][0]["evaluation_route"] = "image_judge"
    _write_json(route_path, route)
    with pytest.raises(selector.SelectorError, match="route_map file SHA-256 mismatch"):
        selector.load_input_package(package_path, pins)


def test_protected_membership_relabel_fails_against_preregistered_projection(
    tmp_path: Path, monkeypatch
) -> None:
    package_path, pins, _ = _build_package(tmp_path, monkeypatch)
    membership_path = tmp_path / "source_union_membership_v1_1.json"
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    membership["task_ids"] = ["task-B"]
    _write_json(membership_path, membership)
    with pytest.raises(selector.SelectorError, match="source_union_membership file SHA-256 mismatch"):
        selector.load_input_package(package_path, pins)


def test_candidate_swap_fails_per_upstream_row_binding_even_if_pool_is_repinned(
    tmp_path: Path, monkeypatch
) -> None:
    package_path, pins, rows = _build_package(tmp_path, monkeypatch)
    rows[0]["anchor"], rows[1]["anchor"] = rows[1]["anchor"], rows[0]["anchor"]
    pool_path = tmp_path / "candidate_pool_v1_1.jsonl"
    _write_jsonl(pool_path, rows)
    pins = _repin_package(
        package_path,
        copy.deepcopy(pins),
        candidate_pool=pool_path,
    )
    with pytest.raises(selector.SelectorError, match="violates pinned row binding"):
        selector.load_input_package(package_path, pins)


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered"])
def test_missing_extra_or_reordered_pool_rows_fail_closed(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    package_path, pins, rows = _build_package(tmp_path, monkeypatch)
    if mutation == "missing":
        rows = rows[:-1]
    elif mutation == "extra":
        rows.append(copy.deepcopy(rows[-1]))
    else:
        rows[0], rows[1] = rows[1], rows[0]
    pool_path = tmp_path / "candidate_pool_v1_1.jsonl"
    _write_jsonl(pool_path, rows)
    pins = _repin_package(
        package_path,
        copy.deepcopy(pins),
        candidate_pool=pool_path,
    )
    with pytest.raises(selector.SelectorError, match="missing or extra|missing, extra, or reordered"):
        selector.load_input_package(package_path, pins)


def test_binding_task_id_relabel_fails_authoritative_order_join(
    tmp_path: Path, monkeypatch
) -> None:
    package_path, pins, _ = _build_package(tmp_path, monkeypatch)
    binding_path = tmp_path / "row_bindings_v1_1.json"
    bindings = json.loads(binding_path.read_text(encoding="utf-8"))
    bindings["rows"][0]["task_id"] = "different-task"
    _write_json(binding_path, bindings)
    pins = _repin_package(
        package_path,
        copy.deepcopy(pins),
        row_bindings=binding_path,
    )
    with pytest.raises(selector.SelectorError, match="identity/order mismatch"):
        selector.load_input_package(package_path, pins)


def test_package_manifest_cannot_add_outcome_payload(tmp_path: Path, monkeypatch) -> None:
    package_path, pins, _ = _build_package(tmp_path, monkeypatch)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["score"] = {"overall": 238}
    _write_json(package_path, package)
    pins["input_package_manifest_sha256"] = _sha(package_path)
    with pytest.raises(selector.SelectorError, match="excluded runtime fields"):
        selector.load_input_package(package_path, pins)


def test_traversal_and_unpinned_package_manifest_fail_closed(tmp_path: Path, monkeypatch) -> None:
    package_path, pins, _ = _build_package(tmp_path, monkeypatch)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["candidate_pool"]["path"] = "../candidate_pool_v1_1.jsonl"
    _write_json(package_path, package)
    pins["input_package_manifest_sha256"] = _sha(package_path)
    with pytest.raises(selector.SelectorError, match="traversal-free"):
        selector.load_input_package(package_path, pins)

    pins["input_package_manifest_sha256"] = "0" * 64
    with pytest.raises(selector.SelectorError, match="manifest differs"):
        selector.load_input_package(package_path, pins)


def test_actual_frozen_input_package_strictly_loads_without_running_selection() -> None:
    profile = selector.load_profile(ROOT / "profile_v1_1.json", require_ready=True)
    package = ROOT / "input" / "frozen" / "input_package_v1_1.json"
    rows, ordered_ids, routes, protected, pool_sha = selector.load_input_package(
        package,
        profile["authority_pins"],
    )
    assert len(rows) == len(ordered_ids) == len(routes) == 274
    assert len(set(ordered_ids)) == 274
    assert Counter(routes) == {"deterministic": 177, "image_judge": 97}
    assert len(protected) == 156
    assert pool_sha == profile["authority_pins"]["candidate_pool_sha256"]


def test_runtime_does_not_parse_gold_bearing_source_aggregate() -> None:
    source = (ROOT / "selector_v1_1.py").read_text(encoding="utf-8")
    assert "load_protected_source_union" not in source
    assert '_read_json(path, "protected scope aggregate")' not in source
    assert "source_union_membership" in source


def test_active_v1_1_freeze_rejects_code_profile_test_or_supersession_tamper(
    tmp_path: Path,
) -> None:
    report = selector.verify_preregistered_freeze(ROOT)
    assert report["status"] == "preregistered_freeze_v1_1_verified"
    for filename in (
        "profile_v1_1.json",
        "selector_v1_1.py",
        "test_selector_v1_1.py",
        "SUPERSEDED_FREEZE_v1.json",
        "PREREGISTERED_FREEZE.json",
    ):
        (tmp_path / filename).write_bytes((ROOT / filename).read_bytes())
    (tmp_path / "selector_v1_1.py").write_bytes(
        (ROOT / "selector_v1_1.py").read_bytes() + b"\n"
    )
    with pytest.raises(selector.SelectorError, match="file SHA-256 mismatch"):
        selector.verify_preregistered_freeze(tmp_path)
