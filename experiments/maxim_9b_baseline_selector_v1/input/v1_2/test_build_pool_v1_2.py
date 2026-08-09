from __future__ import annotations

import copy
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_pool_v1_2 as pool  # noqa: E402


FROZEN = HERE / "frozen"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _fixture() -> tuple[dict, list[dict], dict, dict, dict, dict, dict[str, list[dict]]]:
    package = _json(FROZEN / "input_package_v1_2.json")
    candidates = _jsonl(FROZEN / package["candidate_pool"]["path"])
    order = _json(FROZEN / package["benchmark_order"]["path"])
    routes = _json(FROZEN / package["route_map"]["path"])
    membership = _json(FROZEN / package["source_union_membership"]["path"])
    bindings = _json(FROZEN / package["row_bindings"]["path"])
    upstream = {
        role: _jsonl(FROZEN / descriptor["path"])
        for role, descriptor in package["upstream_artifacts"].items()
    }
    return package, candidates, order, routes, membership, bindings, upstream


def test_frozen_v1_2_verifies_without_selection_or_evaluation() -> None:
    result = pool.verify_existing(FROZEN)
    assert result["status"] == "normalized_input_v1_2_verified_not_selected_not_evaluated"
    assert result["rows"] == 274
    assert result["roles"][-1] == "structural_strict_9b"


def test_v1_1_pins_remain_byte_identical() -> None:
    assert pool.v11.sha256_file(pool.BASE_PACKAGE["path"]) == pool.BASE_PACKAGE["sha256"]
    for descriptor in pool.BASE_ARTIFACTS.values():
        assert pool.v11.sha256_file(pool.BASE_FROZEN / descriptor["path"]) == descriptor["sha256"]
    for descriptor in pool.BASE_UPSTREAM.values():
        assert pool.v11.sha256_file(pool.BASE_FROZEN / descriptor["path"]) == descriptor["sha256"]


def test_authorities_are_exact_v1_1_bytes() -> None:
    package = _json(FROZEN / "input_package_v1_2.json")
    for key in ("benchmark_order", "route_map", "source_union_membership"):
        descriptor = package[key]
        assert (FROZEN / descriptor["path"]).read_bytes() == (
            pool.BASE_FROZEN / pool.BASE_ARTIFACTS[key]["path"]
        ).read_bytes()


def test_structural_projection_is_full_9b_and_honest_about_source_access() -> None:
    package, candidates, order, routes, membership, bindings, upstream = _fixture()
    assert len(candidates) == len(upstream[pool.STRUCTURAL_ROLE]) == 274
    assert package["upstream_artifacts"][pool.STRUCTURAL_ROLE]["sha256"] == pool.STRUCTURAL_SOURCE["sha256"]
    assert {row["structural"]["model"] for row in candidates} == {pool.MODEL}
    assert Counter(
        row["structural"]["generation"]["source_access"] for row in candidates
    ) == {True: 256, False: 18}
    assert all(
        row["structural"] == pool.structural_candidate(upstream[pool.STRUCTURAL_ROLE][index])
        for index, row in enumerate(candidates)
    )
    pool.validate_closure(
        pool_rows=candidates,
        benchmark_order=order,
        route_map=routes,
        source_union_membership=membership,
        row_bindings=bindings,
        upstream_rows=upstream,
    )


def test_inherited_five_role_projections_are_unchanged() -> None:
    _, _, _, _, _, bindings, _ = _fixture()
    base = _json(pool.BASE_FROZEN / pool.BASE_ARTIFACTS["row_bindings"]["path"])
    for index in range(274):
        inherited = {
            role: bindings["rows"][index]["role_projection_sha256"][role]
            for role in pool.BASE_UPSTREAM
        }
        assert inherited == base["rows"][index]["role_projection_sha256"]


def test_deterministic_rebuild_is_byte_identical(tmp_path: Path) -> None:
    rebuilt = tmp_path / "frozen"
    result = pool.build(rebuilt)
    for relative in (
        "input_package_v1_2.json",
        "candidate_pool_v1_2.jsonl",
        "row_bindings_v1_2.json",
        "benchmark_order_v1_1.json",
        "evaluator_route_map_v1_1.json",
        "source_union_membership_v1_1.json",
        "build_manifest_v1_2.json",
        "SHA256SUMS.txt",
    ):
        assert (rebuilt / relative).read_bytes() == (FROZEN / relative).read_bytes()
    assert result["input_package_v1_2_sha256"] == pool.v11.sha256_file(
        FROZEN / "input_package_v1_2.json"
    )


def test_structural_candidate_model_tamper_is_rejected() -> None:
    _, candidates, _, _, _, _, _ = _fixture()
    candidate = copy.deepcopy(candidates[0]["structural"])
    candidate["model"] = "Qwen/Qwen3.5-27B"
    with pytest.raises(pool.PoolV12Error, match="model"):
        pool.validate_candidate(candidate, pool.STRUCTURAL_ROLE, allow_source=True)


def test_structural_outcome_field_tamper_is_rejected() -> None:
    _, candidates, _, _, _, _, _ = _fixture()
    candidate = copy.deepcopy(candidates[0]["structural"])
    candidate["score"] = 1
    with pytest.raises(pool.PoolV12Error):
        pool.validate_candidate(candidate, pool.STRUCTURAL_ROLE, allow_source=True)


def test_structural_candidate_swap_is_rejected_by_source_binding() -> None:
    _, candidates, order, routes, membership, bindings, upstream = _fixture()
    tampered = copy.deepcopy(candidates)
    tampered[0]["structural"], tampered[1]["structural"] = (
        tampered[1]["structural"],
        tampered[0]["structural"],
    )
    with pytest.raises(pool.PoolV12Error, match="Structural candidate/source"):
        pool.validate_closure(
            pool_rows=tampered,
            benchmark_order=order,
            route_map=routes,
            source_union_membership=membership,
            row_bindings=bindings,
            upstream_rows=upstream,
        )


def test_route_relabel_is_rejected() -> None:
    _, candidates, order, routes, membership, bindings, upstream = _fixture()
    tampered = copy.deepcopy(routes)
    current = tampered["rows"][0]["evaluation_route"]
    tampered["rows"][0]["evaluation_route"] = (
        "image_judge" if current == "deterministic" else "deterministic"
    )
    with pytest.raises(pool.PoolV12Error, match="route authority"):
        pool.validate_closure(
            pool_rows=candidates,
            benchmark_order=order,
            route_map=tampered,
            source_union_membership=membership,
            row_bindings=bindings,
            upstream_rows=upstream,
        )


def test_source_union_membership_tamper_is_rejected() -> None:
    _, candidates, order, routes, membership, bindings, upstream = _fixture()
    tampered = copy.deepcopy(membership)
    tampered["task_ids"] = tampered["task_ids"][1:]
    with pytest.raises(pool.PoolV12Error, match="source-union authority"):
        pool.validate_closure(
            pool_rows=candidates,
            benchmark_order=order,
            route_map=routes,
            source_union_membership=tampered,
            row_bindings=bindings,
            upstream_rows=upstream,
        )


def test_structural_upstream_row_misalignment_is_rejected() -> None:
    _, candidates, order, routes, membership, bindings, upstream = _fixture()
    tampered = copy.deepcopy(upstream)
    tampered[pool.STRUCTURAL_ROLE][0], tampered[pool.STRUCTURAL_ROLE][1] = (
        tampered[pool.STRUCTURAL_ROLE][1],
        tampered[pool.STRUCTURAL_ROLE][0],
    )
    with pytest.raises(
        pool.PoolV12Error,
        match="Structural candidate/source projection mismatch|source/task misalignment",
    ):
        pool.validate_closure(
            pool_rows=candidates,
            benchmark_order=order,
            route_map=routes,
            source_union_membership=membership,
            row_bindings=bindings,
            upstream_rows=tampered,
        )


def test_upstream_byte_tamper_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "frozen"
    shutil.copytree(FROZEN, copied)
    target = copied / "upstream" / "structural_strict_9b.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(pool.PoolV12Error, match="SHA-256 mismatch"):
        pool.verify_existing(copied)


def test_path_traversal_descriptor_is_rejected() -> None:
    with pytest.raises(pool.PoolV12Error, match="unsafe descriptor"):
        pool._safe_descriptor(
            {"path": "../structural.jsonl", "sha256": "0" * 64}, "tampered"
        )


def test_schema_documents_are_valid_json() -> None:
    for name in (
        "candidate_pool_v1_2.schema.json",
        "input_package_v1_2.schema.json",
        "row_bindings_v1_2.schema.json",
    ):
        schema = _json(HERE / name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
