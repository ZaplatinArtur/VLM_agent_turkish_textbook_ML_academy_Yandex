from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


INPUT_ROOT = Path(__file__).resolve().parent
SELECTOR_ROOT = INPUT_ROOT.parent
FROZEN = INPUT_ROOT / "frozen"

sys.path.insert(0, str(INPUT_ROOT))
sys.path.insert(0, str(SELECTOR_ROOT))
import build_pool  # noqa: E402
import selector  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_frozen_pool_rebuild_is_byte_identical(tmp_path: Path) -> None:
    rebuilt = tmp_path / "rebuilt"
    report = build_pool.build_pool(rebuilt)
    assert report["rows"] == 274
    for relative in (
        "candidate_pool.jsonl",
        "candidate_pool_v1_1.jsonl",
        "input_package_v1_1.json",
        "benchmark_order_v1_1.json",
        "evaluator_route_map_v1_1.json",
        "source_union_membership_v1_1.json",
        "row_bindings_v1_1.json",
        "evaluator_route_map.jsonl",
        "source_union_membership.jsonl",
        "row_bindings.jsonl",
        "binding_manifest.json",
        "pool_manifest.json",
        "normalization_manifest.json",
        "SHA256SUMS.txt",
    ):
        assert (rebuilt / relative).read_bytes() == (FROZEN / relative).read_bytes()
    assert build_pool.verify_frozen(rebuilt)["status"].endswith("not_evaluated")


def test_superseded_v1_loader_accepts_diagnostic_pool_shape_only() -> None:
    manifest, rows, pool_sha = selector.load_candidate_pool(
        FROZEN / "pool_manifest.json",
        freeze_sha256=build_pool.SELECTOR_FREEZE_SHA256,
        profile_sha256=build_pool.SELECTOR_PROFILE_SHA256,
    )
    protected, authority = selector.load_protected_source_union(build_pool.REPO_ROOT)
    membership = _read_jsonl(FROZEN / "source_union_membership.jsonl")
    projected = {row["opaque_id"] for row in membership if row["protected_by_source_union"]}
    assert len(rows) == manifest["rows"] == 274
    assert pool_sha == manifest["candidate_pool"]["sha256"]
    assert protected == projected
    assert len(protected) == 156
    assert authority["source_union_sha256"] == build_pool.SOURCE_UNION_PROJECTION_SHA256
    assert all(len(row["parallel_batches"]["parallel8_v1"]["raw_votes"]) == 8 for row in rows)
    assert all(
        len(row["parallel_batches"]["parallel8_reasoning_first_v2"]["raw_votes"]) == 8
        for row in rows
    )


def test_upstream_byte_tamper_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "pool"
    shutil.copytree(FROZEN, copied)
    target = copied / "upstream" / "active_crop_v2.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(selector.SelectorError, match="SHA-256 mismatch"):
        selector.load_candidate_pool(
            copied / "pool_manifest.json",
            freeze_sha256=build_pool.SELECTOR_FREEZE_SHA256,
            profile_sha256=build_pool.SELECTOR_PROFILE_SHA256,
        )


def test_row_tamper_27b_gold_and_outcome_are_rejected() -> None:
    source = _read_jsonl(FROZEN / "upstream" / "active_crop_v2.jsonl")
    source[0]["model"] = "Qwen/Qwen3.5-27B"
    with pytest.raises(build_pool.PoolBuildError, match="model is not pinned"):
        build_pool.validate_answer_source_rows(source, "tampered")

    source = _read_jsonl(FROZEN / "upstream" / "active_crop_v2.jsonl")
    source[0]["generation"]["gold_access"] = True
    with pytest.raises(build_pool.PoolBuildError, match="gold_access"):
        build_pool.validate_answer_source_rows(source, "tampered")

    source = _read_jsonl(FROZEN / "upstream" / "active_crop_v2.jsonl")
    source[0]["outcome"] = "correct"
    with pytest.raises(build_pool.PoolBuildError, match="forbidden field outcome"):
        build_pool.validate_answer_source_rows(source, "tampered")


def test_pool_schema_and_stable_projection_are_exact() -> None:
    schema = json.loads((INPUT_ROOT / "candidate_pool.schema.json").read_text(encoding="utf-8"))
    manifest = json.loads((FROZEN / "normalization_manifest.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(FROZEN / "candidate_pool.jsonl")
    assert schema["$id"] == build_pool.POOL_ROW_SCHEMA
    assert len({row["opaque_id"] for row in rows}) == 274
    assert all(row["schema_version"] == schema["$id"] for row in rows)
    assert manifest["content_projection_sha256"] == build_pool.canonical_projection_sha256(
        manifest["content_projection"]
    )
    assert manifest["artifacts"]["candidate_pool"]["sha256"] == _sha(
        FROZEN / "candidate_pool.jsonl"
    )


def _binding_fixture() -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, list[dict]]]:
    pool = _read_jsonl(FROZEN / "candidate_pool.jsonl")
    routes = _read_jsonl(FROZEN / "evaluator_route_map.jsonl")
    membership = _read_jsonl(FROZEN / "source_union_membership.jsonl")
    bindings = _read_jsonl(FROZEN / "row_bindings.jsonl")
    upstream = {
        role: _read_jsonl(FROZEN / "upstream" / f"{role}.jsonl")
        for role in build_pool.UPSTREAM
    }
    return pool, routes, membership, bindings, upstream


def test_independent_route_relabel_tamper_is_rejected() -> None:
    pool, routes, membership, bindings, upstream = _binding_fixture()
    routes[0]["evaluation_route"] = (
        "image_judge" if routes[0]["evaluation_route"] == "deterministic" else "deterministic"
    )
    with pytest.raises(build_pool.PoolBuildError, match="evaluator route relabel"):
        build_pool.validate_binding_projections(
            pool_rows=pool,
            route_rows=routes,
            membership_rows=membership,
            binding_rows=bindings,
            upstream_rows=upstream,
        )


def test_protected_candidate_swap_tamper_is_rejected() -> None:
    pool, routes, membership, bindings, upstream = _binding_fixture()
    protected = [index for index, row in enumerate(membership) if row["protected_by_source_union"]]
    left, right = protected[:2]
    pool[left]["anchor"], pool[right]["anchor"] = pool[right]["anchor"], pool[left]["anchor"]
    with pytest.raises(build_pool.PoolBuildError, match="candidate swap"):
        build_pool.validate_binding_projections(
            pool_rows=pool,
            route_rows=routes,
            membership_rows=membership,
            binding_rows=bindings,
            upstream_rows=upstream,
        )


def test_missing_extra_or_misaligned_id_tamper_is_rejected() -> None:
    pool, routes, membership, bindings, upstream = _binding_fixture()
    bindings[0]["row_index"] = 1
    with pytest.raises(build_pool.PoolBuildError, match="row index/task binding"):
        build_pool.validate_binding_projections(
            pool_rows=pool,
            route_rows=routes,
            membership_rows=membership,
            binding_rows=bindings,
            upstream_rows=upstream,
        )


def test_benchmark_sha_and_ordered_id_projection_are_pinned() -> None:
    manifest = json.loads((FROZEN / "binding_manifest.json").read_text(encoding="utf-8"))
    authority = manifest["benchmark_task_id_authority"]
    benchmark = build_pool.assert_pinned(
        build_pool.BENCHMARK_ID_AUTHORITY, "test benchmark authority"
    )
    order = build_pool.read_ordered_task_id_prefixes(benchmark, "test benchmark authority")
    assert authority["sha256"] == build_pool.BENCHMARK_ID_AUTHORITY["sha256"]
    assert authority["ordered_task_ids_sha256"] == build_pool.canonical_projection_sha256(order)
    assert len(order) == len(set(order)) == 274


def test_v11_pool_has_no_task_id_route_or_outcome_surface() -> None:
    rows = _read_jsonl(FROZEN / "candidate_pool_v1_1.jsonl")
    assert len(rows) == 274
    for index, row in enumerate(rows):
        assert set(row) == {
            "schema_version",
            "row_index",
            "anchor",
            "routers",
            "parallel_batches",
        }
        assert row["row_index"] == index
        serialized = json.dumps(row, ensure_ascii=False, sort_keys=True).casefold()
        assert '"task_id"' not in serialized
        assert '"evaluation_route"' not in serialized
        assert '"reference_answer"' not in serialized
        assert '"outcome":' not in serialized
        assert "27b" not in serialized


def test_v11_projection_binding_rejects_candidate_swap() -> None:
    pool = _read_jsonl(FROZEN / "candidate_pool_v1_1.jsonl")
    order = json.loads((FROZEN / "benchmark_order_v1_1.json").read_text(encoding="utf-8"))
    route_map = json.loads(
        (FROZEN / "evaluator_route_map_v1_1.json").read_text(encoding="utf-8")
    )
    bindings = json.loads((FROZEN / "row_bindings_v1_1.json").read_text(encoding="utf-8"))
    pool[0]["anchor"], pool[1]["anchor"] = pool[1]["anchor"], pool[0]["anchor"]
    image_ids = frozenset(
        row["task_id"] for row in route_map["rows"] if row["evaluation_route"] == "image_judge"
    )
    with pytest.raises(build_pool.PoolBuildError, match="row binding mismatch"):
        build_pool.validate_v11_artifacts(
            pool_rows=pool,
            benchmark_order=order,
            benchmark_order_sha256=_sha(FROZEN / "benchmark_order_v1_1.json"),
            route_map=route_map,
            row_bindings=bindings,
            expected_task_order=order["rows"],
            image_ids=image_ids,
        )


def test_v11_route_map_relabel_is_rejected_independently() -> None:
    pool = _read_jsonl(FROZEN / "candidate_pool_v1_1.jsonl")
    order = json.loads((FROZEN / "benchmark_order_v1_1.json").read_text(encoding="utf-8"))
    route_map = json.loads(
        (FROZEN / "evaluator_route_map_v1_1.json").read_text(encoding="utf-8")
    )
    bindings = json.loads((FROZEN / "row_bindings_v1_1.json").read_text(encoding="utf-8"))
    image_ids = frozenset(
        row["task_id"] for row in route_map["rows"] if row["evaluation_route"] == "image_judge"
    )
    route_map["rows"][0]["evaluation_route"] = (
        "image_judge"
        if route_map["rows"][0]["evaluation_route"] == "deterministic"
        else "deterministic"
    )
    with pytest.raises(build_pool.PoolBuildError, match="evaluator route authority"):
        build_pool.validate_v11_artifacts(
            pool_rows=pool,
            benchmark_order=order,
            benchmark_order_sha256=_sha(FROZEN / "benchmark_order_v1_1.json"),
            route_map=route_map,
            row_bindings=bindings,
            expected_task_order=order["rows"],
            image_ids=image_ids,
        )
