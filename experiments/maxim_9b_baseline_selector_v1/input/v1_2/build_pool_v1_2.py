from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


INPUT_ROOT = Path(__file__).resolve().parents[1]
if str(INPUT_ROOT) not in sys.path:
    sys.path.insert(0, str(INPUT_ROOT))

import build_pool as v11  # noqa: E402


MODEL = "Qwen/Qwen3.5-9B"
EXPECTED_ROWS = 274
STRUCTURAL_ROLE = "structural_strict_9b"
POOL_ROW_SCHEMA = "maxim-9b-baseline-selector-pool-row-v1.2"
PACKAGE_SCHEMA = "maxim-9b-baseline-selector-input-package-v1.2"
BINDINGS_SCHEMA = "maxim-9b-baseline-selector-combined-row-bindings-v1.2"
MANIFEST_SCHEMA = "maxim-9b-baseline-selector-input-build-manifest-v1.2"

REPO_ROOT = v11.REPO_ROOT
BASE_FROZEN = INPUT_ROOT / "frozen"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "frozen"

BASE_PACKAGE = {
    "path": BASE_FROZEN / "input_package_v1_1.json",
    "sha256": "f2e7bdf8ea0cd8d44d073c3cc3f7a6933a98de2b032d44e1e5625e98eb869f0e",
}
BASE_ARTIFACTS: dict[str, dict[str, str]] = {
    "benchmark_order": {
        "path": "benchmark_order_v1_1.json",
        "sha256": "7140c7c01b48053f6a15a3b0113f68cad37bbb887744828b570a6eaa0447d62b",
    },
    "route_map": {
        "path": "evaluator_route_map_v1_1.json",
        "sha256": "f89ef00f95b9d83610b66948fcb11667dc927f2452b000ef62e031a1a0de26f6",
    },
    "source_union_membership": {
        "path": "source_union_membership_v1_1.json",
        "sha256": "93a1018a63e2b9dfeef841541df3b566d6bd6275471accb9f167c7c60c44416a",
    },
    "candidate_pool": {
        "path": "candidate_pool_v1_1.jsonl",
        "sha256": "b755e730c0841fc154ce83f3f82ec8a136cbb177c54b0c0233bb3e118926c1b3",
    },
    "row_bindings": {
        "path": "row_bindings_v1_1.json",
        "sha256": "ababaf1ff19b48275f5a6718177a4391984460e475ba475a40620dffab2e9aa6",
    },
}

BASE_UPSTREAM: dict[str, dict[str, str]] = {
    role: {
        "path": f"upstream/{role}.jsonl",
        "sha256": descriptor["sha256"],
    }
    for role, descriptor in v11.UPSTREAM.items()
}
STRUCTURAL_SOURCE = {
    "path": "reports/maxim_structural_evidence_rag_v1_20260803/solver.jsonl",
    "sha256": "4c73b6eb326e5790b19e14c01a79df853be23bb5f55b498ce4c58b78ebc3dff5",
}
STRUCTURAL_PROVENANCE = {
    "preparation_manifest": {
        "path": "reports/maxim_structural_evidence_rag_v1_20260803/preparation_manifest_v1.json",
        "sha256": "dfb5b48c18dcf527ca78a524ca4c51d414ffd978dc35a2d846c355351c277b82",
    },
    "determinism_verification": {
        "path": "reports/maxim_structural_evidence_rag_v1_20260803/determinism_verification_v1.json",
        "sha256": "d3f2ed019477d7b6e9dbac9054ccc7bf7e217b32f08685621fa351827950d484",
    },
    "failclosed_manifest": {
        "path": (
            "reports/maxim_structural_evidence_rag_v1_20260803/"
            "generation_remote/solver.failclosed_manifest.json"
        ),
        "sha256": "b7ce5773812cf917cb4fe9c7c1c4b6c79bb4019553e1fb9575939b671dba45d2",
    },
    "raw_solver": {
        "path": "reports/maxim_structural_evidence_rag_v1_20260803/solver.raw.jsonl",
        "sha256": "02824db2e3ad36d77a5541caeab737710d5227fff32d9c21fdb3942d1088e5ef",
    },
}

EXPECTED_BASE_PACKAGE_KEYS = {
    "schema_version",
    "rows",
    "benchmark_sha256",
    "created_before_new_arm_evaluation",
    "runtime_outcome_access",
    "benchmark_order",
    "route_map",
    "source_union_membership",
    "upstream_artifacts",
    "row_bindings",
    "candidate_pool",
}
EXPECTED_PACKAGE_KEYS = set(EXPECTED_BASE_PACKAGE_KEYS)
V12_GENERATION_KEYS = set(v11.V11_GENERATION_KEYS)


class PoolV12Error(RuntimeError):
    pass


def _raise(message: str) -> None:
    raise PoolV12Error(message)


def _assert_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        _raise(f"{label}: missing {path}")
    actual = v11.sha256_file(path)
    if actual != expected:
        _raise(f"{label}: SHA-256 mismatch expected={expected} actual={actual}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PoolV12Error(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        _raise(f"{label}: expected JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise PoolV12Error(f"{label}: invalid JSONL: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        _raise(f"{label}: every row must be an object")
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.write_bytes(v11.canonical_bytes(value))
    return v11.sha256_file(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(v11.canonical_bytes(row))
    return v11.sha256_file(path)


def _safe_descriptor(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        _raise(f"{label}: descriptor shape mismatch")
    path = value.get("path")
    sha = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        or not isinstance(sha, str)
        or len(sha) != 64
    ):
        _raise(f"{label}: unsafe descriptor")
    return {"path": path, "sha256": sha}


def _assert_no_forbidden_structure(value: Any, label: str) -> None:
    for key, item in v11._walk(value):
        lowered = key.casefold() if key is not None else ""
        if lowered in v11.FORBIDDEN_KEYS:
            _raise(f"{label}: forbidden outcome field {key}")
        if lowered == "gold_access" and item is not False:
            _raise(f"{label}: non-false gold access")
        if lowered == "new_arm_gold_reference_correctness_outcome_or_judge_access" and item is not False:
            _raise(f"{label}: non-false normalized outcome access")
        if lowered in v11.MODEL_KEYS and item not in (None, MODEL):
            _raise(f"{label}: forbidden model closure value {item!r}")
        if lowered == "inherited_27b_outputs" and item is not False:
            _raise(f"{label}: inherited 27B output marker")


def _verify_no_27b_bytes(path: Path, label: str) -> None:
    try:
        v11._assert_no_forbidden_bytes(path, label)
    except v11.PoolBuildError as exc:
        raise PoolV12Error(str(exc)) from exc


def _load_base() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
]:
    _assert_sha(BASE_PACKAGE["path"], BASE_PACKAGE["sha256"], "v1.1 base package")
    package = _read_json(BASE_PACKAGE["path"], "v1.1 base package")
    if set(package) != EXPECTED_BASE_PACKAGE_KEYS:
        _raise("v1.1 base package key set mismatch")
    if package.get("schema_version") != v11.INPUT_PACKAGE_V11_SCHEMA:
        _raise("v1.1 base package schema mismatch")
    if package.get("rows") != EXPECTED_ROWS or package.get("runtime_outcome_access") is not False:
        _raise("v1.1 base package row/outcome contract mismatch")
    if package.get("benchmark_sha256") != v11.BENCHMARK_ID_AUTHORITY["sha256"]:
        _raise("v1.1 base package benchmark pin mismatch")

    loaded: dict[str, Any] = {}
    for key, expected in BASE_ARTIFACTS.items():
        source = BASE_FROZEN / expected["path"]
        _assert_sha(source, expected["sha256"], f"v1.1 {key}")
        if package.get(key) != expected:
            _raise(f"v1.1 package descriptor changed for {key}")
        loaded[key] = (
            _read_jsonl(source, f"v1.1 {key}")
            if source.suffix == ".jsonl"
            else _read_json(source, f"v1.1 {key}")
        )

    upstream_rows: dict[str, list[dict[str, Any]]] = {}
    if package.get("upstream_artifacts") != BASE_UPSTREAM:
        _raise("v1.1 upstream descriptor set changed")
    for role, descriptor in BASE_UPSTREAM.items():
        source = BASE_FROZEN / descriptor["path"]
        _assert_sha(source, descriptor["sha256"], f"v1.1 upstream {role}")
        _verify_no_27b_bytes(source, f"v1.1 upstream {role}")
        rows = _read_jsonl(source, f"v1.1 upstream {role}")
        try:
            v11.validate_answer_source_rows(rows, f"v1.1 upstream {role}")
        except v11.PoolBuildError as exc:
            raise PoolV12Error(str(exc)) from exc
        upstream_rows[role] = rows

    return (
        package,
        loaded["candidate_pool"],
        loaded["benchmark_order"],
        loaded["route_map"],
        loaded["source_union_membership"],
        loaded["row_bindings"],
        upstream_rows,
    )


def _load_structural() -> list[dict[str, Any]]:
    source = REPO_ROOT / STRUCTURAL_SOURCE["path"]
    _assert_sha(source, STRUCTURAL_SOURCE["sha256"], "Structural final solver")
    _verify_no_27b_bytes(source, "Structural final solver")
    rows = _read_jsonl(source, "Structural final solver")
    try:
        v11.validate_answer_source_rows(rows, "Structural final solver")
    except v11.PoolBuildError as exc:
        raise PoolV12Error(str(exc)) from exc
    for label, descriptor in STRUCTURAL_PROVENANCE.items():
        _assert_sha(REPO_ROOT / descriptor["path"], descriptor["sha256"], label)
    return rows


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _source_access(row: Mapping[str, Any]) -> bool:
    generation = row.get("generation")
    if not isinstance(generation, dict):
        _raise("Structural row generation is missing")
    composition = generation.get("failclosed_composition")
    if not isinstance(composition, dict):
        _raise(f"Structural {row.get('task_id')}: failclosed composition is missing")
    chosen = composition.get("chosen_source")
    if chosen == "candidate":
        route = generation.get("route")
        if route not in {"structural_evidence", "router_page_rag_fallback"}:
            _raise(f"Structural {row.get('task_id')}: unknown candidate route {route!r}")
        return True
    if chosen == "frozen_subject_router":
        provenance = row.get("offline_provenance")
        decision = provenance.get("decision") if isinstance(provenance, dict) else None
        fallback = decision.get("chosen_source") if isinstance(decision, dict) else None
        if fallback == "page_rag":
            return True
        if fallback == "no_tools":
            return False
        _raise(f"Structural {row.get('task_id')}: unknown frozen fallback {fallback!r}")
    _raise(f"Structural {row.get('task_id')}: unknown chosen source {chosen!r}")


def structural_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    generation = row.get("generation")
    usage = row.get("usage")
    if not isinstance(generation, dict):
        _raise(f"Structural {row.get('task_id')}: generation is missing")
    if not isinstance(usage, dict):
        usage = {}
    trace = v11._trace(row)
    finish_reason = trace.get("finish_reason") if isinstance(trace, dict) else None
    if finish_reason not in {"stop", "length", "error", "missing"}:
        finish_reason = "stop"
    temperature = generation.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        temperature = None
    seed = generation.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        seed = None
    answer = row.get("final_answer")
    available = isinstance(answer, str) and bool(answer.strip()) and row.get("error") is None
    candidate = {
        "available": available,
        "model": MODEL,
        "final_answer": answer if available else None,
        "generation": {
            "finish_reason": finish_reason,
            "error": row.get("error"),
            "forced_answer": bool(row.get("forced_answer")),
            "input_tokens": _nonnegative_int(usage.get("input_tokens")),
            "output_tokens": _nonnegative_int(usage.get("output_tokens")),
            "call_count": _nonnegative_int(generation.get("call_count")),
            "temperature": temperature,
            "seed": seed,
            "prompt_version": str(row.get("prompt_version") or ""),
            "upstream_artifact_sha256": STRUCTURAL_SOURCE["sha256"],
            "new_arm_gold_reference_correctness_outcome_or_judge_access": False,
            "source_access": _source_access(row),
        },
    }
    validate_candidate(candidate, STRUCTURAL_ROLE, allow_source=True)
    return candidate


def validate_candidate(value: Any, role: str, *, allow_source: bool) -> None:
    if not isinstance(value, dict) or set(value) != {
        "available",
        "model",
        "final_answer",
        "generation",
    }:
        _raise(f"{role}: candidate shape mismatch")
    if value.get("model") != MODEL or not isinstance(value.get("available"), bool):
        _raise(f"{role}: candidate model/availability mismatch")
    answer = value.get("final_answer")
    if value["available"]:
        if not isinstance(answer, str) or not answer.strip():
            _raise(f"{role}: available candidate answer is empty")
    elif answer is not None:
        _raise(f"{role}: unavailable candidate has an answer")
    generation = value.get("generation")
    if not isinstance(generation, dict) or set(generation) != V12_GENERATION_KEYS:
        _raise(f"{role}: generation shape mismatch")
    expected_sha = (
        STRUCTURAL_SOURCE["sha256"]
        if role == STRUCTURAL_ROLE
        else BASE_UPSTREAM[role]["sha256"]
    )
    if generation.get("upstream_artifact_sha256") != expected_sha:
        _raise(f"{role}: upstream binding mismatch")
    if generation.get("new_arm_gold_reference_correctness_outcome_or_judge_access") is not False:
        _raise(f"{role}: outcome access is not false")
    if not isinstance(generation.get("source_access"), bool):
        _raise(f"{role}: source access is not boolean")
    if not allow_source and generation.get("source_access") is not False:
        _raise(f"{role}: unexpected source access")
    _assert_no_forbidden_structure(value, role)


def role_projection(row: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    if role == STRUCTURAL_ROLE:
        value = row.get("structural")
    else:
        try:
            value = v11.normalized_role_projection_v11(row, role)
        except v11.PoolBuildError as exc:
            raise PoolV12Error(str(exc)) from exc
    if not isinstance(value, dict):
        _raise(f"row {row.get('row_index')}: role {role} missing")
    return value


def _role_names() -> tuple[str, ...]:
    return (*BASE_UPSTREAM.keys(), STRUCTURAL_ROLE)


def validate_closure(
    *,
    pool_rows: Sequence[dict[str, Any]],
    benchmark_order: Mapping[str, Any],
    route_map: Mapping[str, Any],
    source_union_membership: Mapping[str, Any],
    row_bindings: Mapping[str, Any],
    upstream_rows: Mapping[str, Sequence[dict[str, Any]]],
) -> None:
    roles = _role_names()
    if len(pool_rows) != EXPECTED_ROWS:
        _raise("v1.2 pool is not full274")
    order = benchmark_order.get("rows") if isinstance(benchmark_order, dict) else None
    if not isinstance(order, list) or len(order) != EXPECTED_ROWS or len(set(order)) != EXPECTED_ROWS:
        _raise("v1.2 benchmark order is incomplete or duplicate")
    if benchmark_order.get("benchmark_sha256") != v11.BENCHMARK_ID_AUTHORITY["sha256"]:
        _raise("v1.2 benchmark authority mismatch")
    if route_map.get("benchmark_order_sha256") != BASE_ARTIFACTS["benchmark_order"]["sha256"]:
        _raise("v1.2 route map order authority mismatch")
    route_rows = route_map.get("rows")
    if not isinstance(route_rows, list) or len(route_rows) != EXPECTED_ROWS:
        _raise("v1.2 route map is incomplete")
    for index, (task_id, route) in enumerate(zip(order, route_rows, strict=True)):
        if route != {
            "row_index": index,
            "task_id": task_id,
            "evaluation_route": route.get("evaluation_route"),
        } or route.get("evaluation_route") not in {"deterministic", "image_judge"}:
            _raise(f"v1.2 route relabel/misalignment at row {index}")
    if route_map != _read_json(BASE_FROZEN / BASE_ARTIFACTS["route_map"]["path"], "base route map"):
        _raise("v1.2 route authority is not byte-equivalent v1.1 content")
    if source_union_membership != _read_json(
        BASE_FROZEN / BASE_ARTIFACTS["source_union_membership"]["path"],
        "base source membership",
    ):
        _raise("v1.2 source-union authority differs from v1.1")

    if set(upstream_rows) != set(roles):
        _raise("v1.2 upstream role closure mismatch")
    if any(len(rows) != EXPECTED_ROWS for rows in upstream_rows.values()):
        _raise("v1.2 upstream row count mismatch")

    if set(row_bindings) != {
        "schema_version",
        "benchmark_sha256",
        "benchmark_order_sha256",
        "projection_contract",
        "upstream_artifact_sha256",
        "rows",
    }:
        _raise("v1.2 row-bindings shape mismatch")
    if row_bindings.get("schema_version") != BINDINGS_SCHEMA:
        _raise("v1.2 row-bindings schema mismatch")
    expected_upstream_sha = {
        **{role: descriptor["sha256"] for role, descriptor in BASE_UPSTREAM.items()},
        STRUCTURAL_ROLE: STRUCTURAL_SOURCE["sha256"],
    }
    if row_bindings.get("upstream_artifact_sha256") != expected_upstream_sha:
        _raise("v1.2 row-bindings upstream closure mismatch")
    binding_rows = row_bindings.get("rows")
    if not isinstance(binding_rows, list) or len(binding_rows) != EXPECTED_ROWS:
        _raise("v1.2 row-bindings are incomplete")

    base_bindings = _read_json(
        BASE_FROZEN / BASE_ARTIFACTS["row_bindings"]["path"], "v1.1 row bindings"
    )
    base_binding_rows = base_bindings["rows"]
    for index, (task_id, pool, binding) in enumerate(
        zip(order, pool_rows, binding_rows, strict=True)
    ):
        if set(pool) != {
            "schema_version",
            "row_index",
            "anchor",
            "routers",
            "parallel_batches",
            "structural",
        }:
            _raise(f"v1.2 pool row {index}: exact shape mismatch")
        if pool.get("schema_version") != POOL_ROW_SCHEMA or pool.get("row_index") != index:
            _raise(f"v1.2 pool row {index}: schema/index mismatch")
        if set(pool.get("routers", {})) != {"v4", "v5"}:
            _raise(f"v1.2 pool row {index}: router roles mismatch")
        if set(pool.get("parallel_batches", {})) != {
            "parallel8_v1",
            "parallel8_reasoning_first_v2",
        }:
            _raise(f"v1.2 pool row {index}: parallel roles mismatch")
        validate_candidate(pool["anchor"], "active_crop_v2", allow_source=False)
        validate_candidate(
            pool["routers"]["v4"], "native_thinking_math_router_v4", allow_source=False
        )
        validate_candidate(
            pool["routers"]["v5"], "native_thinking_math_router_v5", allow_source=False
        )
        for role in ("parallel8_v1", "parallel8_reasoning_first_v2"):
            batch = pool["parallel_batches"][role]
            if not isinstance(batch, dict) or set(batch) != {"final", "raw_votes"}:
                _raise(f"v1.2 row {index}: {role} batch shape mismatch")
            if not isinstance(batch["raw_votes"], list) or len(batch["raw_votes"]) != 8:
                _raise(f"v1.2 row {index}: {role} must have eight raw votes")
            validate_candidate(batch["final"], role, allow_source=False)
            for vote in batch["raw_votes"]:
                validate_candidate(vote, role, allow_source=False)
        validate_candidate(pool["structural"], STRUCTURAL_ROLE, allow_source=True)
        expected_structural = structural_candidate(upstream_rows[STRUCTURAL_ROLE][index])
        if pool["structural"] != expected_structural:
            _raise(f"v1.2 row {index}: Structural candidate/source projection mismatch")

        source_hashes: dict[str, str] = {}
        for role in roles:
            source_row = upstream_rows[role][index]
            if source_row.get("task_id") != task_id:
                _raise(f"v1.2 row {index}: source/task misalignment for {role}")
            source_hashes[role] = v11.canonical_row_sha256(role_projection(pool, role))
        expected_binding = {
            "row_index": index,
            "task_id": task_id,
            "role_projection_sha256": source_hashes,
        }
        if binding != expected_binding:
            _raise(f"v1.2 row {index}: candidate swap/binding mismatch")
        if {
            role: source_hashes[role] for role in BASE_UPSTREAM
        } != base_binding_rows[index]["role_projection_sha256"]:
            _raise(f"v1.2 row {index}: inherited v1.1 role projection changed")


def _copy_exact(source: Path, destination: Path, expected_sha: str, label: str) -> None:
    _assert_sha(source, expected_sha, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    _assert_sha(destination, expected_sha, f"copied {label}")


def build(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        _raise(f"refusing to overwrite existing v1.2 package: {output_dir}")
    (
        _base_package,
        base_pool,
        benchmark_order,
        route_map,
        source_union_membership,
        base_bindings,
        base_upstream_rows,
    ) = _load_base()
    structural_rows = _load_structural()
    order = benchmark_order["rows"]
    if [row.get("task_id") for row in structural_rows] != order:
        _raise("Structural solver task order differs from v1.1 benchmark order")

    pool_rows: list[dict[str, Any]] = []
    for index, (base_row, structural_row) in enumerate(
        zip(base_pool, structural_rows, strict=True)
    ):
        if base_row.get("row_index") != index:
            _raise(f"v1.1 base pool row index mismatch at {index}")
        pool_rows.append(
            {
                "schema_version": POOL_ROW_SCHEMA,
                "row_index": index,
                "anchor": base_row["anchor"],
                "routers": base_row["routers"],
                "parallel_batches": base_row["parallel_batches"],
                "structural": structural_candidate(structural_row),
            }
        )

    roles = _role_names()
    upstream_sha = {
        **{role: descriptor["sha256"] for role, descriptor in BASE_UPSTREAM.items()},
        STRUCTURAL_ROLE: STRUCTURAL_SOURCE["sha256"],
    }
    row_bindings = {
        "schema_version": BINDINGS_SCHEMA,
        "benchmark_sha256": v11.BENCHMARK_ID_AUTHORITY["sha256"],
        "benchmark_order_sha256": BASE_ARTIFACTS["benchmark_order"]["sha256"],
        "projection_contract": "sha256_of_utf8_json_sort_keys_compact_role_projection",
        "upstream_artifact_sha256": upstream_sha,
        "rows": [
            {
                "row_index": index,
                "task_id": task_id,
                "role_projection_sha256": {
                    role: v11.canonical_row_sha256(role_projection(pool_rows[index], role))
                    for role in roles
                },
            }
            for index, task_id in enumerate(order)
        ],
    }
    for index, base_binding in enumerate(base_bindings["rows"]):
        inherited = {
            role: row_bindings["rows"][index]["role_projection_sha256"][role]
            for role in BASE_UPSTREAM
        }
        if inherited != base_binding["role_projection_sha256"]:
            _raise(f"v1.2 build changed v1.1 role projection at row {index}")

    upstream_rows = {**base_upstream_rows, STRUCTURAL_ROLE: structural_rows}
    validate_closure(
        pool_rows=pool_rows,
        benchmark_order=benchmark_order,
        route_map=route_map,
        source_union_membership=source_union_membership,
        row_bindings=row_bindings,
        upstream_rows=upstream_rows,
    )

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v1_2_build_", dir=output_parent))
    try:
        for key in ("benchmark_order", "route_map", "source_union_membership"):
            descriptor = BASE_ARTIFACTS[key]
            _copy_exact(
                BASE_FROZEN / descriptor["path"],
                temporary / descriptor["path"],
                descriptor["sha256"],
                f"v1.1 authority {key}",
            )
        for role, descriptor in BASE_UPSTREAM.items():
            _copy_exact(
                BASE_FROZEN / descriptor["path"],
                temporary / descriptor["path"],
                descriptor["sha256"],
                f"v1.1 upstream {role}",
            )
        structural_snapshot = temporary / "upstream" / f"{STRUCTURAL_ROLE}.jsonl"
        _copy_exact(
            REPO_ROOT / STRUCTURAL_SOURCE["path"],
            structural_snapshot,
            STRUCTURAL_SOURCE["sha256"],
            "Structural final solver",
        )

        pool_sha = _write_jsonl(temporary / "candidate_pool_v1_2.jsonl", pool_rows)
        bindings_sha = _write_json(temporary / "row_bindings_v1_2.json", row_bindings)
        upstream_descriptors = {
            **BASE_UPSTREAM,
            STRUCTURAL_ROLE: {
                "path": f"upstream/{STRUCTURAL_ROLE}.jsonl",
                "sha256": STRUCTURAL_SOURCE["sha256"],
            },
        }
        package = {
            "schema_version": PACKAGE_SCHEMA,
            "rows": EXPECTED_ROWS,
            "benchmark_sha256": v11.BENCHMARK_ID_AUTHORITY["sha256"],
            "created_before_new_arm_evaluation": True,
            "runtime_outcome_access": False,
            "benchmark_order": BASE_ARTIFACTS["benchmark_order"],
            "route_map": BASE_ARTIFACTS["route_map"],
            "source_union_membership": BASE_ARTIFACTS["source_union_membership"],
            "upstream_artifacts": upstream_descriptors,
            "row_bindings": {
                "path": "row_bindings_v1_2.json",
                "sha256": bindings_sha,
            },
            "candidate_pool": {
                "path": "candidate_pool_v1_2.jsonl",
                "sha256": pool_sha,
            },
        }
        package_sha = _write_json(temporary / "input_package_v1_2.json", package)

        source_access_counts = Counter(
            row["structural"]["generation"]["source_access"] for row in pool_rows
        )
        structural_origin_counts = Counter()
        for row in structural_rows:
            generation = row["generation"]
            chosen = generation["failclosed_composition"]["chosen_source"]
            if chosen == "candidate":
                origin = generation["route"]
            else:
                origin = f"frozen_subject_router_{row['offline_provenance']['decision']['chosen_source']}"
            structural_origin_counts[origin] += 1
        content_projection = {
            "base_v1_1_input_package_sha256": BASE_PACKAGE["sha256"],
            "benchmark_sha256": v11.BENCHMARK_ID_AUTHORITY["sha256"],
            "authority_sha256": {
                key: descriptor["sha256"]
                for key, descriptor in BASE_ARTIFACTS.items()
                if key in {"benchmark_order", "route_map", "source_union_membership"}
            },
            "upstream_artifact_sha256": upstream_sha,
            "candidate_pool_sha256": pool_sha,
            "row_bindings_sha256": bindings_sha,
            "structural_source_access_counts": {
                "false": source_access_counts[False],
                "true": source_access_counts[True],
            },
            "structural_origin_counts": dict(sorted(structural_origin_counts.items())),
            "inherited_v1_1_role_projection_changes": 0,
            "runtime_outcome_access": False,
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "normalized_input_v1_2_locked_not_selected_not_evaluated",
            "rows": EXPECTED_ROWS,
            "upstream_generation_model_closure": [MODEL],
            "inherited_27b_outputs": False,
            "runtime_outcome_access": False,
            "base_v1_1_input_package": {
                "sha256": BASE_PACKAGE["sha256"],
                "v1_1_bytes_modified": False,
            },
            "structural_provenance_authorities": STRUCTURAL_PROVENANCE,
            "artifacts": {
                "input_package_v1_2": {
                    "path": "input_package_v1_2.json",
                    "sha256": package_sha,
                },
                "candidate_pool_v1_2": {
                    "path": "candidate_pool_v1_2.jsonl",
                    "sha256": pool_sha,
                },
                "row_bindings_v1_2": {
                    "path": "row_bindings_v1_2.json",
                    "sha256": bindings_sha,
                },
            },
            "content_projection": content_projection,
            "content_projection_sha256": v11.canonical_projection_sha256(content_projection),
        }
        manifest_sha = _write_json(temporary / "build_manifest_v1_2.json", manifest)

        files_for_sums = sorted(
            path for path in temporary.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        sums = "".join(
            f"{v11.sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n"
            for path in files_for_sums
        )
        (temporary / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    result = verify_existing(output_dir)
    result["build_manifest_v1_2_sha256"] = manifest_sha
    return result


def verify_existing(output_dir: Path) -> dict[str, Any]:
    if not output_dir.is_dir():
        _raise(f"v1.2 package directory missing: {output_dir}")
    package_path = output_dir / "input_package_v1_2.json"
    package = _read_json(package_path, "v1.2 input package")
    if set(package) != EXPECTED_PACKAGE_KEYS or package.get("schema_version") != PACKAGE_SCHEMA:
        _raise("v1.2 package exact shape/schema mismatch")
    if package.get("rows") != EXPECTED_ROWS:
        _raise("v1.2 package row count mismatch")
    if package.get("benchmark_sha256") != v11.BENCHMARK_ID_AUTHORITY["sha256"]:
        _raise("v1.2 package benchmark pin mismatch")
    if package.get("created_before_new_arm_evaluation") is not True:
        _raise("v1.2 package chronology flag mismatch")
    if package.get("runtime_outcome_access") is not False:
        _raise("v1.2 package outcome access is not false")
    _assert_no_forbidden_structure(package, "v1.2 package")

    expected_authorities = {
        key: BASE_ARTIFACTS[key]
        for key in ("benchmark_order", "route_map", "source_union_membership")
    }
    for key, descriptor in expected_authorities.items():
        if package.get(key) != descriptor:
            _raise(f"v1.2 {key} authority descriptor changed")
        _assert_sha(output_dir / descriptor["path"], descriptor["sha256"], f"v1.2 {key}")
    benchmark_order = _read_json(
        output_dir / package["benchmark_order"]["path"], "v1.2 benchmark order"
    )
    route_map = _read_json(output_dir / package["route_map"]["path"], "v1.2 route map")
    source_union_membership = _read_json(
        output_dir / package["source_union_membership"]["path"],
        "v1.2 source membership",
    )

    expected_upstream = {
        **BASE_UPSTREAM,
        STRUCTURAL_ROLE: {
            "path": f"upstream/{STRUCTURAL_ROLE}.jsonl",
            "sha256": STRUCTURAL_SOURCE["sha256"],
        },
    }
    if package.get("upstream_artifacts") != expected_upstream:
        _raise("v1.2 package upstream descriptor closure mismatch")
    upstream_rows: dict[str, list[dict[str, Any]]] = {}
    for role, raw_descriptor in package["upstream_artifacts"].items():
        descriptor = _safe_descriptor(raw_descriptor, f"v1.2 upstream {role}")
        path = output_dir / descriptor["path"]
        _assert_sha(path, descriptor["sha256"], f"v1.2 upstream {role}")
        _verify_no_27b_bytes(path, f"v1.2 upstream {role}")
        rows = _read_jsonl(path, f"v1.2 upstream {role}")
        try:
            v11.validate_answer_source_rows(rows, f"v1.2 upstream {role}")
        except v11.PoolBuildError as exc:
            raise PoolV12Error(str(exc)) from exc
        upstream_rows[role] = rows

    pool_descriptor = _safe_descriptor(package.get("candidate_pool"), "v1.2 candidate pool")
    bindings_descriptor = _safe_descriptor(package.get("row_bindings"), "v1.2 row bindings")
    pool_path = output_dir / pool_descriptor["path"]
    bindings_path = output_dir / bindings_descriptor["path"]
    _assert_sha(pool_path, pool_descriptor["sha256"], "v1.2 candidate pool")
    _assert_sha(bindings_path, bindings_descriptor["sha256"], "v1.2 row bindings")
    pool_rows = _read_jsonl(pool_path, "v1.2 candidate pool")
    row_bindings = _read_json(bindings_path, "v1.2 row bindings")
    validate_closure(
        pool_rows=pool_rows,
        benchmark_order=benchmark_order,
        route_map=route_map,
        source_union_membership=source_union_membership,
        row_bindings=row_bindings,
        upstream_rows=upstream_rows,
    )

    manifest_path = output_dir / "build_manifest_v1_2.json"
    manifest = _read_json(manifest_path, "v1.2 build manifest")
    projection = manifest.get("content_projection")
    if (
        not isinstance(projection, dict)
        or manifest.get("content_projection_sha256")
        != v11.canonical_projection_sha256(projection)
    ):
        _raise("v1.2 build manifest content projection mismatch")
    if manifest.get("upstream_generation_model_closure") != [MODEL]:
        _raise("v1.2 model closure is not exactly Qwen3.5-9B")
    if manifest.get("inherited_27b_outputs") is not False:
        _raise("v1.2 inherited 27B marker is not false")
    if manifest.get("runtime_outcome_access") is not False:
        _raise("v1.2 manifest outcome access is not false")
    if manifest.get("structural_provenance_authorities") != STRUCTURAL_PROVENANCE:
        _raise("v1.2 Structural provenance authority set mismatch")
    for label, descriptor in STRUCTURAL_PROVENANCE.items():
        _assert_sha(REPO_ROOT / descriptor["path"], descriptor["sha256"], label)
    for label, descriptor in manifest.get("artifacts", {}).items():
        safe = _safe_descriptor(descriptor, f"v1.2 manifest artifact {label}")
        _assert_sha(output_dir / safe["path"], safe["sha256"], label)

    sums_path = output_dir / "SHA256SUMS.txt"
    seen: set[str] = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in seen:
            _raise(f"SHA256SUMS duplicate path {relative}")
        seen.add(relative)
        descriptor = _safe_descriptor({"path": relative, "sha256": expected}, "SHA256SUMS")
        _assert_sha(output_dir / descriptor["path"], descriptor["sha256"], relative)
    expected_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if seen != expected_files:
        _raise("SHA256SUMS file closure mismatch")

    return {
        "status": "normalized_input_v1_2_verified_not_selected_not_evaluated",
        "rows": len(pool_rows),
        "roles": list(_role_names()),
        "input_package_v1_2_sha256": v11.sha256_file(package_path),
        "candidate_pool_v1_2_sha256": v11.sha256_file(pool_path),
        "row_bindings_v1_2_sha256": v11.sha256_file(bindings_path),
        "benchmark_order_sha256": BASE_ARTIFACTS["benchmark_order"]["sha256"],
        "route_map_sha256": BASE_ARTIFACTS["route_map"]["sha256"],
        "source_union_membership_sha256": BASE_ARTIFACTS["source_union_membership"]["sha256"],
        "structural_upstream_sha256": STRUCTURAL_SOURCE["sha256"],
        "sha256sums_sha256": v11.sha256_file(sums_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the isolated gold/outcome-blind selector input v1.2"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    result = verify_existing(args.output_dir) if args.verify_existing else build(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
