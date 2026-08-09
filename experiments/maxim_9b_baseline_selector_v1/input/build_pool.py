from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL = "Qwen/Qwen3.5-9B"
EXPECTED_ROWS = 274
POOL_ROW_SCHEMA = "maxim-9b-baseline-candidate-pool-row-v1"
POOL_MANIFEST_SCHEMA = "maxim-9b-baseline-candidate-pool-manifest-v1"
NORMALIZATION_SCHEMA = "maxim-9b-baseline-candidate-pool-normalization-v1"
MEMBERSHIP_SCHEMA = "maxim-9b-source-union-membership-row-v1"
EVALUATOR_ROUTE_SCHEMA = "maxim-9b-evaluator-route-row-v1"
ROW_BINDING_SCHEMA = "maxim-9b-candidate-pool-row-binding-v1"
BINDING_MANIFEST_SCHEMA = "maxim-9b-candidate-pool-binding-manifest-v1"
INPUT_PACKAGE_V11_SCHEMA = "maxim-9b-baseline-selector-input-package-v1.1"
BENCHMARK_ORDER_V11_SCHEMA = "maxim-9b-baseline-selector-benchmark-order-v1.1"
ROUTE_MAP_V11_SCHEMA = "maxim-9b-baseline-selector-route-map-v1.1"
ROW_BINDINGS_V11_SCHEMA = "maxim-9b-baseline-selector-row-bindings-v1.1"
POOL_ROW_V11_SCHEMA = "maxim-9b-baseline-selector-pool-row-v1.1"
SOURCE_UNION_MEMBERSHIP_V11_SCHEMA = (
    "maxim-9b-baseline-selector-source-union-membership-v1.1"
)
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTOR_ROOT = Path(__file__).resolve().parent.parent

UPSTREAM: dict[str, dict[str, str]] = {
    "active_crop_v2": {
        "path": "reports/maxim_query_active_crop_v2_20260803/solver.jsonl",
        "sha256": "6697c043f3142a736b817ead5da494eea334f5349e0db833bd72f23fe35cb17c",
        "provenance": "fully_pinned_preregistered_9b_active_crop_with_no_tools_fallback",
    },
    "native_thinking_math_router_v4": {
        "path": "reports/maxim_native_thinking_math_router_v4_20260803/solver.jsonl",
        "sha256": "0fd0e6fef6b220749faa015e7a163cdf596e070afaaad407e4d36bc9b1337307",
        "provenance": "fully_pinned_9b_native_thinking_math_router_primary",
    },
    "native_thinking_math_router_v5": {
        "path": "reports/maxim_native_thinking_math_router_v5_20260803/solver.jsonl",
        "sha256": "45dc8c16f834d27e5d114f9162b7984c8baec4037fef011730e91ba13f192845",
        "provenance": "fully_pinned_9b_correlated_v4_token_budget_ablation",
    },
    "parallel8_v1": {
        "path": "reports/maxim_ideas_full274_20260731/parallel8_v1/solver.jsonl",
        "sha256": "b1b7a1b785a9a3fc076c04c37fa72f4a82169f137651ff57b40e984901b0d645",
        "provenance": "fresh_9b_eight_route_batch_partial_endpoint_attestation_diversity_donor",
    },
    "parallel8_reasoning_first_v2": {
        "path": "reports/maxim_ideas_full274_20260731/parallel8_reasoning_first_v2/solver.jsonl",
        "sha256": "6115effd03d7eac3e11e9726ecd9822802a04f235a0b361e1863b9bc7e221023",
        "provenance": "fresh_9b_eight_route_batch_partial_endpoint_attestation_core",
    },
}

PUBLIC_QUEUE = {
    "path": "reports/maxim_visual_sketchpad_v2_20260803/public_parse_queue.jsonl",
    "sha256": "172183440d95f863f8c7d895d4dbe2ec9b5161cdff19827252d5c7562868993d",
}
BENCHMARK_ID_AUTHORITY = {
    "path": "artifacts/baselines/basic_page_rag_v1/validation_274.jsonl",
    "sha256": "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9",
    "projection": "ordered task_id prefix only; no other benchmark field is parsed",
}
IMAGE_ROUTE_TEMPLATE = {
    "path": "reports/maxim_ideas_full274_20260731/parallel8_reasoning_first_v2/image97_input.jsonl",
    "sha256": "41f35172092f67bc14368d7312cb91ad50256166dc42d0f630ed5e7a9965aa46",
    "projection": "task_id_only; no reference, candidate, score, or verdict field is copied",
}
EVALUATION_CONTRACT = {
    "path": "scripts/prepare_maxim_online_evaluation_v1.py",
    "sha256": "f5d3acafcc92e80609c0b98f4e8d69a7edc616abdb8de90936952b9aa470a9fc",
    "contract": "frozen 177 deterministic plus 97 image-judge task split",
}
GEOMETRY_SOURCE = {
    "path": "reports/maxim_tiled_vision_v1_20260803/solver.raw.jsonl",
    "sha256": "a28fef62caf6752769e5cfd6012d32bb7b53499ccd74f0acd1a1f0faeae45dad",
    "projection": "generation.tiles[0].source_size only; answers are not read into the pool",
}
TRUNCATED_IMAGE_AMENDMENT = {
    "path": (
        "reports/maxim_document_parser_v1_20260803/parser_augmented_solver_v1/"
        "parser_artifacts/execution_amendment_v5_truncated_png.json"
    ),
    "sha256": "9d3bf9438f2e2568d7884f27faf48d13e6c63360578e681feed99e5395a656ce",
    "projection": "val_0197 decoded_rgb shape only",
}

SOURCE_CERTIFICATES: tuple[dict[str, str], ...] = (
    {
        "stage": "ogm",
        "path": "reports/maxim_9b_source_replay_v1_20260809/active_crop/ogm_resolver/certificates.jsonl",
        "sha256": "223cd659758921e7e6a80d17acf2c5258838b0a8dfc0d553b977ecafa016a84a",
    },
    {
        "stage": "direct_pdf",
        "path": "reports/maxim_9b_source_replay_v1_20260809/active_crop/direct_pdf_resolver/certificates.jsonl",
        "sha256": "79ac3c76f5d372bacae0dece9b3fc55c21666b661fe42e3773dd417f94ba1a32",
    },
    {
        "stage": "source_v6_cumulative",
        "path": "reports/maxim_9b_source_replay_v1_20260809/active_crop/v6_resolver/certificates.jsonl",
        "sha256": "8c0eb70b970c7699575defdef9d937ae377b2e2b81d326a2a876e723c8576608",
    },
    {
        "stage": "source_v7_main",
        "path": "reports/maxim_9b_source_replay_v1_20260809/active_crop/v7_resolver/certificates.jsonl",
        "sha256": "f6fdcb7c8dcbb699e00aacd7ca96a5e01ca51a3f46697cb898ff7753f7a6e363",
    },
    {
        "stage": "source_v7_fill_history",
        "path": "reports/maxim_9b_source_replay_v1_20260809/active_crop/fill_resolver/certificates.jsonl",
        "sha256": "292ab092f1f8a42a05dfe09e2023f179a3cfdb9806b357c98bc2c758a950bc36",
    },
)

SELECTOR_PROFILE_SHA256 = "bbe251ffdca45df5dd0cc5438435a69a2bf3b8a4770b7f4ee7119d010057f77c"
SELECTOR_FREEZE_SHA256 = "001b4d5664c44e7ba686a4457a838e6f50991611751e7cb6fb8ad70f1a5619b4"
SOURCE_AGGREGATE_SHA256 = "3de5129dee80d2f2fda544bdf7eecfa7d0f467d56bb7e43afc8eac89a6a5dacd"
SOURCE_UNION_PROJECTION_SHA256 = "7e5e0c972e82d87d164cb3ef03b13fbb4c8084bf07c2512129958882871508cd"
SOURCE_UNION_SIZE = 156

EXPECTED_PARALLEL_ROUTES = (
    "literal_direct",
    "formal_structure",
    "option_elimination",
    "visual_evidence",
    "domain_expert",
    "counterexample_check",
    "independent_rederive",
    "concise_verifier",
)

FORBIDDEN_KEYS = frozenset(
    {
        "reference_answer",
        "reference_answers",
        "gold_answer",
        "correct_answer",
        "expected_answer",
        "correctness",
        "is_correct",
        "judge_verdict",
        "verdict",
        "score",
        "scores",
        "outcome",
        "outcomes",
        "reward",
    }
)
MODEL_KEYS = frozenset({"model", "base_row_model", "answer_producing_model"})
FORBIDDEN_BYTE_MARKERS = (b"qwen/qwen3.5-27b", b"qwen3.5-27b", b"inherited_27b_outputs\":true")


class PoolBuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return compact_json_bytes(value) + b"\n"


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_projection_sha256(value: Any) -> str:
    return hashlib.sha256(compact_json_bytes(value)).hexdigest()


def _path(descriptor: Mapping[str, str]) -> Path:
    return REPO_ROOT / descriptor["path"]


def assert_pinned(descriptor: Mapping[str, str], label: str) -> Path:
    path = _path(descriptor)
    if not path.is_file():
        raise PoolBuildError(f"{label}: missing pinned file {path}")
    actual = sha256_file(path)
    if actual != descriptor["sha256"]:
        raise PoolBuildError(
            f"{label}: SHA-256 mismatch; expected={descriptor['sha256']} actual={actual}"
        )
    return path


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PoolBuildError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PoolBuildError(f"{label}: expected JSON object")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PoolBuildError(f"{label}:{line_number}: row is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise PoolBuildError(f"{label}: invalid JSONL: {exc}") from exc
    return rows


TASK_ID_PREFIX_RE = re.compile(rb'^\s*\{\s*"task_id"\s*:\s*"([A-Za-z0-9_.:-]+)"')


def read_ordered_task_id_prefixes(path: Path, label: str) -> list[str]:
    """Project only the leading task_id token from each JSONL row.

    This deliberately does not deserialize benchmark records, so reference and
    outcome fields are outside the semantic access surface of the builder.
    """

    ids: list[str] = []
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                match = TASK_ID_PREFIX_RE.match(raw_line)
                if match is None:
                    raise PoolBuildError(
                        f"{label}:{line_number}: task_id is not the leading JSON field"
                    )
                ids.append(match.group(1).decode("ascii"))
    except OSError as exc:
        raise PoolBuildError(f"{label}: cannot read task-id projection: {exc}") from exc
    if len(ids) != len(set(ids)):
        raise PoolBuildError(f"{label}: duplicate projected task_id")
    return ids


def canonical_row_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(compact_json_bytes(row)).hexdigest()


def _walk(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from _walk(item)


def validate_answer_source_rows(rows: Sequence[dict[str, Any]], label: str) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise PoolBuildError(f"{label}: expected {EXPECTED_ROWS} rows, found {len(rows)}")
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen:
            raise PoolBuildError(f"{label}:{index}: missing or duplicate task_id")
        seen.add(task_id)
        if row.get("model") != MODEL:
            raise PoolBuildError(f"{label}:{task_id}: model is not pinned Qwen3.5-9B")
        if row.get("error") is not None:
            raise PoolBuildError(f"{label}:{task_id}: final solver row has an error")
        if not isinstance(row.get("final_answer"), str) or not row["final_answer"].strip():
            raise PoolBuildError(f"{label}:{task_id}: final answer is unavailable")
        generation = row.get("generation")
        if not isinstance(generation, dict) or generation.get("gold_access") is not False:
            raise PoolBuildError(f"{label}:{task_id}: generation.gold_access is not false")
        for key, value in _walk(row):
            lowered = key.casefold() if key is not None else ""
            if lowered in FORBIDDEN_KEYS:
                raise PoolBuildError(f"{label}:{task_id}: forbidden field {key}")
            if lowered == "gold_access" and value is not False:
                raise PoolBuildError(f"{label}:{task_id}: non-false gold_access")
            if lowered in MODEL_KEYS and value not in (None, MODEL):
                raise PoolBuildError(
                    f"{label}:{task_id}: mixed answer-producing model field {key}={value!r}"
                )
            if lowered == "inherited_27b_outputs" and value is not False:
                raise PoolBuildError(f"{label}:{task_id}: inherited 27B output marker")


def _assert_no_forbidden_bytes(path: Path, label: str) -> None:
    lowered = path.read_bytes().lower().replace(b" ", b"")
    for marker in FORBIDDEN_BYTE_MARKERS:
        if marker in lowered:
            raise PoolBuildError(f"{label}: forbidden 27B byte marker {marker!r}")


def _index(rows: Sequence[dict[str, Any]], label: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in indexed:
            raise PoolBuildError(f"{label}: missing or duplicate task_id")
        order.append(task_id)
        indexed[task_id] = row
    return order, indexed


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _safe_prompt(value: Any, label: str) -> str:
    prompt = str(value or "")
    if not SAFE_ID_RE.fullmatch(prompt):
        raise PoolBuildError(f"{label}: unsafe or missing prompt_version {prompt!r}")
    return prompt


def _trace(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    generation = row.get("generation")
    if not isinstance(generation, dict):
        return None
    calls = generation.get("calls")
    if isinstance(calls, list) and calls and isinstance(calls[-1], dict):
        return calls[-1]
    for key in ("selector_trace", "call_trace"):
        value = generation.get(key)
        if isinstance(value, dict):
            return value
    traces = generation.get("call_traces")
    if isinstance(traces, list) and traces and isinstance(traces[-1], dict):
        return traces[-1]
    return None


def _finish_reason(trace: Mapping[str, Any] | None, available: bool, error: Any) -> str:
    if error is not None:
        return "error"
    if not available:
        return "missing"
    value = trace.get("finish_reason") if trace else None
    return value if value in {"stop", "length", "error", "missing"} else "stop"


def _generation(
    *,
    row: Mapping[str, Any],
    role: str,
    trace: Mapping[str, Any] | None = None,
    prompt_version: str | None = None,
    raw_vote: bool = False,
) -> dict[str, Any]:
    generation = row.get("generation") if isinstance(row.get("generation"), dict) else {}
    usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
    error = trace.get("parse_error") if raw_vote and trace else row.get("error")
    if error is not None and not isinstance(error, str):
        error = str(error)
    available = bool(
        isinstance(trace.get("final_answer"), str) and trace["final_answer"].strip()
        if raw_vote and trace
        else isinstance(row.get("final_answer"), str) and row["final_answer"].strip()
    )
    if raw_vote and trace:
        input_tokens = _nonnegative_int(trace.get("input_tokens"))
        output_tokens = _nonnegative_int(trace.get("output_tokens"))
        call_count = 1 if trace.get("endpoint") else 0
        forced_answer = bool(trace.get("recovered_partial"))
    else:
        input_tokens = _nonnegative_int(usage.get("input_tokens"))
        output_tokens = _nonnegative_int(usage.get("output_tokens"))
        call_count = _nonnegative_int(generation.get("call_count"))
        forced_answer = bool(row.get("forced_answer"))
    temperature = generation.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        temperature = None
    seed = generation.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        seed = None
    return {
        "finish_reason": _finish_reason(trace, available, error),
        "error": error,
        "forced_answer": forced_answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "call_count": call_count,
        "temperature": temperature,
        "seed": seed,
        "prompt_version": _safe_prompt(
            prompt_version if prompt_version is not None else row.get("prompt_version"),
            role,
        ),
        "upstream_artifact_sha256": UPSTREAM[role]["sha256"],
        "gold_access": False,
        "score_or_outcome_access": False,
        "known_error_memory_access": False,
        "source_access": False,
    }


def _candidate(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    answer = row.get("final_answer")
    available = isinstance(answer, str) and bool(answer.strip()) and row.get("error") is None
    return {
        "available": available,
        "model": MODEL,
        "final_answer": answer if available else None,
        "generation": _generation(row=row, role=role, trace=_trace(row)),
    }


def _raw_vote(row: Mapping[str, Any], trace: Mapping[str, Any], role: str) -> dict[str, Any]:
    answer = trace.get("final_answer")
    available = isinstance(answer, str) and bool(answer.strip())
    route = str(trace.get("route") or "")
    prompt = f"{row.get('prompt_version')}:{route}"
    return {
        "available": available,
        "model": MODEL,
        "final_answer": answer if available else None,
        "generation": _generation(
            row=row,
            role=role,
            trace=trace,
            prompt_version=prompt,
            raw_vote=True,
        ),
    }


def _parallel_batch(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    generation = row.get("generation")
    traces = generation.get("candidate_traces") if isinstance(generation, dict) else None
    if not isinstance(traces, list) or len(traces) != 8:
        raise PoolBuildError(f"{role}:{row.get('task_id')}: expected exactly 8 raw traces")
    indices = [trace.get("index") for trace in traces if isinstance(trace, dict)]
    routes = [trace.get("route") for trace in traces if isinstance(trace, dict)]
    if indices != list(range(1, 9)) or tuple(routes) != EXPECTED_PARALLEL_ROUTES:
        raise PoolBuildError(f"{role}:{row.get('task_id')}: raw trace route/index mismatch")
    return {
        "final": _candidate(row, role),
        "raw_votes": [_raw_vote(row, trace, role) for trace in traces],
    }


def load_source_union() -> tuple[frozenset[str], dict[str, Any]]:
    stage_sets: dict[str, set[str]] = {}
    union: set[str] = set()
    sources: list[dict[str, Any]] = []
    for descriptor in SOURCE_CERTIFICATES:
        path = assert_pinned(descriptor, f"source certificates {descriptor['stage']}")
        rows = read_jsonl(path, f"source certificates {descriptor['stage']}")
        ids = {str(row.get("task_id") or "") for row in rows}
        if "" in ids or len(ids) != len(rows):
            raise PoolBuildError(f"source certificates {descriptor['stage']}: invalid task IDs")
        stage_sets[descriptor["stage"]] = ids
        union.update(ids)
        sources.append(
            {
                "stage": descriptor["stage"],
                "path": descriptor["path"],
                "sha256": descriptor["sha256"],
                "rows": len(rows),
            }
        )
    if len(union) != SOURCE_UNION_SIZE:
        raise PoolBuildError(
            f"source certificate union mismatch: expected={SOURCE_UNION_SIZE} actual={len(union)}"
        )
    return frozenset(union), {
        "stage_counts": {key: len(value) for key, value in stage_sets.items()},
        "certificate_sources": sources,
    }


def load_geometry(task_order: Sequence[str]) -> dict[str, tuple[list[int], list[int]]]:
    path = assert_pinned(GEOMETRY_SOURCE, "geometry source")
    rows = read_jsonl(path, "geometry source")
    order, indexed = _index(rows, "geometry source")
    if order != list(task_order):
        raise PoolBuildError("geometry source task order mismatch")
    result: dict[str, tuple[list[int], list[int]]] = {}
    for task_id, row in indexed.items():
        generation = row.get("generation")
        tiles = generation.get("tiles") if isinstance(generation, dict) else None
        sizes: dict[int, tuple[int, int]] = {}
        if isinstance(tiles, list):
            for tile in tiles:
                if not isinstance(tile, dict):
                    continue
                image_index = tile.get("source_image_index")
                source_size = tile.get("source_size")
                if (
                    isinstance(image_index, int)
                    and isinstance(source_size, list)
                    and len(source_size) == 2
                    and all(isinstance(item, int) and item > 0 for item in source_size)
                ):
                    size = (source_size[0], source_size[1])
                    if image_index in sizes and sizes[image_index] != size:
                        raise PoolBuildError(f"geometry source {task_id}: inconsistent size")
                    sizes[image_index] = size
        if sizes:
            expected_indices = list(range(len(sizes)))
            if sorted(sizes) != expected_indices:
                raise PoolBuildError(f"geometry source {task_id}: non-contiguous image indices")
            result[task_id] = (
                [sizes[index][0] for index in expected_indices],
                [sizes[index][1] for index in expected_indices],
            )

    amendment_path = assert_pinned(TRUNCATED_IMAGE_AMENDMENT, "truncated image amendment")
    amendment = read_json(amendment_path, "truncated image amendment")
    task_id = str(amendment.get("original", {}).get("task_id") or "")
    shape = amendment.get("decoded_rgb", {}).get("shape")
    if task_id != "val_0197" or not (
        isinstance(shape, list)
        and len(shape) == 3
        and all(isinstance(item, int) and item > 0 for item in shape)
    ):
        raise PoolBuildError("truncated image amendment shape mismatch")
    result[task_id] = ([shape[1]], [shape[0]])
    if set(result) != set(task_order):
        missing = sorted(set(task_order) - set(result))
        raise PoolBuildError(f"geometry coverage mismatch; missing={missing[:10]}")
    return result


def _task_id_projection(path: Path, label: str) -> list[str]:
    rows = read_jsonl(path, label)
    ids: list[str] = []
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise PoolBuildError(f"{label}: missing task_id")
        ids.append(task_id)
    if len(ids) != len(set(ids)):
        raise PoolBuildError(f"{label}: duplicate task_id")
    return ids


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            data = canonical_bytes(row)
            handle.write(data)
            digest.update(data)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    data = canonical_bytes(value)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _answer(row: Mapping[str, Any]) -> str:
    value = row.get("final_answer")
    return value.strip() if isinstance(value, str) else ""


def normalized_role_projection(pool_row: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    if role == "active_crop_v2":
        value = pool_row.get("anchor")
    elif role == "native_thinking_math_router_v4":
        value = pool_row.get("routers", {}).get("v4")
    elif role == "native_thinking_math_router_v5":
        value = pool_row.get("routers", {}).get("v5")
    else:
        value = pool_row.get("parallel_batches", {}).get(role)
    if not isinstance(value, dict):
        raise PoolBuildError(f"pool row {pool_row.get('opaque_id')}: missing projection for {role}")
    return value


def _candidate_v11(candidate: Mapping[str, Any]) -> dict[str, Any]:
    generation = candidate.get("generation")
    if not isinstance(generation, dict):
        raise PoolBuildError("v1.1 candidate generation is missing")
    return {
        "available": candidate.get("available"),
        "model": candidate.get("model"),
        "final_answer": candidate.get("final_answer"),
        "generation": {
            "finish_reason": generation.get("finish_reason"),
            "error": generation.get("error"),
            "forced_answer": generation.get("forced_answer"),
            "input_tokens": generation.get("input_tokens"),
            "output_tokens": generation.get("output_tokens"),
            "call_count": generation.get("call_count"),
            "temperature": generation.get("temperature"),
            "seed": generation.get("seed"),
            "prompt_version": generation.get("prompt_version"),
            "upstream_artifact_sha256": generation.get("upstream_artifact_sha256"),
            "new_arm_gold_reference_correctness_outcome_or_judge_access": False,
            "source_access": False,
        },
    }


def _parallel_batch_v11(batch: Mapping[str, Any]) -> dict[str, Any]:
    final = batch.get("final")
    votes = batch.get("raw_votes")
    if not isinstance(final, dict) or not isinstance(votes, list) or len(votes) != 8:
        raise PoolBuildError("v1.1 parallel batch shape mismatch")
    return {
        "final": _candidate_v11(final),
        "raw_votes": [_candidate_v11(value) for value in votes],
    }


def pool_row_v11(pool_row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    routers = pool_row.get("routers")
    batches = pool_row.get("parallel_batches")
    if not isinstance(routers, dict) or not isinstance(batches, dict):
        raise PoolBuildError("v1.1 pool source row shape mismatch")
    return {
        "schema_version": POOL_ROW_V11_SCHEMA,
        "row_index": row_index,
        "anchor": _candidate_v11(pool_row["anchor"]),
        "routers": {
            "v4": _candidate_v11(routers["v4"]),
            "v5": _candidate_v11(routers["v5"]),
        },
        "parallel_batches": {
            "parallel8_v1": _parallel_batch_v11(batches["parallel8_v1"]),
            "parallel8_reasoning_first_v2": _parallel_batch_v11(
                batches["parallel8_reasoning_first_v2"]
            ),
        },
    }


def normalized_role_projection_v11(
    pool_row: Mapping[str, Any], role: str
) -> Mapping[str, Any]:
    if role == "active_crop_v2":
        value = pool_row.get("anchor")
    elif role == "native_thinking_math_router_v4":
        value = pool_row.get("routers", {}).get("v4")
    elif role == "native_thinking_math_router_v5":
        value = pool_row.get("routers", {}).get("v5")
    else:
        value = pool_row.get("parallel_batches", {}).get(role)
    if not isinstance(value, dict):
        raise PoolBuildError(f"v1.1 pool row {pool_row.get('row_index')}: missing {role}")
    return value


def validate_binding_projections(
    *,
    pool_rows: Sequence[dict[str, Any]],
    route_rows: Sequence[dict[str, Any]],
    membership_rows: Sequence[dict[str, Any]],
    binding_rows: Sequence[dict[str, Any]],
    upstream_rows: Mapping[str, Sequence[dict[str, Any]]],
) -> None:
    if not all(
        len(values) == EXPECTED_ROWS
        for values in (pool_rows, route_rows, membership_rows, binding_rows)
    ):
        raise PoolBuildError("binding closure is not complete full274")
    if set(upstream_rows) != set(UPSTREAM):
        raise PoolBuildError("binding closure upstream role set mismatch")
    if any(len(values) != EXPECTED_ROWS for values in upstream_rows.values()):
        raise PoolBuildError("binding closure upstream row count mismatch")

    seen: set[str] = set()
    for index, (pool, route, membership, binding) in enumerate(
        zip(pool_rows, route_rows, membership_rows, binding_rows, strict=True)
    ):
        opaque_id = pool.get("opaque_id")
        if not isinstance(opaque_id, str) or not opaque_id or opaque_id in seen:
            raise PoolBuildError("binding closure has missing or duplicate opaque_id")
        seen.add(opaque_id)
        if route != {
            "schema_version": EVALUATOR_ROUTE_SCHEMA,
            "opaque_id": opaque_id,
            "evaluation_route": pool.get("observable", {}).get("evaluation_route"),
        }:
            raise PoolBuildError(f"binding closure {opaque_id}: evaluator route relabel detected")
        if membership.get("schema_version") != MEMBERSHIP_SCHEMA or membership.get(
            "opaque_id"
        ) != opaque_id or not isinstance(membership.get("protected_by_source_union"), bool):
            raise PoolBuildError(f"binding closure {opaque_id}: source-union row mismatch")
        if binding.get("schema_version") != ROW_BINDING_SCHEMA:
            raise PoolBuildError(f"binding closure {opaque_id}: row binding schema mismatch")
        if binding.get("opaque_id") != opaque_id or binding.get("row_index") != index:
            raise PoolBuildError(f"binding closure {opaque_id}: row index/task binding mismatch")
        if binding.get("evaluator_route_row_sha256") != canonical_row_sha256(route):
            raise PoolBuildError(f"binding closure {opaque_id}: evaluator route hash mismatch")
        if binding.get("source_union_membership_row_sha256") != canonical_row_sha256(membership):
            raise PoolBuildError(f"binding closure {opaque_id}: source-union hash mismatch")
        sources = binding.get("upstream_rows")
        if not isinstance(sources, dict) or set(sources) != set(UPSTREAM):
            raise PoolBuildError(f"binding closure {opaque_id}: upstream binding set mismatch")
        for role in UPSTREAM:
            source_row = upstream_rows[role][index]
            source_binding = sources[role]
            if not isinstance(source_binding, dict):
                raise PoolBuildError(f"binding closure {opaque_id}: missing {role} binding")
            expected = {
                "source_task_id": source_row.get("task_id"),
                "source_row_index": index,
                "source_row_canonical_sha256": canonical_row_sha256(source_row),
                "normalized_projection_sha256": canonical_row_sha256(
                    normalized_role_projection(pool, role)
                ),
            }
            if source_binding != expected or expected["source_task_id"] != opaque_id:
                raise PoolBuildError(
                    f"binding closure {opaque_id}: candidate swap or source misalignment for {role}"
                )


V11_GENERATION_KEYS = frozenset(
    {
        "finish_reason",
        "error",
        "forced_answer",
        "input_tokens",
        "output_tokens",
        "call_count",
        "temperature",
        "seed",
        "prompt_version",
        "upstream_artifact_sha256",
        "new_arm_gold_reference_correctness_outcome_or_judge_access",
        "source_access",
    }
)


def _validate_candidate_v11(value: Any, role: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "available",
        "model",
        "final_answer",
        "generation",
    }:
        raise PoolBuildError(f"v1.1 {role}: candidate shape mismatch")
    if value.get("model") != MODEL or not isinstance(value.get("available"), bool):
        raise PoolBuildError(f"v1.1 {role}: candidate model/availability mismatch")
    generation = value.get("generation")
    if not isinstance(generation, dict) or set(generation) != V11_GENERATION_KEYS:
        raise PoolBuildError(f"v1.1 {role}: generation shape mismatch")
    if generation["upstream_artifact_sha256"] != UPSTREAM[role]["sha256"]:
        raise PoolBuildError(f"v1.1 {role}: upstream profile mismatch")
    if generation["new_arm_gold_reference_correctness_outcome_or_judge_access"] is not False:
        raise PoolBuildError(f"v1.1 {role}: outcome access is not false")
    if generation["source_access"] is not False:
        raise PoolBuildError(f"v1.1 {role}: source access is not false")


def validate_v11_artifacts(
    *,
    pool_rows: Sequence[dict[str, Any]],
    benchmark_order: Mapping[str, Any],
    benchmark_order_sha256: str,
    route_map: Mapping[str, Any],
    row_bindings: Mapping[str, Any],
    expected_task_order: Sequence[str],
    image_ids: frozenset[str],
) -> None:
    if set(benchmark_order) != {"schema_version", "benchmark_sha256", "rows"}:
        raise PoolBuildError("v1.1 benchmark order shape mismatch")
    if benchmark_order.get("schema_version") != BENCHMARK_ORDER_V11_SCHEMA:
        raise PoolBuildError("v1.1 benchmark order schema mismatch")
    if benchmark_order.get("benchmark_sha256") != BENCHMARK_ID_AUTHORITY["sha256"]:
        raise PoolBuildError("v1.1 benchmark SHA mismatch")
    if benchmark_order.get("rows") != list(expected_task_order):
        raise PoolBuildError("v1.1 benchmark order task projection mismatch")

    expected_route_rows = [
        {
            "row_index": index,
            "task_id": task_id,
            "evaluation_route": "image_judge" if task_id in image_ids else "deterministic",
        }
        for index, task_id in enumerate(expected_task_order)
    ]
    expected_route_map = {
        "schema_version": ROUTE_MAP_V11_SCHEMA,
        "benchmark_sha256": BENCHMARK_ID_AUTHORITY["sha256"],
        "benchmark_order_sha256": benchmark_order_sha256,
        "derivation": "question_image_format_metadata_only_no_gold_score_correctness_outcome_or_judge",
        "rows": expected_route_rows,
    }
    if route_map != expected_route_map:
        raise PoolBuildError("v1.1 evaluator route authority mismatch")

    if len(pool_rows) != EXPECTED_ROWS:
        raise PoolBuildError("v1.1 candidate pool is not full274")
    role_projections: list[dict[str, str]] = []
    for index, row in enumerate(pool_rows):
        if not isinstance(row, dict) or set(row) != {
            "schema_version",
            "row_index",
            "anchor",
            "routers",
            "parallel_batches",
        }:
            raise PoolBuildError(f"v1.1 pool row {index}: exact shape mismatch")
        if row.get("schema_version") != POOL_ROW_V11_SCHEMA or row.get("row_index") != index:
            raise PoolBuildError(f"v1.1 pool row {index}: row-index/schema mismatch")
        if set(row["routers"]) != {"v4", "v5"} or set(row["parallel_batches"]) != {
            "parallel8_v1",
            "parallel8_reasoning_first_v2",
        }:
            raise PoolBuildError(f"v1.1 pool row {index}: candidate role set mismatch")
        _validate_candidate_v11(row["anchor"], "active_crop_v2")
        _validate_candidate_v11(row["routers"]["v4"], "native_thinking_math_router_v4")
        _validate_candidate_v11(row["routers"]["v5"], "native_thinking_math_router_v5")
        for role in ("parallel8_v1", "parallel8_reasoning_first_v2"):
            batch = row["parallel_batches"][role]
            if not isinstance(batch, dict) or set(batch) != {"final", "raw_votes"}:
                raise PoolBuildError(f"v1.1 pool row {index}: {role} batch shape mismatch")
            if not isinstance(batch["raw_votes"], list) or len(batch["raw_votes"]) != 8:
                raise PoolBuildError(f"v1.1 pool row {index}: {role} vote count mismatch")
            _validate_candidate_v11(batch["final"], role)
            for candidate in batch["raw_votes"]:
                _validate_candidate_v11(candidate, role)
        role_projections.append(
            {
                role: canonical_row_sha256(normalized_role_projection_v11(row, role))
                for role in UPSTREAM
            }
        )

    expected_bindings = {
        "schema_version": "maxim-9b-baseline-selector-combined-row-bindings-v1.1",
        "benchmark_sha256": BENCHMARK_ID_AUTHORITY["sha256"],
        "benchmark_order_sha256": benchmark_order_sha256,
        "projection_contract": "sha256_of_utf8_json_sort_keys_compact_role_projection",
        "upstream_artifact_sha256": {
            role: descriptor["sha256"] for role, descriptor in UPSTREAM.items()
        },
        "rows": [
            {
                "row_index": index,
                "task_id": task_id,
                "role_projection_sha256": role_projections[index],
            }
            for index, task_id in enumerate(expected_task_order)
        ],
    }
    if row_bindings != expected_bindings:
        raise PoolBuildError("v1.1 row binding mismatch or candidate swap")


def build_pool(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise PoolBuildError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # v1 failed independent audit and was intentionally superseded without
    # execution. v1.1 is frozen only after this package receives an audit PASS,
    # so the input package cannot depend on a profile/freeze hash circularly.

    public_path = assert_pinned(PUBLIC_QUEUE, "public task queue")
    public_rows = read_jsonl(public_path, "public task queue")
    task_order, public = _index(public_rows, "public task queue")
    if len(task_order) != EXPECTED_ROWS:
        raise PoolBuildError("public task queue is not full274")
    benchmark_path = assert_pinned(BENCHMARK_ID_AUTHORITY, "benchmark task-id authority")
    benchmark_task_order = read_ordered_task_id_prefixes(
        benchmark_path, "benchmark task-id authority"
    )
    if benchmark_task_order != task_order:
        raise PoolBuildError("public queue order differs from pinned benchmark task-id projection")
    assert_pinned(EVALUATION_CONTRACT, "evaluator split contract")

    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_index: dict[str, dict[str, dict[str, Any]]] = {}
    for role, descriptor in UPSTREAM.items():
        path = assert_pinned(descriptor, role)
        _assert_no_forbidden_bytes(path, role)
        rows = read_jsonl(path, role)
        validate_answer_source_rows(rows, role)
        order, indexed = _index(rows, role)
        if order != task_order:
            raise PoolBuildError(f"{role}: task order differs from public queue")
        if role.startswith("parallel8"):
            for row in rows:
                _parallel_batch(row, role)
        source_rows[role] = rows
        source_index[role] = indexed

    image_template_path = assert_pinned(IMAGE_ROUTE_TEMPLATE, "image evaluator route template")
    image_ids = _task_id_projection(image_template_path, "image evaluator route template")
    if len(image_ids) != 97 or not set(image_ids).issubset(task_order):
        raise PoolBuildError("image evaluator route task projection mismatch")
    image_set = frozenset(image_ids)

    source_union, source_union_meta = load_source_union()
    if not source_union.issubset(task_order):
        raise PoolBuildError("source-union contains task IDs outside full274")
    geometry = load_geometry(task_order)

    rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for task_id in task_order:
        task = public[task_id]
        images = task.get("question_images")
        if not isinstance(images, list):
            raise PoolBuildError(f"public queue {task_id}: question_images is not a list")
        widths, heights = geometry[task_id]
        if len(widths) != len(images) or len(heights) != len(images):
            raise PoolBuildError(f"public queue {task_id}: image geometry count mismatch")
        question = task.get("question")
        question_text = question if isinstance(question, str) else ""
        answer_type = "choice" if str(task.get("answer_type") or "").casefold() == "choice" else "other"
        grade = task.get("grade")
        if grade is not None:
            grade = str(grade)
        subject = task.get("subject")
        if subject is not None:
            subject = str(subject)
        row = {
            "schema_version": POOL_ROW_SCHEMA,
            "opaque_id": task_id,
            "observable": {
                "answer_type": answer_type,
                "option_count": None,
                "evaluation_route": "image_judge" if task_id in image_set else "deterministic",
                "subject": subject,
                "grade": grade,
                "question_char_count": len(question_text),
                "ocr_char_count": None,
                "image_count": len(images),
                "image_widths": widths,
                "image_heights": heights,
            },
            "anchor": _candidate(source_index["active_crop_v2"][task_id], "active_crop_v2"),
            "routers": {
                "v4": _candidate(
                    source_index["native_thinking_math_router_v4"][task_id],
                    "native_thinking_math_router_v4",
                ),
                "v5": _candidate(
                    source_index["native_thinking_math_router_v5"][task_id],
                    "native_thinking_math_router_v5",
                ),
            },
            "parallel_batches": {
                "parallel8_v1": _parallel_batch(
                    source_index["parallel8_v1"][task_id], "parallel8_v1"
                ),
                "parallel8_reasoning_first_v2": _parallel_batch(
                    source_index["parallel8_reasoning_first_v2"][task_id],
                    "parallel8_reasoning_first_v2",
                ),
            },
        }
        rows.append(row)
        route_rows.append(
            {
                "schema_version": EVALUATOR_ROUTE_SCHEMA,
                "opaque_id": task_id,
                "evaluation_route": row["observable"]["evaluation_route"],
            }
        )
        membership_rows.append(
            {
                "schema_version": MEMBERSHIP_SCHEMA,
                "opaque_id": task_id,
                "protected_by_source_union": task_id in source_union,
            }
        )

    binding_rows: list[dict[str, Any]] = []
    for index, (task_id, pool_row, route_row, membership_row) in enumerate(
        zip(task_order, rows, route_rows, membership_rows, strict=True)
    ):
        binding_rows.append(
            {
                "schema_version": ROW_BINDING_SCHEMA,
                "opaque_id": task_id,
                "row_index": index,
                "evaluator_route_row_sha256": canonical_row_sha256(route_row),
                "source_union_membership_row_sha256": canonical_row_sha256(membership_row),
                "upstream_rows": {
                    role: {
                        "source_task_id": source_index[role][task_id]["task_id"],
                        "source_row_index": index,
                        "source_row_canonical_sha256": canonical_row_sha256(
                            source_index[role][task_id]
                        ),
                        "normalized_projection_sha256": canonical_row_sha256(
                            normalized_role_projection(pool_row, role)
                        ),
                    }
                    for role in UPSTREAM
                },
            }
        )
    validate_binding_projections(
        pool_rows=rows,
        route_rows=route_rows,
        membership_rows=membership_rows,
        binding_rows=binding_rows,
        upstream_rows=source_rows,
    )

    rows_v11 = [pool_row_v11(row, index) for index, row in enumerate(rows)]
    benchmark_order_v11 = {
        "schema_version": BENCHMARK_ORDER_V11_SCHEMA,
        "benchmark_sha256": BENCHMARK_ID_AUTHORITY["sha256"],
        "rows": task_order,
    }

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        upstream_dir = temporary / "upstream"
        upstream_dir.mkdir()
        upstream_descriptors: dict[str, dict[str, str]] = {}
        for role, descriptor in UPSTREAM.items():
            destination = upstream_dir / f"{role}.jsonl"
            shutil.copyfile(_path(descriptor), destination)
            if sha256_file(destination) != descriptor["sha256"]:
                raise PoolBuildError(f"{role}: copied upstream bytes changed")
            upstream_descriptors[role] = {
                "path": f"upstream/{role}.jsonl",
                "sha256": descriptor["sha256"],
            }

        pool_sha = _write_jsonl(temporary / "candidate_pool.jsonl", rows)
        pool_v11_sha = _write_jsonl(temporary / "candidate_pool_v1_1.jsonl", rows_v11)
        benchmark_order_v11_sha = _write_json(
            temporary / "benchmark_order_v1_1.json", benchmark_order_v11
        )
        route_map_v11 = {
            "schema_version": ROUTE_MAP_V11_SCHEMA,
            "benchmark_sha256": BENCHMARK_ID_AUTHORITY["sha256"],
            "benchmark_order_sha256": benchmark_order_v11_sha,
            "derivation": "question_image_format_metadata_only_no_gold_score_correctness_outcome_or_judge",
            "rows": [
                {
                    "row_index": index,
                    "task_id": task_id,
                    "evaluation_route": "image_judge" if task_id in image_set else "deterministic",
                }
                for index, task_id in enumerate(task_order)
            ],
        }
        route_map_v11_sha = _write_json(
            temporary / "evaluator_route_map_v1_1.json", route_map_v11
        )
        source_union_membership_v11 = {
            "schema_version": SOURCE_UNION_MEMBERSHIP_V11_SCHEMA,
            "authority": {
                "aggregate_sha256": SOURCE_AGGREGATE_SHA256,
                "source_union_projection_sha256": SOURCE_UNION_PROJECTION_SHA256,
                "source_union_size": SOURCE_UNION_SIZE,
            },
            "derivation": "projection_task_id_only_no_answer_score_correctness_outcome_or_judge",
            "task_ids": sorted(source_union),
        }
        source_union_membership_v11_sha = _write_json(
            temporary / "source_union_membership_v1_1.json",
            source_union_membership_v11,
        )
        row_bindings_v11 = {
            "schema_version": "maxim-9b-baseline-selector-combined-row-bindings-v1.1",
            "benchmark_sha256": BENCHMARK_ID_AUTHORITY["sha256"],
            "benchmark_order_sha256": benchmark_order_v11_sha,
            "projection_contract": "sha256_of_utf8_json_sort_keys_compact_role_projection",
            "upstream_artifact_sha256": {
                role: descriptor["sha256"] for role, descriptor in UPSTREAM.items()
            },
            "rows": [
                {
                    "row_index": index,
                    "task_id": task_id,
                    "role_projection_sha256": {
                        role: canonical_row_sha256(
                            normalized_role_projection_v11(rows_v11[index], role)
                        )
                        for role in UPSTREAM
                    },
                }
                for index, task_id in enumerate(task_order)
            ],
        }
        validate_v11_artifacts(
            pool_rows=rows_v11,
            benchmark_order=benchmark_order_v11,
            benchmark_order_sha256=benchmark_order_v11_sha,
            route_map=route_map_v11,
            row_bindings=row_bindings_v11,
            expected_task_order=task_order,
            image_ids=image_set,
        )
        row_bindings_v11_sha = _write_json(
            temporary / "row_bindings_v1_1.json", row_bindings_v11
        )
        route_map_sha = _write_jsonl(temporary / "evaluator_route_map.jsonl", route_rows)
        membership_sha = _write_jsonl(
            temporary / "source_union_membership.jsonl", membership_rows
        )
        row_bindings_sha = _write_jsonl(temporary / "row_bindings.jsonl", binding_rows)
        pool_manifest = {
            "schema_version": POOL_MANIFEST_SCHEMA,
            "pool_id": "maxim_9b_baseline_candidate_pool_v1",
            "created_after_selector_freeze": True,
            "created_before_evaluation": True,
            "rows": EXPECTED_ROWS,
            "model_closure": [MODEL],
            "selector_freeze_sha256": SELECTOR_FREEZE_SHA256,
            "selector_profile_sha256": SELECTOR_PROFILE_SHA256,
            "source_union_authority": {
                "aggregate_sha256": SOURCE_AGGREGATE_SHA256,
                "source_union_sha256": SOURCE_UNION_PROJECTION_SHA256,
                "source_union_size": SOURCE_UNION_SIZE,
            },
            "upstream_artifacts": upstream_descriptors,
            "candidate_pool": {"path": "candidate_pool.jsonl", "sha256": pool_sha},
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
        }
        pool_manifest_sha = _write_json(temporary / "pool_manifest.json", pool_manifest)

        input_package_v11 = {
            "schema_version": INPUT_PACKAGE_V11_SCHEMA,
            "rows": EXPECTED_ROWS,
            "benchmark_sha256": BENCHMARK_ID_AUTHORITY["sha256"],
            "created_before_new_arm_evaluation": True,
            "runtime_outcome_access": False,
            "benchmark_order": {
                "path": "benchmark_order_v1_1.json",
                "sha256": benchmark_order_v11_sha,
            },
            "route_map": {
                "path": "evaluator_route_map_v1_1.json",
                "sha256": route_map_v11_sha,
            },
            "source_union_membership": {
                "path": "source_union_membership_v1_1.json",
                "sha256": source_union_membership_v11_sha,
            },
            "upstream_artifacts": upstream_descriptors,
            "row_bindings": {
                "path": "row_bindings_v1_1.json",
                "sha256": row_bindings_v11_sha,
            },
            "candidate_pool": {
                "path": "candidate_pool_v1_1.jsonl",
                "sha256": pool_v11_sha,
            },
        }
        input_package_v11_sha = _write_json(
            temporary / "input_package_v1_1.json", input_package_v11
        )

        benchmark_task_order_sha = canonical_projection_sha256(benchmark_task_order)
        binding_manifest = {
            "schema_version": BINDING_MANIFEST_SCHEMA,
            "status": "frozen_before_selector_or_evaluation",
            "rows": EXPECTED_ROWS,
            "benchmark_task_id_authority": {
                "repository_relative_path": BENCHMARK_ID_AUTHORITY["path"],
                "sha256": BENCHMARK_ID_AUTHORITY["sha256"],
                "ordered_task_ids_sha256": benchmark_task_order_sha,
                "projection": BENCHMARK_ID_AUTHORITY["projection"],
            },
            "evaluator_route_authority": {
                "evaluation_contract": EVALUATION_CONTRACT,
                "image97_template": IMAGE_ROUTE_TEMPLATE,
                "route_map": {
                    "path": "evaluator_route_map.jsonl",
                    "sha256": route_map_sha,
                    "rows": EXPECTED_ROWS,
                    "deterministic_rows": 177,
                    "image_judge_rows": 97,
                },
            },
            "source_union_veto_authority": {
                "aggregate_sha256": SOURCE_AGGREGATE_SHA256,
                "aggregate_source_union_projection_sha256": SOURCE_UNION_PROJECTION_SHA256,
                "source_union_size": SOURCE_UNION_SIZE,
                "membership_map": {
                    "path": "source_union_membership.jsonl",
                    "sha256": membership_sha,
                    "rows": EXPECTED_ROWS,
                },
            },
            "row_binding_authority": {
                "path": "row_bindings.jsonl",
                "sha256": row_bindings_sha,
                "rows": EXPECTED_ROWS,
                "canonicalization": "UTF-8 sorted-key compact JSON plus LF",
                "bindings": [
                    "opaque_id_to_ordered_benchmark_index",
                    "opaque_id_to_independent_evaluator_route_row",
                    "opaque_id_to_source_union_veto_row",
                    "opaque_id_and_index_to_each_upstream_source_row_content",
                    "each_upstream_source_row_to_normalized_candidate_projection",
                ],
            },
            "candidate_pool": {"path": "candidate_pool.jsonl", "sha256": pool_sha},
            "upstream_artifacts": upstream_descriptors,
            "model_closure": [MODEL],
            "inherited_27b_outputs": False,
            "gold_reference_score_correctness_judge_outcome_used": False,
        }
        binding_manifest_sha = _write_json(
            temporary / "binding_manifest.json", binding_manifest
        )

        answers = {
            role: [_answer(source_index[role][task_id]) for task_id in task_order]
            for role in UPSTREAM
        }
        disagreements: dict[str, int] = {}
        roles = list(UPSTREAM)
        for left_index, left in enumerate(roles):
            for right in roles[left_index + 1 :]:
                disagreements[f"{left}__vs__{right}"] = sum(
                    a != b for a, b in zip(answers[left], answers[right], strict=True)
                )
        vote_diagnostics: dict[str, Any] = {}
        for role in ("parallel8_v1", "parallel8_reasoning_first_v2"):
            parse_errors = 0
            distinct_distribution: Counter[int] = Counter()
            total_input = total_output = 0
            for row in source_rows[role]:
                traces = row["generation"]["candidate_traces"]
                distinct_distribution[len({_answer(trace) for trace in traces})] += 1
                for trace in traces:
                    parse_errors += trace.get("parse_error") is not None
                    total_input += _nonnegative_int(trace.get("input_tokens"))
                    total_output += _nonnegative_int(trace.get("output_tokens"))
            vote_diagnostics[role] = {
                "raw_votes": EXPECTED_ROWS * 8,
                "parse_error_or_recovery_markers": parse_errors,
                "distinct_answer_count_distribution": {
                    str(key): distinct_distribution[key] for key in sorted(distinct_distribution)
                },
                "raw_vote_input_tokens": total_input,
                "raw_vote_output_tokens": total_output,
            }

        task_order_sha = canonical_projection_sha256(task_order)
        image_ids_sha = canonical_projection_sha256(sorted(image_set))
        source_union_ids_sha = canonical_projection_sha256(sorted(source_union))
        content_projection = {
            "schema_version": NORMALIZATION_SCHEMA,
            "rows": EXPECTED_ROWS,
            "task_order_sha256": task_order_sha,
            "model_closure": [MODEL],
            "upstream_sha256": {role: descriptor["sha256"] for role, descriptor in UPSTREAM.items()},
            "candidate_pool_sha256": pool_sha,
            "candidate_pool_v1_1_sha256": pool_v11_sha,
            "input_package_v1_1_sha256": input_package_v11_sha,
            "benchmark_order_v1_1_sha256": benchmark_order_v11_sha,
            "evaluator_route_map_v1_1_sha256": route_map_v11_sha,
            "source_union_membership_v1_1_sha256": source_union_membership_v11_sha,
            "row_bindings_v1_1_sha256": row_bindings_v11_sha,
            "binding_manifest_sha256": binding_manifest_sha,
            "row_bindings_sha256": row_bindings_sha,
            "evaluator_route_map_sha256": route_map_sha,
            "source_union_membership_sha256": membership_sha,
            "source_union_task_ids_sha256": source_union_ids_sha,
            "image_evaluator_task_ids_sha256": image_ids_sha,
            "benchmark_sha256": BENCHMARK_ID_AUTHORITY["sha256"],
            "benchmark_ordered_task_ids_sha256": benchmark_task_order_sha,
            "selector_target_schema": "maxim-9b-baseline-selector-profile-v1.1",
        }
        normalization_manifest = {
            "schema_version": NORMALIZATION_SCHEMA,
            "status": "frozen_normalized_input_not_evaluated",
            "rows": EXPECTED_ROWS,
            "unique_task_ids": EXPECTED_ROWS,
            "model_closure": [MODEL],
            "answer_producing_model_closure": [MODEL],
            "inherited_27b_outputs": False,
            "source_or_outcome_routing_used": False,
            "content_projection_contract": (
                "compact UTF-8 JSON with sorted keys and no projection newline; excludes paths outside pinned repository-relative "
                "descriptors and excludes timestamps"
            ),
            "content_projection": content_projection,
            "content_projection_sha256": canonical_projection_sha256(content_projection),
            "artifacts": {
                "candidate_pool": {"path": "candidate_pool.jsonl", "sha256": pool_sha},
                "candidate_pool_v1_1": {
                    "path": "candidate_pool_v1_1.jsonl",
                    "sha256": pool_v11_sha,
                },
                "input_package_v1_1": {
                    "path": "input_package_v1_1.json",
                    "sha256": input_package_v11_sha,
                },
                "benchmark_order_v1_1": {
                    "path": "benchmark_order_v1_1.json",
                    "sha256": benchmark_order_v11_sha,
                },
                "evaluator_route_map_v1_1": {
                    "path": "evaluator_route_map_v1_1.json",
                    "sha256": route_map_v11_sha,
                },
                "source_union_membership_v1_1": {
                    "path": "source_union_membership_v1_1.json",
                    "sha256": source_union_membership_v11_sha,
                },
                "row_bindings_v1_1": {
                    "path": "row_bindings_v1_1.json",
                    "sha256": row_bindings_v11_sha,
                },
                "binding_manifest": {
                    "path": "binding_manifest.json",
                    "sha256": binding_manifest_sha,
                },
                "row_bindings": {"path": "row_bindings.jsonl", "sha256": row_bindings_sha},
                "evaluator_route_map": {
                    "path": "evaluator_route_map.jsonl",
                    "sha256": route_map_sha,
                    "deterministic_rows": EXPECTED_ROWS - len(image_set),
                    "image_judge_rows": len(image_set),
                },
                "source_union_membership": {
                    "path": "source_union_membership.jsonl",
                    "sha256": membership_sha,
                    "protected_rows": SOURCE_UNION_SIZE,
                },
                "selector_pool_manifest": {
                    "path": "pool_manifest.json",
                    "sha256": pool_manifest_sha,
                },
            },
            "selector_chronology": {
                "v1_status": "superseded_not_executed_after_independent_audit_fail",
                "v1_1_status": "input_locked_before_profile_authority_pins_and_freeze",
                "selection_or_evaluation_performed": False,
            },
            "upstream_artifacts": {
                role: {
                    "repository_relative_path": descriptor["path"],
                    "snapshot_path": f"upstream/{role}.jsonl",
                    "sha256": descriptor["sha256"],
                    "model_closure": [MODEL],
                    "provenance": descriptor["provenance"],
                }
                for role, descriptor in UPSTREAM.items()
            },
            "metadata_authorities": {
                "benchmark_task_id_authority": {
                    **BENCHMARK_ID_AUTHORITY,
                    "rows": EXPECTED_ROWS,
                    "ordered_task_ids_sha256": benchmark_task_order_sha,
                },
                "public_task_queue": PUBLIC_QUEUE,
                "image_evaluator_route": {
                    **IMAGE_ROUTE_TEMPLATE,
                    "evaluation_contract": EVALUATION_CONTRACT,
                    "task_ids": 97,
                    "task_id_projection_sha256": image_ids_sha,
                },
                "image_geometry": GEOMETRY_SOURCE,
                "truncated_image_geometry": TRUNCATED_IMAGE_AMENDMENT,
            },
            "source_union_veto": {
                "aggregate_sha256": SOURCE_AGGREGATE_SHA256,
                "aggregate_source_union_projection_sha256": SOURCE_UNION_PROJECTION_SHA256,
                "task_id_membership_projection_sha256": source_union_ids_sha,
                "size": SOURCE_UNION_SIZE,
                "policy": "safety_veto_only; protected rows must preserve ActiveCrop anchor",
                **source_union_meta,
            },
            "candidate_matrix_without_outcomes": {
                "pairwise_exact_answer_disagreements": disagreements,
                "parallel_vote_diagnostics": vote_diagnostics,
                "evaluator_route_counts": {
                    "deterministic": EXPECTED_ROWS - len(image_set),
                    "image_judge": len(image_set),
                },
                "source_union_protected_rows": SOURCE_UNION_SIZE,
            },
            "access_attestations": {
                "gold_or_reference_answer_used": False,
                "score_correctness_or_outcome_used": False,
                "judge_verdict_used": False,
                "known_error_memory_used": False,
                "only_task_id_projected_from_image_evaluator_template": True,
                "only_task_id_projected_from_source_certificates": True,
            },
            "provenance_caveats": {
                "native_v4_v5": "correlated ablations, not independent systems",
                "parallel8_v1_v2": (
                    "fresh generations without inherited solver rows, but exact endpoint revision was not "
                    "immutably recorded; v2 is core and v1 is a diversity donor"
                ),
            },
        }
        normalization_sha = _write_json(
            temporary / "normalization_manifest.json", normalization_manifest
        )
        checksum_entries = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS.txt":
                checksum_entries.append(
                    f"{sha256_file(path)}  {path.relative_to(temporary).as_posix()}"
                )
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(checksum_entries) + "\n", encoding="utf-8", newline="\n"
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "status": "normalized_candidate_pool_frozen_not_evaluated",
        "output_dir": str(output_dir),
        "rows": EXPECTED_ROWS,
        "candidate_pool_sha256": pool_sha,
        "candidate_pool_v1_1_sha256": pool_v11_sha,
        "input_package_v1_1_sha256": input_package_v11_sha,
        "pool_manifest_sha256": pool_manifest_sha,
        "binding_manifest_sha256": binding_manifest_sha,
        "normalization_manifest_sha256": normalization_sha,
        "content_projection_sha256": normalization_manifest["content_projection_sha256"],
        "source_union_membership_sha256": membership_sha,
    }


def verify_frozen(input_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise PoolBuildError(f"frozen input directory does not exist: {input_dir}")
    normalization = read_json(input_dir / "normalization_manifest.json", "normalization manifest")
    binding_manifest = read_json(input_dir / "binding_manifest.json", "binding manifest")
    pool_manifest = read_json(input_dir / "pool_manifest.json", "selector pool manifest")
    if normalization.get("schema_version") != NORMALIZATION_SCHEMA:
        raise PoolBuildError("normalization manifest schema mismatch")
    if binding_manifest.get("schema_version") != BINDING_MANIFEST_SCHEMA:
        raise PoolBuildError("binding manifest schema mismatch")
    if pool_manifest.get("schema_version") != POOL_MANIFEST_SCHEMA:
        raise PoolBuildError("selector pool manifest schema mismatch")
    if normalization.get("content_projection_sha256") != canonical_projection_sha256(
        normalization.get("content_projection")
    ):
        raise PoolBuildError("normalization content projection hash mismatch")

    artifact_paths = {
        "candidate_pool": input_dir / "candidate_pool.jsonl",
        "candidate_pool_v1_1": input_dir / "candidate_pool_v1_1.jsonl",
        "input_package_v1_1": input_dir / "input_package_v1_1.json",
        "benchmark_order_v1_1": input_dir / "benchmark_order_v1_1.json",
        "evaluator_route_map_v1_1": input_dir / "evaluator_route_map_v1_1.json",
        "source_union_membership_v1_1": input_dir / "source_union_membership_v1_1.json",
        "row_bindings_v1_1": input_dir / "row_bindings_v1_1.json",
        "evaluator_route_map": input_dir / "evaluator_route_map.jsonl",
        "source_union_membership": input_dir / "source_union_membership.jsonl",
        "row_bindings": input_dir / "row_bindings.jsonl",
        "binding_manifest": input_dir / "binding_manifest.json",
        "selector_pool_manifest": input_dir / "pool_manifest.json",
    }
    for key, path in artifact_paths.items():
        descriptor = normalization.get("artifacts", {}).get(key)
        if not isinstance(descriptor, dict) or descriptor.get("sha256") != sha256_file(path):
            raise PoolBuildError(f"frozen artifact {key} hash mismatch")

    pool_rows = read_jsonl(artifact_paths["candidate_pool"], "candidate pool")
    route_rows = read_jsonl(artifact_paths["evaluator_route_map"], "evaluator route map")
    membership_rows = read_jsonl(
        artifact_paths["source_union_membership"], "source-union membership"
    )
    binding_rows = read_jsonl(artifact_paths["row_bindings"], "row bindings")
    pool_rows_v11 = read_jsonl(
        artifact_paths["candidate_pool_v1_1"], "v1.1 candidate pool"
    )
    input_package_v11 = read_json(
        artifact_paths["input_package_v1_1"], "v1.1 input package"
    )
    benchmark_order_v11 = read_json(
        artifact_paths["benchmark_order_v1_1"], "v1.1 benchmark order"
    )
    route_map_v11 = read_json(
        artifact_paths["evaluator_route_map_v1_1"], "v1.1 evaluator route map"
    )
    source_union_membership_v11 = read_json(
        artifact_paths["source_union_membership_v1_1"],
        "v1.1 source-union membership",
    )
    row_bindings_v11 = read_json(
        artifact_paths["row_bindings_v1_1"], "v1.1 row bindings"
    )

    upstream_rows: dict[str, list[dict[str, Any]]] = {}
    for role, descriptor in UPSTREAM.items():
        snapshot = input_dir / "upstream" / f"{role}.jsonl"
        if sha256_file(snapshot) != descriptor["sha256"]:
            raise PoolBuildError(f"frozen upstream snapshot {role} hash mismatch")
        rows = read_jsonl(snapshot, f"frozen upstream snapshot {role}")
        validate_answer_source_rows(rows, f"frozen upstream snapshot {role}")
        upstream_rows[role] = rows
    validate_binding_projections(
        pool_rows=pool_rows,
        route_rows=route_rows,
        membership_rows=membership_rows,
        binding_rows=binding_rows,
        upstream_rows=upstream_rows,
    )

    benchmark_path = assert_pinned(BENCHMARK_ID_AUTHORITY, "benchmark task-id authority")
    benchmark_order = read_ordered_task_id_prefixes(
        benchmark_path, "benchmark task-id authority"
    )
    pool_order = [str(row.get("opaque_id") or "") for row in pool_rows]
    if pool_order != benchmark_order:
        raise PoolBuildError("candidate pool order differs from benchmark task-id projection")

    image_path = assert_pinned(IMAGE_ROUTE_TEMPLATE, "image evaluator route template")
    image_ids = frozenset(_task_id_projection(image_path, "image evaluator route template"))
    expected_routes = [
        {
            "schema_version": EVALUATOR_ROUTE_SCHEMA,
            "opaque_id": task_id,
            "evaluation_route": "image_judge" if task_id in image_ids else "deterministic",
        }
        for task_id in benchmark_order
    ]
    if route_rows != expected_routes:
        raise PoolBuildError("evaluator route map differs from independent frozen authority")
    validate_v11_artifacts(
        pool_rows=pool_rows_v11,
        benchmark_order=benchmark_order_v11,
        benchmark_order_sha256=sha256_file(artifact_paths["benchmark_order_v1_1"]),
        route_map=route_map_v11,
        row_bindings=row_bindings_v11,
        expected_task_order=benchmark_order,
        image_ids=image_ids,
    )

    expected_package_keys = {
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
    if set(input_package_v11) != expected_package_keys:
        raise PoolBuildError("v1.1 input package exact key set mismatch")
    if input_package_v11.get("schema_version") != INPUT_PACKAGE_V11_SCHEMA:
        raise PoolBuildError("v1.1 input package schema mismatch")
    if input_package_v11.get("rows") != EXPECTED_ROWS:
        raise PoolBuildError("v1.1 input package row count mismatch")
    if input_package_v11.get("benchmark_sha256") != BENCHMARK_ID_AUTHORITY["sha256"]:
        raise PoolBuildError("v1.1 input package benchmark pin mismatch")
    if input_package_v11.get("created_before_new_arm_evaluation") is not True:
        raise PoolBuildError("v1.1 input package chronology mismatch")
    if input_package_v11.get("runtime_outcome_access") is not False:
        raise PoolBuildError("v1.1 input package outcome access is not false")
    expected_descriptors = {
        "benchmark_order": artifact_paths["benchmark_order_v1_1"],
        "route_map": artifact_paths["evaluator_route_map_v1_1"],
        "source_union_membership": artifact_paths["source_union_membership_v1_1"],
        "row_bindings": artifact_paths["row_bindings_v1_1"],
        "candidate_pool": artifact_paths["candidate_pool_v1_1"],
    }
    for key, path in expected_descriptors.items():
        if input_package_v11.get(key) != {
            "path": path.name,
            "sha256": sha256_file(path),
        }:
            raise PoolBuildError(f"v1.1 input package {key} descriptor mismatch")
    expected_upstream_descriptors = {
        role: {
            "path": f"upstream/{role}.jsonl",
            "sha256": descriptor["sha256"],
        }
        for role, descriptor in UPSTREAM.items()
    }
    if input_package_v11.get("upstream_artifacts") != expected_upstream_descriptors:
        raise PoolBuildError("v1.1 input package upstream descriptor mismatch")

    source_union, _ = load_source_union()
    expected_membership_v11 = {
        "schema_version": SOURCE_UNION_MEMBERSHIP_V11_SCHEMA,
        "authority": {
            "aggregate_sha256": SOURCE_AGGREGATE_SHA256,
            "source_union_projection_sha256": SOURCE_UNION_PROJECTION_SHA256,
            "source_union_size": SOURCE_UNION_SIZE,
        },
        "derivation": "projection_task_id_only_no_answer_score_correctness_outcome_or_judge",
        "task_ids": sorted(source_union),
    }
    if source_union_membership_v11 != expected_membership_v11:
        raise PoolBuildError("v1.1 source-union membership authority mismatch")
    expected_membership = [
        {
            "schema_version": MEMBERSHIP_SCHEMA,
            "opaque_id": task_id,
            "protected_by_source_union": task_id in source_union,
        }
        for task_id in benchmark_order
    ]
    if membership_rows != expected_membership:
        raise PoolBuildError("source-union veto map differs from pinned certificates")
    return {
        "status": "frozen_normalized_input_verified_not_evaluated",
        "rows": len(pool_rows),
        "candidate_pool_sha256": sha256_file(artifact_paths["candidate_pool"]),
        "candidate_pool_v1_1_sha256": sha256_file(artifact_paths["candidate_pool_v1_1"]),
        "input_package_v1_1_sha256": sha256_file(artifact_paths["input_package_v1_1"]),
        "binding_manifest_sha256": sha256_file(artifact_paths["binding_manifest"]),
        "content_projection_sha256": normalization["content_projection_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen gold-blind 9B selector input")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "frozen",
    )
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    try:
        report = (
            verify_frozen(args.output_dir)
            if args.verify_existing
            else build_pool(args.output_dir)
        )
    except PoolBuildError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
