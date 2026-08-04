"""Preregister the blind route for query-conditioned active crops.

This preparation step is deliberately unable to consume scores or judge
artifacts.  It derives a narrow route from the frozen no-tools solver's own
public generation signals and native image geometry, then emits a queue that
contains only the question view and the provisional no-tools answer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

try:
    import run_maxim_agent_ideas as core
except ModuleNotFoundError:  # Imported as scripts.prepare_maxim_query_active_crop_v2.
    from scripts import run_maxim_agent_ideas as core


FROZEN_BENCHMARK_SHA256 = (
    "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
)
FROZEN_NO_TOOLS_SHA256 = (
    "496236da966ed68aa81af3d33da1c40b85c5a11b342de253ada244f97320de8f"
)
CONDITION = "maxim_query_conditioned_active_crop_verifier_v2"
PROFILE_SCHEMA = "maxim-query-active-crop-preregistered-profile-v2"
QUEUE_SCHEMA = "maxim-query-active-crop-public-request-v2"
MIN_LARGE_IMAGE_PIXELS = 600_000
MAX_FALLBACK_CONTEXT_CHARS = 5_000

VISUAL_ANCHORS = (
    "görsel",
    "şekil",
    "grafik",
    "tablo",
    "harita",
    "diyagram",
    "resim",
    "figure",
    "chart",
    "map",
    "diagram",
)
UNCERTAINTY_ANCHORS = (
    "muhtemelen",
    "muhtemel",
    "olabilir",
    "görünüyor",
    "emin değil",
    "belirsiz",
    "net değil",
    "yaklaşık",
    "likely",
    "probably",
    "uncertain",
)
FORBIDDEN_KEY_RE = re.compile(
    r"(?:reference(?:_answer|_solution)?|gold(?:_answer)?|judge|verdict|score|correct)",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError(f"{label} row {row_number}: missing task_id")
        if task_id in result:
            raise ValueError(f"{label}: duplicate task_id {task_id}")
        result[task_id] = row
    return result


def assert_public_payload(value: Any, path: str = "$") -> None:
    """Reject gold, score, and judge-shaped keys anywhere in public requests."""
    if isinstance(value, dict):
        for key, child in value.items():
            if FORBIDDEN_KEY_RE.search(str(key)):
                raise ValueError(f"forbidden key at {path}.{key}")
            assert_public_payload(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_public_payload(child, f"{path}[{index}]")


def _image_geometry(task: dict[str, Any], image_root: Path) -> list[dict[str, Any]]:
    geometry: list[dict[str, Any]] = []
    for image_index, image in enumerate(task.get("question_images") or []):
        if not isinstance(image, dict):
            continue
        name = Path(str(image.get("data") or "")).name
        if not name:
            continue
        path = (image_root / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as loaded:
            native = ImageOps.exif_transpose(loaded)
            width, height = native.size
        geometry.append(
            {
                "image_index": image_index,
                "file_name": name,
                "width": width,
                "height": height,
                "pixels": width * height,
            }
        )
    if not geometry:
        raise ValueError(f"task {task.get('task_id')}: no usable image")
    return geometry


def route_reasons(
    fallback: dict[str, Any], image_geometry: list[dict[str, Any]]
) -> list[str]:
    """Return the frozen, label-free route reasons in deterministic order."""
    reasons: list[str] = []
    if fallback.get("forced_answer") is True:
        reasons.append("no_tools_forced_answer_parse_signal")
    text = " ".join(
        str(fallback.get(key) or "") for key in ("reasoning", "solution_steps")
    ).casefold()
    large_image = max(int(item["pixels"]) for item in image_geometry) >= MIN_LARGE_IMAGE_PIXELS
    if large_image and _contains_anchor(text, VISUAL_ANCHORS):
        reasons.append("visual_anchor_plus_large_native_image")
    if large_image and _contains_anchor(text, UNCERTAINTY_ANCHORS):
        reasons.append("uncertainty_anchor_plus_large_native_image")
    return reasons


def _contains_anchor(text: str, anchors: tuple[str, ...]) -> bool:
    """Match complete Unicode tokens/phrases, never substrings like map/yapmak."""
    return any(
        re.search(r"(?<!\w)" + re.escape(anchor.casefold()) + r"(?!\w)", text)
        for anchor in anchors
    )


def _fallback_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_answer": str(row.get("final_answer") or "").strip(),
        "reasoning": str(row.get("reasoning") or "")[:MAX_FALLBACK_CONTEXT_CHARS],
        "solution_steps": str(row.get("solution_steps") or "")[:MAX_FALLBACK_CONTEXT_CHARS],
        "forced_answer": bool(row.get("forced_answer")),
        "source_condition": str(row.get("condition") or ""),
    }


def build_queue(
    benchmark_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    image_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    fallback_index = _index(fallback_rows, "fallback")
    benchmark_ids = [str(row.get("task_id") or "") for row in benchmark_rows]
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError("benchmark contains duplicate task IDs")
    if set(benchmark_ids) != set(fallback_index):
        raise ValueError("fallback task-id set differs from benchmark")

    queue: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for raw_task in benchmark_rows:
        unordered_task = core._task_view(raw_task)
        task = {
            key: unordered_task.get(key)
            for key in (
                "task_id", "subject", "grade", "question",
                "question_images", "answer_type",
            )
        }
        task_id = str(task["task_id"])
        fallback = _fallback_view(fallback_index[task_id])
        if not fallback["final_answer"]:
            raise ValueError(f"fallback {task_id}: empty final_answer")
        geometry = _image_geometry(task, image_root)
        reasons = route_reasons(fallback, geometry)
        if not reasons:
            continue
        request = {
            "schema_version": QUEUE_SCHEMA,
            "task_id": task_id,
            "condition": CONDITION,
            "task": task,
            "fallback": fallback,
            "image_geometry": geometry,
            "route_reasons": reasons,
            "generation_contract": {
                "max_zoom_regions": 2,
                "locator_calls": 1,
                "solver_verifier_calls": 1,
                "max_logical_model_calls": 2,
            },
        }
        assert_public_payload(request)
        request["request_sha256"] = canonical_sha256(request)
        queue.append(request)
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return queue, counts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--composer", type=Path, required=True)
    parser.add_argument("--active-dependency", type=Path, required=True)
    parser.add_argument("--core-dependency", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-frozen-sha-check", action="store_true")
    args = parser.parse_args(argv)
    if not args.skip_frozen_sha_check:
        benchmark_sha = sha256_file(args.benchmark)
        fallback_sha = sha256_file(args.fallback)
        if benchmark_sha != FROZEN_BENCHMARK_SHA256:
            raise SystemExit(f"benchmark SHA mismatch: {benchmark_sha}")
        if fallback_sha != FROZEN_NO_TOOLS_SHA256:
            raise SystemExit(f"fallback SHA mismatch: {fallback_sha}")
    code_paths = (
        args.runner, args.composer, args.active_dependency, args.core_dependency
    )
    if not all(path.is_file() for path in code_paths):
        raise SystemExit("all runner/composer/transitive code must exist before preregistration")

    benchmark_rows = core._load_jsonl(args.benchmark)
    fallback_rows = core._load_jsonl(args.fallback)
    queue, route_counts = build_queue(benchmark_rows, fallback_rows, args.image_root)
    if not queue:
        raise SystemExit("blind route selected zero tasks")

    output_dir = args.output_dir.resolve()
    queue_path = output_dir / "public_queue.jsonl"
    profile_path = output_dir / "preregistered_profile.json"
    manifest_path = output_dir / "preparation_manifest.json"
    _write_jsonl(queue_path, queue)
    profile = {
        "schema_version": PROFILE_SCHEMA,
        "status": "frozen_before_generation_and_before_score",
        "condition": CONDITION,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark": {"rows": 274, "sha256": FROZEN_BENCHMARK_SHA256},
        "fallback": {
            "identity": "frozen no-tools Qwen3.5-9B v2_cot",
            "sha256": FROZEN_NO_TOOLS_SHA256,
            "default_for_nonroute_and_every_gate_failure": True,
        },
        "blind_routing": {
            "selected_rows": len(queue),
            "minimum_large_image_pixels": MIN_LARGE_IMAGE_PIXELS,
            "rules_any": [
                "no-tools forced_answer is exactly true",
                "no-tools public reasoning contains a pinned visual anchor and max native image pixels >= 600000",
                "no-tools public reasoning contains a pinned uncertainty anchor and max native image pixels >= 600000",
            ],
            "visual_anchors": list(VISUAL_ANCHORS),
            "uncertainty_anchors": list(UNCERTAINTY_ANCHORS),
            "route_reason_counts": route_counts,
            "labels_scores_or_judge_used": False,
        },
        "model": {
            "name": "Qwen/Qwen3.5-9B",
            "expected_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "backend": "openai-compatible-vllm",
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 0.95,
            "enable_thinking": False,
            "structured_mode": "response_format_json_schema",
            "locator_max_tokens": 1200,
            "solver_verifier_max_tokens": 3072,
            "locator_seed": 26080311,
            "solver_verifier_seed": 26080329,
            "logical_calls_per_routed_task": 2,
            "maximum_http_attempts_per_logical_call": 2,
            "retry_policy": "transport_or_schema_error_only_inside_frozen call",
            "max_fallback_context_chars_per_field": MAX_FALLBACK_CONTEXT_CHARS,
        },
        "active_crop": {
            "maximum_zoom_regions": 2,
            "bbox_coordinates": "[left,top,right,bottom] integer 0..1000",
            "minimum_normalized_side": 60,
            "padding_fraction": 0.12,
            "native_upscale_rule": "4x if min crop side <=450px; 3x if <=750px; otherwise 2x",
            "maximum_upscale": 4,
            "resampling": "Pillow LANCZOS",
            "contrast": 1.08,
            "sharpness": 1.18,
            "solver_inputs": "untouched original image(s) first, then 1-2 deterministic native crops",
        },
        "selection_gate": {
            "default": "copy exact frozen no-tools row",
            "select_active_answer_only_if": [
                "runner has no error and answer differs canonically from no-tools",
                "verifier says baseline_supported is false",
                "verifier confidence >= 0.90",
                "locator overall confidence >= 0.80",
                "every used region confidence >= 0.70",
                "all_required_evidence_visible is true",
                "original_crop_consistent is true",
                "answer_format_verified is true",
                "at least two nonempty visible facts",
                "at least two nonempty verification checks",
                "candidate answer passes deterministic answer-type validation",
            ],
            "any_failure": "copy exact frozen no-tools row",
        },
        "gold_blind_contract": {
            "queue_forbidden_key_regex": FORBIDDEN_KEY_RE.pattern,
            "reference_gold_score_judge_fields_in_queue": False,
            "generation_gold_access": False,
            "score_conditioned_routing": False,
            "prepare_runner_composer_do_not_load_score_or_judge_artifacts": True,
        },
        "evaluation": {
            "benchmark": "frozen274",
            "required_lineage": "frozen-judge-v2-qwen3.5-9b-seed20260714",
            "fresh_judge_only_after_complete_274_row_composition": True,
        },
        "freeze_rule": "No route, prompt, schema, crop, decoding, retry, threshold, composition, or fallback rule may change after generation begins.",
    }
    _write_json(profile_path, profile)
    manifest = {
        "schema_version": "maxim-query-active-crop-preparation-manifest-v2",
        "condition": CONDITION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scoring_performed": False,
        "gold_access": False,
        "sources": {
            "benchmark": {"path": str(args.benchmark.resolve()), "sha256": sha256_file(args.benchmark)},
            "fallback": {"path": str(args.fallback.resolve()), "sha256": sha256_file(args.fallback)},
            "image_root": str(args.image_root.resolve()),
        },
        "code": {
            "prepare": sha256_file(Path(__file__)),
            "runner": sha256_file(args.runner),
            "composer": sha256_file(args.composer),
            "active_crop_dependency": sha256_file(args.active_dependency),
            "endpoint_client_dependency": sha256_file(args.core_dependency),
        },
        "artifacts": {
            "profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
            "queue": {"path": str(queue_path), "rows": len(queue), "sha256": sha256_file(queue_path)},
        },
        "route_reason_counts": route_counts,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
