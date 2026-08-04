"""Run the preregistered query-conditioned active-crop verifier.

The runner accepts only the public, gold-free queue.  One call locates at most
two evidence regions; deterministic Pillow transforms create 2x--4x crops; a
second call compares the frozen no-tools proposal against the original pixels
and crops.  Selection is intentionally deferred to the frozen composer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps

try:
    import prepare_maxim_query_active_crop_v2 as prepare
    import run_maxim_active_vision_v1 as active
    import run_maxim_agent_ideas as core
except ModuleNotFoundError:  # Imported as scripts.run_maxim_query_active_crop_v2.
    from scripts import prepare_maxim_query_active_crop_v2 as prepare
    from scripts import run_maxim_active_vision_v1 as active
    from scripts import run_maxim_agent_ideas as core


CONDITION = prepare.CONDITION + "_raw"
LOCATOR_SEED = 26080311
SOLVER_SEED = 26080329

REGION_SCHEMA = {
    "type": "object",
    "properties": {
        "image_index": {"type": "integer", "minimum": 0, "maximum": 7},
        "evidence_role": {
            "type": "string",
            "enum": [
                "stem_and_options",
                "diagram",
                "graph",
                "table",
                "formula",
                "map",
                "dense_text",
                "mixed",
            ],
        },
        "bbox_1000": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            "minItems": 4,
            "maxItems": 4,
        },
        "visible_anchor": {"type": "string", "minLength": 1, "maxLength": 240},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "image_index",
        "evidence_role",
        "bbox_1000",
        "visible_anchor",
        "confidence",
    ],
    "additionalProperties": False,
}

LOCATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "items": REGION_SCHEMA,
            "minItems": 1,
            "maxItems": 2,
        },
        "second_zoom_needed": {"type": "boolean"},
        "missing_evidence_risk": {"type": "string", "maxLength": 360},
        "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "regions",
        "second_zoom_needed",
        "missing_evidence_risk",
        "overall_confidence",
    ],
    "additionalProperties": False,
}

VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "visible_facts": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 280},
            "minItems": 2,
            "maxItems": 8,
        },
        "verification_checks": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 360},
            "minItems": 2,
            "maxItems": 4,
        },
        "baseline_supported": {"type": "boolean"},
        "baseline_error": {"type": "string", "maxLength": 500},
        "all_required_evidence_visible": {"type": "boolean"},
        "original_crop_consistent": {"type": "boolean"},
        "answer_format_verified": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string", "maxLength": 2200},
        "solution_steps": {"type": "string", "maxLength": 2600},
        "candidate_answer": {"type": "string", "minLength": 1, "maxLength": 120},
    },
    "required": [
        "visible_facts",
        "verification_checks",
        "baseline_supported",
        "baseline_error",
        "all_required_evidence_visible",
        "original_crop_consistent",
        "answer_format_verified",
        "confidence",
        "reasoning",
        "solution_steps",
        "candidate_answer",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a careful Turkish-school-question visual verifier. There is no "
    "answer key, reference answer, gold label, score, or judge feedback. Use "
    "only the supplied question pixels and the explicitly marked provisional "
    "answer. A crop is only a deterministic enlargement of original pixels. "
    "Do not change the provisional answer merely to be different: change it "
    "only when visible evidence proves a concrete error. Return exactly the "
    "requested JSON schema."
)


def _normalised_bbox(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox_1000 must contain four values")
    x1, y1, x2, y2 = (max(0, min(1000, int(item))) for item in value)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 < 60:
        centre = (x1 + x2) // 2
        x1, x2 = max(0, centre - 30), min(1000, centre + 30)
    if y2 - y1 < 60:
        centre = (y1 + y2) // 2
        y1, y2 = max(0, centre - 30), min(1000, centre + 30)
    return x1, y1, x2, y2


def _pixel_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int, padding: float = 0.12
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left, top = x1 * width / 1000, y1 * height / 1000
    right, bottom = x2 * width / 1000, y2 * height / 1000
    pad_x = max(8.0, (right - left) * padding)
    pad_y = max(8.0, (bottom - top) * padding)
    return (
        max(0, int(left - pad_x)),
        max(0, int(top - pad_y)),
        min(width, int(right + pad_x + 0.999)),
        min(height, int(bottom + pad_y + 0.999)),
    )


def _upscale_factor(crop_size: tuple[int, int]) -> int:
    shortest = min(crop_size)
    if shortest <= 450:
        return 4
    if shortest <= 750:
        return 3
    return 2


def _native_crop(
    image_path: Path, bbox_1000: Any
) -> tuple[str, dict[str, Any]]:
    with Image.open(image_path) as loaded:
        original = ImageOps.exif_transpose(loaded).convert("RGB")
    normalised = _normalised_bbox(bbox_1000)
    pixel_bbox = _pixel_bbox(normalised, *original.size)
    crop = original.crop(pixel_bbox)
    scale = _upscale_factor(crop.size)
    crop = crop.resize(
        (crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS
    )
    crop = ImageEnhance.Contrast(crop).enhance(1.08)
    crop = ImageEnhance.Sharpness(crop).enhance(1.18)
    metadata = {
        "source_image": image_path.name,
        "source_size": list(original.size),
        "bbox_1000": list(normalised),
        "pixel_bbox": list(pixel_bbox),
        "native_crop_size": [pixel_bbox[2] - pixel_bbox[0], pixel_bbox[3] - pixel_bbox[1]],
        "upscale_factor": scale,
        "output_crop_size": list(crop.size),
        "padding_fraction": 0.12,
        "resampling": "LANCZOS",
    }
    return active._png_data_url(crop), metadata


def _validate_request(row: dict[str, Any]) -> None:
    expected = str(row.get("request_sha256") or "")
    unhashed = {key: value for key, value in row.items() if key != "request_sha256"}
    if prepare.canonical_sha256(unhashed) != expected:
        raise ValueError(f"request hash mismatch for {row.get('task_id')}")
    prepare.assert_public_payload(unhashed)
    if row.get("schema_version") != prepare.QUEUE_SCHEMA:
        raise ValueError("queue schema mismatch")
    if row.get("condition") != prepare.CONDITION:
        raise ValueError("queue condition mismatch")


def _image_paths(task: dict[str, Any], image_root: Path) -> list[Path]:
    return active._image_paths(task, image_root)


def _fallback_text(fallback: dict[str, Any]) -> str:
    return (
        f"Provisional answer: {fallback.get('final_answer')}\n"
        f"Provisional reasoning:\n{str(fallback.get('reasoning') or '')[:prepare.MAX_FALLBACK_CONTEXT_CHARS]}\n"
        f"Provisional steps:\n{str(fallback.get('solution_steps') or '')[:prepare.MAX_FALLBACK_CONTEXT_CHARS]}"
    )


def _locator_messages(
    request: dict[str, Any], *, image_root: Path, image_url_root: str
) -> list[dict[str, Any]]:
    task = request["task"]
    instruction = (
        "Inspect every untouched original image. Locate the smallest region "
        "that can verify OR falsify the provisional answer. Include the exact "
        "stem/options when answer-letter mapping matters. A second region is "
        "allowed only when essential evidence is spatially separate (for "
        "example graph plus options, table plus stem, or diagram plus labels). "
        "Use integer [left,top,right,bottom] coordinates on a 0..1000 scale. "
        "Do not decide which answer is correct in this locator call."
    )
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"{core._task_prompt(task)}\n\n{_fallback_text(request['fallback'])}"
            f"\n\n{instruction}"
        ),
    }]
    content.extend(core._image_blocks(task, image_root=image_root, image_url_root=image_url_root))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def _solver_messages(
    request: dict[str, Any],
    locator: dict[str, Any],
    crop_urls: list[str],
    crop_metadata: list[dict[str, Any]],
    *,
    image_root: Path,
    image_url_root: str,
) -> list[dict[str, Any]]:
    task = request["task"]
    instruction = (
        "Independently verify the provisional answer against visible pixels. "
        "The untouched original image(s) come first and are authoritative; "
        "then come one or two deterministic enlarged crops in metadata order. "
        "First transcribe at least two concrete visible facts, then perform at "
        "least two checks (such as recomputation and option mapping, or graph "
        "reading and stem-negation check). Set baseline_supported=false only "
        "when you can name its concrete visible or logical error. If evidence "
        "is incomplete, crops disagree with originals, or confidence is below "
        "0.90, preserve the provisional answer. For multiple choice, "
        "candidate_answer is only the option letter."
    )
    safe_locator = {
        "regions": locator.get("regions"),
        "second_zoom_needed": locator.get("second_zoom_needed"),
        "missing_evidence_risk": locator.get("missing_evidence_risk"),
        "overall_confidence": locator.get("overall_confidence"),
    }
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"{core._task_prompt(task)}\n\n{_fallback_text(request['fallback'])}\n\n"
            f"{instruction}\n\nLocator metadata (not an answer): "
            f"{json.dumps(safe_locator, ensure_ascii=False)}\n"
            f"Crop metadata: {json.dumps(crop_metadata, ensure_ascii=False)}"
        ),
    }]
    content.extend(core._image_blocks(task, image_root=image_root, image_url_root=image_url_root))
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in crop_urls)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def _compact(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": call.get("endpoint"),
        "finish_reason": call.get("finish_reason"),
        "attempt": call.get("attempt"),
        "latency_s": call.get("latency_s"),
        "input_tokens": call.get("input_tokens"),
        "output_tokens": call.get("output_tokens"),
        "recovered_partial": bool(call.get("recovered_partial")),
        "parse_error": call.get("parse_error"),
    }


def run_request(
    request: dict[str, Any], *, pool: core.EndpointPool, image_root: Path,
    image_url_root: str, locator_max_tokens: int, solver_max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []
    locator: dict[str, Any] = {}
    verifier: dict[str, Any] = {}
    crop_metadata: list[dict[str, Any]] = []
    error: str | None = None
    try:
        _validate_request(request)
        locator_call = pool.complete(
            messages=_locator_messages(request, image_root=image_root, image_url_root=image_url_root),
            schema_name="query_active_crop_locator_v2",
            schema=LOCATOR_SCHEMA,
            max_tokens=locator_max_tokens,
            temperature=0.0,
            seed=LOCATOR_SEED,
            retries=1,
        )
        calls.append(locator_call)
        locator = locator_call["parsed"]
        regions = list(locator.get("regions") or [])[:2]
        if not bool(locator.get("second_zoom_needed")):
            regions = regions[:1]
        if not regions:
            raise ValueError("locator returned zero used regions")
        paths = _image_paths(request["task"], image_root)
        crop_urls: list[str] = []
        used_regions: list[dict[str, Any]] = []
        for region in regions:
            image_index = max(0, min(len(paths) - 1, int(region["image_index"])))
            crop_url, metadata = _native_crop(paths[image_index], region["bbox_1000"])
            metadata["image_index"] = image_index
            metadata["evidence_role"] = region.get("evidence_role")
            metadata["visible_anchor"] = region.get("visible_anchor")
            metadata["locator_confidence"] = region.get("confidence")
            crop_urls.append(crop_url)
            crop_metadata.append(metadata)
            used = copy.deepcopy(region)
            used["image_index"] = image_index
            used_regions.append(used)
        locator["used_regions"] = used_regions
        verifier_call = pool.complete(
            messages=_solver_messages(
                request, locator, crop_urls, crop_metadata,
                image_root=image_root, image_url_root=image_url_root,
            ),
            schema_name="query_active_crop_verifier_v2",
            schema=VERIFIER_SCHEMA,
            max_tokens=solver_max_tokens,
            temperature=0.0,
            seed=SOLVER_SEED,
            retries=1,
        )
        calls.append(verifier_call)
        verifier = verifier_call["parsed"]
        if not str(verifier.get("candidate_answer") or "").strip():
            raise ValueError("verifier returned empty candidate_answer")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    answer = str(verifier.get("candidate_answer") or "").strip() or None
    return {
        "task_id": str(request.get("task_id") or ""),
        "condition": CONDITION,
        "model": pool.model,
        "prompt_version": CONDITION,
        "final_answer": answer,
        "solution_steps": str(verifier.get("solution_steps") or "").strip() or None,
        "reasoning": str(verifier.get("reasoning") or "").strip() or None,
        "forced_answer": False,
        "raw_response": json.dumps(verifier, ensure_ascii=False) if verifier else None,
        "generation": {
            "temperature": 0.0,
            "top_p": 0.95,
            "structured_mode": "response_format_json_schema",
            "enable_thinking": False,
            "gold_access": False,
            "idea": "query_conditioned_active_crop_plus_conservative_baseline_verifier",
            "request_sha256": request.get("request_sha256"),
            "route_reasons": request.get("route_reasons"),
            "baseline_answer": request.get("fallback", {}).get("final_answer"),
            "locator": locator or None,
            "active_crops": crop_metadata,
            "selection_evidence": {
                key: verifier.get(key)
                for key in (
                    "visible_facts", "verification_checks", "baseline_supported",
                    "baseline_error", "all_required_evidence_visible",
                    "original_crop_consistent", "answer_format_verified", "confidence",
                )
            } if verifier else None,
            "logical_call_count": len(calls),
            "call_traces": [_compact(call) for call in calls],
        },
        "tool_calls": [],
        "usage": {
            "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls),
            "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls),
            "latency_s": round(time.perf_counter() - started, 3),
        },
        "error": error,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-url-root", default="file:///images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", default=core.MODEL)
    parser.add_argument("--task-concurrency", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--locator-max-tokens", type=int, default=1200)
    parser.add_argument("--solver-max-tokens", type=int, default=3072)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args(argv)
    if active.sha256_file(args.queue) != args.queue_sha256:
        raise SystemExit("queue SHA256 mismatch")
    if not 1 <= args.task_concurrency <= 16:
        raise SystemExit("--task-concurrency must be in [1, 16]")
    requests = core._load_jsonl(args.queue)
    for request in requests:
        _validate_request(request)
    if args.task_id:
        selected = set(args.task_id)
        requests = [row for row in requests if str(row["task_id"]) in selected]
        missing = selected - {str(row["task_id"]) for row in requests}
        if missing:
            raise SystemExit(f"unknown task IDs: {sorted(missing)}")
    if args.limit is not None:
        requests = requests[: args.limit]
    order = [str(row["task_id"]) for row in requests]
    if len(order) != len(set(order)):
        raise SystemExit("duplicate queue task IDs")
    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        if not (args.resume or args.retry_errors):
            raise SystemExit("output exists; pass --resume or --retry-errors")
        for row in core._load_jsonl(args.output):
            task_id = str(row.get("task_id") or "")
            if task_id in set(order) and not (args.retry_errors and row.get("error")):
                existing[task_id] = row
    pool = core.EndpointPool(args.base_url, model=args.model, timeout_s=args.timeout_s)
    output = dict(existing)
    pending = [row for row in requests if str(row["task_id"]) not in existing]
    lock = threading.Lock()

    def execute(request: dict[str, Any]) -> dict[str, Any]:
        return run_request(
            request, pool=pool, image_root=args.image_root,
            image_url_root=args.image_url_root,
            locator_max_tokens=args.locator_max_tokens,
            solver_max_tokens=args.solver_max_tokens,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.task_concurrency) as executor:
        futures = {executor.submit(execute, request): request for request in pending}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            request = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "task_id": str(request["task_id"]), "condition": CONDITION,
                    "model": pool.model, "prompt_version": CONDITION,
                    "final_answer": None, "solution_steps": None, "reasoning": None,
                    "forced_answer": False, "raw_response": None,
                    "generation": {"gold_access": False, "logical_call_count": 0},
                    "tool_calls": [],
                    "usage": {"input_tokens": 0, "output_tokens": 0, "latency_s": 0},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with lock:
                output[str(request["task_id"])] = row
                ordered = [output[task_id] for task_id in order if task_id in output]
                _write_rows(args.output, ordered)
                completed += 1
                print(json.dumps({
                    "completed_now": completed, "total_saved": len(ordered),
                    "target": len(requests), "task_id": request["task_id"],
                    "answer": row.get("final_answer"), "error": row.get("error"),
                }, ensure_ascii=False), flush=True)
    ordered = [output[task_id] for task_id in order if task_id in output]
    _write_rows(args.output, ordered)
    errors = sum(bool(row.get("error")) for row in ordered)
    print(json.dumps({
        "output": str(args.output.resolve()), "rows": len(ordered), "errors": errors,
        "sha256": active.sha256_file(args.output),
    }, ensure_ascii=False))
    return 0 if len(ordered) == len(requests) and errors == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
