#!/usr/bin/env python3
"""Gold-blind, conservative Visual Sketchpad treatment for full274.

The model first plans simple marks on the untouched page.  The runner renders
those marks deterministically, then asks the model to solve from the original
pixels plus the rendered sketch.  A frozen active-crop row remains the default:
the sketch candidate is selected only for explicitly visual sketch kinds and a
pre-registered conjunctive evidence gate.  No retrieval or parser transcript is
used, making this treatment independent of element-RAG.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

try:
    import run_maxim_active_vision_v1 as active
    import run_maxim_agent_ideas as core
except ModuleNotFoundError:  # Imported as scripts.run_maxim_visual_sketchpad_v2.
    from scripts import run_maxim_active_vision_v1 as active
    from scripts import run_maxim_agent_ideas as core


FROZEN_PUBLIC_QUEUE_SHA256 = (
    "172183440d95f863f8c7d895d4dbe2ec9b5161cdff19827252d5c7562868993d"
)
FROZEN_FALLBACK_SHA256 = (
    "6697c043f3142a736b817ead5da494eea334f5349e0db833bd72f23fe35cb17c"
)
CONDITION = "maxim_visual_sketchpad_v2_conservative_active_crop_v2"
SCHEMA_VERSION = "maxim-visual-sketchpad-v2"
PLAN_SEED = 26_080_381
SOLVE_SEED = 26_080_382
ELIGIBLE_SKETCH_KINDS = {
    "auxiliary_lines",
    "coordinate_grid",
    "table_guides",
}
FORBIDDEN_KEYS = {
    "reference_answer",
    "reference_solution",
    "gold_answer",
    "gold_solution",
}

POINT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "minLength": 1, "maxLength": 24},
        "x": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y": {"type": "integer", "minimum": 0, "maximum": 1000},
    },
    "required": ["label", "x", "y"],
    "additionalProperties": False,
}

LINE_SCHEMA = {
    "type": "object",
    "properties": {
        "x1": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y1": {"type": "integer", "minimum": 0, "maximum": 1000},
        "x2": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y2": {"type": "integer", "minimum": 0, "maximum": 1000},
        "label": {"type": "string", "maxLength": 40},
    },
    "required": ["x1", "y1", "x2", "y2", "label"],
    "additionalProperties": False,
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "sketch_kind": {
            "type": "string",
            "enum": [
                "auxiliary_lines",
                "coordinate_grid",
                "table_guides",
                "crop_box",
                "none",
            ],
        },
        "image_index": {"type": "integer", "minimum": 0, "maximum": 7},
        "focus_bbox_1000": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            "minItems": 4,
            "maxItems": 4,
        },
        "points": {"type": "array", "items": POINT_SCHEMA, "maxItems": 12},
        "lines": {"type": "array", "items": LINE_SCHEMA, "maxItems": 12},
        "plan_notes": {"type": "string", "minLength": 1, "maxLength": 1000},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "sketch_kind",
        "image_index",
        "focus_bbox_1000",
        "points",
        "lines",
        "plan_notes",
        "confidence",
    ],
    "additionalProperties": False,
}

SOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "visual_facts": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 320},
            "minItems": 2,
            "maxItems": 12,
        },
        "sketch_specific_fact": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
        },
        "verification_checks": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 320},
            "minItems": 2,
            "maxItems": 8,
        },
        "sketch_helpful": {"type": "boolean"},
        "original_sketch_consistent": {"type": "boolean"},
        "answer_format_verified": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "final_answer": {"type": "string", "minLength": 1, "maxLength": 1600},
        "reasoning": {"type": "string", "minLength": 1, "maxLength": 2600},
        "solution_steps": {"type": "string", "minLength": 1, "maxLength": 3000},
    },
    "required": [
        "visual_facts",
        "sketch_specific_fact",
        "verification_checks",
        "sketch_helpful",
        "original_sketch_consistent",
        "answer_format_verified",
        "confidence",
        "final_answer",
        "reasoning",
        "solution_steps",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a careful expert solving Turkish school questions. There is no "
    "answer key, reference answer, gold label, judge feedback, or score. The "
    "untouched original pixels are authoritative; rendered marks are only a "
    "scratchpad and may reflect an imperfect plan. Verify the stem, negation, "
    "symbols, units, every option, and the option-to-letter mapping. Return "
    "exactly the requested JSON schema."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_gold_blind(value: Any, *, location: str) -> None:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            leaked = FORBIDDEN_KEYS.intersection(current)
            if leaked:
                raise ValueError(f"{location}: forbidden keys {sorted(leaked)}")
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def rows_by_task(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        assert_gold_blind(row, location=label)
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in indexed:
            raise ValueError(f"{label}: missing or duplicate task_id {task_id!r}")
        indexed[task_id] = row
    return indexed


def _plan_messages(
    task: dict[str, Any], *, image_root: Path, image_url_root: str
) -> list[dict[str, Any]]:
    instruction = (
        "Inspect the original page but do not solve the question. Plan a small "
        "visual scratchpad operation that could materially reduce a spatial, "
        "graph, geometry, or table-reading error. Coordinates use a 0..1000 "
        "page scale. Use auxiliary_lines for geometric construction, "
        "coordinate_grid for graphs/maps/spatial correspondence, table_guides "
        "for row/column alignment, crop_box for text/formula zoom only, and "
        "none when drawing would not help. Proposed marks must follow visible "
        "anchors; never invent missing geometry."
    )
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"{core._task_prompt(task)}\n\n{instruction}"}
    ]
    content.extend(
        core._image_blocks(
            task, image_root=image_root, image_url_root=image_url_root
        )
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _coord(value: Any, extent: int) -> int:
    return max(0, min(extent - 1, round(int(value) * (extent - 1) / 1000)))


def _render_sketch(
    image_path: Path, plan: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    with Image.open(image_path) as loaded:
        original = ImageOps.exif_transpose(loaded).convert("RGB")
    width, height = original.size
    sketch = original.copy()
    draw = ImageDraw.Draw(sketch)
    stroke = max(4, round(min(original.size) * 0.005))
    kind = str(plan["sketch_kind"])
    bbox = active._normalised_bbox(plan["focus_bbox_1000"])
    pixel_bbox = active._pixel_bbox(bbox, width, height, padding=0.08)

    if kind == "coordinate_grid":
        grid_colour = (80, 130, 180)
        for step in range(100, 1000, 100):
            x = _coord(step, width)
            y = _coord(step, height)
            draw.line((x, 0, x, height - 1), fill=grid_colour, width=max(1, stroke // 3))
            draw.line((0, y, width - 1, y), fill=grid_colour, width=max(1, stroke // 3))

    draw.rectangle(pixel_bbox, outline=(220, 30, 30), width=stroke)
    for line in plan.get("lines") or []:
        draw.line(
            (
                _coord(line["x1"], width),
                _coord(line["y1"], height),
                _coord(line["x2"], width),
                _coord(line["y2"], height),
            ),
            fill=(20, 80, 230),
            width=stroke,
        )
    radius = max(6, stroke * 2)
    for point in plan.get("points") or []:
        x = _coord(point["x"], width)
        y = _coord(point["y"], height)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 210, 20),
            outline=(20, 20, 20),
            width=max(1, stroke // 2),
        )
        draw.text((x + radius + 2, y - radius), str(point["label"]), fill=(0, 0, 0))

    crop = original.crop(pixel_bbox)
    if min(crop.size) < 1200:
        scale = min(4.0, 1200 / max(1, min(crop.size)))
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
    crop = ImageEnhance.Contrast(crop).enhance(1.10)
    crop = ImageEnhance.Sharpness(crop).enhance(1.20)
    metadata = {
        "source_image": image_path.name,
        "source_size": [width, height],
        "sketch_kind": kind,
        "focus_bbox_1000": list(bbox),
        "pixel_bbox": list(pixel_bbox),
        "points_rendered": len(plan.get("points") or []),
        "lines_rendered": len(plan.get("lines") or []),
        "render_version": "pil-lines-grid-points-v1",
    }
    return active._png_data_url(sketch), active._png_data_url(crop), metadata


def _solve_messages(
    task: dict[str, Any],
    plan: dict[str, Any],
    *,
    sketch_url: str,
    crop_url: str,
    image_root: Path,
    image_url_root: str,
) -> list[dict[str, Any]]:
    safe_plan = {
        key: plan.get(key)
        for key in (
            "sketch_kind",
            "image_index",
            "focus_bbox_1000",
            "points",
            "lines",
            "plan_notes",
            "confidence",
        )
    }
    instruction = (
        "Solve independently. First compare the untouched original page with "
        "the rendered scratchpad and enhanced focus crop. Treat every drawn "
        "line, point, and grid as a hypothesis rather than source evidence. "
        "State concrete visible facts, name one sketch-specific inference, and "
        "perform at least two verification checks against the original. Set "
        "sketch_helpful=false if the marks add no reliable information. For "
        "multiple choice final_answer must be only the option letter; for open "
        "questions return the complete requested answer."
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{core._task_prompt(task)}\n\n{instruction}\n\n"
                "Gold-blind sketch plan: "
                + json.dumps(safe_plan, ensure_ascii=False, separators=(",", ":"))
            ),
        }
    ]
    content.extend(
        core._image_blocks(
            task, image_root=image_root, image_url_root=image_url_root
        )
    )
    content.extend(
        [
            {"type": "image_url", "image_url": {"url": sketch_url}},
            {"type": "image_url", "image_url": {"url": crop_url}},
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def gate_failures(
    task: dict[str, Any],
    plan: dict[str, Any],
    solve: dict[str, Any],
    fallback: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if plan.get("sketch_kind") not in ELIGIBLE_SKETCH_KINDS:
        failures.append("ineligible_sketch_kind")
    if float(plan.get("confidence") or 0.0) < 0.85:
        failures.append("planner_confidence_below_0.85")
    if solve.get("sketch_helpful") is not True:
        failures.append("sketch_not_helpful")
    if solve.get("original_sketch_consistent") is not True:
        failures.append("original_sketch_inconsistent")
    if solve.get("answer_format_verified") is not True:
        failures.append("answer_format_not_verified")
    if float(solve.get("confidence") or 0.0) < 0.90:
        failures.append("solver_confidence_below_0.90")
    if len(solve.get("visual_facts") or []) < 3:
        failures.append("fewer_than_3_visual_facts")
    if len(solve.get("verification_checks") or []) < 2:
        failures.append("fewer_than_2_verification_checks")
    candidate_answer = str(solve.get("final_answer") or "").strip()
    fallback_answer = str(fallback.get("final_answer") or "").strip()
    if not candidate_answer:
        failures.append("empty_candidate_answer")
    answer_type = str(task.get("answer_type") or "").casefold()
    is_choice = answer_type in {"choice", "multiple_choice", "multiple-choice"}
    if is_choice and re.fullmatch(r"[A-E]", candidate_answer) is None:
        failures.append("choice_candidate_not_strict_A_to_E")
    if is_choice:
        fallback_match = re.fullmatch(r"\s*([A-Ea-e])(?:[.)])?\s*", fallback_answer)
        canonical_fallback = (
            fallback_match.group(1).upper() if fallback_match else fallback_answer.casefold()
        )
        canonical_candidate = candidate_answer.upper()
    else:
        canonical_fallback = " ".join(fallback_answer.split()).casefold()
        canonical_candidate = " ".join(candidate_answer.split()).casefold()
    if canonical_candidate == canonical_fallback:
        failures.append("same_answer_as_frozen_fallback")
    return failures


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


def _fallback_result(
    fallback: dict[str, Any],
    *,
    plan: dict[str, Any] | None,
    sketch_metadata: dict[str, Any] | None,
    solve: dict[str, Any] | None,
    calls: list[dict[str, Any]],
    failures: list[str],
    error: str | None,
) -> dict[str, Any]:
    row = copy.deepcopy(fallback)
    original_condition = row.get("condition")
    generation = copy.deepcopy(row.get("generation") or {})
    generation["gold_access"] = False
    generation["visual_sketchpad_v2"] = {
        "schema_version": SCHEMA_VERSION,
        "selected_source": "frozen_active_crop_v2",
        "fallback_original_condition": original_condition,
        "plan": plan,
        "sketch_metadata": sketch_metadata,
        "candidate_evidence": solve,
        "gate_failures": failures,
        "candidate_error": error,
        "call_traces": [_compact(call) for call in calls],
    }
    row.update(
        {
            "condition": CONDITION,
            "prompt_version": CONDITION,
            "generation": generation,
            "error": None,
        }
    )
    return row


def _candidate_result(
    task: dict[str, Any],
    *,
    plan: dict[str, Any],
    sketch_metadata: dict[str, Any],
    solve: dict[str, Any],
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "task_id": str(task["task_id"]),
        "condition": CONDITION,
        "model": core.MODEL,
        "prompt_version": CONDITION,
        "final_answer": str(solve["final_answer"]).strip(),
        "solution_steps": str(solve["solution_steps"]).strip(),
        "reasoning": str(solve["reasoning"]).strip(),
        "forced_answer": False,
        "raw_response": json.dumps(solve, ensure_ascii=False),
        "generation": {
            "temperature": 0.0,
            "top_p": 0.95,
            "enable_thinking": False,
            "gold_access": False,
            "visual_sketchpad_v2": {
                "schema_version": SCHEMA_VERSION,
                "selected_source": "visual_sketchpad_candidate",
                "original_pixels_in_both_calls": True,
                "plan": plan,
                "sketch_metadata": sketch_metadata,
                "candidate_evidence": {
                    key: solve.get(key)
                    for key in (
                        "visual_facts",
                        "sketch_specific_fact",
                        "verification_checks",
                        "sketch_helpful",
                        "original_sketch_consistent",
                        "answer_format_verified",
                        "confidence",
                    )
                },
                "gate_failures": [],
                "call_traces": [_compact(call) for call in calls],
            },
        },
        "tool_calls": [],
        "usage": {
            "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls),
            "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls),
            "latency_s": round(sum(float(call.get("latency_s") or 0.0) for call in calls), 3),
        },
        "error": None,
    }


def run_task(
    task: dict[str, Any],
    fallback: dict[str, Any],
    *,
    pool: core.EndpointPool,
    image_root: Path,
    image_url_root: str,
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    plan: dict[str, Any] | None = None
    sketch_metadata: dict[str, Any] | None = None
    solve: dict[str, Any] | None = None
    try:
        plan_call = pool.complete(
            messages=_plan_messages(
                task, image_root=image_root, image_url_root=image_url_root
            ),
            schema_name="maxim_visual_sketchpad_plan_v2",
            schema=PLAN_SCHEMA,
            max_tokens=1600,
            temperature=0.0,
            seed=PLAN_SEED,
            retries=1,
        )
        calls.append(plan_call)
        plan = plan_call["parsed"]
        paths = active._image_paths(task, image_root)
        requested_image_index = int(plan["image_index"])
        image_index = max(0, min(len(paths) - 1, requested_image_index))
        # Every downstream prompt and artifact records the actual selected page.
        plan["image_index"] = image_index
        planner_failures: list[str] = []
        if plan.get("sketch_kind") not in ELIGIBLE_SKETCH_KINDS:
            planner_failures.append("ineligible_sketch_kind")
        if float(plan.get("confidence") or 0.0) < 0.85:
            planner_failures.append("planner_confidence_below_0.85")
        if planner_failures:
            return _fallback_result(
                fallback,
                plan=plan,
                sketch_metadata={
                    "requested_image_index": requested_image_index,
                    "selected_image_index": image_index,
                    "render_skipped": True,
                },
                solve=None,
                calls=calls,
                failures=planner_failures,
                error=None,
            )
        sketch_url, crop_url, sketch_metadata = _render_sketch(
            paths[image_index], plan
        )
        sketch_metadata["requested_image_index"] = requested_image_index
        sketch_metadata["selected_image_index"] = image_index
        solve_call = pool.complete(
            messages=_solve_messages(
                task,
                plan,
                sketch_url=sketch_url,
                crop_url=crop_url,
                image_root=image_root,
                image_url_root=image_url_root,
            ),
            schema_name="maxim_visual_sketchpad_solution_v2",
            schema=SOLVE_SCHEMA,
            max_tokens=4096,
            temperature=0.0,
            seed=SOLVE_SEED,
            retries=1,
        )
        calls.append(solve_call)
        solve = solve_call["parsed"]
        failures = gate_failures(task, plan, solve, fallback)
        if failures:
            return _fallback_result(
                fallback,
                plan=plan,
                sketch_metadata=sketch_metadata,
                solve=solve,
                calls=calls,
                failures=failures,
                error=None,
            )
        return _candidate_result(
            task,
            plan=plan,
            sketch_metadata=sketch_metadata,
            solve=solve,
            calls=calls,
        )
    except Exception as exc:
        return _fallback_result(
            fallback,
            plan=plan,
            sketch_metadata=sketch_metadata,
            solve=solve,
            calls=calls,
            failures=["candidate_error"],
            error=f"{type(exc).__name__}: {exc}",
        )


def write_ordered_atomic(
    output: Path, task_order: list[str], rows: dict[str, dict[str, Any]]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for task_id in task_order:
            if task_id in rows:
                sink.write(json.dumps(rows[task_id], ensure_ascii=False) + "\n")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fallback-solver", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-url-root", default="file:///images")
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", default=core.MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-concurrency", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if sha256_file(args.input) != FROZEN_PUBLIC_QUEUE_SHA256:
        raise SystemExit("frozen public queue SHA mismatch")
    if sha256_file(args.fallback_solver) != FROZEN_FALLBACK_SHA256:
        raise SystemExit("frozen fallback SHA mismatch")
    if args.model != core.MODEL:
        raise SystemExit(f"frozen model mismatch: expected {core.MODEL!r}")
    if not 1 <= args.task_concurrency <= 16:
        raise SystemExit("--task-concurrency must be in [1, 16]")
    raw_tasks = core._load_jsonl(args.input)
    for row in raw_tasks:
        assert_gold_blind(row, location=str(row.get("task_id")))
    tasks = [core._task_view(row) for row in raw_tasks]
    fallback_rows = rows_by_task(
        core._load_jsonl(args.fallback_solver), label="fallback solver"
    )
    task_order = [str(task.get("task_id") or "") for task in tasks]
    if len(tasks) != 274 or len(set(task_order)) != 274:
        raise SystemExit("expected 274 unique public tasks")
    if set(fallback_rows) != set(task_order):
        raise SystemExit("fallback task ID set mismatch")
    for task in tasks:
        active._image_paths(task, args.image_root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "tasks": len(tasks),
                    "public_queue_sha256": FROZEN_PUBLIC_QUEUE_SHA256,
                    "fallback_sha256": FROZEN_FALLBACK_SHA256,
                    "condition": CONDITION,
                    "eligible_sketch_kinds": sorted(ELIGIBLE_SKETCH_KINDS),
                    "generation_gold_access": False,
                    "network_calls": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        if not args.resume:
            raise SystemExit("output exists; pass --resume")
        existing_rows = core._load_jsonl(args.output)
        existing = rows_by_task(existing_rows, label="existing output")
        if not set(existing).issubset(set(task_order)):
            raise SystemExit("existing output contains unexpected task IDs")
        expected_existing_order = [
            task_id for task_id in task_order if task_id in existing
        ]
        if [str(row.get("task_id") or "") for row in existing_rows] != expected_existing_order:
            raise SystemExit("existing output does not preserve public queue order")
        for task_id, row in existing.items():
            generation = row.get("generation") or {}
            treatment = generation.get("visual_sketchpad_v2") or {}
            if (
                row.get("condition") != CONDITION
                or row.get("prompt_version") != CONDITION
                or generation.get("gold_access") is not False
                or treatment.get("schema_version") != SCHEMA_VERSION
                or treatment.get("selected_source")
                not in {"visual_sketchpad_candidate", "frozen_active_crop_v2"}
                or row.get("error") is not None
                or not str(row.get("final_answer") or "").strip()
            ):
                raise SystemExit(f"existing output row binding mismatch: {task_id}")
    pool = core.EndpointPool(
        args.base_url, model=args.model, timeout_s=args.timeout_s
    )
    output_rows = dict(existing)
    pending = [task for task in tasks if str(task["task_id"]) not in existing]
    write_lock = threading.Lock()

    def execute(task: dict[str, Any]) -> dict[str, Any]:
        task_id = str(task["task_id"])
        return run_task(
            task,
            fallback_rows[task_id],
            pool=pool,
            image_root=args.image_root,
            image_url_root=args.image_url_root,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.task_concurrency
    ) as executor:
        futures = {executor.submit(execute, task): task for task in pending}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task = futures[future]
            row = future.result()
            task_id = str(task["task_id"])
            with write_lock:
                output_rows[task_id] = row
                write_ordered_atomic(args.output, task_order, output_rows)
            selected = (
                (row.get("generation") or {})
                .get("visual_sketchpad_v2", {})
                .get("selected_source")
            )
            print(
                f"[{completed}/{len(pending)}] {task_id} "
                f"answer={row.get('final_answer')!r} source={selected}",
                flush=True,
            )

    write_ordered_atomic(args.output, task_order, output_rows)
    candidate_count = sum(
        (row.get("generation") or {})
        .get("visual_sketchpad_v2", {})
        .get("selected_source")
        == "visual_sketchpad_candidate"
        for row in output_rows.values()
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(output_rows),
                "visual_sketchpad_candidates": candidate_count,
                "frozen_fallbacks": len(output_rows) - candidate_count,
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "generation_gold_access": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
