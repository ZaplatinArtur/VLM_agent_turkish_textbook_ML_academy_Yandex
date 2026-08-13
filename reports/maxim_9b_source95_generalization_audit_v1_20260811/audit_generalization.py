from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image, PngImagePlugin


REPORT_ROOT = Path(__file__).resolve().parent
REPO = REPORT_ROOT.parents[1]
EXPERIMENT_ROOT = REPO / "experiments" / "maxim_9b_source95_tool_wave_v1_20260811"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import tool_wave as wave  # noqa: E402


SCHEMA = "maxim-9b-source95-generalization-audit-v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_native(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _reidentified_pairs(
    base_pairs: list[tuple[bytes, dict[str, Any]]], task_id: str
) -> tuple[list[tuple[bytes, dict[str, Any]]], str]:
    replacement = f"unseen_reid__{task_id}"
    output: list[tuple[bytes, dict[str, Any]]] = []
    for raw, row in base_pairs:
        if row["task_id"] != task_id:
            output.append((raw, row))
            continue
        changed = copy.deepcopy(row)
        changed["task_id"] = replacement
        output.append((wave.canonical(changed), changed))
    return output, replacement


def audit_reidentification() -> dict[str, Any]:
    order, routes = wave.load_routes()
    base_pairs, _ = wave.load_base_solver(order)
    certificates = wave.build_certificates(routes)

    _, positive_candidates = wave.compose_solver(base_pairs, certificates, "combined")
    if set(positive_candidates) != set(wave.COMBINED_TARGETS):
        raise AssertionError("positive-control overlay did not select all 12 frozen IDs")

    rows: list[dict[str, Any]] = []
    for task_id in wave.COMBINED_TARGETS:
        changed_pairs, replacement = _reidentified_pairs(base_pairs, task_id)
        rejected = False
        error = None
        try:
            wave.compose_solver(changed_pairs, certificates, "combined")
        except wave.WaveError as exc:
            rejected = True
            error = str(exc)
        rows.append(
            {
                "original_task_id": task_id,
                "replacement_task_id": replacement,
                "semantic_payload_unchanged_except_task_id": True,
                "combined_overlay_rejected": rejected,
                "error": error,
            }
        )
    return {
        "positive_control_selected": len(positive_candidates),
        "counterfactuals": rows,
        "reidentified_rejected": sum(row["combined_overlay_rejected"] for row in rows),
        "reidentified_accepted": sum(not row["combined_overlay_rejected"] for row in rows),
    }


def _pixel_identical_png(original: bytes) -> tuple[bytes, dict[str, Any]]:
    with Image.open(io.BytesIO(original)) as source:
        original_rgba = source.convert("RGBA")
        size = original_rgba.size
        pixels = original_rgba.tobytes()
        variant_image = original_rgba.copy()

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("source95_counterfactual", "pixel-identical re-encode")
    buffer = io.BytesIO()
    variant_image.save(
        buffer,
        format="PNG",
        compress_level=9,
        pnginfo=metadata,
    )
    variant = buffer.getvalue()
    with Image.open(io.BytesIO(variant)) as decoded:
        decoded_rgba = decoded.convert("RGBA")
        same_pixels = decoded_rgba.size == size and decoded_rgba.tobytes() == pixels
    return variant, {"width": size[0], "height": size[1], "same_rgba_pixels": same_pixels}


def audit_pixel_identical_images() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="source95_cf_", dir=wave.REPO) as temporary:
        temporary_path = Path(temporary)
        for task_id in wave.COMBINED_TARGETS:
            expected = wave.IMAGE_PINS[task_id]
            source_path = wave._workspace_path(f"{wave.IMAGE_ROOT_RELATIVE}/{task_id}.png")
            original = source_path.read_bytes()
            variant, pixels = _pixel_identical_png(original)
            variant_path = temporary_path / f"{task_id}.png"
            variant_path.write_bytes(variant)
            relative = variant_path.resolve().relative_to(wave.REPO.resolve()).as_posix()
            rejected = False
            error = None
            try:
                wave.stable_pinned_bytes(relative, expected, f"counterfactual {task_id}")
            except wave.WaveError as exc:
                rejected = True
                error = str(exc)
            rows.append(
                {
                    "task_id": task_id,
                    "width": pixels["width"],
                    "height": pixels["height"],
                    "same_rgba_pixels": pixels["same_rgba_pixels"],
                    "original_sha256": _sha256(original),
                    "variant_sha256": _sha256(variant),
                    "bytes_differ": original != variant,
                    "pin_gate_rejected": rejected,
                    "error": error,
                }
            )
    return {
        "counterfactuals": rows,
        "pixel_identical": sum(row["same_rgba_pixels"] for row in rows),
        "byte_distinct": sum(row["bytes_differ"] for row in rows),
        "pin_gate_rejected": sum(row["pin_gate_rejected"] for row in rows),
    }


def _kernel_case(
    *,
    task_id: str,
    family: str,
    call: Callable[[], Any],
    expected: Any,
) -> dict[str, Any]:
    try:
        actual = call()
        passed = actual == expected
        error = None
    except Exception as exc:  # pragma: no cover - retained in audit output
        actual = None
        passed = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "task_id": task_id,
        "kernel_family": family,
        "unseen_structured_input": True,
        "passed": passed,
        "actual": _json_native(actual),
        "expected": _json_native(expected),
        "error": error,
    }


def audit_parameterized_kernels() -> dict[str, Any]:
    v = lambda name: ("var", name)
    logic_values = {"a": True, "b": False, "c": False}
    logic_expression = ("iff", ("and", v("a"), ("not", v("b"))), ("not", v("c")))
    number_specs = [
        {"display": "7/13", "kind": "fraction"},
        {"display": "sqrt(9/16)", "kind": "sqrt_fraction", "numerator": 9, "denominator": 16},
        {"display": "sqrt(2)", "kind": "sqrt_fraction", "numerator": 2, "denominator": 1},
        {"display": "pi", "kind": "pi"},
        {"display": "0.(27)", "kind": "repeating_decimal"},
    ]
    cases = [
        _kernel_case(
            task_id="val_0253",
            family="gcd_equal_thickness_optimizer",
            call=lambda: wave.solve_cabinet_counts(506, (3, 3), 758, (4, 4)),
            expected={
                "usable_a": 500,
                "usable_b": 750,
                "max_box_thickness": 250,
                "count_a": 2,
                "count_b": 3,
                "minimum_total": 5,
            },
        ),
        _kernel_case(
            task_id="val_0216",
            family="integer_length_lcm_solver",
            call=lambda: wave.least_common_arrangement_length(3, 7, 8),
            expected=168,
        ),
        _kernel_case(
            task_id="val_0232",
            family="semicircle_midpoint_theorem_solver",
            call=lambda: wave.semicircle_center_distance(10, 6),
            expected={"distance_squared": 16, "outside": 4, "inside": 1, "exact": "4"},
        ),
        _kernel_case(
            task_id="val_0245",
            family="right_triangle_barrier_height_solver",
            call=lambda: wave.barrier_tip_interval(130, 50, 10, {"Q": 160, "P": 100, "R": 40}),
            expected={"rise_cm": 120, "tip_height_cm": 130, "upper": "Q", "lower": "P", "interval": "Q - P"},
        ),
        _kernel_case(
            task_id="val_0086",
            family="exact_integer_domain_quadratic_inequality_solver",
            call=lambda: wave.excluded_natural_values_for_positive_quadratic(1, -7, 12),
            expected=[3, 4],
        ),
        _kernel_case(
            task_id="val_0213",
            family="signed_integer_calculator",
            call=lambda: wave.signed_updates(5, (-12, 8, 1)),
            expected=2,
        ),
        _kernel_case(
            task_id="val_0230",
            family="integer_equation_bead_function_solver",
            call=lambda: wave.solve_bead_function((1, 0, 2, 1), None, 2, -5, 5),
            expected={
                "rod1_length": 10,
                "bead_height": 3,
                "f_values": {1: 4, 2: 9, 3: 5, 4: 10},
                "f_of_f1": 10,
            },
        ),
        _kernel_case(
            task_id="val_0067",
            family="lexicographic_digit_inequality_enumerator",
            call=lambda: wave.solve_placeholder_digit_sum(350, lambda digit: 100 + 100 * digit, {3: "Z"}),
            expected={"valid_digits": [0, 1, 2], "sum": 3, "answer": "Z"},
        ),
        _kernel_case(
            task_id="val_0267",
            family="boolean_ast_truth_table",
            call=lambda: wave.eval_boolean_expression(logic_expression, logic_values),
            expected=True,
        ),
        _kernel_case(
            task_id="val_0205",
            family="mixed_radix_duration_arithmetic",
            call=lambda: {
                "add": wave.duration_add((2, 11, 25), (1, 2, 10)),
                "subtract": wave.duration_subtract((2030, 3, 5), (2028, 11, 20)),
            },
            expected={"add": (4, 2, 5), "subtract": (1, 3, 15)},
        ),
        _kernel_case(
            task_id="val_0218",
            family="exact_rationality_classifier",
            call=lambda: wave.classify_exact_numbers(number_specs),
            expected={
                "rational": ["7/13", "sqrt(9/16)", "0.(27)"],
                "irrational": ["sqrt(2)", "pi"],
            },
        ),
    ]

    column_signature = inspect.signature(wave.solve_column_arithmetic)
    hardcoded = {
        "task_id": "val_0204",
        "kernel_family": "column_arithmetic_constraint_enumerator",
        "unseen_structured_input": False,
        "passed": False,
        "reason": "public function has no input parameters and embeds all three equations as constants",
        "signature": str(column_signature),
    }
    return {
        "parameterized_cases": cases,
        "hardcoded_cases": [hardcoded],
        "reusable_kernel_task_slots_passed": sum(case["passed"] for case in cases),
        "reusable_kernel_task_slots_total": len(cases),
        "hardcoded_task_slots": 1,
    }


def _production_files() -> list[Path]:
    project_roots = [
        REPO,
        REPO.parent / "VLM_agent_turkish_textbook_source95",
        REPO.parent / "VLM",
    ]
    output: list[Path] = []
    for project in project_roots:
        source_root = project / "src"
        if source_root.is_dir():
            output.extend(sorted(source_root.rglob("*.py")))
        for name in ("main.py", "pyproject.toml"):
            candidate = project / name
            if candidate.is_file():
                output.append(candidate)
    return output


def audit_production_wiring() -> dict[str, Any]:
    markers = [
        wave.EXPERIMENT_ID,
        "source95_tool_wave",
        *wave.COMBINED_TARGETS,
        *sorted({wave.build_certificates()[task_id]["tool"] for task_id in wave.COMBINED_TARGETS}),
    ]
    hits: list[dict[str, str]] = []
    files = _production_files()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")
        for marker in markers:
            if marker in text:
                hits.append({"path": str(path), "marker": marker})

    solver_registry = (REPO / "src" / "mla_baseline" / "solvers" / "__init__.py").read_text(encoding="utf-8")
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    return {
        "project_roots": [str(REPO), str(REPO.parent / "VLM_agent_turkish_textbook_source95"), str(REPO.parent / "VLM")],
        "production_files_scanned": len(files),
        "source95_marker_hits": hits,
        "source95_marker_hit_count": len(hits),
        "basic_rag_solver_registry_mentions_source95": "source95" in solver_registry.casefold(),
        "basic_rag_package_discovery_is_src_only": 'where = ["src"]' in pyproject,
        "experiment_directory_packaged_by_setuptools": False,
        "wired_into_production_service": False if not hits else "not_proven_false_due_to_marker_hits",
    }


def build_result() -> dict[str, Any]:
    reid = audit_reidentification()
    images = audit_pixel_identical_images()
    kernels = audit_parameterized_kernels()
    production = audit_production_wiring()
    strict_e2e = 0
    return {
        "schema_version": SCHEMA,
        "audited_experiment": wave.EXPERIMENT_ID,
        "audited_claim": {"correct": 261, "denominator": 274, "accuracy": 261 / 274},
        "cpu_only": True,
        "gpu_used": False,
        "model_calls": 0,
        "network_calls": 0,
        "findings": {
            "metric_arithmetic_reproduced": True,
            "benchmark_specific_overlay": True,
            "tool_selection_uses_exact_task_id": True,
            "all_12_certificates_bind_exact_benchmark_image_sha256": True,
            "semantic_or_ocr_dispatcher_present": False,
            "production_service_wired": False,
            "strict_end_to_end_generalizable_fixes": strict_e2e,
            "strict_end_to_end_generalizable_fixes_denominator": len(wave.COMBINED_TARGETS),
            "reusable_parameterized_kernel_task_slots": kernels["reusable_kernel_task_slots_passed"],
            "reusable_parameterized_kernel_task_slots_denominator": len(wave.COMBINED_TARGETS),
            "same_textbook_unseen_task_automatically_gets_overlay": False,
            "honest_interpretation": "valid benchmark-specific model-plus-offline-proof overlay score; not an unseen-task model or production-pipeline score",
        },
        "reidentification": reid,
        "pixel_identical_image_reencode": images,
        "kernel_counterfactuals": kernels,
        "production_wiring": production,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only counterfactual audit of the source95 261/274 overlay")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    result = build_result()
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":") if args.compact else None,
            indent=None if args.compact else 2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
