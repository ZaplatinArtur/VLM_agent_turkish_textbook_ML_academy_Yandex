#!/usr/bin/env python3
"""Freeze the Bio9/Physics12 task-ID-free source resolver before opaque use.

This command accepts only official source PDFs and public source artifacts.  It
rebuilds the source census, verifies all page payloads, runs the adversarial
unit suite and hashes the complete implementation/profile.  It has no opaque
input, selection map, gold, prediction, evaluation or scorer argument.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_PACKAGES = REPO_ROOT / "tmp" / "portfolio_official_sources" / "python_pkgs"
TEST_PACKAGES = REPO_ROOT / "tmp" / "maxim_math12_test_pkgs"
for candidate in (
    PINNED_PACKAGES,
    TEST_PACKAGES,
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from build_mcq_fullpage_source_v1 import build_source  # noqa: E402
from evidence_os.mcq_fullpage_source import (  # noqa: E402
    EXPECTED_CHOICE_KEY_COUNT,
    EXPECTED_CONTENT_PAGE_COUNT,
    EXPECTED_NUMPY_VERSION,
    EXPECTED_OPENCV_VERSION,
    EXPECTED_PDFPLUMBER_VERSION,
    EXPECTED_PDFTOPPM_SHA256,
    EXPECTED_POPPLER_VERSION,
    EXPECTED_PROTOCOL_RECORD_COUNT,
    EXPECTED_PYTHON_VERSION,
    FROZEN_SIFT_RUNTIME_PROFILE,
    FROZEN_VISUAL_THRESHOLDS,
    assert_mcq_runtime,
    load_mcq_inventory,
    load_mcq_key_index,
    load_mcq_render_manifest,
    write_canonical_json,
)
from evidence_os.official_ogm import canonical_json_sha256, sha256_file  # noqa: E402


SCHEMA = "mcq-fullpage-source-adapter-freeze-v1"
CODE_PATHS = (
    "src/evidence_os/mcq_fullpage_source.py",
    "src/evidence_os/mcq_opaque_batch.py",
    "scripts/build_mcq_fullpage_source_v1.py",
    "scripts/mcq_fullpage_source_adapter.py",
    "scripts/run_mcq_opaque_batch_v1.py",
    "scripts/freeze_mcq_fullpage_source_v1.py",
    "tests/test_mcq_fullpage_source.py",
    "tests/test_mcq_opaque_batch.py",
)
MINIMUM_TEST_COUNT = 43
EXPECTED_PYTEST_VERSION = "8.4.2"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read freeze JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"freeze JSON input is not an object: {path}")
    return value


def _file_entry(path: Path, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"freeze file is missing: {path}")
    try:
        relative = path.relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _run_tests() -> dict[str, Any]:
    try:
        import pytest  # type: ignore
    except ImportError as exc:
        raise ValueError("pinned pytest runtime is unavailable") from exc
    if str(getattr(pytest, "__version__", "")) != EXPECTED_PYTEST_VERSION:
        raise ValueError("pytest version differs from the freeze test runtime")
    test_paths = [str(REPO_ROOT / path) for path in CODE_PATHS if path.startswith("tests/")]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(path)
        for path in (PINNED_PACKAGES, TEST_PACKAGES, REPO_ROOT / "src")
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *test_paths],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    matches = re.findall(r"([0-9]+) passed", output)
    passed = int(matches[-1]) if matches else 0
    if result.returncode != 0 or passed < MINIMUM_TEST_COUNT:
        raise ValueError("MCQ adversarial unit suite did not pass the freeze floor")
    return {
        "command": "python -m pytest -q tests/test_mcq_fullpage_source.py tests/test_mcq_opaque_batch.py",
        "returncode": result.returncode,
        "passed": passed,
        "minimum_required": MINIMUM_TEST_COUNT,
        "pytest": EXPECTED_PYTEST_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--page-root", type=Path, required=True)
    parser.add_argument("--biology-pdf", type=Path, required=True)
    parser.add_argument("--physics-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(strict=False)
    if output.exists():
        raise ValueError("freeze manifest output must be absent")

    report_dir = args.report_dir.resolve()
    inventory_path = report_dir / "inventory.json"
    key_index_path = report_dir / "official_key_index.json"
    source_audit_path = report_dir / "source_build_audit.json"
    render_manifest_path = report_dir / "render_manifest.json"
    report_path = report_dir / "REPORT_RU.md"

    observed_runtime = assert_mcq_runtime(
        require_pdfplumber=True, require_visual=True
    )
    inventory = load_mcq_inventory(inventory_path)
    key_index = load_mcq_key_index(key_index_path, inventory)
    render_manifest = load_mcq_render_manifest(
        render_manifest_path, inventory, page_root=args.page_root
    )
    rebuilt_inventory, rebuilt_key_index, rebuilt_audit = build_source(
        args.biology_pdf, args.physics_pdf
    )
    if (
        inventory.to_mapping() != rebuilt_inventory.to_mapping()
        or key_index.to_mapping() != rebuilt_key_index.to_mapping()
    ):
        raise ValueError("source artifacts do not reproduce from official PDFs")
    source_audit = _load_object(source_audit_path)
    if source_audit != rebuilt_audit:
        raise ValueError("source build audit does not reproduce")
    if (
        source_audit.get("protocol_addresses") != EXPECTED_PROTOCOL_RECORD_COUNT
        or source_audit.get("official_choice_records")
        != EXPECTED_CHOICE_KEY_COUNT
        or source_audit.get("unsupported_open_response_records") != 4
        or source_audit.get("candidate_content_pages")
        != EXPECTED_CONTENT_PAGE_COUNT
        or source_audit.get("protocol_defect", {}).get("selected_holdout_impact")
        != "not_inspected_before_resolver_freeze"
    ):
        raise ValueError("source census/protocol-defect declaration changed")

    code_files = [_file_entry(REPO_ROOT / path) for path in CODE_PATHS]
    for path in CODE_PATHS:
        if path.endswith(".py"):
            py_compile.compile(str(REPO_ROOT / path), doraise=True)
    tests = _run_tests()
    page_payloads = [
        {
            **_file_entry(
                args.page_root.resolve() / Path(page.relative_path),
                root=args.page_root,
            ),
            "document_id": page.document_id,
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
        }
        for page in render_manifest.pages
    ]
    if len(page_payloads) != EXPECTED_CONTENT_PAGE_COUNT:
        raise ValueError("freeze payload set does not contain all 28 pages")

    profile: dict[str, Any] = {
        "schema_version": "mcq-fullpage-exact-runtime-profile-v1",
        "candidate_page_policy": "all_28_content_pages_no_shortlist",
        "decision_features": ["observable_prompt", "image_bytes"],
        "forbidden_policy_features": [
            "task_id",
            "input_id",
            "selection_map",
            "gold",
            "prediction",
            "evaluation",
        ],
        "dependencies": {
            "python": EXPECTED_PYTHON_VERSION,
            "pdfplumber": EXPECTED_PDFPLUMBER_VERSION,
            "numpy": EXPECTED_NUMPY_VERSION,
            "opencv": EXPECTED_OPENCV_VERSION,
            "poppler": EXPECTED_POPPLER_VERSION,
            "pdftoppm_sha256": EXPECTED_PDFTOPPM_SHA256,
            "pytest_test_only": EXPECTED_PYTEST_VERSION,
        },
        "visual_thresholds": asdict(FROZEN_VISUAL_THRESHOLDS),
        "sift_runtime_profile": asdict(FROZEN_SIFT_RUNTIME_PROFILE),
        "render": {
            "dpi": 144,
            "color_mode": "poppler_gray_rgb_png",
            "pages": EXPECTED_CONTENT_PAGE_COUNT,
        },
        "abstention": {
            "incomplete_page_sweep": True,
            "ambiguous_page": True,
            "failed_geometry": True,
            "prompt_page_miss": True,
            "missing_or_cross_bound_key": True,
            "unsupported_open_response": True,
        },
    }
    profile["profile_projection_sha256"] = canonical_json_sha256(profile)

    projection: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "ready_for_freeze_commit_no_opaque_run",
        "created_date": "2026-08-08",
        "scope": "task-id-free exact page/question/official-key source resolution",
        "holdout_status": "not_read_not_run",
        "correctness_status": "not_computed",
        "accuracy_claim": None,
        "network_used": False,
        "gpu_used": False,
        "source_census": {
            "protocol_addresses": EXPECTED_PROTOCOL_RECORD_COUNT,
            "official_choice_records": EXPECTED_CHOICE_KEY_COUNT,
            "unsupported_open_response_records": 4,
            "candidate_content_pages": EXPECTED_CONTENT_PAGE_COUNT,
            "physical_choice_key_cells_verified": source_audit[
                "physical_choice_key_cells_verified"
            ],
            "inventory_projection_sha256": (
                inventory.inventory_projection_sha256
            ),
            "key_index_projection_sha256": (
                key_index.key_index_projection_sha256
            ),
            "render_manifest_projection_sha256": (
                render_manifest.render_manifest_projection_sha256
            ),
            "protocol_defect": source_audit["protocol_defect"],
        },
        "frozen_profile": profile,
        "runtime_observed": observed_runtime,
        "code": {
            "files": code_files,
            "combined_code_projection_sha256": canonical_json_sha256(code_files),
        },
        "artifacts": [
            _file_entry(inventory_path),
            _file_entry(key_index_path),
            _file_entry(source_audit_path),
            _file_entry(render_manifest_path),
            _file_entry(report_path),
        ],
        "page_payloads": {
            "count": len(page_payloads),
            "files": page_payloads,
            "combined_projection_sha256": canonical_json_sha256(page_payloads),
        },
        "checks": {
            "source_rebuild_exact": True,
            "all_page_payload_hashes_and_png_headers": True,
            "python_compile": True,
            "adversarial_unit_tests": tests,
        },
        "post_freeze_order": [
            "commit_this_manifest_and_every_hashed_artifact",
            "only_then_read_and_run_opaque_prompt_image_bundle",
            "do_not_change_thresholds_code_or_source_inventory",
            "evaluate_with_sealed_scorer_as_a_separate_step",
            "publish_protocol_defect_impact_without_post_hoc_replacement",
        ],
    }
    projection["manifest_projection_sha256"] = canonical_json_sha256(projection)
    write_canonical_json(output, projection)
    print(
        json.dumps(
            {
                "status": projection["status"],
                "tests_passed": tests["passed"],
                "manifest_projection_sha256": projection[
                    "manifest_projection_sha256"
                ],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
