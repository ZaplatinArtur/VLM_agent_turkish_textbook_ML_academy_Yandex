#!/usr/bin/env python3
"""Freeze the hardened MCQ v1.1 trust boundary before opaque evaluation.

This command has no opaque input, asset, alignment map, gold, prediction,
scorer or accuracy argument.  It attests the already-published source bundle,
rebuilds source records from the two official PDFs, verifies an independent
28-page Poppler reproduction and runs the adversarial v1.1 test suite.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import tempfile
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
    EXPECTED_CONTENT_PAGE_COUNT,
    EXPECTED_FROZEN_BUNDLE_MANIFEST_PROJECTION_SHA256,
    EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256,
    EXPECTED_INVENTORY_FILE_SHA256,
    EXPECTED_INVENTORY_PROJECTION_SHA256,
    EXPECTED_KEY_INDEX_FILE_SHA256,
    EXPECTED_KEY_INDEX_PROJECTION_SHA256,
    EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256,
    EXPECTED_PDFTOPPM_SHA256,
    EXPECTED_POPPLER_VERSION,
    EXPECTED_RENDER_MANIFEST_FILE_SHA256,
    EXPECTED_RENDER_MANIFEST_PROJECTION_SHA256,
    EXPECTED_SOURCE_AUDIT_FILE_SHA256,
    assert_frozen_mcq_bundle,
    assert_frozen_mcq_objects,
    assert_mcq_runtime,
    load_mcq_render_manifest,
    write_canonical_json,
)
from evidence_os.official_ogm import canonical_json_sha256, sha256_file  # noqa: E402


SCHEMA = "mcq-fullpage-source-adapter-freeze-v1.1"
MINIMUM_TEST_COUNT = 56
EXPECTED_PYTEST_VERSION = "8.4.2"
CODE_PATHS = (
    "src/evidence_os/__init__.py",
    "src/evidence_os/certificates.py",
    "src/evidence_os/contracts.py",
    "src/evidence_os/mcq_fullpage_source.py",
    "src/evidence_os/mcq_opaque_batch.py",
    "src/evidence_os/official_ogm.py",
    "src/evidence_os/policy.py",
    "src/evidence_os/source_first.py",
    "src/evidence_os/visual_coordinate_binding.py",
    "scripts/build_mcq_fullpage_source_v1.py",
    "scripts/mcq_fullpage_source_adapter.py",
    "scripts/run_mcq_opaque_batch_v1.py",
    "scripts/freeze_mcq_fullpage_source_v1_1.py",
    "tests/test_mcq_fullpage_source.py",
    "tests/test_mcq_opaque_batch.py",
    "tests/test_mcq_frozen_bundle_v11.py",
)
TEST_PATHS = (
    "tests/test_mcq_fullpage_source.py",
    "tests/test_mcq_opaque_batch.py",
    "tests/test_mcq_frozen_bundle_v11.py",
)


def _file_entry(path: Path, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"freeze file is missing: {resolved}")
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = resolved.name
    return {
        "path": relative,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _run_tests() -> dict[str, Any]:
    try:
        import pytest  # type: ignore
    except ImportError as exc:
        raise ValueError("pinned pytest runtime is unavailable") from exc
    if str(getattr(pytest, "__version__", "")) != EXPECTED_PYTEST_VERSION:
        raise ValueError("pytest version differs from the v1.1 freeze runtime")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(item) for item in (PINNED_PACKAGES, TEST_PACKAGES, REPO_ROOT / "src")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *(str(REPO_ROOT / path) for path in TEST_PATHS),
        ],
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
        raise ValueError("MCQ v1.1 adversarial suite failed its freeze floor")
    return {
        "command": (
            "python -m pytest -q tests/test_mcq_fullpage_source.py "
            "tests/test_mcq_opaque_batch.py tests/test_mcq_frozen_bundle_v11.py"
        ),
        "returncode": result.returncode,
        "passed": passed,
        "minimum_required": MINIMUM_TEST_COUNT,
        "pytest": EXPECTED_PYTEST_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report-dir", type=Path, required=True)
    parser.add_argument("--page-root", type=Path, required=True)
    parser.add_argument("--biology-pdf", type=Path, required=True)
    parser.add_argument("--physics-pdf", type=Path, required=True)
    parser.add_argument("--pdftoppm", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(strict=False)
    if output.exists():
        raise ValueError("v1.1 freeze output must be absent")

    source_report_dir = args.source_report_dir.resolve()
    bundle = assert_frozen_mcq_bundle(
        freeze_manifest_path=source_report_dir / "freeze_manifest.json",
        inventory_path=source_report_dir / "inventory.json",
        key_index_path=source_report_dir / "official_key_index.json",
        render_manifest_path=source_report_dir / "render_manifest.json",
        page_root=args.page_root,
    )
    runtime = assert_mcq_runtime(require_pdfplumber=True, require_visual=True)

    rebuilt_inventory, rebuilt_key, rebuilt_audit = build_source(
        args.biology_pdf, args.physics_pdf
    )
    if (
        rebuilt_inventory.to_mapping() != bundle.inventory.to_mapping()
        or rebuilt_key.to_mapping() != bundle.key_index.to_mapping()
    ):
        raise ValueError("official-PDF source rebuild differs from frozen artifacts")
    source_audit_path = source_report_dir / "source_build_audit.json"
    try:
        observed_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen source-build audit cannot be read") from exc
    if observed_audit != rebuilt_audit:
        raise ValueError("official-PDF source audit did not reproduce")

    pdftoppm = args.pdftoppm.resolve()
    if (
        not pdftoppm.is_file()
        or sha256_file(pdftoppm) != EXPECTED_PDFTOPPM_SHA256
    ):
        raise ValueError("pdftoppm binary differs from the exact runtime pin")
    render_environment = os.environ.copy()
    render_environment["PYTHONPATH"] = os.pathsep.join(
        str(item) for item in (PINNED_PACKAGES, TEST_PACKAGES, REPO_ROOT / "src")
    )
    with tempfile.TemporaryDirectory(prefix="mcq_v11_fresh_render_") as raw_temp:
        fresh_root = Path(raw_temp) / "renders"
        fresh_manifest = Path(raw_temp) / "render_manifest.json"
        render_result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "mcq_fullpage_source_adapter.py"),
                "render-pages",
                "--biology-pdf",
                str(args.biology_pdf.resolve()),
                "--physics-pdf",
                str(args.physics_pdf.resolve()),
                "--inventory",
                str(source_report_dir / "inventory.json"),
                "--pdftoppm",
                str(pdftoppm),
                "--output-dir",
                str(fresh_root),
                "--manifest",
                str(fresh_manifest),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=render_environment,
        )
        if render_result.returncode != 0:
            raise ValueError(
                "fresh pinned-Poppler 28-page reproduction failed: "
                f"{render_result.stderr.strip()}"
            )
        reproduced_render = load_mcq_render_manifest(
            fresh_manifest,
            bundle.inventory,
            page_root=fresh_root,
        )
        assert_frozen_mcq_objects(
            bundle.inventory, bundle.key_index, reproduced_render
        )
        if reproduced_render.to_mapping() != bundle.render_manifest.to_mapping():
            raise ValueError("fresh Poppler render manifest did not reproduce")
        reproduced_pages = [
            {
                "document_id": item.document_id,
                "height": item.height,
                "page_number": item.page_number,
                "path": item.relative_path,
                "sha256": sha256_file(item.resolved_path),
                "size_bytes": item.resolved_path.stat().st_size,
                "width": item.width,
            }
            for item in reproduced_render.pages
            if item.resolved_path is not None
        ]
        if (
            len(reproduced_pages) != EXPECTED_CONTENT_PAGE_COUNT
            or canonical_json_sha256(reproduced_pages)
            != EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256
        ):
            raise ValueError("fresh render is not the exact complete 28-page set")
        reproduced_manifest_file_sha256 = sha256_file(fresh_manifest)
        reproduced_manifest_projection_sha256 = (
            reproduced_render.render_manifest_projection_sha256
        )

    for path in CODE_PATHS:
        py_compile.compile(str(REPO_ROOT / path), doraise=True)
    tests = _run_tests()
    code_files = [_file_entry(REPO_ROOT / path) for path in CODE_PATHS]
    report_entry = _file_entry(args.report)
    source_artifacts = [
        _file_entry(source_report_dir / "freeze_manifest.json"),
        _file_entry(source_report_dir / "inventory.json"),
        _file_entry(source_report_dir / "official_key_index.json"),
        _file_entry(source_report_dir / "source_build_audit.json"),
        _file_entry(source_report_dir / "render_manifest.json"),
    ]
    if {
        item["sha256"] for item in source_artifacts
    } != {
        EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256,
        EXPECTED_INVENTORY_FILE_SHA256,
        EXPECTED_KEY_INDEX_FILE_SHA256,
        EXPECTED_SOURCE_AUDIT_FILE_SHA256,
        EXPECTED_RENDER_MANIFEST_FILE_SHA256,
    }:
        raise ValueError("v1.1 source artifact hash set changed")

    projection: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "ready_for_commit_no_opaque_read_or_run",
        "created_date": "2026-08-08",
        "scope": "exact frozen-bundle and opaque-MCQ trust-boundary hardening",
        "holdout_status": "not_read_not_run_by_v1.1_fix",
        "correctness_status": "not_computed",
        "accuracy_claim": None,
        "network_used": False,
        "gpu_used": False,
        "runtime_observed": runtime,
        "trust_anchor": {
            "freeze_manifest_sha256": EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256,
            "freeze_manifest_projection_sha256": (
                EXPECTED_FROZEN_BUNDLE_MANIFEST_PROJECTION_SHA256
            ),
            "inventory_file_sha256": EXPECTED_INVENTORY_FILE_SHA256,
            "inventory_projection_sha256": (
                EXPECTED_INVENTORY_PROJECTION_SHA256
            ),
            "key_index_file_sha256": EXPECTED_KEY_INDEX_FILE_SHA256,
            "key_index_projection_sha256": (
                EXPECTED_KEY_INDEX_PROJECTION_SHA256
            ),
            "render_manifest_file_sha256": (
                EXPECTED_RENDER_MANIFEST_FILE_SHA256
            ),
            "render_manifest_projection_sha256": (
                EXPECTED_RENDER_MANIFEST_PROJECTION_SHA256
            ),
            "page_payloads_projection_sha256": (
                EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256
            ),
            "attestation_projection_sha256": (
                bundle.attestation_projection_sha256
            ),
        },
        "hardening": {
            "bundle_attested_before_opaque_read": True,
            "self_consistent_alternate_bundle_rejected": True,
            "direct_execute_reparses_raw_jsonl": True,
            "direct_execute_path_safe_unique_ids": True,
            "direct_execute_prompt_and_image_pins_recomputed": True,
            "run_manifest_pins_raw_jsonl_sha_size_and_ordered_projection": True,
            "certificate_replay_requires_expected_image_bytes": True,
            "certificate_replay_recomputes_sift": False,
            "certificate_replay_scope": "recorded evidence and binding replay only",
        },
        "reproduction": {
            "official_source_rebuild_exact": True,
            "independent_poppler_render_exact": True,
            "fresh_render_created_inside_freeze_process": True,
            "poppler_version": EXPECTED_POPPLER_VERSION,
            "pdftoppm_sha256": EXPECTED_PDFTOPPM_SHA256,
            "rendered_page_count": len(reproduced_pages),
            "render_manifest_file_sha256": reproduced_manifest_file_sha256,
            "render_manifest_projection_sha256": (
                reproduced_manifest_projection_sha256
            ),
            "page_payloads_projection_sha256": (
                EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256
            ),
            "page_payloads_exact_byte_match": True,
        },
        "source_artifacts": source_artifacts,
        "code": {
            "files": code_files,
            "combined_code_projection_sha256": canonical_json_sha256(code_files),
        },
        "report": report_entry,
        "checks": {
            "python_compile": True,
            "adversarial_unit_tests": tests,
            "source_rebuild_exact": True,
            "all_28_renders_reproduced_exactly": True,
        },
        "post_freeze_order": [
            "commit_and_push_v1.1_code_report_and_this_manifest",
            "obtain_independent_v1.1_audit_pass",
            "only_then_read_and_run_opaque_mcq_bundle",
            "seal_outputs_before_opening_private_alignment_or_gold",
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
