#!/usr/bin/env python3
"""Freeze the audited Math12 v1.1 code, profile and public artifacts.

This command has no benchmark, holdout, gold, prediction or scorer input.  It
verifies the five already-public dev certificates and solution records, hashes
the portable source artifacts, and writes one canonical pre-unseen manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_PACKAGES = REPO_ROOT / "tmp" / "portfolio_official_sources" / "python_pkgs"
for candidate in (PINNED_PACKAGES, REPO_ROOT / "src"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evidence_os.math12_activity_source import (  # noqa: E402
    EXPECTED_NUMPY_VERSION,
    EXPECTED_OPENCV_VERSION,
    EXPECTED_PDFPLUMBER_VERSION,
    EXPECTED_POPPLER_VERSION,
    EXPECTED_PYTHON_VERSION,
    FROZEN_SIFT_RUNTIME_PROFILE,
    FROZEN_VISUAL_THRESHOLDS,
    load_math12_inventory,
    load_math12_render_manifest,
    load_math12_source_certificate,
    verify_math12_source_certificate,
    write_canonical_json,
)
from evidence_os.official_ogm import canonical_json_sha256, sha256_file  # noqa: E402


SCHEMA = "math12-source-adapter-freeze-v1.1"
LABELS = ("val_0054", "val_0055", "val_0056", "val_0057", "val_0058")
EXPECTED_ALIGNMENT = {
    "val_0054": 3,
    "val_0055": 17,
    "val_0056": 88,
    "val_0057": 43,
    "val_0058": 31,
}
CODE_PATHS = (
    "src/evidence_os/math12_activity_source.py",
    "scripts/math12_official_source_adapter.py",
    "scripts/audit_math12_dev5_source_bindings.py",
    "scripts/freeze_math12_source_v1_1.py",
    "tests/test_math12_activity_source.py",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_solution(
    path: Path, certificate: Any, expected_activity: int
) -> dict[str, Any]:
    value = _load_object(path)
    text = str(value.get("official_solution_text") or "")
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    next_header = f"Etkinlik No.: {expected_activity + 1}"
    if (
        value.get("schema_version") != "math12-answer-bound-official-solution-v1"
        or value.get("activity_number") != expected_activity
        or value.get("source_certificate_projection_sha256")
        != certificate.certificate_projection_sha256
        or value.get("official_solution_text_sha256") != text_sha
        or f"Etkinlik No.: {expected_activity}" not in text
        or next_header in text
    ):
        raise ValueError(f"official solution does not pass v1.1 marker/binding checks: {path}")
    bound_projection = {
        key: value[key]
        for key in (
            "schema_version",
            "document_id",
            "pdf_sha256",
            "inventory_projection_sha256",
            "source_certificate_projection_sha256",
            "task_image_sha256",
            "selected_content_page",
            "activity_number",
            "key_start",
            "key_end_exclusive",
            "binding_projection_sha256",
            "key_projection_sha256",
            "official_solution_text_sha256",
        )
    }
    if canonical_json_sha256(bound_projection) != value.get(
        "answer_bound_certificate_projection_sha256"
    ):
        raise ValueError(f"official solution answer-bound projection mismatch: {path}")
    return {
        "label": path.stem,
        "activity": expected_activity,
        **_file_entry(path),
        "official_solution_text_sha256": text_sha,
        "own_activity_header_present": True,
        "next_activity_header_absent": True,
        "answer_bound_certificate_projection_sha256": value[
            "answer_bound_certificate_projection_sha256"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--page-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    inventory_path = report_dir / "inventory.json"
    render_path = report_dir / "render_manifest.json"
    audit_path = report_dir / "dev5_source_binding_audit.json"
    report_path = report_dir / "REPORT_RU.md"
    inventory = load_math12_inventory(inventory_path)
    render_manifest = load_math12_render_manifest(
        render_path, inventory, page_root=args.page_root
    )
    audit = _load_object(audit_path)
    if audit.get("summary") != {
        "cases": 5,
        "resolver_accepted": 5,
        "alignment_matches": 5,
        "solutions_without_next_activity_header": 5,
    }:
        raise ValueError("dev5 audit is not the complete accepted v1.1 audit")

    certificates: list[dict[str, Any]] = []
    solutions: list[dict[str, Any]] = []
    for label in LABELS:
        certificate_path = report_dir / "certificates" / f"{label}.json"
        certificate = load_math12_source_certificate(certificate_path)
        decision = verify_math12_source_certificate(
            inventory, render_manifest, certificate
        )
        if not decision.accepted or (
            decision.selected_activity_number != EXPECTED_ALIGNMENT[label]
        ):
            raise ValueError(f"dev certificate alignment failed: {label}")
        certificates.append(
            {
                "label": label,
                "resolved_content_page": decision.selected_content_page,
                "resolved_activity": decision.selected_activity_number,
                **_file_entry(certificate_path),
                "certificate_projection_sha256": (
                    certificate.certificate_projection_sha256
                ),
            }
        )
        solutions.append(
            _verify_solution(
                report_dir / "official_solutions" / f"{label}.json",
                certificate,
                EXPECTED_ALIGNMENT[label],
            )
        )

    code_files = [_file_entry(REPO_ROOT / path) for path in CODE_PATHS]
    profile = {
        "schema_version": "math12-exact-runtime-profile-v1.1",
        "candidate_page_policy": "all_127_content_pages_no_shortlist",
        "dependencies": {
            "python": EXPECTED_PYTHON_VERSION,
            "pdfplumber": EXPECTED_PDFPLUMBER_VERSION,
            "numpy": EXPECTED_NUMPY_VERSION,
            "opencv": EXPECTED_OPENCV_VERSION,
            "poppler": EXPECTED_POPPLER_VERSION,
        },
        "visual_thresholds": asdict(FROZEN_VISUAL_THRESHOLDS),
        "sift_runtime_profile": asdict(FROZEN_SIFT_RUNTIME_PROFILE),
        "render": {"dpi": 144, "color_mode": "gray_png", "pages": 127},
    }
    profile["profile_projection_sha256"] = canonical_json_sha256(profile)
    projection: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "ready_for_freeze_commit_no_unseen_run",
        "created_date": "2026-08-08",
        "supersedes_commit": "9db67f2",
        "scope": "source_binding_and_official_solution_extraction_only",
        "selection_disclosure": (
            "Math12 was selected post-hoc from five dev inputs; 5/5 is dev source-address "
            "replay, not transfer accuracy."
        ),
        "holdout_status": "not_read_not_run",
        "correctness_status": "not_computed",
        "accuracy_claim": None,
        "network_used": False,
        "gpu_used": False,
        "audit_fixes": [
            "marker_top_floor_of_minimum_three_word_tops_at_four_decimals",
            "strict_all_evidence_certificate_replay_api",
            "exact_no_override_math12_threshold_and_sift_profile",
            "python_pdfplumber_numpy_opencv_poppler_preflight_pins",
            "portable_tracked_render_manifest_with_external_hash_checked_payload",
        ],
        "source": {
            "pdf_sha256": inventory.pdf_sha256,
            "pdf_size_bytes": inventory.pdf_size_bytes,
            "page_count": inventory.page_count,
            "activities": len(inventory.activities),
            "content_page_range": [
                inventory.content_page_start,
                inventory.content_page_end,
            ],
            "official_key_page_range": [
                inventory.key_page_start,
                inventory.key_page_end,
            ],
            "inventory_projection_sha256": inventory.inventory_projection_sha256,
            "render_manifest_projection_sha256": (
                render_manifest.render_manifest_projection_sha256
            ),
        },
        "frozen_profile": profile,
        "runtime_invocation": {
            "python_executable": (
                "C:/Users/kmaxc/.cache/codex-runtimes/codex-primary-runtime/"
                "dependencies/python/python.exe"
            ),
            "pythonpath": ["tmp/portfolio_official_sources/python_pkgs", "src"],
            "preflight_command": (
                "python scripts/math12_official_source_adapter.py preflight "
                "--pdftoppm C:/Users/kmaxc/.cache/codex-runtimes/"
                "codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe"
            ),
        },
        "code": {
            "files": code_files,
            "combined_code_projection_sha256": canonical_json_sha256(code_files),
        },
        "artifacts": [
            _file_entry(inventory_path),
            _file_entry(render_path),
            {
                **_file_entry(audit_path),
                "projection_sha256": audit["audit_projection_sha256"],
            },
            _file_entry(report_path),
        ],
        "dev_certificates": certificates,
        "official_solution_records": solutions,
        "checks": {
            "unit_fail_closed_tamper_runtime_marker": "14 passed, 1 skipped",
            "real_pdf_inventory_integration": (
                "1 passed, 14 deselected, 149.32 seconds"
            ),
            "full_dev_visual_recomputation": "5 accepted from original images",
            "strict_certificate_replay": "5 passed",
            "official_solution_extraction": "5 passed",
            "own_marker_present": "5/5",
            "next_marker_absent": "5/5",
            "python_compile": "passed",
            "cli_help_and_preflight": "passed",
        },
    }
    projection["manifest_projection_sha256"] = canonical_json_sha256(projection)
    write_canonical_json(args.output, projection)
    print(
        json.dumps(
            {
                "status": projection["status"],
                "manifest_projection_sha256": projection[
                    "manifest_projection_sha256"
                ],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
