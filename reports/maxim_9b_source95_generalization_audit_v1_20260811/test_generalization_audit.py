from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import audit_generalization as audit  # noqa: E402


def test_all_twelve_reidentified_tasks_fail_the_frozen_overlay() -> None:
    result = audit.audit_reidentification()
    assert result["positive_control_selected"] == 12
    assert result["reidentified_rejected"] == 12
    assert result["reidentified_accepted"] == 0
    assert all(
        row["error"] == "combined: candidate solver target mismatch"
        for row in result["counterfactuals"]
    )


def test_pixel_identical_reencodes_all_fail_exact_image_pins() -> None:
    result = audit.audit_pixel_identical_images()
    assert result["pixel_identical"] == 12
    assert result["byte_distinct"] == 12
    assert result["pin_gate_rejected"] == 12
    assert all(row["error"].endswith("SHA/size pin mismatch") for row in result["counterfactuals"])


def test_eleven_narrow_kernels_accept_unseen_structured_inputs() -> None:
    result = audit.audit_parameterized_kernels()
    assert result["reusable_kernel_task_slots_passed"] == 11
    assert result["reusable_kernel_task_slots_total"] == 11
    assert result["hardcoded_task_slots"] == 1
    assert all(row["passed"] for row in result["parameterized_cases"])
    assert result["hardcoded_cases"][0]["task_id"] == "val_0204"
    assert result["hardcoded_cases"][0]["signature"] == "() -> 'dict[str, dict[str, int]]'"


def test_no_source95_overlay_wiring_in_three_production_source_trees() -> None:
    result = audit.audit_production_wiring()
    assert result["production_files_scanned"] > 0
    assert result["source95_marker_hits"] == []
    assert result["basic_rag_solver_registry_mentions_source95"] is False
    assert result["basic_rag_package_discovery_is_src_only"] is True
    assert result["wired_into_production_service"] is False


def test_summary_does_not_conflate_kernel_reuse_with_end_to_end_generalization() -> None:
    result = audit.build_result()
    finding = result["findings"]
    assert finding["strict_end_to_end_generalizable_fixes"] == 0
    assert finding["strict_end_to_end_generalizable_fixes_denominator"] == 12
    assert finding["reusable_parameterized_kernel_task_slots"] == 11
    assert finding["reusable_parameterized_kernel_task_slots_denominator"] == 12
    assert finding["same_textbook_unseen_task_automatically_gets_overlay"] is False
