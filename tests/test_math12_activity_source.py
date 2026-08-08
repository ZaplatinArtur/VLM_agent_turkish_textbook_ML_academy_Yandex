from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import evidence_os.math12_activity_source as math12_source

from evidence_os.math12_activity_source import (
    INVENTORY_SCHEMA,
    KeyLocator,
    Math12ActivityRecord,
    Math12Inventory,
    Math12SourceError,
    build_math12_inventory,
    decide_math12_source_binding,
    extract_official_solution,
    issue_math12_source_certificate,
    load_math12_inventory,
    load_math12_source_certificate,
    resolve_math12_image_bytes,
)
from evidence_os.official_ogm import canonical_json_sha256
from evidence_os.visual_coordinate_binding import (
    VisualBindingThresholds,
    VisualCoordinateBindingError,
    VisualPageEvidence,
)


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _inventory() -> Math12Inventory:
    records: list[Math12ActivityRecord] = []
    starts = list(range(4, 98)) + [98]
    for number, start in enumerate(starts, start=1):
        end = start if number < 95 else 130
        key_page = 131 + min(48, (number - 1) // 2)
        key_end = KeyLocator(
            key_page if number % 2 else min(180, key_page + 1),
            "right" if number % 2 else "left",
            200.0,
        )
        records.append(
            Math12ActivityRecord(
                activity_number=number,
                index_page_number=2 if number < 56 else 3,
                index_column="left" if number % 2 else "right",
                index_top=float(number),
                content_page_start=start,
                content_page_end=end,
                key_page_start=key_page,
                key_page_end=key_end.page_number,
                key_start=KeyLocator(key_page, "left", 80.0),
                key_end_exclusive=key_end,
                index_projection_sha256=HEX_A,
                content_projection_sha256=HEX_B,
                key_projection_sha256=HEX_C,
                binding_projection_sha256=canonical_json_sha256(
                    {"activity_number": number}
                ),
            )
        )
    pins = tuple((page, HEX_A) for page in range(4, 180))
    prototype = Math12Inventory(
        schema_version=INVENTORY_SCHEMA,
        document_id="meb_math12_bbbbbbbbbbbb",
        pdf_basename="math12.pdf",
        pdf_sha256=HEX_B,
        pdf_size_bytes=1,
        page_count=182,
        index_page_start=2,
        index_page_end=3,
        content_page_start=4,
        content_page_end=130,
        key_page_start=131,
        key_page_end=179,
        source_page_projection_sha256=pins,
        activities=tuple(records),
        inventory_projection_sha256=HEX_C,
    )
    return replace(
        prototype,
        inventory_projection_sha256=canonical_json_sha256(prototype.projection()),
    )


def _evidence(page: int, *, strong: bool = False) -> VisualPageEvidence:
    good, inliers = (100, 90) if strong else (0, 0)
    return VisualPageEvidence(
        task_image_sha256=HEX_A,
        document_id="meb_math12_bbbbbbbbbbbb",
        pdf_sha256=HEX_B,
        page_number=page,
        rendered_page_sha256=canonical_json_sha256({"page": page}),
        good_matches=good,
        inliers=inliers,
        inlier_ratio=inliers / good if good else 0.0,
        task_hull_fraction=0.50 if strong else 0.0,
        median_reprojection_error=0.20 if strong else None,
        mapped_inside_fraction=1.0 if strong else 0.0,
        scale_anisotropy=1.01 if strong else None,
        orientation_preserved=strong,
        convex_mapping=strong,
        mapped_polygon=(
            ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
            if strong
            else None
        ),
    )


def _all_page_evidence(selected_page: int = 54) -> tuple[VisualPageEvidence, ...]:
    return tuple(_evidence(page, strong=page == selected_page) for page in range(4, 131))


def test_all_page_source_binding_returns_range_and_generic_key_span() -> None:
    inventory = _inventory()
    decision = decide_math12_source_binding(
        inventory, _all_page_evidence(54), task_image_sha256=HEX_A
    )
    assert decision.accepted
    assert decision.selected_content_page == 54
    assert decision.selected_activity_number == 51
    assert decision.key_start is not None
    assert decision.key_end_exclusive is not None
    assert decision.key_page_start is not None
    assert decision.key_page_end is not None
    assert all(passed for _, passed in decision.checks)


def test_incomplete_page_sweep_abstains() -> None:
    decision = decide_math12_source_binding(
        _inventory(), _all_page_evidence()[:-1], task_image_sha256=HEX_A
    )
    assert not decision.accepted
    assert decision.reason == "incomplete_or_duplicate_all_page_sweep"


def test_safety_floor_cannot_be_weakened() -> None:
    with pytest.raises(VisualCoordinateBindingError):
        decide_math12_source_binding(
            _inventory(),
            _all_page_evidence(),
            task_image_sha256=HEX_A,
            thresholds=VisualBindingThresholds(min_inliers=39),
        )


def test_certificate_is_answer_free_and_pins_all_evidence() -> None:
    certificate = issue_math12_source_certificate(
        _inventory(),
        _all_page_evidence(),
        task_image_sha256=HEX_A,
        render_manifest_projection_sha256=HEX_C,
    )
    mapping = certificate.to_mapping()
    assert certificate.decision.accepted
    assert len(mapping["evidences"]) == 127
    forbidden = {"answer", "correct", "gold", "score", "task_id", "expected_activity"}
    assert forbidden.isdisjoint(mapping)
    assert mapping["component_scope"] == "source_binding_only_no_answer_no_correctness"


def test_official_solution_extraction_rejects_an_abstained_certificate_before_pdf_io() -> None:
    certificate = issue_math12_source_certificate(
        _inventory(),
        _all_page_evidence()[:-1],
        task_image_sha256=HEX_A,
        render_manifest_projection_sha256=HEX_C,
    )
    assert not certificate.decision.accepted
    with pytest.raises(Math12SourceError, match="fully accepted"):
        extract_official_solution(Path("does-not-exist.pdf"), _inventory(), certificate)


def test_official_solution_success_is_bound_to_only_the_pinned_key_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = {
        "activity_number": 51,
        "start": {"page_number": 156, "column": "left", "top": 80.0},
        "end_exclusive": {"page_number": 156, "column": "right", "top": 200.0},
        "words": [
            {
                "physical_page": 156,
                "column": "left",
                "text": "Etkinlik",
                "x0": 10.0,
                "x1": 30.0,
                "top": 80.0,
                "bottom": 90.0,
            },
            {
                "physical_page": 156,
                "column": "left",
                "text": "çözüm",
                "x0": 10.0,
                "x1": 30.0,
                "top": 100.0,
                "bottom": 110.0,
            },
        ],
    }
    inventory = _inventory()
    records = list(inventory.activities)
    target = records[50]
    records[50] = replace(
        target,
        key_projection_sha256=canonical_json_sha256(projection),
        binding_projection_sha256=canonical_json_sha256({"binding": 51}),
    )
    prototype = replace(
        inventory,
        activities=tuple(records),
        inventory_projection_sha256=HEX_C,
    )
    inventory = replace(
        prototype,
        inventory_projection_sha256=canonical_json_sha256(prototype.projection()),
    )
    certificate = issue_math12_source_certificate(
        inventory,
        _all_page_evidence(54),
        task_image_sha256=HEX_A,
        render_manifest_projection_sha256=HEX_C,
    )

    class _FakeDocument:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(math12_source, "sha256_file", lambda _path: inventory.pdf_sha256)
    monkeypatch.setattr(
        math12_source,
        "_key_section_projection",
        lambda *_args, **_kwargs: projection,
    )
    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda _path: _FakeDocument()))
    solution = extract_official_solution(Path("pinned.pdf"), inventory, certificate)
    assert solution.activity_number == 51
    assert solution.official_solution_text == "Etkinlik\nçözüm"
    assert solution.source_certificate_projection_sha256 == (
        certificate.certificate_projection_sha256
    )
    assert len(solution.answer_bound_certificate_projection_sha256) == 64
    assert solution.component_scope == "official_source_solution_text_no_gold_no_correctness"


def test_generic_resolver_signature_has_no_benchmark_route_inputs() -> None:
    parameters = set(inspect.signature(resolve_math12_image_bytes).parameters)
    assert parameters == {
        "image_bytes",
        "inventory",
        "render_manifest",
        "thresholds",
        "runtime_profile",
    }
    assert not {"task_id", "gold", "expected_activity", "answer", "score"} & parameters


def test_certificate_loader_rejects_post_hoc_route_tampering(tmp_path: Path) -> None:
    certificate = issue_math12_source_certificate(
        _inventory(),
        _all_page_evidence(),
        task_image_sha256=HEX_A,
        render_manifest_projection_sha256=HEX_C,
    ).to_mapping()
    certificate["decision"]["selected_activity_number"] = 95
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(certificate), encoding="utf-8")
    with pytest.raises(Math12SourceError, match="projection pin mismatch"):
        load_math12_source_certificate(path)


@pytest.mark.skipif(
    os.environ.get("MATH12_RUN_SOURCE_INTEGRATION") != "1",
    reason="opt-in source-PDF integration test",
)
def test_official_pdf_inventory_is_complete_and_target_dev_addresses_are_source_derived() -> None:
    root = Path(__file__).resolve().parents[1]
    matches = list(
        (root / "tmp" / "remaining_official_source_audit" / "pdfs").glob(
            "matematik 12*.pdf"
        )
    )
    assert len(matches) == 1
    inventory = build_math12_inventory(matches[0])
    assert inventory.pdf_sha256 == (
        "16d650177e62dc04b9a8b42fd7aafc3c1a8a38ec8c7040f92d5a26b120cde548"
    )
    assert len(inventory.activities) == 95
    expected = {
        3: (7, 7, 131),
        17: (24, 25, 139),
        31: (40, 41, 145),
        43: (60, 60, 152),
        88: (119, 119, 174),
    }
    for number, (content_start, content_end, key_start) in expected.items():
        record = inventory.activities[number - 1]
        assert (record.content_page_start, record.content_page_end) == (
            content_start,
            content_end,
        )
        assert record.key_page_start == key_start


def test_generated_inventory_round_trip_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "reports"
        / "maxim_math12_activity_source_v1_20260808"
        / "inventory.json"
    )
    if not path.is_file():
        pytest.skip("generated inventory is not present")
    inventory = load_math12_inventory(path)
    assert len(inventory.activities) == 95
    assert inventory.content_page_start == 4
    assert inventory.content_page_end == 130
