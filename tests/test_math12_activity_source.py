from __future__ import annotations

from dataclasses import replace
import hashlib
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
    FROZEN_SIFT_RUNTIME_PROFILE,
    FROZEN_VISUAL_THRESHOLDS,
    KeyLocator,
    Math12ActivityRecord,
    Math12Inventory,
    Math12RenderedPage,
    Math12RenderManifest,
    Math12SourceError,
    build_math12_inventory,
    decide_math12_source_binding,
    extract_official_solution,
    issue_math12_source_certificate,
    load_math12_inventory,
    load_math12_source_certificate,
    resolve_math12_image_bytes,
    verify_math12_source_certificate,
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
        rendered_page_sha256=hashlib.sha256(f"page-{page}".encode("ascii")).hexdigest(),
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


def _render_manifest(tmp_path: Path, inventory: Math12Inventory) -> Math12RenderManifest:
    pages: list[Math12RenderedPage] = []
    for page_number in range(4, 131):
        payload = f"page-{page_number}".encode("ascii")
        name = f"page-{page_number:03d}.png"
        path = tmp_path / name
        path.write_bytes(payload)
        pages.append(
            Math12RenderedPage(
                page_number=page_number,
                manifest_path=name,
                path=path,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        )
    prototype = Math12RenderManifest(
        schema_version=math12_source.RENDER_MANIFEST_SCHEMA,
        document_id=inventory.document_id,
        pdf_sha256=inventory.pdf_sha256,
        inventory_projection_sha256=inventory.inventory_projection_sha256,
        render_dpi=FROZEN_SIFT_RUNTIME_PROFILE.render_dpi,
        color_mode="gray_png",
        poppler_version=math12_source.EXPECTED_POPPLER_VERSION,
        pages=tuple(pages),
        render_manifest_projection_sha256=HEX_C,
    )
    return replace(
        prototype,
        render_manifest_projection_sha256=canonical_json_sha256(prototype.projection()),
    )


def _repin_certificate(certificate):
    evidence_pin = canonical_json_sha256(
        [math12_source._evidence_to_mapping(item) for item in certificate.evidences]
    )
    provisional = replace(
        certificate,
        evidence_projection_sha256=evidence_pin,
        certificate_projection_sha256=HEX_C,
    )
    mapping = provisional.to_mapping()
    projection = {
        key: mapping[key]
        for key in (
            "schema_version",
            "document_id",
            "pdf_sha256",
            "inventory_projection_sha256",
            "render_manifest_projection_sha256",
            "task_image_sha256",
            "thresholds",
            "runtime_profile",
            "decision",
            "evidence_projection_sha256",
        )
    }
    return replace(
        provisional,
        certificate_projection_sha256=canonical_json_sha256(projection),
    )


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


def test_exact_frozen_thresholds_and_runtime_cannot_be_overridden() -> None:
    with pytest.raises(VisualCoordinateBindingError, match="exact frozen profile"):
        decide_math12_source_binding(
            _inventory(),
            _all_page_evidence(),
            task_image_sha256=HEX_A,
            thresholds=replace(FROZEN_VISUAL_THRESHOLDS, min_inliers=41),
        )
    with pytest.raises(VisualCoordinateBindingError, match="exact frozen profile"):
        issue_math12_source_certificate(
            _inventory(),
            _all_page_evidence(),
            task_image_sha256=HEX_A,
            render_manifest_projection_sha256=HEX_C,
            runtime_profile=replace(FROZEN_SIFT_RUNTIME_PROFILE, nfeatures=12_001),
        )
    with pytest.raises(VisualCoordinateBindingError, match="exact frozen profile"):
        resolve_math12_image_bytes(
            b"not-decoded-because-profile-fails-first",
            _inventory(),
            None,  # type: ignore[arg-type]
            runtime_profile=replace(FROZEN_SIFT_RUNTIME_PROFILE, rng_seed=7),
        )


def test_marker_floor_includes_own_header_and_excludes_next_header_and_body() -> None:
    def word(text: str, x0: float, top: float) -> dict[str, object]:
        return {"text": text, "x0": x0, "x1": x0 + 10.0, "top": top, "bottom": top + 1.0}

    own = (
        word("Etkinlik", 10.0, 10.00019),
        word("No.:", 25.0, 10.00005),
        word("1", 38.0, 10.00012),
    )
    following = (
        word("Etkinlik", 10.0, 20.00019),
        word("No.:", 25.0, 20.00005),
        word("2", 38.0, 20.00012),
    )
    words = [*own, word("own-body", 10.0, 15.0), *following, word("next-body", 10.0, 21.0)]

    class _Page:
        width = 100.0

        def extract_words(self, **_kwargs):
            return words

    document = SimpleNamespace(pages=[_Page()])
    start = math12_source._key_marker_locator(1, 100.0, own)
    end = math12_source._key_marker_locator(1, 100.0, following)
    assert start.top == 10.0
    assert end.top == 20.0
    projection = math12_source._key_section_projection(document, 1, start, end)
    selected = [item["text"] for item in projection["words"]]
    assert selected == ["Etkinlik", "No.:", "1", "own-body"]
    assert "2" not in selected
    assert "next-body" not in selected


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
        extract_official_solution(
            Path("does-not-exist.pdf"), _inventory(), None, certificate  # type: ignore[arg-type]
        )


def test_official_solution_success_is_bound_to_only_the_pinned_key_span(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    manifest = _render_manifest(tmp_path, inventory)
    certificate = issue_math12_source_certificate(
        inventory,
        _all_page_evidence(54),
        task_image_sha256=HEX_A,
        render_manifest_projection_sha256=manifest.render_manifest_projection_sha256,
    )

    class _FakeDocument:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        math12_source,
        "sha256_file",
        lambda path: (
            inventory.pdf_sha256
            if Path(path).name == "pinned.pdf"
            else hashlib.sha256(Path(path).read_bytes()).hexdigest()
        ),
    )
    monkeypatch.setattr(
        math12_source,
        "_key_section_projection",
        lambda *_args, **_kwargs: projection,
    )
    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda _path: _FakeDocument()))
    monkeypatch.setattr(math12_source, "assert_math12_runtime", lambda **_kwargs: {})
    solution = extract_official_solution(
        Path("pinned.pdf"), inventory, manifest, certificate
    )
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


def test_strict_replay_rejects_self_consistent_post_hoc_decision(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    manifest = _render_manifest(tmp_path, inventory)
    certificate = issue_math12_source_certificate(
        inventory,
        _all_page_evidence(),
        task_image_sha256=HEX_A,
        render_manifest_projection_sha256=manifest.render_manifest_projection_sha256,
    )
    tampered = _repin_certificate(
        replace(
            certificate,
            decision=replace(certificate.decision, selected_activity_number=95),
        )
    )
    with pytest.raises(Math12SourceError, match="does not replay"):
        verify_math12_source_certificate(inventory, manifest, tampered)


def test_strict_replay_rejects_evidence_rebound_to_other_render_bytes(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    manifest = _render_manifest(tmp_path, inventory)
    certificate = issue_math12_source_certificate(
        inventory,
        _all_page_evidence(),
        task_image_sha256=HEX_A,
        render_manifest_projection_sha256=manifest.render_manifest_projection_sha256,
    )
    evidences = list(certificate.evidences)
    evidences[0] = replace(evidences[0], rendered_page_sha256=HEX_C)
    tampered = _repin_certificate(replace(certificate, evidences=tuple(evidences)))
    with pytest.raises(Math12SourceError, match="not bound to rendered page bytes"):
        verify_math12_source_certificate(inventory, manifest, tampered)


def test_generated_dev_solutions_do_not_contain_the_next_activity_header() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = (
        root
        / "reports"
        / "maxim_math12_activity_source_v1_20260808"
        / "official_solutions"
    )
    if not directory.is_dir():
        pytest.skip("generated official solution records are not present")
    files = sorted(directory.glob("val_*.json"))
    assert len(files) == 5
    for path in files:
        value = json.loads(path.read_text(encoding="utf-8"))
        activity = int(value["activity_number"])
        assert f"Etkinlik No.: {activity + 1}" not in value["official_solution_text"]


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
