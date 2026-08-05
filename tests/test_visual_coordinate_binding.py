from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from evidence_os.visual_coordinate_binding import (
    ActivityVisualObservationRef,
    ActivityVisualRecordRef,
    IndexedQuestionRef,
    PdfQuestionMarker,
    SiftRuntimeProfile,
    VisualBindingThresholds,
    VisualCoordinateBindingError,
    VisualPageEvidence,
    compute_sift_page_evidence,
    decide_visual_activity_binding,
    decide_visual_binding,
    load_activity_visual_artifact_json,
    unique_indexed_markers,
    verify_activity_visual_binding,
    verified_activity_bindings_from_artifact,
    visual_page_evidence_from_mapping,
)


TASK_SHA = "a" * 64
PDF_SHA = "b" * 64
RENDER_SHA = "c" * 64
DOCUMENT_ID = "workbook_deadbeef"


def _evidence(
    *,
    page: int = 15,
    rendered_sha: str = RENDER_SHA,
    good: int = 100,
    inliers: int = 80,
    ratio: float = 0.80,
    hull: float = 0.40,
    error: float | None = 0.20,
    inside: float = 1.0,
    anisotropy: float | None = 1.01,
    orientation: bool = True,
    convex: bool = True,
    polygon: tuple[tuple[float, float], ...] | None = (
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (0.0, 100.0),
    ),
    task_sha: str = TASK_SHA,
    document_id: str = DOCUMENT_ID,
    pdf_sha: str = PDF_SHA,
) -> VisualPageEvidence:
    return VisualPageEvidence(
        task_image_sha256=task_sha,
        document_id=document_id,
        pdf_sha256=pdf_sha,
        page_number=page,
        rendered_page_sha256=rendered_sha,
        good_matches=good,
        inliers=inliers,
        inlier_ratio=ratio,
        task_hull_fraction=hull,
        median_reprojection_error=error,
        mapped_inside_fraction=inside,
        scale_anisotropy=anisotropy,
        orientation_preserved=orientation,
        convex_mapping=convex,
        mapped_polygon=polygon,  # type: ignore[arg-type]
    )


def _record(
    *,
    page: int = 15,
    number: int = 14,
    record_id: str | None = None,
    answer_format: str = "choice",
    key_binding_kind: str = "inline_solution",
    key_projection_sha256: str | None = None,
    visually_checked: bool = True,
    key_bbox: tuple[float, float, float, float] | None = (10.0, 10.0, 20.0, 20.0),
) -> IndexedQuestionRef:
    return IndexedQuestionRef(
        document_id=DOCUMENT_ID,
        record_id=record_id or f"{DOCUMENT_ID}:p{page}:q{number}",
        content_page_number=page,
        question_number=number,
        visually_checked=visually_checked,
        answer_format=answer_format,
        key_binding_kind=key_binding_kind,
        key_page_number=page,
        key_bbox=key_bbox,
        key_projection_sha256=key_projection_sha256,
    )


def _marker(*, page: int = 15, number: int = 14, center: tuple[float, float] = (20.0, 20.0)) -> PdfQuestionMarker:
    return PdfQuestionMarker(page, number, center, f"{number}.")


def _decide(
    evidences: list[VisualPageEvidence],
    markers: list[PdfQuestionMarker] | None = None,
    records: list[IndexedQuestionRef] | None = None,
    observed: int | None = None,
):
    return decide_visual_binding(
        evidences,
        markers or [_marker()],
        records or [_record()],
        expected_task_image_sha256=TASK_SHA,
        expected_document_id=DOCUMENT_ID,
        expected_pdf_sha256=PDF_SHA,
        observed_question_number=observed,
    )


def _evidence_mapping(
    *,
    page: int = 15,
    rendered_sha: str = RENDER_SHA,
    document_id: str = DOCUMENT_ID,
    inliers: int = 80,
) -> dict[str, object]:
    return {
        "task_image_sha256": TASK_SHA,
        "document_id": document_id,
        "pdf_sha256": PDF_SHA,
        "page_number": page,
        "rendered_page_sha256": rendered_sha,
        "good_matches": 100,
        "inliers": inliers,
        "inlier_ratio": 0.80,
        "task_hull_fraction": 0.40,
        "median_reprojection_error": 0.20,
        "mapped_inside_fraction": 1.0,
        "scale_anisotropy": 1.01,
        "orientation_preserved": True,
        "convex_mapping": True,
        "mapped_polygon": [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ],
    }


def _activity_record(
    *,
    page: int = 15,
    number: int = 3,
    content_bbox: tuple[float, float, float, float] = (0.0, 0.0, 50.0, 50.0),
) -> ActivityVisualRecordRef:
    return ActivityVisualRecordRef(
        document_id=DOCUMENT_ID,
        record_id=f"{DOCUMENT_ID}:p{page}:q{number}",
        content_page_number=page,
        activity_number=number,
        key_projection_sha256="d" * 64,
        content_projection_sha256="e" * 64,
        binding_projection_sha256="f" * 64,
        visually_checked=True,
        content_bbox=content_bbox,
    )


def _decide_activity(
    evidences: list[VisualPageEvidence],
    *,
    records: list[ActivityVisualRecordRef] | None = None,
    observed: int = 3,
):
    return decide_visual_activity_binding(
        evidences,
        records or [_activity_record()],
        expected_task_image_sha256=TASK_SHA,
        expected_document_id=DOCUMENT_ID,
        expected_pdf_sha256=PDF_SHA,
        observed_activity_number=observed,
    )


def _minimal_activity_artifact(
    repo_root: Path,
) -> tuple[
    dict[str, object],
    dict[str, ActivityVisualObservationRef],
    list[ActivityVisualRecordRef],
    dict[str, Path],
    dict[str, str],
]:
    pdf_path = repo_root / "source.pdf"
    image_path = repo_root / "task.png"
    pdf_path.write_bytes(b"synthetic pinned PDF bytes")
    image_path.write_bytes(b"synthetic pinned task image bytes")
    pdf_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    pins = {
        "parser": "1" * 64,
        "source_index": "2" * 64,
        "source_locators": "3" * 64,
    }
    observation = ActivityVisualObservationRef(
        task_id="synthetic_task",
        task_image_sha256=image_sha,
        width=100,
        height=100,
        parser_identity="synthetic-parser-v1",
        document_id=DOCUMENT_ID,
        pdf_sha256=pdf_sha,
        marker_kind="activity_label",
        marker_number=3,
    )
    record = _activity_record()
    raw_evidence = _evidence_mapping()
    raw_evidence["task_image_sha256"] = image_sha
    raw_evidence["pdf_sha256"] = pdf_sha
    artifact: dict[str, object] = {
        "schema_version": "maxim-activity-visual-binding-source-evidence-v1",
        "generator": "build-maxim-activity-visual-binding-v1",
        "inputs": {
            "documents": {
                DOCUMENT_ID: {
                    "candidate_content_pages": [15],
                    "content_page_ranges": [[15, 15]],
                    "page_count": 15,
                    "pdf_path": "source.pdf",
                    "pdf_sha256": pdf_sha,
                    "rendered_pages": {
                        "15": {
                            "width": 100,
                            "height": 100,
                            "rendered_page_sha256": RENDER_SHA,
                        }
                    },
                }
            },
            "parser_observations": {
                "path": "parser.jsonl",
                "sha256": pins["parser"],
                "rows": 1,
            },
            "source_index": {
                "path": "source-index.json",
                "sha256": pins["source_index"],
            },
            "source_locators": {
                "path": "source-locators.jsonl",
                "sha256": pins["source_locators"],
                "rows": 1,
            },
            "task_image_dir": ".",
        },
        "profiles": {
            "sift": {
                "render_dpi": 144,
                "nfeatures": 12_000,
                "contrast_threshold": 0.02,
                "edge_threshold": 12.0,
                "ratio_test": 0.72,
                "ransac_reprojection_px": 4.0,
                "ransac_max_iters": 5_000,
                "ransac_confidence": 0.999,
                "rng_seed": 19_870_511,
                "expected_opencv_version": "5.0.0",
            }
        },
        "runtime": {
            "python": {"executable": "python", "version": "3.12.13"},
            "opencv": {
                "module_path": "cv2.pyd",
                "opencl_enabled": False,
                "threads": 1,
                "version": "5.0.0",
            },
            "numpy": {"module_path": "numpy", "version": "2.5.1"},
            "poppler": {
                "pdfinfo": {
                    "path": "pdfinfo",
                    "sha256": "6" * 64,
                    "version": "synthetic",
                    "version_line": "synthetic",
                },
                "pdftoppm": {
                    "path": "pdftoppm",
                    "sha256": "7" * 64,
                    "version": "synthetic",
                    "version_line": "synthetic",
                },
            },
            "package_root": "tmp/portfolio_official_sources/python_pkgs",
            "sift_process_workers": 4,
        },
        "source_only_guards": {
            "benchmark_answer_candidate_outcome_artifacts_read": False,
            "source_answer_value_access": False,
            "task_id_is_policy_feature": False,
            "task_id_role": "alignment_audit_only",
            "parser_observation_filter": (
                "observed_source_question_marker=activity_label"
            ),
            "source_index_record_filter": (
                "key_binding_kind=activity_answer_key AND "
                "question_marker_kind=activity_label"
            ),
            "render_scope": (
                "all indexed physical content pages expanded from each activity "
                "document content_page_ranges"
            ),
        },
        "summary": {
            "activity_documents": 1,
            "activity_observations": 1,
            "activity_records": 1,
            "decision_layer": "not_applied_raw_evidence_only",
            "raw_page_evidences": 1,
        },
        "activity_records": [
            {
                "document_id": DOCUMENT_ID,
                "record_id": record.record_id,
                "content_page_number": 15,
                "activity_number": 3,
                "key_binding_kind": "activity_answer_key",
                "question_marker_kind": "activity_label",
                "content_bbox": [0.0, 0.0, 50.0, 50.0],
                "key_projection_sha256": record.key_projection_sha256,
                "content_projection_sha256": record.content_projection_sha256,
                "binding_projection_sha256": record.binding_projection_sha256,
                "visually_checked": True,
            }
        ],
        "bindings_by_task_image_sha256": {
            image_sha: {
                "alignment_audit": {
                    "task_id": observation.task_id,
                    "task_id_role": (
                        "parser_to_public_source_locator_alignment_only"
                    ),
                    "task_id_used_as_page_or_record_feature": False,
                },
                "document_id": DOCUMENT_ID,
                "observed_source_marker": {"kind": "activity_label", "number": 3},
                "parser_identity": observation.parser_identity,
                "raw_page_evidences": [raw_evidence],
                "source_pins": {
                    "parser_artifact_sha256": pins["parser"],
                    "parser_projection_sha256": "4" * 64,
                    "pdf_sha256": pdf_sha,
                    "source_identity_projection_sha256": "5" * 64,
                    "source_index_sha256": pins["source_index"],
                    "source_locators_artifact_sha256": pins["source_locators"],
                },
                "task_image": {
                    "path": image_path.name,
                    "image_basename": image_path.name,
                    "image_sha256": image_sha,
                    "width": 100,
                    "height": 100,
                },
            }
        },
    }
    return (
        artifact,
        {observation.task_id: observation},
        [record],
        {DOCUMENT_ID: pdf_path},
        pins,
    )


def test_visual_activity_accepts_one_strong_source_attested_record() -> None:
    evidence = visual_page_evidence_from_mapping(_evidence_mapping())

    decision = _decide_activity([evidence])

    assert decision.accepted is True
    assert decision.reason == "accepted"
    assert decision.selected_page_number == 15
    assert decision.selected_question_number == 3
    assert decision.selected_record_id == f"{DOCUMENT_ID}:p15:q3"
    assert all(passed for _, passed in decision.checks)


def test_visual_activity_rejects_wrong_observed_activity() -> None:
    evidence = visual_page_evidence_from_mapping(_evidence_mapping())

    decision = _decide_activity([evidence], observed=4)

    assert decision.accepted is False
    assert decision.reason == "activity_record_binding_failed"
    assert dict(decision.checks)["one_indexed_activity_on_page"] is True
    assert dict(decision.checks)["observed_activity_marker_agrees"] is False


def test_visual_activity_rejects_multiple_records_on_selected_page() -> None:
    evidence = visual_page_evidence_from_mapping(_evidence_mapping())

    decision = _decide_activity(
        [evidence],
        records=[_activity_record(number=3), _activity_record(number=4)],
    )

    assert decision.accepted is False
    assert decision.reason == "activity_record_binding_failed"
    assert dict(decision.checks)["one_indexed_activity_on_page"] is False


def test_visual_activity_rejects_source_identity_mismatch() -> None:
    evidence = visual_page_evidence_from_mapping(
        _evidence_mapping(document_id="another_document")
    )

    decision = _decide_activity([evidence])

    assert decision.accepted is False
    assert decision.reason == "source_identity_mismatch"
    assert decision.checks == (("source_identity", False),)


def test_visual_page_mapping_exact_allowlist_rejects_extra_field() -> None:
    mapping = _evidence_mapping()
    mapping["unexpected"] = "not-source-evidence"

    with pytest.raises(VisualCoordinateBindingError, match="exact allowlist"):
        visual_page_evidence_from_mapping(mapping)


@pytest.mark.parametrize("field", ("page_number", "good_matches", "inliers"))
@pytest.mark.parametrize("invalid_value", (15.0, True, "15"))
def test_visual_page_mapping_rejects_noncanonical_integers(
    field: str,
    invalid_value: object,
) -> None:
    mapping = _evidence_mapping()
    mapping[field] = invalid_value

    with pytest.raises(
        VisualCoordinateBindingError,
        match=rf"{field} is not a canonical integer",
    ):
        visual_page_evidence_from_mapping(mapping)


def test_visual_page_mapping_rejects_inlier_ratio_inconsistent_with_counts() -> None:
    mapping = _evidence_mapping()
    mapping["inlier_ratio"] = 0.79

    with pytest.raises(
        VisualCoordinateBindingError,
        match="inlier ratio does not match integer counts",
    ):
        visual_page_evidence_from_mapping(mapping)


def test_visual_artifact_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    artifact_path = tmp_path / "duplicate.json"
    artifact_path.write_text(
        '{"schema_version":"first","schema_version":"second"}',
        encoding="utf-8",
    )

    with pytest.raises(VisualCoordinateBindingError, match="duplicate JSON key"):
        load_activity_visual_artifact_json(artifact_path)


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_visual_artifact_json_rejects_nonfinite_constants(
    tmp_path: Path,
    constant: str,
) -> None:
    artifact_path = tmp_path / "nonfinite.json"
    artifact_path.write_text(f'{{"value":{constant}}}', encoding="utf-8")

    with pytest.raises(VisualCoordinateBindingError, match="non-finite JSON constant"):
        load_activity_visual_artifact_json(artifact_path)


@pytest.mark.parametrize(
    ("field", "weak_value"),
    (
        ("min_good_matches", 49),
        ("min_inliers", 39),
        ("min_inlier_ratio", 0.64),
        ("min_task_hull_fraction", 0.29),
        ("max_median_reprojection_error", 1.01),
        ("min_mapped_inside_fraction", 0.97),
        ("max_scale_anisotropy", 1.16),
        ("min_rank_score_margin", 9.9),
        ("min_rank_score_ratio", 4.9),
    ),
)
def test_visual_activity_rejects_any_threshold_below_the_safety_floor(
    field: str,
    weak_value: int | float,
) -> None:
    thresholds = VisualBindingThresholds(**{field: weak_value})

    with pytest.raises(VisualCoordinateBindingError, match="weaken the safety floor"):
        verify_activity_visual_binding(
            [_evidence()],
            [_activity_record()],
            expected_task_image_sha256=TASK_SHA,
            expected_document_id=DOCUMENT_ID,
            expected_pdf_sha256=PDF_SHA,
            observed_activity_number=3,
            thresholds=thresholds,
        )


def test_visual_activity_artifact_rejects_duplicate_page_evidence(
    tmp_path: Path,
) -> None:
    artifact, observations, records, pdf_paths, pins = _minimal_activity_artifact(
        tmp_path
    )
    verified = verified_activity_bindings_from_artifact(
        artifact,
        repo_root=tmp_path,
        expected_parser_sha256=pins["parser"],
        expected_source_locators_sha256=pins["source_locators"],
        expected_source_index_sha256=pins["source_index"],
        observations_by_task_id=observations,
        records=records,
        document_pdf_paths=pdf_paths,
    )
    assert len(verified) == 1

    bindings = artifact["bindings_by_task_image_sha256"]
    assert isinstance(bindings, dict)
    raw_binding = next(iter(bindings.values()))
    assert isinstance(raw_binding, dict)
    raw_evidences = raw_binding["raw_page_evidences"]
    assert isinstance(raw_evidences, list)
    assert isinstance(raw_evidences[0], dict)
    raw_evidences.append(dict(raw_evidences[0]))

    with pytest.raises(
        VisualCoordinateBindingError,
        match="visual activity evidence page pins changed",
    ):
        verified_activity_bindings_from_artifact(
            artifact,
            repo_root=tmp_path,
            expected_parser_sha256=pins["parser"],
            expected_source_locators_sha256=pins["source_locators"],
            expected_source_index_sha256=pins["source_index"],
            observations_by_task_id=observations,
            records=records,
            document_pdf_paths=pdf_paths,
        )


@pytest.mark.parametrize("omitted_page", (15, 16))
def test_visual_activity_artifact_requires_every_candidate_page_evidence(
    tmp_path: Path,
    omitted_page: int,
) -> None:
    artifact, observations, records, pdf_paths, pins = _minimal_activity_artifact(
        tmp_path
    )
    inputs = artifact["inputs"]
    assert isinstance(inputs, dict)
    documents = inputs["documents"]
    assert isinstance(documents, dict)
    document = documents[DOCUMENT_ID]
    assert isinstance(document, dict)
    document["candidate_content_pages"] = [15, 16]
    document["content_page_ranges"] = [[15, 16]]
    document["page_count"] = 16
    rendered_pages = document["rendered_pages"]
    assert isinstance(rendered_pages, dict)
    second_render_sha = "8" * 64
    rendered_pages["16"] = {
        "width": 100,
        "height": 100,
        "rendered_page_sha256": second_render_sha,
    }

    bindings = artifact["bindings_by_task_image_sha256"]
    assert isinstance(bindings, dict)
    image_sha, raw_binding = next(iter(bindings.items()))
    assert isinstance(image_sha, str)
    assert isinstance(raw_binding, dict)
    raw_evidences = raw_binding["raw_page_evidences"]
    assert isinstance(raw_evidences, list)
    second_evidence = _evidence_mapping(
        page=16,
        rendered_sha=second_render_sha,
        inliers=8,
    )
    second_evidence["inlier_ratio"] = 0.08
    second_evidence["task_image_sha256"] = image_sha
    second_evidence["pdf_sha256"] = observations["synthetic_task"].pdf_sha256
    raw_evidences.append(second_evidence)
    summary = artifact["summary"]
    assert isinstance(summary, dict)
    summary["raw_page_evidences"] = 2

    verified = verified_activity_bindings_from_artifact(
        artifact,
        repo_root=tmp_path,
        expected_parser_sha256=pins["parser"],
        expected_source_locators_sha256=pins["source_locators"],
        expected_source_index_sha256=pins["source_index"],
        observations_by_task_id=observations,
        records=records,
        document_pdf_paths=pdf_paths,
    )
    assert len(verified) == 1
    assert {item.page_number for item in next(iter(verified.values())).evidences} == {
        15,
        16,
    }

    raw_binding["raw_page_evidences"] = [
        item
        for item in raw_evidences
        if isinstance(item, dict) and item["page_number"] != omitted_page
    ]

    with pytest.raises(
        VisualCoordinateBindingError,
        match="visual activity evidence page pins changed",
    ):
        verified_activity_bindings_from_artifact(
            artifact,
            repo_root=tmp_path,
            expected_parser_sha256=pins["parser"],
            expected_source_locators_sha256=pins["source_locators"],
            expected_source_index_sha256=pins["source_index"],
            observations_by_task_id=observations,
            records=records,
            document_pdf_paths=pdf_paths,
        )


def test_visual_activity_artifact_rejects_candidate_range_mismatch(
    tmp_path: Path,
) -> None:
    artifact, observations, records, pdf_paths, pins = _minimal_activity_artifact(
        tmp_path
    )
    inputs = artifact["inputs"]
    assert isinstance(inputs, dict)
    documents = inputs["documents"]
    assert isinstance(documents, dict)
    document = documents[DOCUMENT_ID]
    assert isinstance(document, dict)
    document["content_page_ranges"] = [[15, 16]]
    document["page_count"] = 16

    with pytest.raises(
        VisualCoordinateBindingError,
        match="visual activity page inventory changed",
    ):
        verified_activity_bindings_from_artifact(
            artifact,
            repo_root=tmp_path,
            expected_parser_sha256=pins["parser"],
            expected_source_locators_sha256=pins["source_locators"],
            expected_source_index_sha256=pins["source_index"],
            observations_by_task_id=observations,
            records=records,
            document_pdf_paths=pdf_paths,
        )


def test_visual_activity_content_bbox_mismatch_fails_closed() -> None:
    evidence = _evidence()
    record = _activity_record(content_bbox=(200.0, 200.0, 250.0, 250.0))

    decision = _decide_activity([evidence], records=[record])

    assert decision.accepted is False
    assert decision.reason == "activity_record_binding_failed"
    assert dict(decision.checks)["mapped_crop_bbox_iou"] is False
    assert decision.selected_record_id is None


def test_visual_activity_rejects_insufficient_page_margin() -> None:
    best = visual_page_evidence_from_mapping(_evidence_mapping())
    runner = visual_page_evidence_from_mapping(
        _evidence_mapping(page=16, rendered_sha="1" * 64)
    )

    decision = _decide_activity([best, runner])

    assert decision.accepted is False
    assert decision.reason == "visual_geometry_or_margin_failed"
    assert dict(decision.checks)["page_rank_margin"] is False
    assert dict(decision.checks)["page_rank_ratio"] is False


def test_accepts_strong_unique_pdf_marker_and_certifiable_record() -> None:
    decision = _decide([_evidence(), _evidence(page=16, rendered_sha="d" * 64, inliers=8, ratio=0.08)])

    assert decision.accepted is True
    assert decision.selected_page_number == 15
    assert decision.selected_question_number == 14
    assert decision.selected_record_id == f"{DOCUMENT_ID}:p15:q14"
    assert all(passed for _, passed in decision.checks)


def test_task_id_is_not_a_policy_feature() -> None:
    evidence = _evidence()
    first = _decide([evidence])
    second = _decide([evidence])

    assert first == second


def test_abstains_when_high_raw_inliers_have_impossible_geometry() -> None:
    hard_negative = _evidence(
        good=227,
        inliers=198,
        ratio=198 / 227,
        hull=0.59,
        error=0.0,
        inside=0.0,
        anisotropy=0.0,
        orientation=False,
        convex=False,
        polygon=None,
    )

    decision = _decide([hard_negative])

    assert decision.accepted is False
    assert decision.reason == "visual_geometry_or_margin_failed"
    assert dict(decision.checks)["mapped_inside_fraction"] is False
    assert dict(decision.checks)["convex_mapping"] is False


def test_frozen_inlier_ratio_fails_closed_below_half() -> None:
    evidence = _evidence(good=362, inliers=174, ratio=174 / 362, hull=0.65, error=0.33)

    decision = _decide([evidence])

    assert decision.accepted is False
    assert dict(decision.checks)["inlier_ratio"] is False


def test_abstains_without_page_margin() -> None:
    best = _evidence(page=15, rendered_sha="1" * 64)
    runner = _evidence(page=16, rendered_sha="2" * 64, inliers=79, ratio=0.79)

    decision = _decide([best, runner])

    assert decision.accepted is False
    assert dict(decision.checks)["page_rank_ratio"] is False


def test_observed_number_must_agree_with_mapped_marker() -> None:
    decision = _decide([_evidence()], observed=13)

    assert decision.accepted is False
    assert dict(decision.checks)["observed_number_agrees"] is False


def test_multiple_indexed_markers_inside_crop_abstain() -> None:
    markers = [_marker(number=14), _marker(number=16, center=(80.0, 80.0))]
    records = [_record(number=14), _record(number=16)]

    decision = _decide([_evidence()], markers, records)

    assert decision.accepted is False
    assert decision.reason == "indexed_pdf_marker_ambiguous_or_absent"


def test_duplicate_source_records_abstain() -> None:
    records = [_record(record_id=f"{DOCUMENT_ID}:p15:q14:a"), _record(record_id=f"{DOCUMENT_ID}:p15:q14:b")]

    decision = _decide([_evidence()], records=records)

    assert decision.accepted is False
    assert dict(decision.checks)["unique_source_record"] is False


def test_short_text_key_requires_coordinate_projection_pin() -> None:
    record = _record(answer_format="short_text", key_binding_kind="coordinate_answer_key")

    decision = _decide([_evidence()], records=[record])

    assert decision.accepted is False
    assert dict(decision.checks)["certifiable_reviewed_key"] is False


def test_short_text_coordinate_key_with_projection_is_certifiable() -> None:
    record = _record(
        answer_format="short_text",
        key_binding_kind="coordinate_answer_key",
        key_projection_sha256="e" * 64,
    )

    decision = _decide([_evidence()], records=[record])

    assert decision.accepted is True


def test_source_identity_mismatch_abstains_before_marker_binding() -> None:
    decision = _decide([_evidence(document_id="another_document")])

    assert decision.accepted is False
    assert decision.reason == "source_identity_mismatch"


def test_pdf_word_projection_keeps_only_exact_indexed_markers() -> None:
    words = [
        {"text": "14.", "x0": 10.0, "x1": 14.0, "top": 20.0, "bottom": 24.0},
        {"text": "16)", "x0": 20.0, "x1": 24.0, "top": 30.0, "bottom": 34.0},
        {"text": "17.", "x0": 30.0, "x1": 34.0, "top": 40.0, "bottom": 44.0},
        {"text": "14", "x0": 40.0, "x1": 44.0, "top": 50.0, "bottom": 54.0},
    ]

    markers = unique_indexed_markers(
        words,
        page_number=15,
        indexed_question_numbers={14, 16},
        render_width=200,
        render_height=400,
        pdf_width=100.0,
        pdf_height=200.0,
    )

    assert [(item.question_number, item.center) for item in markers] == [
        (14, (24.0, 44.0)),
        (16, (44.0, 64.0)),
    ]


def test_compute_sift_evidence_recovers_a_synthetic_crop(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    page = np.full((700, 900, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(731)
    for index in range(140):
        x = int(rng.integers(30, 870))
        y = int(rng.integers(30, 670))
        radius = int(rng.integers(3, 13))
        color = tuple(int(value) for value in rng.integers(0, 170, size=3))
        cv2.circle(page, (x, y), radius, color, 1 + index % 3)
        if index % 7 == 0:
            cv2.putText(page, f"Q{index}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    crop = page[120:560, 170:720].copy()
    page_path = tmp_path / "page.png"
    crop_path = tmp_path / "crop.png"
    assert cv2.imwrite(str(page_path), page)
    assert cv2.imwrite(str(crop_path), crop)
    crop_sha = hashlib.sha256(crop_path.read_bytes()).hexdigest()

    evidence = compute_sift_page_evidence(
        crop_path,
        page_path,
        task_image_sha256=crop_sha,
        document_id=DOCUMENT_ID,
        pdf_sha256=PDF_SHA,
        page_number=15,
        profile=SiftRuntimeProfile(expected_opencv_version=cv2.__version__),
    )

    assert evidence.inliers >= 40
    assert evidence.inlier_ratio >= 0.80
    assert evidence.task_hull_fraction >= 0.30
    assert evidence.median_reprojection_error is not None
    assert evidence.median_reprojection_error <= 1.0
    assert evidence.mapped_inside_fraction >= 0.99
    assert evidence.orientation_preserved is True
    assert evidence.convex_mapping is True
