from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import hashlib
from pathlib import Path

import pytest

from evidence_os.mcq_fullpage_source import (
    EXPECTED_CHOICE_KEY_COUNT,
    EXPECTED_CONTENT_PAGE_COUNT,
    EXPECTED_PDFTOPPM_SHA256,
    EXPECTED_PROTOCOL_RECORD_COUNT,
    FROZEN_VISUAL_THRESHOLDS,
    McqKeyIndex,
    McqRenderManifest,
    McqRenderedPage,
    McqSourceError,
    decide_mcq_page_binding,
    issue_mcq_source_certificate,
    load_mcq_inventory,
    load_mcq_key_index,
    parse_observable_mcq_prompt,
    verify_mcq_source_certificate,
)
from evidence_os.visual_coordinate_binding import (
    VisualBindingThresholds,
    VisualCoordinateBindingError,
    VisualPageEvidence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE_ROOT = (
    REPO_ROOT / "reports" / "maxim_mcq_fullpage_source_v1_20260808"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _prompt(question_number: int) -> str:
    return (
        f"Sayfadaki {question_number}. çoktan seçmeli soruyu çözünüz. "
        "Yalnızca A, B, C, D veya E yazınız."
    )


@pytest.fixture(scope="module")
def source_pair():
    inventory_path = PUBLIC_SOURCE_ROOT / "inventory.json"
    key_index_path = PUBLIC_SOURCE_ROOT / "official_key_index.json"
    assert inventory_path.is_file(), "the public source census must be generated first"
    assert key_index_path.is_file(), "the public official-key index must be generated first"
    inventory = load_mcq_inventory(inventory_path)
    key_index = load_mcq_key_index(key_index_path, inventory)
    return inventory, key_index


@pytest.fixture(scope="module")
def synthetic_source(source_pair):
    inventory, key_index = source_pair
    pages = tuple(
        McqRenderedPage(
            document_id=document_id,
            page_number=page_number,
            relative_path=f"{document_id}/page-{page_number:04d}.png",
            sha256=_sha(f"render:{document_id}:{page_number}"),
            size_bytes=1_000 + index,
            width=1_190,
            height=1_684,
        )
        for index, (document_id, page_number) in enumerate(
            inventory.candidate_pages, start=1
        )
    )
    manifest = McqRenderManifest(
        inventory_projection_sha256=inventory.inventory_projection_sha256,
        render_dpi=144,
        color_mode="poppler_gray_rgb_png",
        poppler_version="26.05.0",
        poppler_executable_sha256=EXPECTED_PDFTOPPM_SHA256,
        pages=pages,
        render_manifest_projection_sha256=_sha("synthetic-render-manifest"),
    )
    return inventory, key_index, manifest


def _evidences(
    inventory,
    manifest,
    selected_address: tuple[str, int],
    *,
    task_sha: str | None = None,
) -> tuple[VisualPageEvidence, ...]:
    task_sha = task_sha or _sha("opaque-task-image")
    result: list[VisualPageEvidence] = []
    for document_id, page_number in inventory.candidate_pages:
        selected = (document_id, page_number) == selected_address
        good_matches = 120 if selected else 20
        inliers = 120 if selected else 10
        result.append(
            VisualPageEvidence(
                task_image_sha256=task_sha,
                document_id=document_id,
                pdf_sha256=inventory.document(document_id).pdf_sha256,
                page_number=page_number,
                rendered_page_sha256=manifest.page(document_id, page_number).sha256,
                good_matches=good_matches,
                inliers=inliers,
                inlier_ratio=inliers / good_matches,
                task_hull_fraction=0.90 if selected else 0.20,
                median_reprojection_error=0.20 if selected else 1.0,
                mapped_inside_fraction=1.0 if selected else 0.50,
                scale_anisotropy=1.0,
                orientation_preserved=True,
                convex_mapping=True,
                mapped_polygon=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
            )
        )
    return tuple(result)


def _supported_record(inventory):
    return next(
        record
        for record in inventory.questions
        if record.source_response_kind == "choice_A-E"
    )


def _check_map(decision) -> dict[str, bool]:
    return dict(decision.checks)


def test_exact_turkish_prompt_grammar_accepts_only_the_frozen_instruction() -> None:
    assert parse_observable_mcq_prompt(_prompt(38)) == 38
    assert (
        parse_observable_mcq_prompt(
            "  Sayfadaki\n38 .  çoktan seçmeli soruyu çözünüz.\t"
            "Yalnızca A, B, C, D veya E yazınız.  "
        )
        == 38
    )


@pytest.mark.parametrize(
    "near_miss",
    [
        "Sayfadaki 038. çoktan seçmeli soruyu çözünüz. Yalnızca A, B, C, D veya E yazınız.",
        "sayfadaki 38. çoktan seçmeli soruyu çözünüz. Yalnızca A, B, C, D veya E yazınız.",
        "Sayfadaki 38. coktan secmeli soruyu cozunuz. Yalnizca A, B, C, D veya E yaziniz.",
        "Sayfadaki 38. çoktan seçmeli soruyu çözünüz; Yalnızca A, B, C, D veya E yazınız.",
        "Sayfadaki 38. çoktan seçmeli soruyu çözünüz. Yalnızca A, B, C veya D yazınız.",
        "Sayfadaki 38. çoktan seçmeli soruyu çözünüz. Yalnızca A, B, C, D veya E yazınız. Açıklayınız.",
    ],
)
def test_turkish_prompt_near_misses_fail_closed(near_miss: str) -> None:
    with pytest.raises(McqSourceError, match="frozen Turkish MCQ grammar"):
        parse_observable_mcq_prompt(near_miss)


def test_public_source_census_is_147_records_143_keys_and_four_open_responses(
    source_pair,
) -> None:
    inventory, key_index = source_pair
    assert len(inventory.questions) == EXPECTED_PROTOCOL_RECORD_COUNT == 147
    assert len(key_index.cells) == EXPECTED_CHOICE_KEY_COUNT == 143
    assert len(inventory.candidate_pages) == EXPECTED_CONTENT_PAGE_COUNT == 28
    assert Counter(record.source_response_kind for record in inventory.questions) == {
        "choice_A-E": 143,
        "unsupported_open_response": 4,
    }

    open_records = [
        record
        for record in inventory.questions
        if record.source_response_kind == "unsupported_open_response"
    ]
    assert {(record.unit_number, record.question_number) for record in open_records} == {
        (2, 24),
        (2, 25),
        (2, 26),
        (2, 27),
    }
    assert {record.content_page_number for record in open_records} == {100}
    assert {record.source_family for record in open_records} == {
        "physics12_textbook"
    }
    key_record_ids = {cell.record_id for cell in key_index.cells}
    assert key_record_ids == {
        record.record_id
        for record in inventory.questions
        if record.source_response_kind == "choice_A-E"
    }
    assert key_record_ids.isdisjoint(record.record_id for record in open_records)
    for record in open_records:
        with pytest.raises(McqSourceError, match="absent or ambiguous"):
            key_index.cell(record.record_id)


def test_complete_28_page_sweep_accepts_a_uniquely_bound_choice(
    synthetic_source,
) -> None:
    inventory, key_index, manifest = synthetic_source
    record = _supported_record(inventory)
    evidences = _evidences(
        inventory,
        manifest,
        (record.document_id, record.content_page_number),
    )

    decision = decide_mcq_page_binding(
        evidences,
        inventory,
        manifest,
        key_index,
        expected_task_image_sha256=_sha("opaque-task-image"),
        observed_question_number=record.question_number,
    )

    assert decision.accepted is True
    assert decision.reason == "accepted"
    assert decision.selected_record_id == record.record_id
    assert all(_check_map(decision).values())


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign_address"])
def test_incomplete_duplicate_or_foreign_page_sweep_abstains(
    synthetic_source,
    mutation: str,
) -> None:
    inventory, key_index, manifest = synthetic_source
    record = _supported_record(inventory)
    evidences = list(
        _evidences(
            inventory,
            manifest,
            (record.document_id, record.content_page_number),
        )
    )
    if mutation == "missing":
        evidences.pop()
    elif mutation == "duplicate":
        evidences[-1] = evidences[0]
    else:
        evidences[-1] = replace(evidences[-1], page_number=999)

    decision = decide_mcq_page_binding(
        evidences,
        inventory,
        manifest,
        key_index,
        expected_task_image_sha256=_sha("opaque-task-image"),
        observed_question_number=record.question_number,
    )

    assert decision.accepted is False
    assert decision.reason == "incomplete_or_foreign_page_evidence"
    assert _check_map(decision)["complete_candidate_page_sweep"] is False


@pytest.mark.parametrize("foreign_pin", ["task", "pdf", "render"])
def test_foreign_identity_pin_in_a_complete_sweep_abstains(
    synthetic_source,
    foreign_pin: str,
) -> None:
    inventory, key_index, manifest = synthetic_source
    record = _supported_record(inventory)
    evidences = list(
        _evidences(
            inventory,
            manifest,
            (record.document_id, record.content_page_number),
        )
    )
    index = next(
        i
        for i, item in enumerate(evidences)
        if (item.document_id, item.page_number)
        == (record.document_id, record.content_page_number)
    )
    field = {
        "task": "task_image_sha256",
        "pdf": "pdf_sha256",
        "render": "rendered_page_sha256",
    }[foreign_pin]
    evidences[index] = replace(evidences[index], **{field: _sha("foreign-pin")})

    decision = decide_mcq_page_binding(
        evidences,
        inventory,
        manifest,
        key_index,
        expected_task_image_sha256=_sha("opaque-task-image"),
        observed_question_number=record.question_number,
    )

    assert decision.accepted is False
    assert _check_map(decision)["complete_candidate_page_sweep"] is True
    assert _check_map(decision)["source_identity"] is False


def test_equal_top_page_scores_fail_the_frozen_margin_and_ratio(
    synthetic_source,
) -> None:
    inventory, key_index, manifest = synthetic_source
    record = _supported_record(inventory)
    evidences = list(
        _evidences(
            inventory,
            manifest,
            (record.document_id, record.content_page_number),
        )
    )
    best = next(
        item
        for item in evidences
        if (item.document_id, item.page_number)
        == (record.document_id, record.content_page_number)
    )
    runner_index = next(
        index
        for index, item in enumerate(evidences)
        if (item.document_id, item.page_number)
        != (record.document_id, record.content_page_number)
    )
    runner = evidences[runner_index]
    evidences[runner_index] = replace(
        runner,
        good_matches=best.good_matches,
        inliers=best.inliers,
        inlier_ratio=best.inlier_ratio,
        task_hull_fraction=best.task_hull_fraction,
        median_reprojection_error=best.median_reprojection_error,
        mapped_inside_fraction=best.mapped_inside_fraction,
        scale_anisotropy=best.scale_anisotropy,
    )

    decision = decide_mcq_page_binding(
        evidences,
        inventory,
        manifest,
        key_index,
        expected_task_image_sha256=_sha("opaque-task-image"),
        observed_question_number=record.question_number,
    )

    assert decision.accepted is False
    assert _check_map(decision)["page_rank_margin"] is False
    assert _check_map(decision)["page_rank_ratio"] is False


@pytest.mark.parametrize(
    ("failed_check", "updates"),
    [
        ("good_matches", {"good_matches": 49, "inliers": 49, "inlier_ratio": 1.0}),
        ("inliers", {"good_matches": 60, "inliers": 39, "inlier_ratio": 0.65}),
        ("inlier_ratio", {"good_matches": 100, "inliers": 64, "inlier_ratio": 0.64}),
        ("task_hull_fraction", {"task_hull_fraction": 0.29}),
        ("median_reprojection_error", {"median_reprojection_error": 1.01}),
        ("mapped_inside_fraction", {"mapped_inside_fraction": 0.97}),
        ("scale_anisotropy", {"scale_anisotropy": 1.16}),
        ("orientation_preserved", {"orientation_preserved": False}),
        ("convex_mapping", {"convex_mapping": False}),
        ("mapped_polygon", {"mapped_polygon": None}),
    ],
)
def test_each_frozen_visual_threshold_fails_closed(
    synthetic_source,
    failed_check: str,
    updates: dict[str, object],
) -> None:
    inventory, key_index, manifest = synthetic_source
    record = _supported_record(inventory)
    evidences = list(
        _evidences(
            inventory,
            manifest,
            (record.document_id, record.content_page_number),
        )
    )
    index = next(
        index
        for index, item in enumerate(evidences)
        if (item.document_id, item.page_number)
        == (record.document_id, record.content_page_number)
    )
    evidences[index] = replace(evidences[index], **updates)

    decision = decide_mcq_page_binding(
        evidences,
        inventory,
        manifest,
        key_index,
        expected_task_image_sha256=_sha("opaque-task-image"),
        observed_question_number=record.question_number,
    )

    assert decision.accepted is False
    assert _check_map(decision)[failed_check] is False


def test_callers_cannot_override_even_with_a_stricter_threshold(
    synthetic_source,
) -> None:
    inventory, key_index, manifest = synthetic_source
    record = _supported_record(inventory)
    evidences = _evidences(
        inventory,
        manifest,
        (record.document_id, record.content_page_number),
    )
    stricter = VisualBindingThresholds(
        **{
            **asdict(FROZEN_VISUAL_THRESHOLDS),
            "min_good_matches": FROZEN_VISUAL_THRESHOLDS.min_good_matches + 1,
        }
    )

    with pytest.raises(VisualCoordinateBindingError, match="exact frozen profile"):
        decide_mcq_page_binding(
            evidences,
            inventory,
            manifest,
            key_index,
            expected_task_image_sha256=_sha("opaque-task-image"),
            observed_question_number=record.question_number,
            thresholds=stricter,
        )


@pytest.mark.parametrize("question_number", [24, 25, 26, 27])
def test_physics_u2_open_response_records_fail_closed_even_with_a_forged_key(
    synthetic_source,
    question_number: int,
) -> None:
    inventory, key_index, manifest = synthetic_source
    record = next(
        item
        for item in inventory.questions
        if item.source_family == "physics12_textbook"
        and item.unit_number == 2
        and item.question_number == question_number
        and item.source_response_kind == "unsupported_open_response"
    )
    template = key_index.cells[0]
    forged_key_text = f"{record.question_number} {template.answer}"
    forged_cell = replace(
        template,
        record_id=record.record_id,
        document_id=record.document_id,
        unit_number=record.unit_number,
        question_number=record.question_number,
        key_text=forged_key_text,
        key_text_sha256=hashlib.sha256(
            forged_key_text.encode("utf-8")
        ).hexdigest(),
    )
    forged_index = McqKeyIndex(
        inventory_projection_sha256=key_index.inventory_projection_sha256,
        cells=tuple(
            sorted(
                (forged_cell, *key_index.cells[1:]),
                key=lambda cell: cell.record_id,
            )
        ),
        key_index_projection_sha256=_sha(f"forged-key-index:{question_number}"),
    )
    evidences = _evidences(
        inventory,
        manifest,
        (record.document_id, record.content_page_number),
    )

    decision = decide_mcq_page_binding(
        evidences,
        inventory,
        manifest,
        forged_index,
        expected_task_image_sha256=_sha("opaque-task-image"),
        observed_question_number=question_number,
    )

    assert decision.accepted is False
    assert _check_map(decision)["unique_prompt_number_on_selected_page"] is True
    assert _check_map(decision)["source_key_cell_bound"] is False


def test_answer_bound_certificate_replays_exactly_and_rejects_tampering(
    synthetic_source,
) -> None:
    inventory, key_index, manifest = synthetic_source
    record = _supported_record(inventory)
    task_sha = _sha("opaque-task-image")
    prompt = _prompt(record.question_number)
    evidences = _evidences(
        inventory,
        manifest,
        (record.document_id, record.content_page_number),
        task_sha=task_sha,
    )
    certificate = issue_mcq_source_certificate(
        prompt,
        task_sha,
        evidences,
        inventory,
        manifest,
        key_index,
    )

    replayed = verify_mcq_source_certificate(
        prompt, inventory, manifest, key_index, certificate
    )
    assert replayed.accepted is True
    assert certificate.answer == key_index.cell(record.record_id).answer

    other_answer = next(choice for choice in "ABCDE" if choice != certificate.answer)
    with pytest.raises(McqSourceError, match="answer hash mismatch"):
        replace(certificate, answer_sha256=_sha("forged-answer"))

    other_record = next(
        item
        for item in inventory.questions
        if item.source_response_kind == "choice_A-E"
        and item.record_id != record.record_id
    )
    tampered = [
        replace(
            certificate,
            answer=other_answer,
            answer_sha256=hashlib.sha256(other_answer.encode("utf-8")).hexdigest(),
        ),
        replace(certificate, selected_key_projection_sha256=_sha("forged-key")),
        replace(
            certificate,
            decision=replace(
                certificate.decision,
                selected_record_id=other_record.record_id,
            ),
        ),
        replace(certificate, certificate_projection_sha256=_sha("forged-certificate")),
        replace(
            certificate,
            evidences=(
                replace(certificate.evidences[0], task_image_sha256=_sha("foreign-task")),
                *certificate.evidences[1:],
            ),
        ),
    ]
    for forged in tampered:
        with pytest.raises(McqSourceError):
            verify_mcq_source_certificate(
                prompt, inventory, manifest, key_index, forged
            )

    with pytest.raises(McqSourceError, match="input/source pins changed"):
        verify_mcq_source_certificate(
            _prompt(record.question_number + 1),
            inventory,
            manifest,
            key_index,
            certificate,
        )
