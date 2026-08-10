from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evidence_os.visual_coordinate_binding import (
    ActivityVisualRecordRef,
    VisualBindingThresholds,
    decide_visual_activity_page_binding,
    visual_page_evidence_from_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "reports" / "maxim_official_exact_source_v2_20260805" / "frozen"
ARTIFACT = FROZEN / "activity_visual_binding_biology_activity4_imageonly_candidate_v1.json"
SOURCE_INDEX = (
    FROZEN
    / "public_workbook_source_index_meb_def_10_biology_activity4_imageonly_candidate_v1.json"
)
TASK_IMAGE = ROOT / "tmp" / "blind_visual_binding" / "task_images" / "val_0178.png"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision_projection(decision) -> dict[str, object]:
    return {
        "accepted": decision.accepted,
        "reason": decision.reason,
        "checks": [[name, passed] for name, passed in decision.checks],
        "selected_page_number": decision.selected_page_number,
        "selected_question_number": decision.selected_question_number,
        "selected_record_id": decision.selected_record_id,
        "best_rank_score": decision.best_rank_score,
        "runner_rank_score": decision.runner_rank_score,
    }


def test_frozen_activity4_artifact_replays_the_pure_page_decision() -> None:
    assert _sha256(ARTIFACT) == (
        "4a14842d3e2ea83555b300fb1f1509dc39edb6ff2bc176752b948969d65b184e"
    )
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["source_only_guards"] == {
        "parser_observation_filter": (
            "single_full_page_image_block_without_text_v1"
        ),
        "source_index_record_filter": (
            "key_binding_kind=activity_answer_key AND "
            "question_marker_kind=activity_label"
        ),
        "record_selection": (
            "unique_reviewed_record_and_unique_pdf_activity_marker_on_visual_page_v1"
        ),
        "render_scope": (
            "all indexed physical content pages expanded from each activity "
            "document content_page_ranges"
        ),
        "task_id_role": "alignment_audit_only",
        "task_id_is_policy_feature": False,
        "source_answer_value_access": False,
        "benchmark_answer_candidate_outcome_artifacts_read": False,
    }
    assert artifact["summary"] == {
        "activity_documents": 1,
        "activity_records": 1,
        "image_only_activity_observations": 1,
        "accepted_page_bindings": 1,
        "raw_page_evidences": 178,
        "decision_layer": "strict_unique_page_activity_applied",
    }
    image_sha, binding = next(
        iter(artifact["bindings_by_task_image_sha256"].items())
    )
    assert _sha256(TASK_IMAGE) == image_sha
    assert binding["alignment_audit"]["task_id_used_as_page_or_record_feature"] is False
    assert binding["image_only_observation"]["block_area_coverage"] > 0.94
    assert binding["source_page_activity_marker_inventory"] == {
        "pdf_sha256": (
            "640bb362f2d53d31663326ac303c5065f4670f2a0d506300beb5e41869384e2b"
        ),
        "physical_page_number": 75,
        "canonical_activity_marker_numbers": [4],
        "projection_sha256": (
            "8050f1dbce11c81fef5dddb537fedca3e9e0c4d2f41c0e7d52f7e02be7f97510"
        ),
    }

    records = tuple(
        ActivityVisualRecordRef(
            document_id=record["document_id"],
            record_id=record["record_id"],
            content_page_number=record["content_page_number"],
            activity_number=record["activity_number"],
            key_projection_sha256=record["key_projection_sha256"],
            content_projection_sha256=record["content_projection_sha256"],
            binding_projection_sha256=record["binding_projection_sha256"],
            visually_checked=record["visually_checked"],
            content_bbox=tuple(record["content_bbox"]),
        )
        for record in artifact["activity_records"]
    )
    evidences = tuple(
        visual_page_evidence_from_mapping(value)
        for value in binding["raw_page_evidences"]
    )
    decision = decide_visual_activity_page_binding(
        evidences,
        records,
        expected_task_image_sha256=image_sha,
        expected_document_id=binding["document_id"],
        expected_pdf_sha256=binding["source_pins"]["pdf_sha256"],
        thresholds=VisualBindingThresholds(),
    )

    assert _decision_projection(decision) == binding["page_binding_decision"]
    assert decision.accepted is True
    assert decision.selected_page_number == 75
    assert decision.selected_question_number == 4
    assert decision.best_rank_score / decision.runner_rank_score > 20.0


def test_activity4_source_index_is_task_id_free_and_exactly_one_record() -> None:
    assert _sha256(SOURCE_INDEX) == (
        "2782d62b6b1610af81800c525ffbbecd68947268e925d9e3fedbeb516d88669d"
    )
    source = json.loads(SOURCE_INDEX.read_text(encoding="utf-8"))
    encoded = json.dumps(source, ensure_ascii=False, sort_keys=True).casefold()
    assert "task_id" not in encoded
    document = source["documents"][0]
    assert document["pdf_sha256"] == (
        "640bb362f2d53d31663326ac303c5065f4670f2a0d506300beb5e41869384e2b"
    )
    assert len(document["questions"]) == 1
    record = document["questions"][0]
    assert record["record_id"] == "meb_def_10biyoloji_640bb362f2d5:p75:q4"
    assert record["key_projection_sha256"] == (
        "b9c3d1198dde8d61083470a4f206f2c7843ee7823f0531fb1672f5f981daadda"
    )
    assert record["content_projection_sha256"] == (
        "c9adf8f44e401f901c0451323b51ed060c5312eff327e5ba62612c1d3d5e329c"
    )
    assert record["binding_projection_sha256"] == (
        "4d01ded97c26ea41d0db157c205d2ce07133d12371cb80e2a727d5b7ca2860bc"
    )
