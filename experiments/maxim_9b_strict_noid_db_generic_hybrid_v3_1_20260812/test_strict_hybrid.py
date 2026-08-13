from __future__ import annotations

import json
from pathlib import Path

import pytest

import strict_hybrid as h


def test_selector_projection_exact_and_no_identity() -> None:
    row = {"ocr_text": "visible", "answer_type": "choice", "input_mode": "text_only", "controller_id": "val_1"}
    assert h.selector_observable(row, "maxim274") == {
        "ocr_text": "visible", "answer_type": "choice", "input_mode": "text_only"
    }


def test_ykslop_projection_omits_subject_theory_and_hash() -> None:
    row = {
        "question": "Question", "choices": {label: label for label in "ABCDE"},
        "subject": "secret-route", "theory": [{"text": "retrieval"}], "content_sha256": "alignment",
    }
    observable = h.selector_observable(row, "ykslop_dev185")
    assert set(observable) == {"ocr_text", "answer_type", "input_mode"}
    assert "secret-route" not in observable["ocr_text"] and "retrieval" not in observable["ocr_text"]


def test_noid_router_rejects_identity_field() -> None:
    source_db = json.loads(h.NOID_SOURCE_DB.read_text(encoding="utf-8"))
    module = h.load_noid_module()
    with pytest.raises(Exception):
        module.route_observable({"ocr_text": "x", "answer_type": "choice", "input_mode": "text_only", "task_id": "x"}, source_db)


def test_pinned_noid_source_database_has_no_task_identity() -> None:
    source_db = json.loads(h.NOID_SOURCE_DB.read_text(encoding="utf-8"))
    assert source_db["contains_task_identity"] is False
    forbidden = {"task_id", "controller_id", "benchmark_id", "input_filename", "content_sha256"}
    assert all(not (set(record) & forbidden) for record in source_db["records"])


def test_generic_candidate_requires_exact_audit_pin(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"model_id": "qwen/qwen3.5-9b"}), encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "PASS", "freeze_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(h.HybridError, match="does not pin"):
        h.verify_generic_candidate(candidate, h.sha256(candidate), audit, h.sha256(audit))


def test_base249_cannot_pass_base240_control_pin() -> None:
    freeze = json.loads((h.NOID_FREEZE).read_text(encoding="utf-8"))
    strict = freeze["frozen_artifacts"]["arm_b"]["solver"]["sha256"]
    base240 = h.sha256(h.BASE240)
    assert strict != base240
