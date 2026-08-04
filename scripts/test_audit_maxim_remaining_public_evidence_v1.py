from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_maxim_remaining_public_evidence_v1 as audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, dict]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "val_0001.png").write_bytes(b"evidence-one")
    (image_root / "val_0002.png").write_bytes(b"evidence-two")
    (image_root / "val_0003.jpg").write_bytes(b"malformed")

    evidence = {
        "val_0001": {
            "answer": "4/9",
            "image_file": "val_0001.png",
            "image_sha256": _sha(image_root / "val_0001.png"),
            "tier": "B",
            "source": {"kind": "test", "locator": "fixture"},
            "proof": "fixture proof one",
        },
        "val_0002": {
            "answer": "B",
            "image_file": "val_0002.png",
            "image_sha256": _sha(image_root / "val_0002.png"),
            "tier": "C",
            "source": {"kind": "test", "locator": "fixture"},
            "proof": "fixture proof two",
        },
    }
    malformed = {
        "val_0003": {
            "image_file": "val_0003.jpg",
            "image_sha256": _sha(image_root / "val_0003.jpg"),
            "tier": "MALFORMED",
            "source": {"kind": "test", "locator": "fixture"},
            "reason": "no prompt",
        }
    }

    solver_path = tmp_path / "solver.jsonl"
    solver_rows = []
    outcomes = []
    wrong_ids = {"val_0001", "val_0002", "val_0003"}
    for number in range(1, 275):
        task_id = f"val_{number:04d}"
        answer = "A"
        if task_id == "val_0001":
            answer = "0.4444444444444444 (exactly 4/9)"
        elif task_id == "val_0002":
            answer = "B"
        solver_rows.append(
            {
                "task_id": task_id,
                "final_answer": answer,
                "error": None,
                "generation": {"gold_access": False},
            }
        )
        outcomes.append(
            {"task_id": task_id, "new_correct": task_id not in wrong_ids}
        )
    solver_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in solver_rows),
        encoding="utf-8",
    )

    score_path = tmp_path / "score.json"
    score_path.write_text(
        json.dumps(
            {
                "overall": {"n": 274, "new_correct": 271, "new_accuracy": 271 / 274},
                "task_outcomes": outcomes,
                # Must be ignored, never followed or opened.
                "provenance": {"benchmark": {"path": "C:/secret/benchmark.jsonl"}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return solver_path, score_path, image_root, evidence, malformed


def test_default_registry_is_exact_and_disjoint() -> None:
    assert set(audit.EVIDENCE_CERTIFICATES) == {
        "val_0063", "val_0073", "val_0076", "val_0165", "val_0170",
        "val_0186", "val_0208", "val_0243", "val_0251", "val_0257",
    }
    assert set(audit.MALFORMED_CERTIFICATES) == {"val_0100"}
    assert audit.MALFORMED_CERTIFICATES["val_0100"]["image_sha256"] == (
        "b781b114b9485c10cd49ceeca3a4f6ff6302e5f9c72c8aa6a9996c2e4dd5f9bb"
    )


def test_answer_matching_handles_exact_fractions_and_normalized_text() -> None:
    assert audit.answer_matches("0.4444444444444444 (exactly 4/9)", "4/9")
    assert audit.answer_matches("0.5", "1/2")
    assert audit.answer_matches("Naneli: 12 kutu, Limonlu: 8 kutu", "Naneli: 12 kutu, Limonlu: 8 kutu")
    assert audit.answer_matches("B", "B")
    assert not audit.answer_matches("C", "B")


def test_forbidden_input_path_is_rejected_before_open(tmp_path: Path) -> None:
    path = tmp_path / "gold_answers.json"
    assert not path.exists()
    with pytest.raises(ValueError, match="forbidden solver input path"):
        audit.reject_forbidden_input_path(path, role="solver")


def test_forbidden_solver_target_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden target-bearing field"):
        audit._reject_forbidden_solver_fields(
            {"task_id": "val_0001", "reference_answer": "A"}, location="solver[0]"
        )
    with pytest.raises(ValueError, match="must be exactly false"):
        audit._reject_forbidden_solver_fields(
            {"generation": {"gold_access": True}}, location="solver[0]"
        )


def test_end_to_end_injected_fixture_preserves_standard_and_adjusts_diagnostic(tmp_path: Path) -> None:
    solver, score, images, evidence, malformed = _write_fixture(tmp_path)
    report = audit.run_audit(
        solver_path=solver,
        expected_solver_sha256=_sha(solver),
        score_path=score,
        expected_score_sha256=_sha(score),
        public_image_root=images,
        evidence_registry=evidence,
        malformed_registry=malformed,
        expected_rows=274,
        expected_standard_correct=271,
    )
    assert report["standard_metric"] == {
        "unchanged": True,
        "correct": 271,
        "denominator": 274,
        "accuracy_reported_in_frozen_score": 271 / 274,
        "accuracy_exact": 271 / 274,
    }
    result = report["public_evidence_audit"]
    assert result["independent_evidence_confirmed_count"] == 2
    assert result["malformed_missing_prompt_count"] == 1
    assert result["evidence_adjusted_fixed_denominator"] == {
        "correct": 273, "denominator": 274, "accuracy": 273 / 274
    }
    assert result["evidence_adjusted_answerable_only"] == {
        "correct": 273, "denominator": 273, "accuracy": 1.0
    }
    assert all(
        row["public_image"]["verified"]
        and row["public_image"]["expected_sha256"] == row["public_image"]["actual_sha256"]
        for row in report["evidence_certificates"] + report["malformed_certificates"]
    )
    assert report["input_policy"]["score_provenance_paths_followed"] is False


def test_sha_mismatch_and_score_correct_registry_row_fail_closed(tmp_path: Path) -> None:
    solver, score, images, evidence, malformed = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="solver SHA mismatch"):
        audit.run_audit(
            solver_path=solver,
            expected_solver_sha256="0" * 64,
            score_path=score,
            expected_score_sha256=_sha(score),
            public_image_root=images,
            evidence_registry=evidence,
            malformed_registry=malformed,
            expected_rows=274,
            expected_standard_correct=271,
        )

    score_data = json.loads(score.read_text(encoding="utf-8"))
    for outcome in score_data["task_outcomes"]:
        if outcome["task_id"] == "val_0001":
            outcome["new_correct"] = True
    score_data["overall"]["new_correct"] = 272
    score_data["overall"]["new_accuracy"] = 272 / 274
    score.write_text(json.dumps(score_data, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="not standard-score wrong"):
        audit.run_audit(
            solver_path=solver,
            expected_solver_sha256=_sha(solver),
            score_path=score,
            expected_score_sha256=_sha(score),
            public_image_root=images,
            evidence_registry=evidence,
            malformed_registry=malformed,
            expected_rows=274,
            expected_standard_correct=272,
        )


def test_atomic_write_replaces_complete_file(tmp_path: Path) -> None:
    output = tmp_path / "report.md"
    output.write_text("old", encoding="utf-8")
    audit.atomic_write_text(output, "new\n")
    assert output.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.glob(".report.md.*.tmp")) == []
