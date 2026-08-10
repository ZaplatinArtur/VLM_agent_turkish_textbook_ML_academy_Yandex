from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_maxim_evidence_os_grouped_router_v1 import (
    BRANCH_ORDER,
    DiagnosticError,
    _load_frozen_inputs,
    _require_bool as require_router_bool,
)
from scripts.evaluate_maxim_evidence_os_grouped_v1 import (
    EvaluationError,
    _require_bool as require_evaluator_bool,
    evaluate,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _frozen_fixture(repo: Path) -> tuple[Path, dict[str, object]]:
    public_tasks = repo / "public_tasks.jsonl"
    metadata = repo / "validation.meta.jsonl"
    _write(public_tasks, '{"task_id":"t1","subject":"Math","answer_type":"numeric"}\n')
    _write(metadata, '{"task_id":"t1","source":"https://example.test/book.pdf"}\n')

    anchor: dict[str, object] | None = None
    legacy: dict[str, object] = {}
    scores: dict[str, object] = {}
    for branch in BRANCH_ORDER:
        raw = repo / "raw" / f"{branch}.jsonl"
        projection = repo / "tmp" / "maxim_evidence_os_v1_stage" / f"{branch}.public.jsonl"
        score = repo / "scores" / f"{branch}.json"
        _write(raw, json.dumps({"task_id": "t1", "final_answer": branch}) + "\n")
        _write(projection, json.dumps({"task_id": "t1", "final_answer": branch}) + "\n")
        raw_sha = _sha(raw)
        _write(
            score,
            json.dumps(
                {
                    "provenance": {"solver_results": {"sha256": raw_sha}},
                    "task_outcomes": [{"task_id": "t1", "new_correct": True}],
                }
            ),
        )
        raw_spec: dict[str, object] = {
            "path": str(raw.relative_to(repo)),
            "sha256": raw_sha,
            "public_projection_sha256": _sha(projection),
        }
        if branch == "anchor":
            anchor = raw_spec
        else:
            legacy[branch] = raw_spec
        scores[branch] = {
            "path": str(score.relative_to(repo)),
            "sha256": _sha(score),
        }

    assert anchor is not None
    config: dict[str, object] = {
        "expected_rows": 1,
        "anchor": anchor,
        "public_tasks": {
            "path": str(public_tasks.relative_to(repo)),
            "sha256": _sha(public_tasks),
        },
        "legacy_modules": legacy,
        "evaluation": {
            "source_family_metadata": {
                "path": str(metadata.relative_to(repo)),
                "sha256": _sha(metadata),
            },
            "grouped_router_score_artifacts": scores,
        },
    }
    config_path = repo / "profile.json"
    _write(config_path, json.dumps(config))
    return config_path, config


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_correctness_labels_require_strict_json_booleans(value: object) -> None:
    with pytest.raises(DiagnosticError, match="strict JSON boolean"):
        require_router_bool(value, "label")
    with pytest.raises(EvaluationError, match="strict JSON boolean"):
        require_evaluator_bool(value, "label")


def test_correctness_labels_accept_both_boolean_values() -> None:
    assert require_router_bool(True, "label") is True
    assert require_router_bool(False, "label") is False
    assert require_evaluator_bool(True, "label") is True
    assert require_evaluator_bool(False, "label") is False


def test_frozen_inputs_require_every_preregistered_branch(tmp_path: Path) -> None:
    config_path, config = _frozen_fixture(tmp_path)
    evaluation = config["evaluation"]
    assert isinstance(evaluation, dict)
    scores = evaluation["grouped_router_score_artifacts"]
    assert isinstance(scores, dict)
    scores.pop("parser")
    _write(config_path, json.dumps(config))

    with pytest.raises(DiagnosticError, match="exactly match preregistered BRANCH_ORDER"):
        _load_frozen_inputs(tmp_path.resolve(), config_path.resolve())


def test_frozen_inputs_pin_all_hashes_and_solver_provenance(tmp_path: Path) -> None:
    config_path, _ = _frozen_fixture(tmp_path)
    *_, audit = _load_frozen_inputs(tmp_path.resolve(), config_path.resolve())

    assert set(audit["branches"]) == set(BRANCH_ORDER)
    assert len(audit["config"]["sha256"]) == 64
    assert len(audit["public_tasks"]["sha256"]) == 64
    assert len(audit["validation_metadata"]["sha256"]) == 64
    for branch in BRANCH_ORDER:
        branch_audit = audit["branches"][branch]
        assert branch_audit["score_provenance_solver_sha256"] == branch_audit["raw_solver"]["sha256"]


def test_frozen_inputs_reject_score_solver_provenance_mismatch(tmp_path: Path) -> None:
    config_path, config = _frozen_fixture(tmp_path)
    evaluation = config["evaluation"]
    assert isinstance(evaluation, dict)
    score_specs = evaluation["grouped_router_score_artifacts"]
    assert isinstance(score_specs, dict)
    parser_score_spec = score_specs["parser"]
    assert isinstance(parser_score_spec, dict)
    parser_score = tmp_path / str(parser_score_spec["path"])
    score = json.loads(parser_score.read_text(encoding="utf-8"))
    score["provenance"]["solver_results"]["sha256"] = "0" * 64
    _write(parser_score, json.dumps(score))
    parser_score_spec["sha256"] = _sha(parser_score)
    _write(config_path, json.dumps(config))

    with pytest.raises(DiagnosticError, match="does not match frozen raw solver SHA"):
        _load_frozen_inputs(tmp_path.resolve(), config_path.resolve())


def test_grouped_evaluator_rejects_partial_subset(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    score = tmp_path / "score.json"
    _write(
        metadata,
        '\n'.join(
            [
                '{"task_id":"t1","source":"https://example.test/book.pdf"}',
                '{"task_id":"t2","source":"https://example.test/book.pdf"}',
                "",
            ]
        ),
    )
    _write(
        score,
        json.dumps(
            {
                "task_outcomes": [
                    {"task_id": "t1", "subject": "Math", "new_correct": True}
                ]
            }
        ),
    )

    with pytest.raises(EvaluationError, match="exactly 2 task outcomes"):
        evaluate(
            metadata,
            score,
            folds=2,
            bootstrap_samples=10,
            seed=1,
            expected_rows=2,
        )
