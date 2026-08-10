from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.evidence_os.math12_binding_eval import (
    Math12BindingEvaluationError,
    evaluate_math12_bindings,
)
from src.evidence_os.official_ogm import canonical_json_bytes, canonical_json_sha256


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, selected: int | None = 12):
    run = tmp_path / "clean"
    result = {
        "schema_version": "math12-opaque-source-batch-result-v1",
        "component_scope": "source_resolution_only",
        "input_id": "input-" + "a" * 20,
        "aggregate": {
            "accepted": selected is not None,
            "selected_activity_number": selected,
            "reason": "accepted_activity_agreement" if selected is not None else "abstain_no_accepted_certificate",
        },
    }
    results = run / "results.jsonl"
    _write(results, result)
    artifacts = {"results.jsonl": _sha(results)}
    manifest = {
        "schema_version": "math12-opaque-source-batch-run-v1",
        "component_scope": "source_resolution_only",
        "input_jsonl_sha256": "1" * 64,
        "input_count": 1,
        "artifacts": artifacts,
        "artifacts_projection_sha256": canonical_json_sha256(artifacts),
    }
    _write(run / "run_manifest.json", manifest)
    private_map = tmp_path / "resolver_input_map_math12.jsonl"
    _write(private_map, {"input_id": "input-" + "a" * 20, "task_id": "h80-math12-a012"})
    seal = tmp_path / "math12.seal.json"
    _write(seal, {
        "schema_version": "holdout80-opaque-resolver-input-seal-v1",
        "family_partition": "math12",
        "count": 1,
        "public_inputs_sha256": "1" * 64,
        "private_task_map_sha256": _sha(private_map),
    })
    return run, seal, private_map


def test_perfect_source_binding(tmp_path: Path) -> None:
    run, seal, private_map = _fixture(tmp_path)
    value = evaluate_math12_bindings(run_dir=run, input_seal_path=seal, private_map_path=private_map)
    assert value["total"] == 1
    assert value["correct"] == 1
    assert value["coverage"] == 1.0
    assert value["source_binding_accuracy"] == 1.0
    assert value["conditional_precision"] == 1.0


@pytest.mark.parametrize(
    ("selected", "coverage", "accuracy", "precision"),
    [(13, 1.0, 0.0, 0.0), (None, 0.0, 0.0, None)],
)
def test_wrong_or_abstained_counts_as_incorrect(
    tmp_path: Path, selected: int | None, coverage: float, accuracy: float, precision: float | None
) -> None:
    run, seal, private_map = _fixture(tmp_path, selected=selected)
    value = evaluate_math12_bindings(run_dir=run, input_seal_path=seal, private_map_path=private_map)
    assert value["coverage"] == coverage
    assert value["source_binding_accuracy"] == accuracy
    assert value["conditional_precision"] == precision


def test_tampered_result_is_rejected(tmp_path: Path) -> None:
    run, seal, private_map = _fixture(tmp_path)
    (run / "results.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(Math12BindingEvaluationError, match="artifacts differ"):
        evaluate_math12_bindings(run_dir=run, input_seal_path=seal, private_map_path=private_map)


def test_private_map_fields_are_strict(tmp_path: Path) -> None:
    run, seal, private_map = _fixture(tmp_path)
    _write(private_map, {
        "input_id": "input-" + "a" * 20,
        "task_id": "h80-math12-a012",
        "official_answer": "forbidden",
    })
    seal_value = json.loads(seal.read_text(encoding="utf-8"))
    seal_value["private_task_map_sha256"] = _sha(private_map)
    _write(seal, seal_value)
    with pytest.raises(Math12BindingEvaluationError, match="unexpected fields"):
        evaluate_math12_bindings(run_dir=run, input_seal_path=seal, private_map_path=private_map)


def test_temporary_partial_run_is_rejected(tmp_path: Path) -> None:
    run, seal, private_map = _fixture(tmp_path)
    partial = run.with_name("clean.tmp-broken")
    run.rename(partial)
    with pytest.raises(Math12BindingEvaluationError, match="temporary partial"):
        evaluate_math12_bindings(run_dir=partial, input_seal_path=seal, private_map_path=private_map)
