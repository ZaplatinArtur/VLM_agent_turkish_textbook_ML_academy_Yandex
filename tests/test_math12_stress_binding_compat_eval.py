from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from src.evidence_os import math12_binding_eval as frozen_eval
from src.evidence_os.math12_stress_binding_compat_eval import (
    CompatibilityPins,
    Math12StressCompatibilityError,
    evaluate_math12_stress_bindings_compat,
    write_compatibility_evaluation,
)
from src.evidence_os.official_ogm import (
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
)


INPUT_ID = "input-" + "a" * 20
TASK_ID = "h80-math12-a012"
STRESS_SCHEMA = "holdout80-opaque-resolver-input-stress-v1"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _fixture(tmp_path: Path, *, selected: int | None = 12) -> dict[str, object]:
    holdout = tmp_path / "holdout"
    clean_dir = holdout / "resolver_inputs"
    stress_dir = holdout / "resolver_inputs_stress_v1"
    sealed_dir = holdout / "sealed"
    run_dir = tmp_path / "blind" / "stress"

    clean_asset = clean_dir / "assets" / f"{INPUT_ID}-01.jpg"
    stress_asset = stress_dir / "assets" / f"{INPUT_ID}-01-stress.jpg"
    clean_asset.parent.mkdir(parents=True)
    stress_asset.parent.mkdir(parents=True)
    clean_asset.write_bytes(b"clean-image-payload")
    stress_asset.write_bytes(b"stress-image-payload")
    clean_asset_sha = sha256_file(clean_asset)
    stress_asset_sha = sha256_file(stress_asset)

    clean_input = clean_dir / "math12.jsonl"
    stress_input = stress_dir / "math12_stress_v1.jsonl"
    common = {
        "expected_response_format": "numbered_multi_part_solution",
        "input_id": INPUT_ID,
        "language": "tr",
        "prompt": "opaque prompt",
    }
    _write_jsonl(
        clean_input,
        [{
            **common,
            "images": [{
                "path": f"resolver_inputs/assets/{clean_asset.name}",
                "sha256": clean_asset_sha,
            }],
            "schema_version": "holdout80-opaque-resolver-input-v1",
        }],
    )
    _write_jsonl(
        stress_input,
        [{
            **common,
            "images": [{
                "path": f"resolver_inputs_stress_v1/assets/{stress_asset.name}",
                "sha256": stress_asset_sha,
            }],
            "schema_version": STRESS_SCHEMA,
        }],
    )

    private_map = sealed_dir / "resolver_input_map_math12.jsonl"
    _write_jsonl(private_map, [{"input_id": INPUT_ID, "task_id": TASK_ID}])
    clean_seal = clean_dir / "math12.seal.json"
    _write_json(
        clean_seal,
        {
            "schema_version": "holdout80-opaque-resolver-input-seal-v1",
            "family_partition": "math12",
            "count": 1,
            "public_inputs": "resolver_inputs/math12.jsonl",
            "public_inputs_sha256": sha256_file(clean_input),
            "private_task_map": "sealed/resolver_input_map_math12.jsonl",
            "private_task_map_sha256": sha256_file(private_map),
        },
    )

    builder = stress_dir / "build_stress_v1.py"
    builder.write_bytes(b"# frozen synthetic builder\n")
    builder_sha = sha256_file(builder)
    fixed_seed_sha = "1" * 64
    prereg = stress_dir / "preregistration.json"
    _write_json(
        prereg,
        {
            "schema_version": STRESS_SCHEMA,
            "builder_code_sha256": builder_sha,
            "fixed_seed_sha256": fixed_seed_sha,
            "opaque_input_jsonl_sha256": sha256_file(clean_input),
            "declared_input_count": 1,
        },
    )
    transform = stress_dir / "transform_manifest.json"
    _write_json(
        transform,
        {
            "schema_version": STRESS_SCHEMA,
            "builder_code_sha256": builder_sha,
            "fixed_seed_sha256": fixed_seed_sha,
            "transforms": [{
                "input_id": INPUT_ID,
                "image_index": 1,
                "input_asset_sha256": clean_asset_sha,
                "output_asset_sha256": stress_asset_sha,
                "output_asset_name": stress_asset.name,
                "parameters": {},
            }],
        },
    )
    counts = stress_dir / "counts.json"
    counts_value = {
        "opaque_inputs": 1,
        "images": 1,
        "unique_output_asset_hashes": 1,
        "duplicate_output_asset_hashes": 0,
    }
    _write_json(counts, counts_value)
    freeze = stress_dir / "freeze.json"
    _write_json(
        freeze,
        {
            "schema_version": STRESS_SCHEMA,
            "builder_code_sha256": builder_sha,
            "artifacts": {
                name: sha256_file(stress_dir / name)
                for name in (
                    "counts.json",
                    "math12_stress_v1.jsonl",
                    "preregistration.json",
                    "transform_manifest.json",
                )
            },
            "assets_merkle_sha256": hashlib.sha256(stress_asset_sha.encode("ascii")).hexdigest(),
            "counts": counts_value,
        },
    )

    result = {
        "schema_version": "math12-opaque-source-batch-result-v1",
        "component_scope": "source_resolution_only",
        "input_id": INPUT_ID,
        "aggregate": {
            "accepted": selected is not None,
            "selected_activity_number": selected,
            "reason": (
                "accepted_activity_agreement"
                if selected is not None
                else "abstain_no_accepted_certificate"
            ),
        },
    }
    results_path = run_dir / "results.jsonl"
    _write_jsonl(results_path, [result])
    run_artifacts = {"results.jsonl": sha256_file(results_path)}
    run_manifest = run_dir / "run_manifest.json"
    run_manifest_value = {
        "schema_version": "math12-opaque-source-batch-run-v1",
        "component_scope": "source_resolution_only",
        "input_jsonl_sha256": sha256_file(stress_input),
        "input_count": 1,
        "image_count": 1,
        "accepted_input_count": 1 if selected is not None else 0,
        "certificate_count": 0,
        "solution_record_count": 0,
        "artifacts": run_artifacts,
        "artifacts_projection_sha256": canonical_json_sha256(run_artifacts),
    }
    _write_json(run_manifest, run_manifest_value)

    output_seal = tmp_path / "blind" / "output_seal_before_map.json"
    output_seal_value = {
        "schema_version": "math12-blind-output-seal-v1",
        "status": "sealed_before_private_map_read",
        "scope": "source_activity_binding_only_not_qa_accuracy",
        "private_map_read": False,
        "accuracy_claim": None,
        "runs": {
            "clean": {"not_used_by_compat_fixture": True},
            "stress": {
                "input_jsonl_sha256": sha256_file(stress_input),
                "run_manifest_sha256": sha256_file(run_manifest),
                "artifacts_projection_sha256": run_manifest_value["artifacts_projection_sha256"],
                "input_count": 1,
                "image_count": 1,
                "accepted_input_count": 1 if selected is not None else 0,
                "certificate_count": 0,
                "solution_record_count": 0,
            },
        },
    }
    output_seal_value["seal_projection_sha256"] = canonical_json_sha256(output_seal_value)
    _write_json(output_seal, output_seal_value)

    evaluator_source = Path(frozen_eval.__file__).resolve()
    pins = CompatibilityPins(
        stress_freeze_sha256=sha256_file(freeze),
        stress_builder_sha256=builder_sha,
        stress_preregistration_sha256=sha256_file(prereg),
        clean_input_seal_sha256=sha256_file(clean_seal),
        clean_input_jsonl_sha256=sha256_file(clean_input),
        private_map_sha256=sha256_file(private_map),
        output_seal_sha256=sha256_file(output_seal),
        output_seal_projection_sha256=output_seal_value["seal_projection_sha256"],
        stress_run_manifest_sha256=sha256_file(run_manifest),
        stress_input_jsonl_sha256=sha256_file(stress_input),
        stress_run_artifacts_projection_sha256=run_manifest_value["artifacts_projection_sha256"],
        frozen_evaluator_source_sha256=sha256_file(evaluator_source),
    )
    return {
        "run_dir": run_dir,
        "stress_dir": stress_dir,
        "clean_input_seal_path": clean_seal,
        "clean_input_jsonl_path": clean_input,
        "private_map_path": private_map,
        "output_seal_path": output_seal,
        "pins": pins,
    }


def _evaluate(fixture: dict[str, object]):
    return evaluate_math12_stress_bindings_compat(**fixture)  # type: ignore[arg-type]


def _rewrite_freeze_and_repins(fixture: dict[str, object]) -> None:
    stress_dir = fixture["stress_dir"]
    assert isinstance(stress_dir, Path)
    freeze_path = stress_dir / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["artifacts"] = {
        name: sha256_file(stress_dir / name)
        for name in (
            "counts.json",
            "math12_stress_v1.jsonl",
            "preregistration.json",
            "transform_manifest.json",
        )
    }
    _write_json(freeze_path, freeze)
    pins = fixture["pins"]
    assert isinstance(pins, CompatibilityPins)
    fixture["pins"] = replace(
        pins,
        stress_freeze_sha256=sha256_file(freeze_path),
        stress_preregistration_sha256=sha256_file(stress_dir / "preregistration.json"),
    )


def test_perfect_binding_uses_frozen_metric_engine(tmp_path: Path) -> None:
    value = _evaluate(_fixture(tmp_path))
    assert value["prediction_status"] == "prediction_sealed_before_map"
    assert value["adapter_status"] == "adapter_extended_after_map"
    assert value["preregistration_status"] == "not_preregistered_evaluator"
    assert value["metric_engine"]["formula_changed"] is False
    assert value["metrics"] == {
        "total": 1,
        "accepted": 1,
        "abstained": 0,
        "correct": 1,
        "incorrect": 0,
        "coverage": 1.0,
        "source_binding_accuracy": 1.0,
        "conditional_precision": 1.0,
    }


@pytest.mark.parametrize(
    ("selected", "coverage", "precision"),
    [(13, 1.0, 0.0), (None, 0.0, None)],
)
def test_wrong_or_abstained_is_an_error(
    tmp_path: Path, selected: int | None, coverage: float, precision: float | None
) -> None:
    value = _evaluate(_fixture(tmp_path, selected=selected))
    assert value["metrics"]["correct"] == 0
    assert value["metrics"]["incorrect"] == 1
    assert value["metrics"]["source_binding_accuracy"] == 0.0
    assert value["metrics"]["coverage"] == coverage
    assert value["metrics"]["conditional_precision"] == precision


def test_tampered_run_result_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    run_dir = fixture["run_dir"]
    assert isinstance(run_dir, Path)
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(Math12StressCompatibilityError, match="artifacts differ"):
        _evaluate(fixture)


def test_alternate_stress_freeze_is_rejected_by_exact_pin(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    stress_dir = fixture["stress_dir"]
    assert isinstance(stress_dir, Path)
    freeze_path = stress_dir / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["extra"] = "self-consistent-looking-alternate"
    _write_json(freeze_path, freeze)
    with pytest.raises(Math12StressCompatibilityError, match="stress freeze SHA mismatch"):
        _evaluate(fixture)


def test_preregistration_must_link_to_clean_inputs_even_if_repinned(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    stress_dir = fixture["stress_dir"]
    assert isinstance(stress_dir, Path)
    prereg_path = stress_dir / "preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    prereg["opaque_input_jsonl_sha256"] = "f" * 64
    _write_json(prereg_path, prereg)
    _rewrite_freeze_and_repins(fixture)
    with pytest.raises(Math12StressCompatibilityError, match="does not bind builder and clean inputs"):
        _evaluate(fixture)


def test_transform_asset_chain_must_match_even_if_repinned(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    stress_dir = fixture["stress_dir"]
    assert isinstance(stress_dir, Path)
    transform_path = stress_dir / "transform_manifest.json"
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    transform["transforms"][0]["input_asset_sha256"] = "f" * 64
    _write_json(transform_path, transform)
    _rewrite_freeze_and_repins(fixture)
    with pytest.raises(Math12StressCompatibilityError, match="transform asset linkage mismatch"):
        _evaluate(fixture)


def test_pre_map_output_seal_is_exactly_pinned(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output_seal = fixture["output_seal_path"]
    assert isinstance(output_seal, Path)
    value = json.loads(output_seal.read_text(encoding="utf-8"))
    value["private_map_read"] = True
    _write_json(output_seal, value)
    with pytest.raises(Math12StressCompatibilityError, match="pre-map output seal SHA mismatch"):
        _evaluate(fixture)


def test_evaluation_writer_is_create_only(tmp_path: Path) -> None:
    value = _evaluate(_fixture(tmp_path / "fixture"))
    output = tmp_path / "evaluation.json"
    write_compatibility_evaluation(output, value)
    with pytest.raises(Math12StressCompatibilityError, match="already exists"):
        write_compatibility_evaluation(output, value)
