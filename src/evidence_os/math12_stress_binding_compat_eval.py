"""Post-seal compatibility evaluation for the frozen Math12 stress run.

This module does not claim that its adapter was preregistered.  It exists only
because the preregistered stress input was sealed and executed before the
private map was read, while the original frozen evaluator accepted only the
clean input seal.  The adapter verifies the complete clean -> stress -> run ->
output-seal chain and delegates scoring to the byte-pinned frozen evaluator.

The measured quantity is source activity binding.  It is not QA accuracy and
does not measure mathematical reasoning or the correctness of a solution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from . import math12_binding_eval as frozen_eval
from .official_ogm import canonical_json_bytes, canonical_json_sha256, sha256_file


STRESS_SCHEMA = "holdout80-opaque-resolver-input-stress-v1"
EVALUATION_SCHEMA = "math12-stress-source-binding-compat-evaluation-v1"
BRIDGE_SCHEMA = "math12-stress-source-binding-post-seal-bridge-v1"
OUTPUT_SEAL_SCHEMA = "math12-blind-output-seal-v1"
PREDICTION_STATUS = "prediction_sealed_before_map"
ADAPTER_STATUS = "adapter_extended_after_map"
PREREGISTRATION_STATUS = "not_preregistered_evaluator"
SCOPE = "source_activity_binding_only_not_qa_accuracy"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_INPUT_ID = re.compile(r"^input-[0-9a-f]{20}$")
_OPAQUE_ROW_KEYS = {
    "expected_response_format",
    "images",
    "input_id",
    "language",
    "prompt",
    "schema_version",
}
_IMAGE_KEYS = {"path", "sha256"}
_STRESS_ARTIFACTS = {
    "counts.json",
    "math12_stress_v1.jsonl",
    "preregistration.json",
    "transform_manifest.json",
}


class Math12StressCompatibilityError(ValueError):
    """The post-seal chain is incomplete, changed, or internally inconsistent."""


@dataclass(frozen=True)
class CompatibilityPins:
    """Exact content pins for one immutable compatibility evaluation."""

    stress_freeze_sha256: str
    stress_builder_sha256: str
    stress_preregistration_sha256: str
    clean_input_seal_sha256: str
    clean_input_jsonl_sha256: str
    private_map_sha256: str
    output_seal_sha256: str
    output_seal_projection_sha256: str
    stress_run_manifest_sha256: str
    stress_input_jsonl_sha256: str
    stress_run_artifacts_projection_sha256: str
    frozen_evaluator_source_sha256: str

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if _HEX64.fullmatch(value) is None:
                raise Math12StressCompatibilityError(f"malformed exact pin: {name}")

    @property
    def projection_sha256(self) -> str:
        return canonical_json_sha256(asdict(self))


DEFAULT_PINS = CompatibilityPins(
    stress_freeze_sha256="6a2568b05c479eb47ae44812478d7348552052d2ebbb77040437b068633e927c",
    stress_builder_sha256="f94d64117b1c036cfc29aa886a48467d600dcf6e12b863530bd1430cf7a3b714",
    stress_preregistration_sha256="08a350e9f6d48b8e7f7c475f224385f0df52c898ffdf5ea11bd6e7f850e2bd27",
    clean_input_seal_sha256="7ae72fd90de09bc863a868bc03ec31bda2021795808b27771a0dc077406ede93",
    clean_input_jsonl_sha256="e0ee22d58187fbe11c951ef8153ad825734f83d50e127a1746f4e38649f11960",
    private_map_sha256="9ab07ca41f3a753d3b8625f05b4e71472bd588b88c18b71e2a40c3ffa6c0d964",
    output_seal_sha256="b7419a76dbffbd1e45daffb6f4476bf13cb4e8c4099dcd1c4be74fcb1ccc60bc",
    output_seal_projection_sha256="93fe423ee744db667002b93ffb8d652abc855f471116f86a9fa40b51d2955f59",
    stress_run_manifest_sha256="f74b4901797e412556c45496660f8829826529e750442eff7d7581d45e18a128",
    stress_input_jsonl_sha256="f4ed135db6c046485efad8c5fc0f67a8195b0d5735e3ddc469705d64d76acf5b",
    stress_run_artifacts_projection_sha256="c81634adc8c87426296ab8c4f9e6e09317ce6bab1920b119437da84b3badded9",
    frozen_evaluator_source_sha256="e4ff8329ab9ebce90e6fed180c395091048ce92ec79455bbfa53a849fa16c8d1",
)


def _require_sha(path: Path, expected: str, label: str) -> None:
    try:
        observed = sha256_file(path)
    except OSError as exc:
        raise Math12StressCompatibilityError(f"cannot read {label}") from exc
    if observed != expected:
        raise Math12StressCompatibilityError(
            f"{label} SHA mismatch: expected {expected}, observed {observed}"
        )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = frozen_eval._load_json(path, label=label)
    except frozen_eval.Math12BindingEvaluationError as exc:
        raise Math12StressCompatibilityError(str(exc)) from exc
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        return frozen_eval._load_jsonl(path, label=label)
    except frozen_eval.Math12BindingEvaluationError as exc:
        raise Math12StressCompatibilityError(str(exc)) from exc


def _safe_payload(root: Path, relative: str, label: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(part in {"", ".", ".."} for part in rel.parts):
        raise Math12StressCompatibilityError(f"unsafe {label} path")
    root = root.resolve(strict=True)
    try:
        path = (root / rel).resolve(strict=True)
    except OSError as exc:
        raise Math12StressCompatibilityError(f"missing {label} payload") from exc
    if not path.is_file() or not path.is_relative_to(root):
        raise Math12StressCompatibilityError(f"{label} payload escapes its root")
    return path


def _validate_opaque_rows(
    *,
    clean_input_jsonl_path: Path,
    stress_dir: Path,
    clean_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    transform_manifest: Mapping[str, Any],
) -> tuple[int, list[str]]:
    if not clean_rows or len(clean_rows) != len(stress_rows):
        raise Math12StressCompatibilityError("clean/stress opaque input count mismatch")
    transforms = transform_manifest.get("transforms")
    if not isinstance(transforms, list):
        raise Math12StressCompatibilityError("transform manifest has no transform rows")

    transform_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for item in transforms:
        if not isinstance(item, dict):
            raise Math12StressCompatibilityError("malformed transform row")
        input_id = str(item.get("input_id") or "")
        image_index = item.get("image_index")
        key = (input_id, image_index) if isinstance(image_index, int) else (input_id, -1)
        if _INPUT_ID.fullmatch(input_id) is None or image_index is None or image_index < 1:
            raise Math12StressCompatibilityError("malformed transform identifier")
        if key in transform_by_key:
            raise Math12StressCompatibilityError("duplicate transform identifier")
        transform_by_key[key] = item

    clean_root = clean_input_jsonl_path.parent.parent
    stress_root = stress_dir.parent
    observed_transform_keys: set[tuple[str, int]] = set()
    stress_hashes: list[str] = []
    input_ids: list[str] = []
    for clean, stress in zip(clean_rows, stress_rows, strict=True):
        if set(clean) != _OPAQUE_ROW_KEYS or set(stress) != _OPAQUE_ROW_KEYS:
            raise Math12StressCompatibilityError("opaque row fields differ from the frozen schema")
        input_id = str(clean.get("input_id") or "")
        if _INPUT_ID.fullmatch(input_id) is None or stress.get("input_id") != input_id:
            raise Math12StressCompatibilityError("clean/stress input ID mismatch")
        if input_id in input_ids:
            raise Math12StressCompatibilityError("duplicate opaque input ID")
        input_ids.append(input_id)
        for key in _OPAQUE_ROW_KEYS - {"images", "schema_version"}:
            if stress.get(key) != clean.get(key):
                raise Math12StressCompatibilityError(f"stress changed opaque field: {key}")
        if stress.get("schema_version") != STRESS_SCHEMA:
            raise Math12StressCompatibilityError("unexpected stress row schema")
        clean_images = clean.get("images")
        stress_images = stress.get("images")
        if (
            not isinstance(clean_images, list)
            or not isinstance(stress_images, list)
            or not clean_images
            or len(clean_images) != len(stress_images)
        ):
            raise Math12StressCompatibilityError("clean/stress image count mismatch")
        for image_index, (clean_image, stress_image) in enumerate(
            zip(clean_images, stress_images, strict=True), start=1
        ):
            if (
                not isinstance(clean_image, dict)
                or not isinstance(stress_image, dict)
                or set(clean_image) != _IMAGE_KEYS
                or set(stress_image) != _IMAGE_KEYS
            ):
                raise Math12StressCompatibilityError("malformed opaque image reference")
            clean_sha = str(clean_image["sha256"])
            stress_sha = str(stress_image["sha256"])
            if _HEX64.fullmatch(clean_sha) is None or _HEX64.fullmatch(stress_sha) is None:
                raise Math12StressCompatibilityError("malformed image SHA")
            clean_path = _safe_payload(clean_root, str(clean_image["path"]), "clean image")
            stress_path = _safe_payload(stress_root, str(stress_image["path"]), "stress image")
            _require_sha(clean_path, clean_sha, "clean image")
            _require_sha(stress_path, stress_sha, "stress image")

            key = (input_id, image_index)
            transform = transform_by_key.get(key)
            if transform is None:
                raise Math12StressCompatibilityError("missing transform linkage")
            if (
                transform.get("input_asset_sha256") != clean_sha
                or transform.get("output_asset_sha256") != stress_sha
                or transform.get("output_asset_name") != stress_path.name
            ):
                raise Math12StressCompatibilityError("transform asset linkage mismatch")
            observed_transform_keys.add(key)
            stress_hashes.append(stress_sha)

    if observed_transform_keys != set(transform_by_key):
        raise Math12StressCompatibilityError("transform manifest has orphan rows")
    if len(stress_hashes) != len(set(stress_hashes)):
        raise Math12StressCompatibilityError("stress image SHA values are not unique")
    return len(stress_hashes), input_ids


def _attest_chain(
    *,
    run_dir: Path,
    stress_dir: Path,
    clean_input_seal_path: Path,
    clean_input_jsonl_path: Path,
    private_map_path: Path,
    output_seal_path: Path,
    pins: CompatibilityPins,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int], dict[str, Any]]:
    pins.validate()
    run_dir = run_dir.resolve(strict=True)
    stress_dir = stress_dir.resolve(strict=True)
    clean_input_seal_path = clean_input_seal_path.resolve(strict=True)
    clean_input_jsonl_path = clean_input_jsonl_path.resolve(strict=True)
    private_map_path = private_map_path.resolve(strict=True)
    output_seal_path = output_seal_path.resolve(strict=True)

    evaluator_path = Path(frozen_eval.__file__).resolve(strict=True)
    _require_sha(evaluator_path, pins.frozen_evaluator_source_sha256, "frozen evaluator source")
    _require_sha(clean_input_seal_path, pins.clean_input_seal_sha256, "clean input seal")
    _require_sha(clean_input_jsonl_path, pins.clean_input_jsonl_sha256, "clean opaque JSONL")
    _require_sha(private_map_path, pins.private_map_sha256, "private source-address map")
    _require_sha(stress_dir / "freeze.json", pins.stress_freeze_sha256, "stress freeze")
    _require_sha(stress_dir / "build_stress_v1.py", pins.stress_builder_sha256, "stress builder")
    _require_sha(
        stress_dir / "preregistration.json",
        pins.stress_preregistration_sha256,
        "stress preregistration",
    )
    _require_sha(run_dir / "run_manifest.json", pins.stress_run_manifest_sha256, "stress run manifest")
    _require_sha(output_seal_path, pins.output_seal_sha256, "pre-map output seal")

    try:
        run_manifest, _ = frozen_eval._validate_run(run_dir)
        clean_seal, expected = frozen_eval._load_expected_map(
            clean_input_seal_path, private_map_path
        )
    except frozen_eval.Math12BindingEvaluationError as exc:
        raise Math12StressCompatibilityError(str(exc)) from exc
    if clean_seal.get("public_inputs_sha256") != pins.clean_input_jsonl_sha256:
        raise Math12StressCompatibilityError("clean seal does not bind the clean opaque JSONL")
    if clean_seal.get("private_task_map_sha256") != pins.private_map_sha256:
        raise Math12StressCompatibilityError("clean seal does not bind the private map")

    freeze = _load_json(stress_dir / "freeze.json", "stress freeze")
    if freeze.get("schema_version") != STRESS_SCHEMA:
        raise Math12StressCompatibilityError("unexpected stress freeze schema")
    if freeze.get("builder_code_sha256") != pins.stress_builder_sha256:
        raise Math12StressCompatibilityError("stress freeze builder pin mismatch")
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _STRESS_ARTIFACTS:
        raise Math12StressCompatibilityError("unexpected stress freeze artifacts")
    for name, expected_sha in artifacts.items():
        if _HEX64.fullmatch(str(expected_sha)) is None:
            raise Math12StressCompatibilityError("malformed stress artifact pin")
        _require_sha(stress_dir / name, str(expected_sha), f"stress artifact {name}")

    preregistration = _load_json(stress_dir / "preregistration.json", "stress preregistration")
    if (
        preregistration.get("schema_version") != STRESS_SCHEMA
        or preregistration.get("builder_code_sha256") != pins.stress_builder_sha256
        or preregistration.get("opaque_input_jsonl_sha256") != pins.clean_input_jsonl_sha256
        or preregistration.get("declared_input_count") != clean_seal.get("count")
    ):
        raise Math12StressCompatibilityError("preregistration does not bind builder and clean inputs")

    stress_input_path = stress_dir / "math12_stress_v1.jsonl"
    _require_sha(stress_input_path, pins.stress_input_jsonl_sha256, "stress opaque JSONL")
    if artifacts.get("math12_stress_v1.jsonl") != pins.stress_input_jsonl_sha256:
        raise Math12StressCompatibilityError("stress freeze does not bind the stress opaque JSONL")
    transform_manifest = _load_json(stress_dir / "transform_manifest.json", "transform manifest")
    if (
        transform_manifest.get("schema_version") != STRESS_SCHEMA
        or transform_manifest.get("builder_code_sha256") != pins.stress_builder_sha256
        or transform_manifest.get("fixed_seed_sha256") != preregistration.get("fixed_seed_sha256")
    ):
        raise Math12StressCompatibilityError("transform manifest is not bound to preregistration")

    clean_rows = _load_jsonl(clean_input_jsonl_path, "clean opaque JSONL")
    stress_rows = _load_jsonl(stress_input_path, "stress opaque JSONL")
    image_count, input_ids = _validate_opaque_rows(
        clean_input_jsonl_path=clean_input_jsonl_path,
        stress_dir=stress_dir,
        clean_rows=clean_rows,
        stress_rows=stress_rows,
        transform_manifest=transform_manifest,
    )
    if set(input_ids) != set(expected):
        raise Math12StressCompatibilityError("opaque inputs and sealed private map differ")
    output_hashes = [
        str(image["sha256"])
        for row in stress_rows
        for image in row["images"]
    ]
    assets_merkle = hashlib.sha256("\n".join(sorted(output_hashes)).encode("ascii")).hexdigest()
    if freeze.get("assets_merkle_sha256") != assets_merkle:
        raise Math12StressCompatibilityError("stress asset Merkle projection mismatch")
    counts = {
        "opaque_inputs": len(stress_rows),
        "images": image_count,
        "unique_output_asset_hashes": len(set(output_hashes)),
        "duplicate_output_asset_hashes": len(output_hashes) - len(set(output_hashes)),
    }
    if freeze.get("counts") != counts or _load_json(stress_dir / "counts.json", "stress counts") != counts:
        raise Math12StressCompatibilityError("stress counts mismatch")

    if (
        run_manifest.get("input_jsonl_sha256") != pins.stress_input_jsonl_sha256
        or run_manifest.get("artifacts_projection_sha256")
        != pins.stress_run_artifacts_projection_sha256
        or run_manifest.get("input_count") != len(stress_rows)
        or run_manifest.get("image_count") != image_count
    ):
        raise Math12StressCompatibilityError("stress run is not bound to frozen stress inputs")

    output_seal = _load_json(output_seal_path, "pre-map output seal")
    seal_projection = dict(output_seal)
    declared_projection = seal_projection.pop("seal_projection_sha256", None)
    if (
        output_seal.get("schema_version") != OUTPUT_SEAL_SCHEMA
        or output_seal.get("status") != "sealed_before_private_map_read"
        or output_seal.get("private_map_read") is not False
        or output_seal.get("accuracy_claim") is not None
        or output_seal.get("scope") != SCOPE
        or declared_projection != pins.output_seal_projection_sha256
        or canonical_json_sha256(seal_projection) != declared_projection
    ):
        raise Math12StressCompatibilityError("pre-map output seal metadata/projection mismatch")
    sealed_stress = (output_seal.get("runs") or {}).get("stress")
    run_artifact_names = set((run_manifest.get("artifacts") or {}).keys())
    certificate_count = sum(name.startswith("certificates/") for name in run_artifact_names)
    solution_record_count = sum(name.startswith("solution_records/") for name in run_artifact_names)
    expected_stress_seal = {
        "input_jsonl_sha256": pins.stress_input_jsonl_sha256,
        "run_manifest_sha256": pins.stress_run_manifest_sha256,
        "artifacts_projection_sha256": pins.stress_run_artifacts_projection_sha256,
        "input_count": run_manifest.get("input_count"),
        "image_count": run_manifest.get("image_count"),
        "accepted_input_count": run_manifest.get("accepted_input_count"),
        "certificate_count": certificate_count,
        "solution_record_count": solution_record_count,
    }
    if sealed_stress != expected_stress_seal:
        raise Math12StressCompatibilityError("pre-map output seal does not exactly bind stress run")

    chain = {
        "clean_input_seal_sha256": pins.clean_input_seal_sha256,
        "clean_input_jsonl_sha256": pins.clean_input_jsonl_sha256,
        "private_map_sha256": pins.private_map_sha256,
        "stress_builder_sha256": pins.stress_builder_sha256,
        "stress_preregistration_sha256": pins.stress_preregistration_sha256,
        "stress_freeze_sha256": pins.stress_freeze_sha256,
        "stress_input_jsonl_sha256": pins.stress_input_jsonl_sha256,
        "stress_run_manifest_sha256": pins.stress_run_manifest_sha256,
        "stress_run_artifacts_projection_sha256": pins.stress_run_artifacts_projection_sha256,
        "pre_map_output_seal_sha256": pins.output_seal_sha256,
        "pre_map_output_seal_projection_sha256": pins.output_seal_projection_sha256,
        "frozen_evaluator_source_sha256": pins.frozen_evaluator_source_sha256,
        "opaque_input_count": len(stress_rows),
        "image_count": image_count,
        "all_links_verified": True,
    }
    chain["chain_projection_sha256"] = canonical_json_sha256(chain)
    return clean_seal, run_manifest, expected, chain


def evaluate_math12_stress_bindings_compat(
    *,
    run_dir: Path,
    stress_dir: Path,
    clean_input_seal_path: Path,
    clean_input_jsonl_path: Path,
    private_map_path: Path,
    output_seal_path: Path,
    pins: CompatibilityPins = DEFAULT_PINS,
) -> dict[str, Any]:
    """Attest the sealed stress chain and score it with the frozen rule.

    A derived seal is written only inside a temporary directory so the frozen
    evaluator can consume the already-sealed stress input hash.  The original
    clean seal, stress freeze, predictions, run manifests, preregistration and
    output seal are never written or replaced.
    """

    clean_seal, _, _, chain = _attest_chain(
        run_dir=run_dir,
        stress_dir=stress_dir,
        clean_input_seal_path=clean_input_seal_path,
        clean_input_jsonl_path=clean_input_jsonl_path,
        private_map_path=private_map_path,
        output_seal_path=output_seal_path,
        pins=pins,
    )
    bridge_seal = dict(clean_seal)
    bridge_seal.update(
        {
            "public_inputs": "post-seal compatibility bridge to frozen stress JSONL",
            "public_inputs_sha256": pins.stress_input_jsonl_sha256,
            "compatibility_bridge_schema": BRIDGE_SCHEMA,
            "prediction_status": PREDICTION_STATUS,
            "adapter_status": ADAPTER_STATUS,
            "preregistration_status": PREREGISTRATION_STATUS,
            "original_clean_input_seal_sha256": pins.clean_input_seal_sha256,
            "stress_freeze_sha256": pins.stress_freeze_sha256,
            "pre_map_output_seal_sha256": pins.output_seal_sha256,
        }
    )
    bridge_projection = canonical_json_sha256(bridge_seal)
    with tempfile.TemporaryDirectory(prefix="math12-stress-compat-") as tmp:
        bridge_path = Path(tmp) / "derived_bridge_seal.json"
        bridge_path.write_bytes(canonical_json_bytes(bridge_seal) + b"\n")
        try:
            frozen = frozen_eval.evaluate_math12_bindings(
                run_dir=run_dir,
                input_seal_path=bridge_path,
                private_map_path=private_map_path,
            )
        except frozen_eval.Math12BindingEvaluationError as exc:
            raise Math12StressCompatibilityError(str(exc)) from exc

    metrics = {
        key: frozen[key]
        for key in (
            "total",
            "accepted",
            "abstained",
            "correct",
            "incorrect",
            "coverage",
            "source_binding_accuracy",
            "conditional_precision",
        )
    }
    value: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA,
        "prediction_status": PREDICTION_STATUS,
        "adapter_status": ADAPTER_STATUS,
        "preregistration_status": PREREGISTRATION_STATUS,
        "scope": SCOPE,
        "claim_limit": (
            "synthetic robustness of source-activity binding on transformed copies; "
            "not QA accuracy, not mathematical reasoning accuracy, and not a new benchmark"
        ),
        "timeline_disclosure": (
            "predictions and the exact output seal predate private-map read; this "
            "compatibility adapter and its evaluation were created after private-map read"
        ),
        "original_artifacts_mutated": False,
        "official_default_pins_used": pins == DEFAULT_PINS,
        "pins_projection_sha256": pins.projection_sha256,
        "chain_attestation": chain,
        "metric_engine": {
            "schema_version": frozen_eval.EVALUATION_SCHEMA,
            "source_sha256": pins.frozen_evaluator_source_sha256,
            "compatibility_bridge_schema": BRIDGE_SCHEMA,
            "compatibility_bridge_projection_sha256": bridge_projection,
            "rule": "correct = accepted AND predicted_activity == expected_activity; abstain = error",
            "formula_changed": False,
        },
        "metrics": metrics,
        "rows": frozen["rows"],
        "rows_projection_sha256": frozen["evaluation_projection_sha256"],
    }
    value["evaluation_projection_sha256"] = canonical_json_sha256(value)
    return value


def write_compatibility_evaluation(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Math12StressCompatibilityError("evaluation output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")
