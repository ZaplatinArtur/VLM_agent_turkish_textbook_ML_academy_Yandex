"""Fail-closed batch orchestration for opaque Math12 image inputs.

This module is intentionally a thin wrapper around the frozen Math12 source
resolver.  It validates opaque input bytes, calls the resolver for every image,
writes each resolver certificate and official-source solution record, and only
then aggregates image decisions.  It never accepts a benchmark answer, label,
expected activity, correctness outcome, or score.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import tempfile
from typing import Any

from .math12_activity_source import (
    FROZEN_SIFT_RUNTIME_PROFILE,
    FROZEN_VISUAL_THRESHOLDS,
    Math12BindingDecision,
    Math12Inventory,
    Math12OfficialSolutionRecord,
    Math12RenderManifest,
    Math12SourceCertificate,
    extract_official_solution,
    load_math12_inventory,
    load_math12_render_manifest,
    resolve_math12_image_bytes,
    verify_math12_source_certificate,
)
from .official_ogm import canonical_json_bytes, canonical_json_sha256, sha256_file


INPUT_SCHEMA_ALLOWLIST = frozenset(
    {
        "holdout80-opaque-resolver-input-v1",
        "holdout80-opaque-resolver-input-stress-v1",
    }
)
BATCH_RESULT_SCHEMA = "math12-opaque-source-batch-result-v1"
BATCH_RUN_SCHEMA = "math12-opaque-source-batch-run-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_INPUT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALLOWED_INPUT_KEYS = frozenset(
    {
        "expected_response_format",
        "images",
        "input_id",
        "language",
        "prompt",
        "schema_version",
    }
)
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "accuracy",
        "answer",
        "answers",
        "correct",
        "correctness",
        "evaluation",
        "expected_activity",
        "gold",
        "label",
        "metric",
        "oracle",
        "outcome",
        "reference_answer",
        "reward",
        "score",
        "task_id",
        "verdict",
    }
)


class Math12OpaqueBatchError(ValueError):
    """Opaque input or batch output violates the frozen safety contract."""


@dataclass(frozen=True, slots=True)
class OpaqueImage:
    relative_path: str
    resolved_path: Path
    sha256: str
    image_bytes: bytes


@dataclass(frozen=True, slots=True)
class OpaqueInput:
    input_id: str
    schema_version: str
    images: tuple[OpaqueImage, ...]


ResolveImage = Callable[
    [bytes, Math12Inventory, Math12RenderManifest], Math12SourceCertificate
]
ExtractSolution = Callable[
    [Path, Math12Inventory, Math12RenderManifest, Math12SourceCertificate],
    Math12OfficialSolutionRecord,
]
VerifyCertificate = Callable[
    [Math12Inventory, Math12RenderManifest, Math12SourceCertificate],
    Math12BindingDecision,
]


def _resolve_with_frozen_profile(
    image_bytes: bytes,
    inventory: Math12Inventory,
    render_manifest: Math12RenderManifest,
) -> Math12SourceCertificate:
    return resolve_math12_image_bytes(
        image_bytes,
        inventory,
        render_manifest,
        thresholds=FROZEN_VISUAL_THRESHOLDS,
        runtime_profile=FROZEN_SIFT_RUNTIME_PROFILE,
    )


def _strict_json_object(line: str, *, line_number: int) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Math12OpaqueBatchError(
                    f"duplicate JSON key at input line {line_number}: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise Math12OpaqueBatchError(
            f"non-finite JSON value at input line {line_number}: {value}"
        )

    try:
        value = json.loads(
            line,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise Math12OpaqueBatchError(
            f"malformed JSON at input line {line_number}"
        ) from exc
    if not isinstance(value, dict):
        raise Math12OpaqueBatchError(
            f"input line {line_number} must be one JSON object"
        )
    return value


def _assert_source_only_keys(value: Mapping[str, Any], *, line_number: int) -> None:
    forbidden = sorted(
        str(key) for key in value if str(key).casefold() in _FORBIDDEN_INPUT_KEYS
    )
    if forbidden:
        raise Math12OpaqueBatchError(
            f"forbidden benchmark field at input line {line_number}: {forbidden[0]}"
        )


def _validate_relative_asset_path(raw_path: str) -> None:
    if not raw_path or "\x00" in raw_path:
        raise Math12OpaqueBatchError("opaque asset path is empty or contains NUL")
    windows = PureWindowsPath(raw_path)
    posix = PurePosixPath(raw_path)
    if (
        windows.is_absolute()
        or posix.is_absolute()
        or windows.drive
        or ".." in windows.parts
        or ".." in posix.parts
    ):
        raise Math12OpaqueBatchError("opaque asset path escapes the asset root")


def _resolve_asset_path(asset_root: Path, raw_path: str) -> Path:
    _validate_relative_asset_path(raw_path)
    candidate = (asset_root / Path(raw_path)).resolve(strict=False)
    try:
        candidate.relative_to(asset_root)
    except ValueError as exc:
        raise Math12OpaqueBatchError("opaque asset path escapes the asset root") from exc
    if not candidate.is_file():
        raise Math12OpaqueBatchError("opaque asset is missing or is not a regular file")
    return candidate


def load_opaque_inputs(input_jsonl: Path, asset_root: Path) -> tuple[OpaqueInput, ...]:
    """Validate a source-only JSONL and pin every listed asset byte before work.

    No path not explicitly listed in ``input_jsonl`` is opened.  Assets are
    loaded once after SHA-256 verification, eliminating a hash-check/use race.
    Duplicate input IDs, file paths, and asset hashes are rejected globally.
    """

    input_jsonl = input_jsonl.resolve(strict=False)
    asset_root = asset_root.resolve(strict=False)
    if not input_jsonl.is_file():
        raise Math12OpaqueBatchError("opaque JSONL is missing")
    if not asset_root.is_dir():
        raise Math12OpaqueBatchError("asset root is missing or is not a directory")
    try:
        lines = input_jsonl.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Math12OpaqueBatchError("opaque JSONL cannot be read as UTF-8") from exc
    if not lines or any(not line.strip() for line in lines):
        raise Math12OpaqueBatchError("opaque JSONL is empty or contains blank lines")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    inputs: list[OpaqueInput] = []
    for line_number, line in enumerate(lines, start=1):
        value = _strict_json_object(line, line_number=line_number)
        _assert_source_only_keys(value, line_number=line_number)
        unexpected_input_keys = set(value) - _ALLOWED_INPUT_KEYS
        if unexpected_input_keys:
            raise Math12OpaqueBatchError(
                f"unknown opaque input fields at line {line_number}"
            )
        schema_version = str(value.get("schema_version") or "")
        if schema_version not in INPUT_SCHEMA_ALLOWLIST:
            raise Math12OpaqueBatchError(
                f"unknown opaque input schema at line {line_number}"
            )
        input_id = str(value.get("input_id") or "")
        if _SAFE_INPUT_ID.fullmatch(input_id) is None or input_id in {".", ".."}:
            raise Math12OpaqueBatchError(f"unsafe input_id at line {line_number}")
        normalized_id = input_id.casefold()
        if normalized_id in seen_ids:
            raise Math12OpaqueBatchError(f"duplicate input_id at line {line_number}")
        seen_ids.add(normalized_id)
        raw_images = value.get("images")
        if not isinstance(raw_images, list) or not raw_images:
            raise Math12OpaqueBatchError(f"input line {line_number} has no images")
        images: list[OpaqueImage] = []
        for image_number, raw_image in enumerate(raw_images, start=1):
            if not isinstance(raw_image, dict):
                raise Math12OpaqueBatchError(
                    f"image {image_number} at line {line_number} is not an object"
                )
            unexpected_image_keys = set(raw_image) - {"path", "sha256"}
            if unexpected_image_keys:
                raise Math12OpaqueBatchError(
                    f"image {image_number} at line {line_number} has unknown fields"
                )
            raw_path = str(raw_image.get("path") or "")
            expected_sha = str(raw_image.get("sha256") or "")
            if _HEX64.fullmatch(expected_sha) is None:
                raise Math12OpaqueBatchError(
                    f"image {image_number} at line {line_number} lacks a SHA-256 pin"
                )
            resolved_path = _resolve_asset_path(asset_root, raw_path)
            normalized_path = os.path.normcase(str(resolved_path))
            if normalized_path in seen_paths:
                raise Math12OpaqueBatchError("duplicate opaque asset path")
            seen_paths.add(normalized_path)
            try:
                image_bytes = resolved_path.read_bytes()
            except OSError as exc:
                raise Math12OpaqueBatchError("opaque asset cannot be read") from exc
            actual_sha = hashlib.sha256(image_bytes).hexdigest()
            if actual_sha != expected_sha:
                raise Math12OpaqueBatchError("opaque asset SHA-256 mismatch")
            if actual_sha in seen_hashes:
                raise Math12OpaqueBatchError("duplicate opaque asset bytes")
            seen_hashes.add(actual_sha)
            images.append(
                OpaqueImage(
                    relative_path=raw_path.replace("\\", "/"),
                    resolved_path=resolved_path,
                    sha256=actual_sha,
                    image_bytes=image_bytes,
                )
            )
        inputs.append(
            OpaqueInput(
                input_id=input_id,
                schema_version=schema_version,
                images=tuple(images),
            )
        )
    return tuple(inputs)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _certificate_is_consistently_accepted(
    certificate: Math12SourceCertificate,
    verified_decision: Math12BindingDecision,
    *,
    expected_image_sha256: str,
) -> tuple[bool, int | None, str]:
    if certificate.task_image_sha256 != expected_image_sha256:
        return False, None, "resolver_certificate_image_hash_mismatch"
    if verified_decision != certificate.decision:
        return False, None, "strict_verifier_decision_mismatch"
    decision = verified_decision
    if not decision.accepted:
        return False, None, str(decision.reason or "resolver_abstained")
    if (
        not decision.checks
        or any(not passed for _, passed in decision.checks)
        or not isinstance(decision.selected_activity_number, int)
        or decision.selected_activity_number < 1
    ):
        return False, None, "resolver_returned_malformed_accepted_certificate"
    return True, decision.selected_activity_number, str(decision.reason)


def _exception_code(prefix: str, exc: Exception) -> str:
    # Exception messages can contain local paths or unstable library text.  The
    # class name is sufficient for an auditable, deterministic failure code.
    return f"{prefix}:{type(exc).__name__}"


def _aggregate_input(image_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if any(item["processing_status"] != "certificate_verified" for item in image_results):
        return {
            "accepted": False,
            "reason": "abstain_incomplete_image_processing",
            "selected_activity_number": None,
        }
    activities = {
        int(item["selected_activity_number"])
        for item in image_results
        if item["certificate_accepted"]
    }
    if not activities:
        return {
            "accepted": False,
            "reason": "abstain_no_accepted_certificate",
            "selected_activity_number": None,
        }
    if len(activities) != 1:
        return {
            "accepted": False,
            "reason": "abstain_conflicting_accepted_activities",
            "selected_activity_number": None,
        }
    return {
        "accepted": True,
        "reason": "accepted_activity_agreement",
        "selected_activity_number": next(iter(activities)),
    }


def _artifact_hashes(root: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.name != "run_manifest.json":
            artifacts[path.relative_to(root).as_posix()] = sha256_file(path)
    return artifacts


def execute_opaque_batch(
    *,
    opaque_inputs: Sequence[OpaqueInput],
    input_jsonl_sha256: str,
    output_dir: Path,
    inventory: Math12Inventory,
    render_manifest: Math12RenderManifest,
    pdf_path: Path,
    resolve_image: ResolveImage = _resolve_with_frozen_profile,
    verify_certificate: VerifyCertificate = verify_math12_source_certificate,
    extract_solution: ExtractSolution = extract_official_solution,
) -> dict[str, Any]:
    """Execute an already validated opaque batch into a new output directory."""

    if _HEX64.fullmatch(input_jsonl_sha256) is None:
        raise Math12OpaqueBatchError("opaque JSONL SHA-256 pin is malformed")
    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists():
        raise Math12OpaqueBatchError("output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        input_results: list[dict[str, Any]] = []
        for opaque_input in opaque_inputs:
            image_results: list[dict[str, Any]] = []
            for image_index, image in enumerate(opaque_input.images, start=1):
                stem = f"image-{image_index:02d}-{image.sha256[:16]}"
                certificate_relative = (
                    Path("certificates") / opaque_input.input_id / f"{stem}.json"
                )
                solution_relative = (
                    Path("solution_records") / opaque_input.input_id / f"{stem}.json"
                )
                image_result: dict[str, Any] = {
                    "image_index": image_index,
                    "opaque_asset_path": image.relative_path,
                    "image_sha256": image.sha256,
                    "processing_status": "resolver_error",
                    "certificate_path": None,
                    "certificate_file_sha256": None,
                    "certificate_projection_sha256": None,
                    "certificate_accepted": False,
                    "certificate_reason": "resolver_not_completed",
                    "selected_content_page": None,
                    "selected_activity_number": None,
                    "solution_record_status": "not_attempted",
                    "solution_record_path": None,
                    "solution_record_file_sha256": None,
                    "solution_record_projection_sha256": None,
                }
                try:
                    certificate = resolve_image(
                        image.image_bytes,
                        inventory,
                        render_manifest,
                    )
                    certificate_mapping = certificate.to_mapping()
                    certificate_path = staging / certificate_relative
                    _write_json(certificate_path, certificate_mapping)
                    image_result.update(
                        {
                            "processing_status": "certificate_unverified",
                            "certificate_path": certificate_relative.as_posix(),
                            "certificate_file_sha256": sha256_file(certificate_path),
                            "certificate_projection_sha256": (
                                certificate.certificate_projection_sha256
                            ),
                            "certificate_reason": "strict_verifier_not_completed",
                        }
                    )
                    verified_decision = verify_certificate(
                        inventory,
                        render_manifest,
                        certificate,
                    )
                    accepted, activity, reason = _certificate_is_consistently_accepted(
                        certificate,
                        verified_decision,
                        expected_image_sha256=image.sha256,
                    )
                    image_result.update(
                        {
                            "processing_status": "certificate_verified",
                            "certificate_accepted": accepted,
                            "certificate_reason": reason,
                            "selected_content_page": (
                                certificate.decision.selected_content_page
                                if accepted
                                else None
                            ),
                            "selected_activity_number": activity,
                        }
                    )
                    if accepted:
                        try:
                            solution = extract_solution(
                                pdf_path,
                                inventory,
                                render_manifest,
                                certificate,
                            )
                            if (
                                solution.task_image_sha256 != image.sha256
                                or solution.activity_number != activity
                            ):
                                raise Math12OpaqueBatchError(
                                    "official solution record disagrees with certificate"
                                )
                            solution_path = staging / solution_relative
                            _write_json(solution_path, solution.to_mapping())
                            image_result.update(
                                {
                                    "solution_record_status": "written",
                                    "solution_record_path": solution_relative.as_posix(),
                                    "solution_record_file_sha256": sha256_file(solution_path),
                                    "solution_record_projection_sha256": (
                                        solution.answer_bound_certificate_projection_sha256
                                    ),
                                }
                            )
                        except Exception as exc:  # fail closed, preserve other images
                            image_result["solution_record_status"] = _exception_code(
                                "extraction_error", exc
                            )
                    else:
                        image_result["solution_record_status"] = (
                            "not_attempted_certificate_abstained"
                        )
                except Exception as exc:  # fail closed, preserve other images
                    if image_result["processing_status"] == "certificate_unverified":
                        image_result["processing_status"] = (
                            "certificate_verification_error"
                        )
                        image_result["certificate_reason"] = _exception_code(
                            "verification_error", exc
                        )
                    else:
                        image_result["certificate_reason"] = _exception_code(
                            "resolver_error", exc
                        )
                image_results.append(image_result)

            aggregate = _aggregate_input(image_results)
            input_results.append(
                {
                    "schema_version": BATCH_RESULT_SCHEMA,
                    "component_scope": "source_resolution_only",
                    "input_id": opaque_input.input_id,
                    "input_schema_version": opaque_input.schema_version,
                    "image_count": len(image_results),
                    "accepted_certificate_count": sum(
                        bool(item["certificate_accepted"]) for item in image_results
                    ),
                    "images": image_results,
                    "aggregate": aggregate,
                }
            )

        results_path = staging / "results.jsonl"
        results_path.write_bytes(
            b"".join(canonical_json_bytes(item) + b"\n" for item in input_results)
        )
        artifacts = _artifact_hashes(staging)
        manifest = {
            "schema_version": BATCH_RUN_SCHEMA,
            "component_scope": "source_resolution_only",
            "input_jsonl_sha256": input_jsonl_sha256,
            "inventory_projection_sha256": inventory.inventory_projection_sha256,
            "render_manifest_projection_sha256": (
                render_manifest.render_manifest_projection_sha256
            ),
            "pdf_sha256": inventory.pdf_sha256,
            "frozen_visual_thresholds": asdict(FROZEN_VISUAL_THRESHOLDS),
            "frozen_sift_runtime_profile": asdict(FROZEN_SIFT_RUNTIME_PROFILE),
            "resolver_policy_projection_sha256": canonical_json_sha256(
                {
                    "frozen_visual_thresholds": asdict(FROZEN_VISUAL_THRESHOLDS),
                    "frozen_sift_runtime_profile": asdict(
                        FROZEN_SIFT_RUNTIME_PROFILE
                    ),
                }
            ),
            "input_count": len(input_results),
            "image_count": sum(item["image_count"] for item in input_results),
            "accepted_input_count": sum(
                bool(item["aggregate"]["accepted"]) for item in input_results
            ),
            "abstained_input_count": sum(
                not bool(item["aggregate"]["accepted"]) for item in input_results
            ),
            "artifacts": artifacts,
            "artifacts_projection_sha256": canonical_json_sha256(artifacts),
        }
        _write_json(staging / "run_manifest.json", manifest)
        staging.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_opaque_batch(
    *,
    input_jsonl: Path,
    asset_root: Path,
    inventory_path: Path,
    render_manifest_path: Path,
    render_page_root: Path | None,
    pdf_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Load frozen source artifacts and run one opaque source-only batch."""

    inputs = load_opaque_inputs(input_jsonl, asset_root)
    inventory = load_math12_inventory(inventory_path)
    render_manifest = load_math12_render_manifest(
        render_manifest_path,
        inventory,
        page_root=render_page_root,
    )
    if not pdf_path.is_file() or sha256_file(pdf_path) != inventory.pdf_sha256:
        raise Math12OpaqueBatchError("Math12 PDF differs from the inventory pin")
    return execute_opaque_batch(
        opaque_inputs=inputs,
        input_jsonl_sha256=sha256_file(input_jsonl),
        output_dir=output_dir,
        inventory=inventory,
        render_manifest=render_manifest,
        pdf_path=pdf_path,
    )
