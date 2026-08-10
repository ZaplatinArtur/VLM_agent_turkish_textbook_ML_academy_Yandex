"""Fail-closed batch runner for task-ID-free full-page MCQ inputs.

Repeated page-image bytes are expected: one official textbook page can contain
several questions.  The policy feature is the observable ``(prompt, image)``
pair; ``input_id`` is used only to align outputs and is never passed to the
resolver.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import tempfile
from typing import Any

from .mcq_fullpage_source import (
    EXPECTED_FROZEN_BUNDLE_MANIFEST_PROJECTION_SHA256,
    EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256,
    EXPECTED_INVENTORY_FILE_SHA256,
    EXPECTED_KEY_INDEX_FILE_SHA256,
    EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256,
    EXPECTED_RENDER_MANIFEST_FILE_SHA256,
    McqInventory,
    McqKeyIndex,
    McqRenderManifest,
    McqSourceCertificate,
    McqSourceError,
    assert_frozen_mcq_bundle,
    assert_frozen_mcq_objects,
    parse_observable_mcq_prompt,
    resolve_mcq_image_bytes,
    verify_mcq_source_certificate,
    write_canonical_json,
)
from .official_ogm import canonical_json_bytes, canonical_json_sha256, sha256_file
from .visual_coordinate_binding import VisualCoordinateBindingError


INPUT_SCHEMA_ALLOWLIST = frozenset(
    {
        "holdout80-opaque-resolver-input-v1",
        "holdout80-opaque-resolver-input-stress-v1",
    }
)
BATCH_RESULT_SCHEMA = "mcq-fullpage-opaque-source-batch-result-v1"
BATCH_RUN_SCHEMA = "mcq-fullpage-opaque-source-batch-run-v1.1"
INPUT_PROJECTION_SCHEMA = "mcq-fullpage-opaque-input-projection-v1.1"
V11_CODE_FREEZE_SCHEMA = "mcq-fullpage-source-adapter-freeze-v1.1"
EXPECTED_V11_RUNTIME_CODE_PATHS = frozenset(
    {
        "src/evidence_os/__init__.py",
        "src/evidence_os/certificates.py",
        "src/evidence_os/contracts.py",
        "src/evidence_os/mcq_fullpage_source.py",
        "src/evidence_os/mcq_opaque_batch.py",
        "src/evidence_os/official_ogm.py",
        "src/evidence_os/policy.py",
        "src/evidence_os/source_first.py",
        "src/evidence_os/visual_coordinate_binding.py",
        "scripts/build_mcq_fullpage_source_v1.py",
        "scripts/mcq_fullpage_source_adapter.py",
        "scripts/run_mcq_opaque_batch_v1.py",
        "scripts/freeze_mcq_fullpage_source_v1_1.py",
    }
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_INPUT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
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
_FORBIDDEN_KEYS = frozenset(
    {
        "accuracy",
        "activityid",
        "answer",
        "answers",
        "bookid",
        "correct",
        "correctness",
        "evaluation",
        "expectedanswer",
        "gold",
        "label",
        "metric",
        "oracle",
        "outcome",
        "page",
        "pagenumber",
        "prediction",
        "referenceanswer",
        "reward",
        "score",
        "selected",
        "source",
        "sourcefamily",
        "sourcepdf",
        "taskid",
        "unit",
        "verdict",
    }
)


class McqOpaqueBatchError(ValueError):
    """Opaque input or source-only output violates the batch contract."""


@dataclass(frozen=True, slots=True)
class McqV11CodeAttestation:
    freeze_file_sha256: str
    freeze_projection_sha256: str
    code_projection_sha256: str
    code_file_count: int

    def __post_init__(self) -> None:
        for value in (
            self.freeze_file_sha256,
            self.freeze_projection_sha256,
            self.code_projection_sha256,
        ):
            if _HEX64.fullmatch(value) is None:
                raise McqOpaqueBatchError("v1.1 code attestation contains a bad SHA")
        if self.code_file_count < len(EXPECTED_V11_RUNTIME_CODE_PATHS):
            raise McqOpaqueBatchError("v1.1 code attestation is incomplete")


@dataclass(frozen=True, slots=True)
class McqOpaqueImage:
    relative_path: str
    resolved_path: Path
    sha256: str
    image_bytes: bytes


@dataclass(frozen=True, slots=True)
class McqOpaqueInput:
    input_id: str
    schema_version: str
    prompt: str
    prompt_sha256: str
    language: str
    expected_response_format: str
    image: McqOpaqueImage


def _compact_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _reject_forbidden(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise McqOpaqueBatchError("opaque input contains a non-string key")
            if _compact_key(raw_key) in _FORBIDDEN_KEYS:
                raise McqOpaqueBatchError(
                    "forbidden benchmark field in opaque input: "
                    f"{'.'.join(path + (raw_key,))}"
                )
            _reject_forbidden(child, path + (raw_key,))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_forbidden(child, path + (f"[{index}]",))


def _strict_line(line: str, line_number: int) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise McqOpaqueBatchError(
                    f"duplicate JSON key at opaque line {line_number}: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise McqOpaqueBatchError(
            f"non-finite JSON value at opaque line {line_number}: {value}"
        )

    try:
        value = json.loads(
            line,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise McqOpaqueBatchError(
            f"malformed JSON at opaque line {line_number}"
        ) from exc
    if not isinstance(value, dict):
        raise McqOpaqueBatchError(f"opaque line {line_number} is not an object")
    return value


def _strict_json_file(path: Path, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise McqOpaqueBatchError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise McqOpaqueBatchError(f"non-finite JSON value in {label}: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise McqOpaqueBatchError(f"cannot load {label}") from exc
    if not isinstance(value, dict):
        raise McqOpaqueBatchError(f"{label} root is not an object")
    return value


def assert_mcq_v11_code_freeze(
    *,
    freeze_manifest_path: Path,
    expected_freeze_sha256: str,
    expected_freeze_projection_sha256: str,
    code_root: Path | None = None,
) -> McqV11CodeAttestation:
    """Verify externally pinned v1.1 code bytes before parsing opaque input.

    The expected file/projection hashes are mandatory command-level trust pins;
    they cannot be embedded in the runner because the freeze hashes the runner
    itself.  A published commit plus these pins closes that self-hash cycle.
    """

    if (
        _HEX64.fullmatch(expected_freeze_sha256) is None
        or _HEX64.fullmatch(expected_freeze_projection_sha256) is None
    ):
        raise McqOpaqueBatchError("v1.1 freeze command pins are not SHA-256 values")
    freeze_manifest_path = freeze_manifest_path.resolve(strict=False)
    if (
        not freeze_manifest_path.is_file()
        or sha256_file(freeze_manifest_path) != expected_freeze_sha256
    ):
        raise McqOpaqueBatchError("v1.1 freeze bytes differ from the command pin")
    raw = _strict_json_file(freeze_manifest_path, "v1.1 freeze manifest")
    if (
        raw.get("schema_version") != V11_CODE_FREEZE_SCHEMA
        or raw.get("status") != "ready_for_commit_no_opaque_read_or_run"
        or raw.get("accuracy_claim") is not None
    ):
        raise McqOpaqueBatchError("v1.1 freeze status/schema changed")
    projection = dict(raw)
    declared_projection = projection.pop("manifest_projection_sha256", None)
    recomputed_projection = canonical_json_sha256(projection)
    if (
        declared_projection != recomputed_projection
        or recomputed_projection != expected_freeze_projection_sha256
    ):
        raise McqOpaqueBatchError("v1.1 freeze projection differs from command pin")
    raw_code = raw.get("code")
    if not isinstance(raw_code, dict) or set(raw_code) != {
        "files",
        "combined_code_projection_sha256",
    }:
        raise McqOpaqueBatchError("v1.1 freeze code declaration is malformed")
    raw_files = raw_code.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise McqOpaqueBatchError("v1.1 freeze contains no code files")
    declared_code_projection = raw_code.get("combined_code_projection_sha256")
    if (
        not isinstance(declared_code_projection, str)
        or _HEX64.fullmatch(declared_code_projection) is None
        or canonical_json_sha256(raw_files) != declared_code_projection
    ):
        raise McqOpaqueBatchError("v1.1 code-file projection changed")
    resolved_root = (
        code_root.resolve()
        if code_root is not None
        else Path(__file__).resolve().parents[2]
    )
    observed_paths: set[str] = set()
    for raw_entry in raw_files:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise McqOpaqueBatchError("v1.1 code-file entry is malformed")
        relative = raw_entry.get("path")
        expected_sha = raw_entry.get("sha256")
        expected_size = raw_entry.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or PurePosixPath(relative).is_absolute()
            or PureWindowsPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or ".." in PureWindowsPath(relative).parts
            or relative in observed_paths
            or not isinstance(expected_sha, str)
            or _HEX64.fullmatch(expected_sha) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 1
        ):
            raise McqOpaqueBatchError("v1.1 code-file pin/path is unsafe")
        observed_paths.add(relative)
        code_path = (resolved_root / Path(relative)).resolve(strict=False)
        try:
            code_path.relative_to(resolved_root)
        except ValueError as exc:
            raise McqOpaqueBatchError("v1.1 code file escapes repository") from exc
        if (
            not code_path.is_file()
            or code_path.stat().st_size != expected_size
            or sha256_file(code_path) != expected_sha
        ):
            raise McqOpaqueBatchError(f"v1.1 code bytes changed: {relative}")
    if not EXPECTED_V11_RUNTIME_CODE_PATHS.issubset(observed_paths):
        raise McqOpaqueBatchError("v1.1 freeze omits runtime code")
    return McqV11CodeAttestation(
        freeze_file_sha256=expected_freeze_sha256,
        freeze_projection_sha256=expected_freeze_projection_sha256,
        code_projection_sha256=declared_code_projection,
        code_file_count=len(raw_files),
    )


def _safe_asset_path(asset_root: Path, raw_path: str) -> Path:
    if not raw_path or "\x00" in raw_path:
        raise McqOpaqueBatchError("opaque asset path is empty or contains NUL")
    windows = PureWindowsPath(raw_path)
    posix = PurePosixPath(raw_path)
    if (
        windows.is_absolute()
        or posix.is_absolute()
        or windows.drive
        or ".." in windows.parts
        or ".." in posix.parts
    ):
        raise McqOpaqueBatchError("opaque asset path escapes the asset root")
    resolved = (asset_root / Path(raw_path)).resolve(strict=False)
    try:
        resolved.relative_to(asset_root)
    except ValueError as exc:
        raise McqOpaqueBatchError("opaque asset path escapes the asset root") from exc
    if not resolved.is_file():
        raise McqOpaqueBatchError("opaque asset is missing")
    return resolved


def load_mcq_opaque_inputs(
    input_jsonl: Path, asset_root: Path
) -> tuple[McqOpaqueInput, ...]:
    """Load only observable fields and permit a page shared by several prompts."""

    input_jsonl = input_jsonl.resolve(strict=False)
    asset_root = asset_root.resolve(strict=False)
    if not input_jsonl.is_file() or not asset_root.is_dir():
        raise McqOpaqueBatchError("opaque JSONL or asset root is missing")
    try:
        input_bytes = input_jsonl.read_bytes()
    except OSError as exc:
        raise McqOpaqueBatchError("opaque JSONL cannot be read") from exc
    return _parse_mcq_opaque_input_bytes(input_bytes, asset_root)


def _parse_mcq_opaque_input_bytes(
    input_bytes: bytes, asset_root: Path
) -> tuple[McqOpaqueInput, ...]:
    if not isinstance(input_bytes, bytes) or not input_bytes:
        raise McqOpaqueBatchError("opaque JSONL bytes are empty")
    asset_root = asset_root.resolve(strict=False)
    if not asset_root.is_dir():
        raise McqOpaqueBatchError("opaque asset root is missing")
    try:
        lines = input_bytes.decode("utf-8-sig").splitlines()
    except UnicodeError as exc:
        raise McqOpaqueBatchError("opaque JSONL is not strict UTF-8") from exc
    if not lines or any(not line.strip() for line in lines):
        raise McqOpaqueBatchError("opaque JSONL is empty or contains blank lines")

    seen_ids: set[str] = set()
    seen_observations: set[tuple[str, str]] = set()
    inputs: list[McqOpaqueInput] = []
    for line_number, line in enumerate(lines, start=1):
        value = _strict_line(line, line_number)
        _reject_forbidden(value)
        if set(value) != _ALLOWED_INPUT_KEYS:
            raise McqOpaqueBatchError(
                f"opaque input fields are not exact at line {line_number}"
            )
        if not isinstance(value.get("schema_version"), str):
            raise McqOpaqueBatchError(f"opaque schema is not text at line {line_number}")
        schema = value["schema_version"]
        if schema not in INPUT_SCHEMA_ALLOWLIST:
            raise McqOpaqueBatchError(f"unknown opaque schema at line {line_number}")
        if not isinstance(value.get("input_id"), str):
            raise McqOpaqueBatchError(f"opaque input_id is not text at line {line_number}")
        input_id = value["input_id"]
        if (
            _SAFE_INPUT_ID.fullmatch(input_id) is None
            or input_id in {".", ".."}
            or input_id.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_STEMS
            or input_id.casefold() in seen_ids
        ):
            raise McqOpaqueBatchError(f"unsafe or duplicate input_id at line {line_number}")
        seen_ids.add(input_id.casefold())
        if not isinstance(value.get("prompt"), str):
            raise McqOpaqueBatchError(f"opaque prompt is not text at line {line_number}")
        prompt = value["prompt"]
        parse_observable_mcq_prompt(prompt)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if not isinstance(value.get("language"), str):
            raise McqOpaqueBatchError("MCQ opaque language is not text")
        language = value["language"]
        if language not in {"tr", "tur", "Turkish"}:
            raise McqOpaqueBatchError("MCQ opaque language is not Turkish")
        if not isinstance(value.get("expected_response_format"), str):
            raise McqOpaqueBatchError("MCQ response format is not text")
        response_format = value["expected_response_format"]
        if response_format != "single_choice_ABCDE":
            raise McqOpaqueBatchError("MCQ expected response format is not frozen A-E")
        raw_images = value.get("images")
        if (
            not isinstance(raw_images, list)
            or len(raw_images) != 1
            or not isinstance(raw_images[0], dict)
            or set(raw_images[0]) != {"path", "sha256"}
        ):
            raise McqOpaqueBatchError("MCQ opaque input must contain exactly one pinned image")
        raw_image = raw_images[0]
        if not isinstance(raw_image.get("sha256"), str) or not isinstance(
            raw_image.get("path"), str
        ):
            raise McqOpaqueBatchError("opaque image path/SHA is not text")
        expected_sha = raw_image["sha256"]
        if _HEX64.fullmatch(expected_sha) is None:
            raise McqOpaqueBatchError("opaque image lacks a SHA-256 pin")
        raw_path = raw_image["path"]
        resolved_path = _safe_asset_path(asset_root, raw_path)
        try:
            image_bytes = resolved_path.read_bytes()
        except OSError as exc:
            raise McqOpaqueBatchError("opaque image bytes cannot be read") from exc
        actual_sha = hashlib.sha256(image_bytes).hexdigest()
        if actual_sha != expected_sha:
            raise McqOpaqueBatchError("opaque image bytes differ from their pin")
        observation = (prompt_sha, actual_sha)
        if observation in seen_observations:
            raise McqOpaqueBatchError("duplicate observable prompt/image pair")
        seen_observations.add(observation)
        inputs.append(
            McqOpaqueInput(
                input_id=input_id,
                schema_version=schema,
                prompt=prompt,
                prompt_sha256=prompt_sha,
                language=language,
                expected_response_format=response_format,
                image=McqOpaqueImage(
                    relative_path=raw_path.replace("\\", "/"),
                    resolved_path=resolved_path,
                    sha256=actual_sha,
                    image_bytes=image_bytes,
                ),
            )
        )
    return tuple(inputs)


def _opaque_input_projection(inputs: Sequence[McqOpaqueInput]) -> str:
    rows = [
        {
            "input_id": item.input_id,
            "schema_version": item.schema_version,
            "prompt_sha256": item.prompt_sha256,
            "language": item.language,
            "expected_response_format": item.expected_response_format,
            "image_relative_path": item.image.relative_path,
            "image_sha256": item.image.sha256,
        }
        for item in inputs
    ]
    return canonical_json_sha256(
        {"schema_version": INPUT_PROJECTION_SCHEMA, "ordered_inputs": rows}
    )


def _revalidate_direct_inputs(
    inputs: Sequence[McqOpaqueInput],
    *,
    input_jsonl_bytes: bytes,
    asset_root: Path,
) -> tuple[McqOpaqueInput, ...]:
    """Reparse raw JSONL and require exact equality with direct API objects."""

    if isinstance(inputs, (str, bytes, bytearray)) or not isinstance(
        inputs, Sequence
    ):
        raise McqOpaqueBatchError("direct MCQ inputs are not a sequence")
    reparsed = _parse_mcq_opaque_input_bytes(input_jsonl_bytes, asset_root)
    if not inputs or len(inputs) != len(reparsed):
        raise McqOpaqueBatchError("direct MCQ input count differs from raw JSONL")
    for supplied, observed in zip(inputs, reparsed, strict=True):
        if not isinstance(supplied, McqOpaqueInput) or not isinstance(
            supplied.image, McqOpaqueImage
        ):
            raise McqOpaqueBatchError("direct MCQ input object type changed")
        if supplied != observed:
            raise McqOpaqueBatchError(
                "direct MCQ input object differs from reparsed observable bytes"
            )
    return reparsed


def _default_resolver(
    prompt: str,
    image_bytes: bytes,
    inventory: McqInventory,
    render_manifest: McqRenderManifest,
    key_index: McqKeyIndex,
) -> McqSourceCertificate:
    return resolve_mcq_image_bytes(
        prompt, image_bytes, inventory, render_manifest, key_index
    )


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.name != "run_manifest.json"
    }


def execute_mcq_opaque_batch(
    inputs: Sequence[McqOpaqueInput],
    inventory: McqInventory,
    render_manifest: McqRenderManifest,
    key_index: McqKeyIndex,
    output_dir: Path,
    *,
    v11_freeze_manifest_path: Path,
    expected_v11_freeze_sha256: str,
    expected_v11_freeze_projection_sha256: str,
    input_jsonl_bytes: bytes,
    asset_root: Path,
) -> dict[str, Any]:
    """Resolve opaque observations after re-attesting source and raw inputs.

    The supplied object sequence is never trusted.  It must reproduce exactly
    from the raw JSONL bytes and pinned asset bytes.  Source objects must equal
    the published frozen bundle, including all 28 page payload bytes.
    """

    code_attestation = assert_mcq_v11_code_freeze(
        freeze_manifest_path=v11_freeze_manifest_path,
        expected_freeze_sha256=expected_v11_freeze_sha256,
        expected_freeze_projection_sha256=(
            expected_v11_freeze_projection_sha256
        ),
    )
    assert_frozen_mcq_objects(inventory, key_index, render_manifest)
    inputs = _revalidate_direct_inputs(
        inputs,
        input_jsonl_bytes=input_jsonl_bytes,
        asset_root=asset_root,
    )
    input_jsonl_sha256 = hashlib.sha256(input_jsonl_bytes).hexdigest()
    input_jsonl_size_bytes = len(input_jsonl_bytes)
    input_projection_sha256 = _opaque_input_projection(inputs)
    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists():
        raise McqOpaqueBatchError("MCQ output directory must be absent")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="mcq_opaque_batch_", dir=output_dir.parent))
    rows: list[dict[str, Any]] = []
    try:
        certificate_dir = staging / "certificates"
        certificate_dir.mkdir(parents=True)
        for item in inputs:
            try:
                certificate = _default_resolver(
                    item.prompt,
                    item.image.image_bytes,
                    inventory,
                    render_manifest,
                    key_index,
                )
                if certificate.task_image_sha256 != item.image.sha256:
                    raise McqOpaqueBatchError(
                        "resolver certificate is bound to different image bytes"
                    )
                decision = verify_mcq_source_certificate(
                    item.prompt,
                    inventory,
                    render_manifest,
                    key_index,
                    certificate,
                    expected_task_image_bytes=item.image.image_bytes,
                )
                certificate_path = certificate_dir / f"{item.input_id}.json"
                write_canonical_json(certificate_path, certificate.to_mapping())
                rows.append(
                    {
                        "schema_version": BATCH_RESULT_SCHEMA,
                        "input_id": item.input_id,
                        "prompt_sha256": item.prompt_sha256,
                        "image_sha256": item.image.sha256,
                        "accepted": decision.accepted,
                        "reason": decision.reason,
                        "answer": certificate.answer if decision.accepted else None,
                        "certificate": certificate_path.relative_to(staging).as_posix(),
                        "certificate_sha256": sha256_file(certificate_path),
                        "certificate_projection_sha256": (
                            certificate.certificate_projection_sha256
                        ),
                    }
                )
            except (McqSourceError, VisualCoordinateBindingError) as exc:
                rows.append(
                    {
                        "schema_version": BATCH_RESULT_SCHEMA,
                        "input_id": item.input_id,
                        "prompt_sha256": item.prompt_sha256,
                        "image_sha256": item.image.sha256,
                        "accepted": False,
                        "reason": f"resolver_error:{type(exc).__name__}",
                        "answer": None,
                        "certificate": None,
                        "certificate_sha256": None,
                        "certificate_projection_sha256": None,
                    }
                )
        rows.sort(key=lambda value: value["input_id"])
        result_path = staging / "results.jsonl"
        result_path.write_bytes(
            b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        )
        manifest = {
            "schema_version": BATCH_RUN_SCHEMA,
            "input_count": len(inputs),
            "input_jsonl_sha256": input_jsonl_sha256,
            "input_jsonl_size_bytes": input_jsonl_size_bytes,
            "ordered_input_projection_sha256": input_projection_sha256,
            "accepted_count": sum(bool(item["accepted"]) for item in rows),
            "abstained_count": sum(not bool(item["accepted"]) for item in rows),
            "inventory_projection_sha256": inventory.inventory_projection_sha256,
            "key_index_projection_sha256": key_index.key_index_projection_sha256,
            "render_manifest_projection_sha256": (
                render_manifest.render_manifest_projection_sha256
            ),
            "source_bundle": {
                "freeze_manifest_sha256": (
                    EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256
                ),
                "freeze_manifest_projection_sha256": (
                    EXPECTED_FROZEN_BUNDLE_MANIFEST_PROJECTION_SHA256
                ),
                "inventory_file_sha256": EXPECTED_INVENTORY_FILE_SHA256,
                "key_index_file_sha256": EXPECTED_KEY_INDEX_FILE_SHA256,
                "render_manifest_file_sha256": (
                    EXPECTED_RENDER_MANIFEST_FILE_SHA256
                ),
                "page_payloads_projection_sha256": (
                    EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256
                ),
                "exact_source_objects_attested_before_input_parse": True,
            },
            "v11_code_freeze": {
                "freeze_file_sha256": code_attestation.freeze_file_sha256,
                "freeze_projection_sha256": (
                    code_attestation.freeze_projection_sha256
                ),
                "code_projection_sha256": (
                    code_attestation.code_projection_sha256
                ),
                "code_file_count": code_attestation.code_file_count,
                "externally_pinned_before_input_parse": True,
            },
            "policy": {
                "task_id_is_policy_feature": False,
                "input_id_is_policy_feature": False,
                "observable_features": ["prompt", "image_bytes"],
                "repeated_image_sha_across_distinct_prompts_allowed": True,
                "gold_or_evaluation_access": False,
            },
            "artifacts": _artifact_hashes(staging),
        }
        manifest["run_projection_sha256"] = canonical_json_sha256(manifest)
        write_canonical_json(staging / "run_manifest.json", manifest)
        staging.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_mcq_opaque_batch(
    *,
    input_jsonl: Path,
    asset_root: Path,
    v11_freeze_manifest_path: Path,
    expected_v11_freeze_sha256: str,
    expected_v11_freeze_projection_sha256: str,
    freeze_manifest_path: Path,
    inventory_path: Path,
    key_index_path: Path,
    render_manifest_path: Path,
    page_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert_mcq_v11_code_freeze(
        freeze_manifest_path=v11_freeze_manifest_path,
        expected_freeze_sha256=expected_v11_freeze_sha256,
        expected_freeze_projection_sha256=(
            expected_v11_freeze_projection_sha256
        ),
    )
    bundle = assert_frozen_mcq_bundle(
        freeze_manifest_path=freeze_manifest_path,
        inventory_path=inventory_path,
        key_index_path=key_index_path,
        render_manifest_path=render_manifest_path,
        page_root=page_root,
    )
    input_jsonl = input_jsonl.resolve(strict=False)
    try:
        input_jsonl_bytes = input_jsonl.read_bytes()
    except OSError as exc:
        raise McqOpaqueBatchError("opaque JSONL cannot be read") from exc
    inputs = _parse_mcq_opaque_input_bytes(input_jsonl_bytes, asset_root)
    return execute_mcq_opaque_batch(
        inputs,
        bundle.inventory,
        bundle.render_manifest,
        bundle.key_index,
        output_dir,
        v11_freeze_manifest_path=v11_freeze_manifest_path,
        expected_v11_freeze_sha256=expected_v11_freeze_sha256,
        expected_v11_freeze_projection_sha256=(
            expected_v11_freeze_projection_sha256
        ),
        input_jsonl_bytes=input_jsonl_bytes,
        asset_root=asset_root,
    )
