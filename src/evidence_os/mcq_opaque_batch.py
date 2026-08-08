"""Fail-closed batch runner for task-ID-free full-page MCQ inputs.

Repeated page-image bytes are expected: one official textbook page can contain
several questions.  The policy feature is the observable ``(prompt, image)``
pair; ``input_id`` is used only to align outputs and is never passed to the
resolver.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import tempfile
from typing import Any

from .mcq_fullpage_source import (
    McqInventory,
    McqKeyIndex,
    McqRenderManifest,
    McqSourceCertificate,
    McqSourceError,
    load_mcq_inventory,
    load_mcq_key_index,
    load_mcq_render_manifest,
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
BATCH_RUN_SCHEMA = "mcq-fullpage-opaque-source-batch-run-v1"
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


ResolveMcq = Callable[
    [str, bytes, McqInventory, McqRenderManifest, McqKeyIndex],
    McqSourceCertificate,
]


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
        lines = input_jsonl.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise McqOpaqueBatchError("opaque JSONL cannot be read") from exc
    if not lines or any(not line.strip() for line in lines):
        raise McqOpaqueBatchError("opaque JSONL is empty or contains blank lines")

    seen_ids: set[str] = set()
    seen_observations: set[tuple[str, str]] = set()
    inputs: list[McqOpaqueInput] = []
    for line_number, line in enumerate(lines, start=1):
        value = _strict_line(line, line_number)
        _reject_forbidden(value)
        if set(value) - _ALLOWED_INPUT_KEYS:
            raise McqOpaqueBatchError(f"unknown opaque input field at line {line_number}")
        schema = str(value.get("schema_version") or "")
        if schema not in INPUT_SCHEMA_ALLOWLIST:
            raise McqOpaqueBatchError(f"unknown opaque schema at line {line_number}")
        input_id = str(value.get("input_id") or "")
        if (
            _SAFE_INPUT_ID.fullmatch(input_id) is None
            or input_id in {".", ".."}
            or input_id.casefold() in seen_ids
        ):
            raise McqOpaqueBatchError(f"unsafe or duplicate input_id at line {line_number}")
        seen_ids.add(input_id.casefold())
        prompt = str(value.get("prompt") or "")
        parse_observable_mcq_prompt(prompt)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        language = str(value.get("language") or "")
        if language not in {"tr", "tur", "Turkish"}:
            raise McqOpaqueBatchError("MCQ opaque language is not Turkish")
        response_format = str(value.get("expected_response_format") or "")
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
        expected_sha = str(raw_image.get("sha256") or "")
        if _HEX64.fullmatch(expected_sha) is None:
            raise McqOpaqueBatchError("opaque image lacks a SHA-256 pin")
        raw_path = str(raw_image.get("path") or "")
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
    resolver: ResolveMcq = _default_resolver,
) -> dict[str, Any]:
    """Resolve each opaque prompt/image pair without exposing alignment IDs."""

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
                certificate = resolver(
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
            "accepted_count": sum(bool(item["accepted"]) for item in rows),
            "abstained_count": sum(not bool(item["accepted"]) for item in rows),
            "inventory_projection_sha256": inventory.inventory_projection_sha256,
            "key_index_projection_sha256": key_index.key_index_projection_sha256,
            "render_manifest_projection_sha256": (
                render_manifest.render_manifest_projection_sha256
            ),
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
    inventory_path: Path,
    key_index_path: Path,
    render_manifest_path: Path,
    page_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    inventory = load_mcq_inventory(inventory_path)
    key_index = load_mcq_key_index(key_index_path, inventory)
    render_manifest = load_mcq_render_manifest(
        render_manifest_path, inventory, page_root=page_root
    )
    inputs = load_mcq_opaque_inputs(input_jsonl, asset_root)
    return execute_mcq_opaque_batch(
        inputs, inventory, render_manifest, key_index, output_dir
    )
