#!/usr/bin/env python3
"""Freeze source-only provenance for the primary-layout number projection."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_os.official_ogm import (  # noqa: E402
    OfficialSourceError,
    canonical_json_bytes,
    parser_observation_allow_missing_number,
    parser_observation_primary_layout_number,
    sha256_file,
    strict_activity_label_number_text,
)
from evidence_os.official_workbook import (  # noqa: E402
    activity_label_projection_enabled,
    validate_fail_closed_workbook_policy,
)


_MARKER = re.compile(r"^\s*(?:#{1,6}\s*)?(\d{1,3})\s*(?:[.)]|-(?=\s))")
_EXAMPLE_MARKER = re.compile(
    r"^\s*(?:#{1,6}\s*)?\u00d6rnek\s+([1-9]\d{0,2})\s*[.):]?\s*$",
    re.IGNORECASE,
)


class ProjectionAuditError(RuntimeError):
    """Raised when the frozen source-only projection contract is violated."""


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProjectionAuditError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ProjectionAuditError(f"non-object JSONL row {line_number}: {path}")
        rows.append(value)
    return rows


def _primary_evidence(record: Mapping[str, Any], expected_number: int) -> dict[str, Any]:
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
        raise ProjectionAuditError("projected row has malformed image evidence")
    image = images[0]
    width = int(image["width"])
    height = int(image["height"])
    blocks = image.get("parsing_res_list")
    if not isinstance(blocks, list):
        raise ProjectionAuditError("projected row has malformed block evidence")
    candidates = [
        block
        for block in blocks
        if isinstance(block, Mapping)
        and str(block.get("block_label") or "").casefold() != "image"
        and isinstance(block.get("block_order"), int)
        and not isinstance(block.get("block_order"), bool)
        and block.get("block_order") == 1
    ]
    if len(candidates) != 1:
        raise ProjectionAuditError("primary projection lacks one unique order-one block")
    block = candidates[0]
    content = str(block.get("block_content") or "").strip()
    marker = _MARKER.match(content)
    if marker is None or int(marker.group(1)) != expected_number:
        raise ProjectionAuditError("primary projection marker does not match observation")
    bbox = block.get("block_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ProjectionAuditError("primary projection lacks a four-value bbox")
    normalized_bbox = [
        round(float(bbox[0]) / width, 8),
        round(float(bbox[1]) / height, 8),
        round(float(bbox[2]) / width, 8),
        round(float(bbox[3]) / height, 8),
    ]
    evidence: dict[str, Any] = {
        "block_label": str(block.get("block_label") or "").casefold(),
        "block_order": 1,
        "marker": expected_number,
        "normalized_bbox": normalized_bbox,
    }
    block_id = block.get("block_id")
    if isinstance(block_id, (str, int)) and not isinstance(block_id, bool):
        evidence["block_id"] = block_id
    return evidence


def _primary_example_evidence(
    record: Mapping[str, Any], expected_number: int
) -> dict[str, Any]:
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
        raise ProjectionAuditError("example projection has malformed image evidence")
    image = images[0]
    width = int(image["width"])
    height = int(image["height"])
    blocks = image.get("parsing_res_list")
    if not isinstance(blocks, list):
        raise ProjectionAuditError("example projection has malformed block evidence")
    candidates = [
        block
        for block in blocks
        if isinstance(block, Mapping)
        and str(block.get("block_label") or "").casefold() != "image"
        and isinstance(block.get("block_order"), int)
        and not isinstance(block.get("block_order"), bool)
        and block.get("block_order") == 1
    ]
    if len(candidates) != 1:
        raise ProjectionAuditError("example projection lacks one unique order-one block")
    block = candidates[0]
    if str(block.get("block_label") or "").casefold() != "paragraph_title":
        raise ProjectionAuditError("example projection is not a paragraph title")
    marker = _EXAMPLE_MARKER.fullmatch(
        str(block.get("block_content") or "").strip()
    )
    if marker is None or int(marker.group(1)) != expected_number:
        raise ProjectionAuditError("example projection marker does not match observation")
    bbox = block.get("block_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ProjectionAuditError("example projection lacks a four-value bbox")
    normalized_bbox = [
        round(float(bbox[0]) / width, 8),
        round(float(bbox[1]) / height, 8),
        round(float(bbox[2]) / width, 8),
        round(float(bbox[3]) / height, 8),
    ]
    return {
        "block_label": "paragraph_title",
        "block_order": 1,
        "marker_kind": "example_label",
        "marker": expected_number,
        "normalized_bbox": normalized_bbox,
    }


def _primary_activity_evidence(
    record: Mapping[str, Any], expected_number: int
) -> dict[str, Any]:
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
        raise ProjectionAuditError("activity projection has malformed image evidence")
    image = images[0]
    width = int(image["width"])
    height = int(image["height"])
    blocks = image.get("parsing_res_list")
    if not isinstance(blocks, list):
        raise ProjectionAuditError("activity projection has malformed block evidence")
    candidates = [
        block
        for block in blocks
        if isinstance(block, Mapping)
        and str(block.get("block_label") or "").casefold() != "image"
        and isinstance(block.get("block_order"), int)
        and not isinstance(block.get("block_order"), bool)
        and block.get("block_order") == 1
    ]
    if len(candidates) != 1:
        raise ProjectionAuditError("activity projection lacks one unique order-one block")
    block = candidates[0]
    if str(block.get("block_label") or "").casefold() != "paragraph_title":
        raise ProjectionAuditError("activity projection is not a paragraph title")
    if strict_activity_label_number_text(
        str(block.get("block_content") or "").strip()
    ) != expected_number:
        raise ProjectionAuditError("activity projection marker does not match observation")
    bbox = block.get("block_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ProjectionAuditError("activity projection lacks a four-value bbox")
    normalized_bbox = [
        round(float(bbox[0]) / width, 8),
        round(float(bbox[1]) / height, 8),
        round(float(bbox[2]) / width, 8),
        round(float(bbox[3]) / height, 8),
    ]
    return {
        "block_label": "paragraph_title",
        "block_order": 1,
        "marker_kind": "activity_label",
        "marker": expected_number,
        "terminal_s_canonicalized_to_5": (
            str(block.get("block_content") or "").strip().casefold().endswith("-s")
        ),
        "normalized_bbox": normalized_bbox,
    }


def audit(profile_path: Path, output_dir: Path) -> dict[str, Any]:
    profile = _load_json(profile_path)
    if profile.get("schema_version") != "maxim-public-workbook-profile-v1":
        raise ProjectionAuditError("unsupported profile schema")
    policy = profile.get("policy")
    if not isinstance(policy, dict):
        raise ProjectionAuditError("profile policy is missing")
    try:
        number_projection, _allow_example_label = (
            validate_fail_closed_workbook_policy(policy)
        )
        allow_activity_label = activity_label_projection_enabled(policy)
    except OfficialSourceError as exc:
        raise ProjectionAuditError(
            f"profile fail-closed policy is invalid: {exc}"
        ) from exc
    if number_projection != "primary_layout_then_unique_v1":
        raise ProjectionAuditError("profile does not freeze primary-layout projection")
    example_projection = str(policy.get("example_label_projection") or "disabled")
    if example_projection not in {
        "disabled",
        "primary_paragraph_title_order_one_v1",
    }:
        raise ProjectionAuditError("profile has an unsupported example-label projection")
    activity_projection = str(policy.get("activity_label_projection") or "disabled")
    if activity_projection not in {
        "disabled",
        "primary_paragraph_title_order_one_v1",
    }:
        raise ProjectionAuditError("profile has an unsupported activity-label projection")
    decision_schema = (
        "maxim-primary-layout-projection-decision-v4"
        if allow_activity_label
        else "maxim-primary-layout-projection-decision-v3"
    )
    inputs = profile.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("parser_observations"), dict):
        raise ProjectionAuditError("profile parser input is missing")
    parser_spec = inputs["parser_observations"]
    parser_path = _path(str(parser_spec.get("path") or ""))
    expected_parser_sha = str(parser_spec.get("sha256") or "")
    actual_parser_sha = sha256_file(parser_path)
    if actual_parser_sha != expected_parser_sha:
        raise ProjectionAuditError("parser input hash mismatch")

    parser_rows = _load_jsonl(parser_path)
    expected_rows = int(profile.get("expected_rows") or 0)
    if len(parser_rows) != expected_rows:
        raise ProjectionAuditError("parser row-count mismatch")
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    methods: Counter[str] = Counter()
    for raw in parser_rows:
        task_id = str(raw.get("task_id") or "").strip()
        try:
            legacy = parser_observation_allow_missing_number(raw)
            projected = parser_observation_primary_layout_number(raw)
        except OfficialSourceError as exc:
            if not task_id:
                raise ProjectionAuditError("parser-error row has no task_id") from exc
            if task_id in seen:
                raise ProjectionAuditError(f"duplicate task_id: {task_id}") from exc
            seen.add(task_id)
            method = "parser_error_abstain"
            decisions.append(
                {
                    "schema_version": decision_schema,
                    "task_id": task_id,
                    "method": method,
                    "error": str(exc),
                    "legacy_question_number": None,
                    "projected_question_number": None,
                    "projected_example_label_number": None,
                    **(
                        {"projected_activity_label_number": None}
                        if allow_activity_label
                        else {}
                    ),
                }
            )
            methods[method] += 1
            continue
        task_id = projected.task_id
        if task_id in seen:
            raise ProjectionAuditError(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        marker_values = tuple(
            value
            for value in (
                projected.question_number,
                projected.primary_example_label_number,
                projected.primary_activity_label_number
                if allow_activity_label
                else None,
            )
            if value is not None
        )
        if len(marker_values) > 1:
            method = "source_marker_conflict_abstain"
        elif (
            projected.question_number is None
            and projected.primary_example_label_number is not None
            and example_projection == "primary_paragraph_title_order_one_v1"
        ):
            method = "primary_example_label"
        elif (
            projected.question_number is None
            and projected.primary_example_label_number is None
            and projected.primary_activity_label_number is not None
            and allow_activity_label
        ):
            method = "primary_activity_label"
        elif projected.question_number is None:
            method = "abstain_no_unique_number"
        elif projected.primary_example_label_number is not None:
            method = "numeric_marker_precedence"
        elif projected.question_number == legacy.question_number:
            method = "legacy_unique_or_same_primary"
        elif legacy.question_number is None:
            method = "primary_layout_recovery"
        else:
            method = "primary_layout_override"
        decision: dict[str, Any] = {
            "schema_version": decision_schema,
            "task_id": task_id,
            "method": method,
            "legacy_question_number": legacy.question_number,
            "projected_question_number": projected.question_number,
            "projected_example_label_number": (
                projected.primary_example_label_number
            ),
        }
        if allow_activity_label:
            decision["projected_activity_label_number"] = (
                projected.primary_activity_label_number
            )
        if method.startswith("primary_layout_"):
            if projected.question_number is None:
                raise ProjectionAuditError("primary method has no projected number")
            decision["evidence"] = _primary_evidence(raw, projected.question_number)
        elif method == "primary_example_label":
            assert projected.primary_example_label_number is not None
            decision["evidence"] = _primary_example_evidence(
                raw,
                projected.primary_example_label_number,
            )
        elif method == "primary_activity_label":
            assert projected.primary_activity_label_number is not None
            decision["evidence"] = _primary_activity_evidence(
                raw,
                projected.primary_activity_label_number,
            )
        decisions.append(decision)
        methods[method] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "projection_decisions.jsonl"
    decisions_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in decisions))
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": (
            "maxim-primary-layout-projection-audit-v4"
            if allow_activity_label
            else "maxim-primary-layout-projection-audit-v3"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": {"path": str(profile_path.resolve()), "sha256": sha256_file(profile_path)},
        "parser_observations": {"path": str(parser_path.resolve()), "sha256": actual_parser_sha},
        "implementation": {
            "official_ogm": {
                "path": str((ROOT / "src/evidence_os/official_ogm.py").resolve()),
                "sha256": sha256_file(ROOT / "src/evidence_os/official_ogm.py"),
            },
            "audit_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        },
        "rows": len(decisions),
        "methods": dict(sorted(methods.items())),
        "projection_decisions": {
            "path": str(decisions_path.resolve()),
            "sha256": sha256_file(decisions_path),
        },
        "gold_access": False,
        "benchmark_candidate_or_outcome_access": False,
        "task_id_used_for_alignment_only": True,
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return {**manifest, "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.profile_json, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
