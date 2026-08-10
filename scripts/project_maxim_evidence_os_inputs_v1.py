#!/usr/bin/env python3
"""Stage narrow, gold-free public solver projections for Evidence OS.

This command is the only component allowed to inspect legacy solver structure.
It rejects positive/unknown gold attestations and every evaluation-like key,
then emits only fields used by the observable adapters.  The inference
composer consumes the projection, never the legacy artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


SCHEMA = "maxim-evidence-os-public-projection-v1"
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_FORBIDDEN = frozenset(
    {
        "accuracy",
        "adjudication",
        "correct",
        "correctness",
        "evaluation",
        "gold",
        "groundtruth",
        "judge",
        "metric",
        "oracle",
        "outcome",
        "reference",
        "reward",
        "score",
        "verdict",
    }
)
_TOP_LEVEL_COPY = (
    "answer",
    "condition",
    "error",
    "final_answer",
    "forced_answer",
    "model",
    "prediction",
    "prompt_version",
)
_GENERATION_COPY = (
    "active_crops",
    "calculator_sympy",
    "confidence",
    "evidence_citations",
    "selection_evidence",
)


class ProjectionError(RuntimeError):
    pass


def _components(key: str) -> tuple[str, ...]:
    split = _CAMEL_BOUNDARY.sub("_", key).casefold()
    return tuple(part for part in _NON_ALNUM.split(split) if part)


def _scan(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ProjectionError(f"non-string key at {'.'.join(path) or '<root>'}")
            components = _components(raw_key)
            compact = "".join(components)
            if compact == "goldaccess":
                if child is not False:
                    raise ProjectionError(
                        f"{'.'.join(path + (raw_key,))} must be exactly false"
                    )
                # A negative legacy attestation is stripped, wherever an old
                # composer put it.  It never enters the public projection.
                continue
            if any(token in _FORBIDDEN for token in components):
                raise ProjectionError(
                    f"forbidden evaluation key at {'.'.join(path + (raw_key,))}"
                )
            if compact in {"answerkey", "groundtruth", "iscorrect", "taskscore"}:
                raise ProjectionError(
                    f"forbidden evaluation key at {'.'.join(path + (raw_key,))}"
                )
            _scan(child, path + (raw_key,))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _scan(child, path + (f"[{index}]",))


def _sanitize_selected(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Copy JSON-shaped selected fields while stripping negative attestations."""

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, child in value.items():
            compact = "".join(_components(str(key)))
            if compact == "goldaccess":
                if child is not False:
                    raise ProjectionError(
                        f"{'.'.join(path + (str(key),))} must be exactly false"
                    )
                continue
            output[str(key)] = _sanitize_selected(child, path + (str(key),))
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_selected(child, path + ("[]",)) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ProjectionError(f"unsupported value type at {'.'.join(path)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project(input_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProjectionError(f"invalid JSON {input_path}:{line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ProjectionError(f"{input_path}:{line_number}: expected object")
            task_id = str(raw.get("task_id") or "").strip()
            if not task_id or task_id in seen:
                raise ProjectionError(f"{input_path}:{line_number}: missing/duplicate task_id")
            seen.add(task_id)
            without_alignment_key = {key: value for key, value in raw.items() if key != "task_id"}
            _scan(without_alignment_key)

            row: dict[str, Any] = {"task_id": task_id}
            for key in _TOP_LEVEL_COPY:
                if key in raw:
                    row[key] = _sanitize_selected(raw[key], (key,))
            raw_generation = raw.get("generation")
            projected_generation: dict[str, Any] = {"gold_access": False}
            if isinstance(raw_generation, Mapping):
                for key in _GENERATION_COPY:
                    if key in raw_generation:
                        projected_generation[key] = _sanitize_selected(
                            raw_generation[key], ("generation", key)
                        )
            row["generation"] = projected_generation
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(output_path)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "complete": bool(rows),
        "gold_access": False,
        "score_or_judge_access": False,
        "task_id_used_for_alignment_only": True,
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "top_level_fields": list(_TOP_LEVEL_COPY),
        "generation_fields": list(_GENERATION_COPY),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = project(args.input.resolve(), args.output.resolve(), args.manifest.resolve())
    except (ProjectionError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
