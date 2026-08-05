#!/usr/bin/env python3
"""Project benchmark metadata to source locators only.

The downstream exact-source resolver never opens the original metadata file.
This boundary copies only the opaque alignment key and public source URL; all
answer, image-answer, class, and evaluation fields are structurally absent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


SCHEMA = "maxim-official-source-locator-projection-v2"


class ProjectionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project(input_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProjectionError(f"invalid JSON at {input_path}:{line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ProjectionError(f"{input_path}:{line_number}: expected object")
            task_id = str(raw.get("task_id") or "").strip()
            source_url = str(raw.get("source") or "").strip()
            if not task_id or task_id in seen:
                raise ProjectionError(f"{input_path}:{line_number}: missing/duplicate task_id")
            if not source_url:
                raise ProjectionError(f"{input_path}:{line_number}: missing public source URL")
            seen.add(task_id)
            rows.append({"source_url": source_url, "task_id": task_id})
    if not rows:
        raise ProjectionError("source metadata is empty")
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
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "projected_fields": ["task_id", "source_url"],
        "task_id_used_for_alignment_only": True,
        "answer_or_outcome_fields_propagated": False,
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
        result = project(args.input.resolve(), args.output.resolve(), args.manifest.resolve())
    except (ProjectionError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
