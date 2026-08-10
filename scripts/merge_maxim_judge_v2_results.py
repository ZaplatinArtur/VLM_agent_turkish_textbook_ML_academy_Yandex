"""Merge reusable and freshly computed judge-v2 rows without inventing verdicts.

The image-task input/template is the sole authority for membership and output
order.  Every expected task must occur exactly once in either the reusable or
fresh result file.  The utility intentionally has no solver/routing inputs: it
only validates and combines already-produced judge rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "maxim-judge-v2-merge-v1"


class MergeError(ValueError):
    """Raised when judge partitions cannot be merged losslessly."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MergeError(f"{label}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise MergeError(f"{label}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def _task_id(row: dict[str, Any], label: str, position: int) -> str:
    task_id = str(row.get("task_id") or "").strip()
    if not task_id:
        raise MergeError(f"{label}:{position}: missing task_id")
    return task_id


def index_rows(
    rows: Iterable[dict[str, Any]], label: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for position, row in enumerate(rows, 1):
        task_id = _task_id(row, label, position)
        if task_id in indexed:
            raise MergeError(f"{label}: duplicate task_id {task_id}")
        indexed[task_id] = row
        order.append(task_id)
    return indexed, order


def validate_judge_row(
    row: dict[str, Any], *, label: str, task_id: str, expected_prompt_version: str
) -> bool:
    prompt_version = row.get("prompt_version")
    if prompt_version != expected_prompt_version:
        raise MergeError(
            f"{label}: task {task_id} has prompt_version={prompt_version!r}; "
            f"expected {expected_prompt_version!r}"
        )

    if row.get("error"):
        raise MergeError(f"{label}: task {task_id} has error={row['error']!r}")
    judge = row.get("judge")
    if isinstance(judge, dict) and judge.get("error"):
        raise MergeError(
            f"{label}: task {task_id} has judge.error={judge['error']!r}"
        )

    verdict = row.get("verdict")
    if not isinstance(verdict, dict):
        raise MergeError(f"{label}: task {task_id} has no verdict object")
    strict_correct = verdict.get("strict_correct")
    if not isinstance(strict_correct, bool):
        raise MergeError(
            f"{label}: task {task_id} has no boolean verdict.strict_correct"
        )
    return strict_correct


def merge_rows(
    *,
    template_rows: list[dict[str, Any]],
    reusable_rows: list[dict[str, Any]],
    fresh_rows: list[dict[str, Any]],
    expected_prompt_version: str = "judge-v2",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and merge rows in exact template order.

    The returned rows are the original judge row objects in a new list.  No
    fallback verdict is generated for a missing task.
    """

    _, template_order = index_rows(template_rows, "image template")
    reusable, reusable_order = index_rows(reusable_rows, "reusable judge")
    fresh, fresh_order = index_rows(fresh_rows, "fresh judge")

    expected = set(template_order)
    reusable_ids = set(reusable_order)
    fresh_ids = set(fresh_order)

    overlap = reusable_ids & fresh_ids
    if overlap:
        raise MergeError(
            "judge partitions overlap for task_id(s): " + ", ".join(sorted(overlap))
        )

    unexpected = (reusable_ids | fresh_ids) - expected
    if unexpected:
        raise MergeError(
            "judge partitions contain unexpected task_id(s): "
            + ", ".join(sorted(unexpected))
        )

    missing = expected - (reusable_ids | fresh_ids)
    if missing:
        raise MergeError(
            "judge partitions are incomplete; missing task_id(s): "
            + ", ".join(sorted(missing))
        )

    strict_by_source = {"reusable": 0, "fresh": 0}
    for source_label, indexed in (("reusable", reusable), ("fresh", fresh)):
        for task_id, row in indexed.items():
            strict_by_source[source_label] += int(
                validate_judge_row(
                    row,
                    label=f"{source_label} judge",
                    task_id=task_id,
                    expected_prompt_version=expected_prompt_version,
                )
            )

    combined = [reusable.get(task_id, fresh.get(task_id)) for task_id in template_order]
    # Completeness above makes None impossible.  Keep this assertion local so a
    # future refactor cannot silently serialize a fabricated/empty verdict.
    if any(row is None for row in combined):  # pragma: no cover - defensive guard
        raise MergeError("internal error: an expected judge row was not selected")
    merged_rows = [row for row in combined if row is not None]

    summary = {
        "expected_prompt_version": expected_prompt_version,
        "rows": len(merged_rows),
        "template_order_preserved": True,
        "partition": {
            "reusable_rows": len(reusable),
            "fresh_rows": len(fresh),
            "overlap_rows": 0,
            "missing_rows": 0,
            "unexpected_rows": 0,
        },
        "strict_correct": {
            "reusable": strict_by_source["reusable"],
            "fresh": strict_by_source["fresh"],
            "total": sum(strict_by_source.values()),
        },
    }
    return merged_rows, summary


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _validate_output_paths(inputs: list[Path], outputs: list[Path]) -> None:
    resolved_inputs = {path.resolve() for path in inputs}
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise MergeError("output paths must be distinct")
    for output in outputs:
        if output.resolve() in resolved_inputs:
            raise MergeError(f"output path aliases an input: {output}")
        if output.exists():
            raise MergeError(f"refusing to overwrite existing output: {output}")


def _atomic_write_many(items: list[tuple[Path, bytes]]) -> None:
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for destination, data in items:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=destination.name + ".",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_paths.append((Path(temporary.name), destination))
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in temporary_paths:
            if temporary.exists():
                temporary.unlink()


def build_from_paths(
    *,
    template_path: Path,
    reusable_path: Path,
    fresh_path: Path,
    out_jsonl: Path,
    out_manifest: Path,
    out_sha256: Path,
    expected_prompt_version: str = "judge-v2",
) -> dict[str, Any]:
    input_paths = [template_path, reusable_path, fresh_path]
    output_paths = [out_jsonl, out_manifest, out_sha256]
    _validate_output_paths(input_paths, output_paths)

    rows, summary = merge_rows(
        template_rows=read_jsonl(template_path, "image template"),
        reusable_rows=read_jsonl(reusable_path, "reusable judge"),
        fresh_rows=read_jsonl(fresh_path, "fresh judge"),
        expected_prompt_version=expected_prompt_version,
    )
    output_data = jsonl_bytes(rows)
    output_hash = sha256_bytes(output_data)
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operation": "merge-only; no solver or routing inputs are read",
        "sources": {
            "image_template": {
                "path": str(template_path.resolve()),
                "sha256": sha256_file(template_path),
            },
            "reusable_judge_v2": {
                "path": str(reusable_path.resolve()),
                "sha256": sha256_file(reusable_path),
            },
            "fresh_judge_v2": {
                "path": str(fresh_path.resolve()),
                "sha256": sha256_file(fresh_path),
            },
        },
        "validation": summary,
        "output": {
            "path": str(out_jsonl.resolve()),
            "sha256": output_hash,
            "rows": len(rows),
        },
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
    }
    manifest_data = json_bytes(manifest)
    manifest_hash = sha256_bytes(manifest_data)
    checksum_data = (
        f"{output_hash}  {out_jsonl.name}\n"
        f"{manifest_hash}  {out_manifest.name}\n"
    ).encode("utf-8")

    _atomic_write_many(
        [
            (out_jsonl, output_data),
            (out_manifest, manifest_data),
            (out_sha256, checksum_data),
        ]
    )
    return {
        "output": str(out_jsonl.resolve()),
        "output_sha256": output_hash,
        "manifest": str(out_manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "sha256_file": str(out_sha256.resolve()),
        "sha256_file_sha256": sha256_file(out_sha256),
        "validation": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge reusable and fresh judge-v2 results in canonical image-template "
            "order, failing on every missing, duplicate, overlapping or invalid row."
        )
    )
    parser.add_argument("--image-template", type=Path, required=True)
    parser.add_argument("--reusable-judge", type=Path, required=True)
    parser.add_argument("--fresh-judge", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-sha256", type=Path, required=True)
    parser.add_argument("--expected-prompt-version", default="judge-v2")
    args = parser.parse_args(argv)

    report = build_from_paths(
        template_path=args.image_template,
        reusable_path=args.reusable_judge,
        fresh_path=args.fresh_judge,
        out_jsonl=args.out_jsonl,
        out_manifest=args.out_manifest,
        out_sha256=args.out_sha256,
        expected_prompt_version=args.expected_prompt_version,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
