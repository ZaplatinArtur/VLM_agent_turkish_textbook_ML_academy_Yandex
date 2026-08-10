from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


LEGACY_TASK_ID = re.compile(r"validation_sheet1_r(\d{4})")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_by_task(
    rows: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"{label} contains a row without task_id")
        if task_id in indexed:
            raise ValueError(f"duplicate task_id in {label}: {task_id}")
        indexed[task_id] = row
    return indexed


def canonical_legacy_id(task_id: str) -> str:
    match = LEGACY_TASK_ID.fullmatch(task_id)
    if match is None:
        raise ValueError(f"unexpected legacy task_id: {task_id}")
    return f"val_{match.group(1)}"


def prefixed(value: Any, prefix: str) -> str | None:
    text = str(value or "").strip()
    return f"{prefix.rstrip('/')}/{text}" if text else None


def is_image_reference(row: dict[str, Any]) -> bool:
    return (
        str(row.get("reference_kind") or "").strip().casefold() == "image"
        or (
            not str(row.get("reference_answer") or "").strip()
            and bool(str(row.get("reference_image_path") or "").strip())
        )
    )


def canonical_image_row(
    source: dict[str, Any],
    app_row: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    reference_image = prefixed(source.get("reference_image_path"), prefix)
    question_image = prefixed(source.get("question_image_path"), prefix)
    if not question_image or not reference_image:
        raise ValueError(
            f"image-reference task {app_row['task_id']} is missing an image asset path"
        )
    # Asset references come from the authoritative image manifests. Subject,
    # grade and answer type come from the common 274-row scoring manifest so
    # they match the frozen control's judge payload exactly.
    return {
        "task_id": str(app_row["task_id"]),
        "subject": app_row.get("subject"),
        "grade": app_row.get("grade"),
        "question_image_path": question_image,
        "answer_type": app_row.get("answer_type"),
        "reference_answer": "",
        "reference_image_path": reference_image,
        "reference_kind": "image",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the canonical 97 image-reference tasks used by the frozen "
            "full274 scorer (80 legacy image rows plus 17 delta rows)."
        )
    )
    parser.add_argument("--app-manifest", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--delta-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--legacy-prefix", default="validation_v2_274")
    parser.add_argument("--delta-prefix", default="validation_join")
    parser.add_argument("--expected-common", type=int, default=274)
    parser.add_argument("--expected-legacy-images", type=int, default=80)
    parser.add_argument("--expected-delta-images", type=int, default=17)
    args = parser.parse_args()

    app_rows = read_jsonl(args.app_manifest)
    app_by_id = unique_by_task(app_rows, "app manifest")
    if len(app_rows) != args.expected_common:
        raise ValueError(
            f"expected {args.expected_common} common tasks, found {len(app_rows)}"
        )

    results_rows = read_jsonl(args.results)
    results_by_id = unique_by_task(results_rows, "results")
    missing_results = sorted(set(app_by_id) - set(results_by_id))
    unknown_results = sorted(set(results_by_id) - set(app_by_id))
    if missing_results or unknown_results:
        raise ValueError(
            "results must contain exactly the common benchmark IDs; "
            f"missing={missing_results[:10]}, unknown={unknown_results[:10]}"
        )

    image_by_id: dict[str, dict[str, Any]] = {}
    legacy_image_count = 0
    for source in read_jsonl(args.legacy_manifest):
        if not is_image_reference(source):
            continue
        task_id = canonical_legacy_id(str(source["task_id"]))
        if task_id not in app_by_id:
            raise ValueError(f"legacy image task is outside common benchmark: {task_id}")
        if task_id in image_by_id:
            raise ValueError(f"duplicate canonical image task: {task_id}")
        image_by_id[task_id] = canonical_image_row(
            source,
            app_by_id[task_id],
            prefix=args.legacy_prefix,
        )
        legacy_image_count += 1

    delta_image_count = 0
    for source in read_jsonl(args.delta_manifest):
        task_id = str(source.get("task_id") or "").strip()
        if not task_id or task_id not in app_by_id:
            raise ValueError(f"delta image task is outside common benchmark: {task_id}")
        if not is_image_reference(source):
            raise ValueError(f"delta task is not an image reference: {task_id}")
        if task_id in image_by_id:
            raise ValueError(f"duplicate canonical image task: {task_id}")
        image_by_id[task_id] = canonical_image_row(
            source,
            app_by_id[task_id],
            prefix=args.delta_prefix,
        )
        delta_image_count += 1

    if legacy_image_count != args.expected_legacy_images:
        raise ValueError(
            f"expected {args.expected_legacy_images} legacy image tasks, "
            f"found {legacy_image_count}"
        )
    if delta_image_count != args.expected_delta_images:
        raise ValueError(
            f"expected {args.expected_delta_images} delta image tasks, "
            f"found {delta_image_count}"
        )

    ordered_ids = [str(row["task_id"]) for row in app_rows if row["task_id"] in image_by_id]
    expected_images = args.expected_legacy_images + args.expected_delta_images
    if len(ordered_ids) != expected_images or len(set(ordered_ids)) != expected_images:
        raise ValueError(
            f"expected {expected_images} unique canonical image tasks, "
            f"found {len(ordered_ids)}"
        )
    if "val_0058" not in image_by_id:
        raise ValueError("canonical exception val_0058 is missing from the image route")

    write_jsonl(args.output_manifest, [image_by_id[task_id] for task_id in ordered_ids])
    write_jsonl(args.output_results, [results_by_id[task_id] for task_id in ordered_ids])

    report = {
        "common_tasks": len(app_rows),
        "deterministic_tasks": len(app_rows) - len(ordered_ids),
        "image_tasks": len(ordered_ids),
        "legacy_image_tasks": legacy_image_count,
        "delta_image_tasks": delta_image_count,
        "canonical_exception": "val_0058",
        "first_image_task": ordered_ids[0],
        "last_image_task": ordered_ids[-1],
        "output_manifest": str(args.output_manifest),
        "output_manifest_sha256": sha256(args.output_manifest),
        "output_results": str(args.output_results),
        "output_results_sha256": sha256(args.output_results),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
