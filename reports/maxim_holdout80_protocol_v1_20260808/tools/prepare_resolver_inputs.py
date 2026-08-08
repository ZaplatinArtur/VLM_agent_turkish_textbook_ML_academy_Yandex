#!/usr/bin/env python3
"""Create opaque, task-id-free inputs for a frozen resolver replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


REPORT = Path(os.environ.get("VLM_HOLDOUT_REPORT_DIR", Path(__file__).resolve().parents[1])).resolve()
WORKSPACE = Path(os.environ.get("VLM_HOLDOUT_WORKSPACE", Path(__file__).resolve().parents[3])).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["math12", "mcq", "all"], default="all")
    args = parser.parse_args()
    manifest_path = REPORT / "selection_manifest.jsonl"
    freeze = json.loads((REPORT / "freeze.json").read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != freeze["manifest_sha256"]:
        raise RuntimeError("Frozen manifest hash mismatch")
    rows = load_jsonl(manifest_path)
    if args.family == "math12":
        rows = [row for row in rows if row["source_family"] == "math12_beceri"]
    elif args.family == "mcq":
        rows = [row for row in rows if row["task_format"] == "multiple_choice_ABCDE"]

    output_dir = REPORT / "resolver_inputs"
    asset_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    public_rows = []
    private_map = []
    for row in rows:
        identity_payload = {"question_asset_sha256": row["question_asset_sha256"], "prompt": row["prompt"]}
        input_id = "input-" + hashlib.sha256(canonical(identity_payload).encode("utf-8")).hexdigest()[:20]
        images = []
        for index, (path_text, expected) in enumerate(zip(row["question_assets"], row["question_asset_sha256"]), start=1):
            source = (WORKSPACE / path_text).resolve()
            destination = asset_dir / f"{input_id}-{index:02d}{source.suffix.casefold()}"
            if not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            if sha256_file(destination) != expected:
                raise RuntimeError(f"Opaque input asset hash mismatch: {destination}")
            images.append({"path": destination.relative_to(REPORT).as_posix(), "sha256": expected})
        public_rows.append({
            "schema_version": "holdout80-opaque-resolver-input-v1",
            "input_id": input_id,
            "language": row["language"],
            "prompt": row["prompt"],
            "images": images,
            "expected_response_format": "single_choice_ABCDE" if row["task_format"] == "multiple_choice_ABCDE" else "numbered_multi_part_solution",
        })
        private_map.append({"input_id": input_id, "task_id": row["task_id"]})
    public_rows.sort(key=lambda row: row["input_id"])
    private_map.sort(key=lambda row: row["input_id"])
    public_path = output_dir / f"{args.family}.jsonl"
    map_path = REPORT / "sealed" / f"resolver_input_map_{args.family}.jsonl"
    public_path.write_text("".join(canonical(row) + "\n" for row in public_rows), encoding="utf-8")
    map_path.write_text("".join(canonical(row) + "\n" for row in private_map), encoding="utf-8")
    seal = {
        "schema_version": "holdout80-opaque-resolver-input-seal-v1",
        "family_partition": args.family,
        "count": len(public_rows),
        "frozen_manifest_sha256": freeze["manifest_sha256"],
        "public_inputs": public_path.relative_to(REPORT).as_posix(),
        "public_inputs_sha256": sha256_file(public_path),
        "private_task_map": map_path.relative_to(REPORT).as_posix(),
        "private_task_map_sha256": sha256_file(map_path),
        "forbidden_resolver_fields": ["task_id", "source_family", "source_pdf", "activity_id", "unit", "question_pages", "official_answer"],
    }
    seal_path = output_dir / f"{args.family}.seal.json"
    seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(seal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
