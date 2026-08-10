#!/usr/bin/env python3
"""Evaluate predictions without silently treating manual Math tasks as wrong/right."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


REPORT = Path(os.environ.get("VLM_HOLDOUT_REPORT_DIR", Path(__file__).resolve().parents[1])).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def normalized_choice(value: object) -> str | None:
    text = str(value or "").strip().upper()
    direct = re.fullmatch(r"[ABCDE]", text)
    if direct:
        return direct.group(0)
    matches = re.findall(r"(?:^|\b)([ABCDE])(?:\b|$)", text)
    return matches[-1] if len(matches) == 1 else None


def reportability(
    *,
    manual_scored: int,
    manual_required: int,
    duplicates: list[str],
    unknown: list[str],
    missing: list[str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if manual_scored != manual_required:
        reasons.append(f"manual_math_incomplete:{manual_scored}/{manual_required}")
    if duplicates:
        reasons.append(f"duplicate_task_ids:{len(duplicates)}")
    if unknown:
        reasons.append(f"unknown_task_ids:{len(unknown)}")
    if missing:
        reasons.append(f"missing_task_ids:{len(missing)}")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="JSONL: task_id, answer; optional manual_score 0/1 for Math")
    parser.add_argument("--output", type=Path, default=REPORT / "evaluation.json")
    args = parser.parse_args()

    freeze = json.loads((REPORT / "freeze.json").read_text(encoding="utf-8"))
    manifest_path = REPORT / "selection_manifest.jsonl"
    if sha256_file(manifest_path) != freeze["manifest_sha256"]:
        raise RuntimeError("Frozen manifest hash mismatch")
    seal = json.loads((REPORT / "sealed" / "gold_seal.json").read_text(encoding="utf-8"))
    gold_path = REPORT / "sealed" / "sealed_gold.jsonl"
    if sha256_file(gold_path) != seal["sealed_gold_sha256"]:
        raise RuntimeError("Sealed gold hash mismatch")
    if seal["frozen_manifest_sha256"] != freeze["manifest_sha256"]:
        raise RuntimeError("Gold belongs to a different manifest")

    gold = {r["task_id"]: r for r in load_jsonl(gold_path)}
    predictions = load_jsonl(args.predictions)
    pred_map = {}
    duplicates = []
    for row in predictions:
        task_id = row["task_id"]
        if task_id in pred_map:
            duplicates.append(task_id)
        pred_map[task_id] = row
    unknown = sorted(set(pred_map) - set(gold))
    missing = sorted(set(gold) - set(pred_map))
    details = []
    automatic_correct = 0
    automatic_total = 0
    manual_correct = 0
    manual_scored = 0
    for task_id, target in sorted(gold.items()):
        pred = pred_map.get(task_id)
        if target["scoring_type"] == "exact_choice":
            automatic_total += 1
            value = normalized_choice(pred.get("answer") if pred else None)
            correct = value == target["official_answer"]
            automatic_correct += int(correct)
            details.append({"task_id": task_id, "scoring_type": "exact_choice", "prediction": value, "correct": correct})
        else:
            manual_score = pred.get("manual_score") if pred else None
            if manual_score in (0, 1, 0.0, 1.0):
                manual_scored += 1
                manual_correct += int(manual_score)
            details.append({"task_id": task_id, "scoring_type": "manual", "manual_score": manual_score, "correct": None})

    manual_required = 20
    reportable, invalid_reasons = reportability(
        manual_scored=manual_scored,
        manual_required=manual_required,
        duplicates=duplicates,
        unknown=unknown,
        missing=missing,
    )
    result = {
        "schema_version": "holdout80-evaluation-v1",
        "frozen_manifest_sha256": freeze["manifest_sha256"],
        "sealed_gold_sha256": seal["sealed_gold_sha256"],
        "prediction_file": str(args.predictions.resolve()),
        "prediction_sha256": sha256_file(args.predictions),
        "duplicates": duplicates,
        "unknown_task_ids": unknown,
        "missing_task_ids": missing,
        "automatic_mcq": {
            "correct": automatic_correct,
            "total": automatic_total,
            "accuracy": automatic_correct / automatic_total if automatic_total else None,
        },
        "manual_math": {
            "correct": manual_correct,
            "scored": manual_scored,
            "required": manual_required,
            "complete": manual_scored == manual_required,
        },
        "overall": {
            "correct": automatic_correct + manual_correct if reportable else None,
            "total": 80 if reportable else None,
            "accuracy": (automatic_correct + manual_correct) / 80 if reportable else None,
            "reportable": reportable,
            "invalid_reasons": invalid_reasons,
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("automatic_mcq", "manual_math", "overall", "missing_task_ids", "unknown_task_ids")}, ensure_ascii=False, indent=2))
    if not reportable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
