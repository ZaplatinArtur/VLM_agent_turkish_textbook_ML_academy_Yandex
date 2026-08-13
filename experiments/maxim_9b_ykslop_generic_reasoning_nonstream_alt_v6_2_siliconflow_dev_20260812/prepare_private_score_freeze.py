"""Prepare the DEV private scorer freeze without opening private gold.

The gold descriptor was attested and frozen before this candidate existed.  This
preparation step records that exact descriptor as an external pin; the private
scorer verifies and opens it only after a complete public-only run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from nonstream_protocol import exclusive_bytes, exclusive_json, sha256_file, utc_now


HERE = Path(__file__).resolve().parent
GOLD = HERE.parent / "maxim_9b_ykslop_no_overlap_theory_openrouter_v1_20260811" / "frozen" / "development_gold_private.jsonl"
SCORER = HERE / "score_dev_private.py"
EXECUTION_FREEZE = HERE / "DEV_EXECUTION_FREEZE.json"
OUTPUT = HERE / "DEV_PRIVATE_SCORE_FREEZE.json"
SIDECAR = HERE / "DEV_PRIVATE_SCORE_FREEZE_SHA256.txt"
KNOWN_GOLD_SHA256 = "e723c025d298a890f706f97773ccaa87b9d5bd0c7935ef9fbfada12963a18793"
KNOWN_GOLD_SIZE = 18315


def descriptor(path: Path, *, rows: int | None = None) -> dict:
    value = {
        "path": Path(os.path.relpath(path, HERE)).as_posix(),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = rows
    return value


def main() -> None:
    if OUTPUT.exists() or SIDECAR.exists():
        raise RuntimeError("private score freeze already exists")
    execution_sha = sha256_file(EXECUTION_FREEZE)
    value = {
        "schema_version": "generic-medium-nonstream-private-score-freeze-v6",
        "state": "private_score_frozen_unexecuted",
        "created_utc": utc_now(),
        "execution_freeze_sha256": execution_sha,
        "artifacts": {
            "development_gold": {
                "path": Path(os.path.relpath(GOLD, HERE)).as_posix(),
                "sha256": KNOWN_GOLD_SHA256,
                "size": KNOWN_GOLD_SIZE,
                "rows": 185,
            },
            "scorer": descriptor(SCORER),
        },
        "verification": {
            "preparer": descriptor(Path(__file__)),
        },
        "scoring_contract": {
            "denominator": 185,
            "success_correct_at_least": 148,
            "success_accuracy_at_least": 0.8,
            "malformed_missing_or_error": "wrong",
            "aggregate_only": True,
            "task_ids_in_result": False,
            "gold_answers_in_result": False,
            "per_row_outcomes_in_result": False,
            "threshold_met_in_result": True,
        },
        "guards": {
            "gold_opened_during_freeze": False,
            "no_scoring_before_full_completion": True,
            "runtime_does_not_import_private_scorer": True,
            "final_data_absent": True,
            "prior_outcomes_absent": True,
        },
    }
    exclusive_json(OUTPUT, value)
    digest = sha256_file(OUTPUT)
    exclusive_bytes(SIDECAR, (digest + "\n").encode("ascii"))
    print(json.dumps({"private_score_freeze_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
