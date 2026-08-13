"""Freeze the private aggregate scorer without opening private gold."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXECUTION_FREEZE = HERE / "DEV_FAILOVER_FREEZE.json"
EXECUTION_SHA = HERE / "DEV_FAILOVER_FREEZE_SHA256.txt"
SCORER = HERE / "score_dev_private.py"
REFERENCE_PRIVATE_FREEZE = HERE.parent / "maxim_9b_ykslop_generic_reasoning_nonstream_alt_v6_2_siliconflow_dev_20260812" / "DEV_PRIVATE_SCORE_FREEZE.json"
OUTPUT = HERE / "DEV_PRIVATE_SCORE_FREEZE.json"
OUTPUT_SHA = HERE / "DEV_PRIVATE_SCORE_FREEZE_SHA256.txt"


class FreezeError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    if OUTPUT.exists() or OUTPUT_SHA.exists():
        raise FreezeError("private score freeze already exists")
    execution_hash = sha(EXECUTION_FREEZE)
    if EXECUTION_SHA.read_text(encoding="ascii").strip() != execution_hash:
        raise FreezeError("execution freeze sidecar mismatch")
    reference = json.loads(REFERENCE_PRIVATE_FREEZE.read_text(encoding="utf-8"))
    gold = reference.get("artifacts", {}).get("development_gold")
    if type(gold) is not dict or set(gold) != {"path", "rows", "sha256", "size"} or gold.get("rows") != 185:
        raise FreezeError("reference private gold descriptor mismatch")
    relative_gold = Path("../maxim_9b_ykslop_no_overlap_theory_openrouter_v1_20260811/frozen/development_gold_private.jsonl")
    scorer = {"path": "score_dev_private.py", "sha256": sha(SCORER), "size": SCORER.stat().st_size}
    value = {
        "schema_version": "generic-failover-private-score-freeze-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "state": "private_score_frozen_unexecuted",
        "execution_freeze_sha256": execution_hash,
        "artifacts": {
            "development_gold": {
                "path": relative_gold.as_posix(),
                "rows": gold["rows"],
                "sha256": gold["sha256"],
                "size": gold["size"],
            },
            "scorer": scorer,
        },
        "scoring_contract": {
            "aggregate_only": True,
            "denominator": 185,
            "success_correct_at_least": 148,
            "success_accuracy_at_least": 0.8,
            "task_ids_in_result": False,
            "gold_answers_in_result": False,
            "per_row_outcomes_in_result": False,
        },
        "guards": {
            "gold_opened_during_freeze": False,
            "final_data_opened": False,
            "no_scoring_before_full_completion": True,
            "runtime_does_not_import_private_scorer": True,
        },
    }
    write_new(OUTPUT, canonical(value))
    output_hash = sha(OUTPUT)
    write_new(OUTPUT_SHA, (output_hash + "\n").encode("ascii"))
    print(output_hash)


if __name__ == "__main__":
    main()
