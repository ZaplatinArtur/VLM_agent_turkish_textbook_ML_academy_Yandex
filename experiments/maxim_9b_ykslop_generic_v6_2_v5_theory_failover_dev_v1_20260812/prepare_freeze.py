"""Prepare the content-only fallback artifact and pre-outcome execution freeze."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
V5_DIR = HERE.parent / "maxim_9b_ykslop_generic_content_pipeline_v5_20260811"
V6_DIR = HERE.parent / "maxim_9b_ykslop_generic_reasoning_nonstream_alt_v6_2_siliconflow_dev_20260812"
V5_PUBLIC = V5_DIR / "frozen" / "benchmark_public_dev.jsonl"
V5_PREDICTIONS = V5_DIR / "runs" / "final_predictions.jsonl"
V5_FREEZE = V5_DIR / "DEV_EXECUTION_FREEZE.json"
V5_COMPLETION = V5_DIR / "DEV_WAVE_COMPLETION.json"
V6_QUEUE = V6_DIR / "frozen" / "queue_public_content_only.jsonl"
V6_FREEZE = V6_DIR / "DEV_EXECUTION_FREEZE.json"
V6_PREDICTIONS = V6_DIR / "runs" / "predictions_content_only.jsonl"
V6_COMPLETION = V6_DIR / "DEV_WAVE_COMPLETION.json"
FALLBACK = HERE / "frozen" / "v5_theory_fallback_content_only.jsonl"
FREEZE = HERE / "DEV_FAILOVER_FREEZE.json"
FREEZE_SHA = HERE / "DEV_FAILOVER_FREEZE_SHA256.txt"


EXPECTED = {
    V5_FREEZE: "5d4786e4d5289f0a58612b78af765e1d108b9b656fb5289ee658e3c6a8ef2f3d",
    V5_COMPLETION: "6a95b8d77bef5e20c3d73bc5dee6d1f8f1f5cb327ba9c4e8480d9fff14a4602a",
    V5_PREDICTIONS: "9b260c69143a37f12a84044f251fa07db3f17758b44f40144743297c49a4177d",
    V5_PUBLIC: "eebfda230a10ef98f07c53b5d7ab55cca24a718c8439074192d4f29156acc47c",
    V6_FREEZE: "c1b275c985489a8fd1e534d93138c00c288c91b22c735793bc5b7c24b726f099",
    V6_QUEUE: "b2b03dfb53218e7e099c5129c4ef6acd096d8c44e44b2e82a37c84c746e549a2",
}


class FreezeError(RuntimeError):
    pass


def data(path: Path) -> bytes:
    return path.read_bytes()


def sha(path: Path) -> str:
    return hashlib.sha256(data(path)).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line.decode("utf-8")) for line in data(path).splitlines() if line.strip()]
    if any(type(row) is not dict for row in rows):
        raise FreezeError(f"non-object JSONL row: {path}")
    return rows


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.resolve().relative_to(HERE.resolve()).as_posix(),
        "sha256": sha(path),
        "size": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = rows
    return value


def assert_pre_outcome_absence() -> None:
    forbidden = [
        V6_PREDICTIONS,
        V6_COMPLETION,
        HERE / "runs" / "predictions_content_only.jsonl",
        HERE / "DEV_WAVE_COMPLETION.json",
        HERE / "DEV_RESULT_PRIVATE.json",
    ]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise FreezeError(f"pre-outcome freeze guard failed; outputs already exist: {present}")


def build_fallback() -> list[dict[str, Any]]:
    public = jsonl(V5_PUBLIC)
    v6_queue = jsonl(V6_QUEUE)
    predictions = jsonl(V5_PREDICTIONS)
    if not (len(public) == len(v6_queue) == 185 and len(predictions) == 370):
        raise FreezeError("source denominator mismatch")
    by_outer_alignment: dict[tuple[str, str], dict[str, Any]] = {}
    for row in predictions:
        opaque = row.get("benchmark_id")
        arm = row.get("arm")
        if type(opaque) is not str or arm not in {"no_context", "local_textbook_theory_bm25"}:
            raise FreezeError("V5 prediction alignment contract mismatch")
        key = (opaque, arm)
        if key in by_outer_alignment:
            raise FreezeError("duplicate V5 prediction alignment key")
        by_outer_alignment[key] = row
    fallbacks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for public_row, queue_row in zip(public, v6_queue):
        opaque = public_row.get("benchmark_id")
        if type(opaque) is not str:
            raise FreezeError("public outer alignment key mismatch")
        if any(public_row.get(key) != queue_row.get(key) for key in ("question", "choices", "subject", "has_figure")):
            raise FreezeError("V5 public/V6 content order mismatch")
        prediction = by_outer_alignment.get((opaque, "local_textbook_theory_bm25"))
        if type(prediction) is not dict:
            raise FreezeError("missing V5 theory prediction")
        answer = prediction.get("final_answer")
        content_hash = queue_row.get("content_sha256")
        if answer not in tuple("ABCDE") or type(answer) is not str:
            raise FreezeError("invalid V5 theory fallback answer")
        if type(content_hash) is not str or len(content_hash) != 64 or content_hash in seen:
            raise FreezeError("V6 content hash contract mismatch")
        if prediction.get("gold_access") is not False:
            raise FreezeError("V5 runtime gold guard mismatch")
        seen.add(content_hash)
        fallbacks.append(
            {
                "schema_version": "generic-v5-theory-content-fallback-v1",
                "content_sha256": content_hash,
                "prediction": answer,
                "source_arm": "local_textbook_theory_bm25",
                "gold_access": False,
                "final_access": False,
                "opaque_identifier_retained": False,
            }
        )
    return fallbacks


def main() -> None:
    if any(path.exists() for path in (FALLBACK, FREEZE, FREEZE_SHA)):
        raise FreezeError("freeze artifacts already exist")
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise FreezeError(f"lineage hash mismatch: {path}")
    assert_pre_outcome_absence()
    fallbacks = build_fallback()
    write_new(FALLBACK, b"".join(canonical(row) for row in fallbacks))
    freeze = {
        "schema_version": "generic-v6-v5-theory-failover-dev-freeze-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "state": "pre_outcome_frozen_unexecuted",
        "scope": "full_185_public_DEV_content_only",
        "row_count": 185,
        "v6_execution_freeze_sha256": EXPECTED[V6_FREEZE],
        "selection_rule": {
            "primary": "V6.2 answer only for an exact strict valid successful row",
            "fallback": "fixed V5 local_textbook_theory_bm25 final answer",
            "fallback_on": ["missing", "duplicate content hash", "schema mismatch", "terminal error", "invalid answer"],
            "content_disagreement_arbitration": False,
            "confidence_used": False,
            "gold_or_outcome_used": False,
            "task_identifier_used": False,
            "outer_alignment_only_during_fallback_materialization": True,
        },
        "success_criterion": {"correct_at_least": 148, "total": 185, "accuracy_at_least": 0.8},
        "lineage": {
            "v5_execution_freeze_sha256": EXPECTED[V5_FREEZE],
            "v5_completion_sha256": EXPECTED[V5_COMPLETION],
            "v5_predictions_sha256": EXPECTED[V5_PREDICTIONS],
            "v5_public_source_sha256": EXPECTED[V5_PUBLIC],
            "v6_execution_freeze_sha256": EXPECTED[V6_FREEZE],
            "v6_queue_sha256": EXPECTED[V6_QUEUE],
        },
        "artifacts": {
            "v5_theory_fallback": artifact(FALLBACK, 185),
            "rule": artifact(HERE / "failover_rule.py"),
            "runner": artifact(HERE / "run_failover.py"),
            "private_scorer": artifact(HERE / "score_dev_private.py"),
            "tests": artifact(HERE / "test_failover_rule.py"),
            "dry_run": artifact(HERE / "DRY_RUN.json"),
        },
        "pre_outcome_guards": {
            "v6_atomic_predictions_absent": True,
            "v6_completion_absent": True,
            "ensemble_outputs_absent": True,
            "final80_opened": False,
            "gold_opened_by_rule_or_runner": False,
            "v6_partial_cache_inspected": False,
        },
        "planned_outputs_absent": ["runs/predictions_content_only.jsonl", "DEV_WAVE_COMPLETION.json", "DEV_RESULT_PRIVATE.json"],
    }
    assert_pre_outcome_absence()
    write_new(FREEZE, canonical(freeze))
    freeze_hash = sha(FREEZE)
    write_new(FREEZE_SHA, (freeze_hash + "\n").encode("ascii"))
    print(freeze_hash)


if __name__ == "__main__":
    main()
