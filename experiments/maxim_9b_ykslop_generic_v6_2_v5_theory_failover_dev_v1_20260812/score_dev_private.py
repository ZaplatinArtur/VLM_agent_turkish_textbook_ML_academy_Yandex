"""Private aggregate-only scorer for the frozen failover candidate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PRIVATE_FREEZE = HERE / "DEV_PRIVATE_SCORE_FREEZE.json"
PRIVATE_FREEZE_SHA = HERE / "DEV_PRIVATE_SCORE_FREEZE_SHA256.txt"
EXECUTION_FREEZE = HERE / "DEV_FAILOVER_FREEZE.json"
COMPLETION = HERE / "DEV_WAVE_COMPLETION.json"
PUBLIC_SOURCE = HERE.parent / "maxim_9b_ykslop_generic_content_pipeline_v5_20260811" / "frozen" / "benchmark_public_dev.jsonl"
RESULT = HERE / "DEV_RESULT_PRIVATE.json"


class ScoreError(RuntimeError):
    pass


def stable_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(stable_bytes(path))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(stable_bytes(path).decode("utf-8"))
    if type(value) is not dict:
        raise ScoreError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(stable_bytes(path).splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScoreError(f"invalid JSONL row {number}: {path}") from exc
        if type(value) is not dict:
            raise ScoreError(f"non-object JSONL row {number}: {path}")
        rows.append(value)
    return rows


def verify_descriptor(value: Any, *, base: Path) -> Path:
    if type(value) is not dict or set(value) != {"path", "rows", "sha256", "size"}:
        raise ScoreError("artifact descriptor mismatch")
    path = (base / value["path"]).resolve()
    data = stable_bytes(path)
    if len(data) != value["size"] or sha256_bytes(data) != value["sha256"]:
        raise ScoreError("artifact hash/size mismatch")
    if len([line for line in data.splitlines() if line.strip()]) != value["rows"]:
        raise ScoreError("artifact row mismatch")
    return path


def exclusive_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    if RESULT.exists():
        raise ScoreError("private score output already exists")
    if sha256_file(PRIVATE_FREEZE) != args.expected_private_freeze_sha256:
        raise ScoreError("private score freeze external pin mismatch")
    if PRIVATE_FREEZE_SHA.read_text(encoding="ascii").strip() != args.expected_private_freeze_sha256:
        raise ScoreError("private score freeze sidecar mismatch")
    private = read_json(PRIVATE_FREEZE)
    if (
        private.get("schema_version") != "generic-failover-private-score-freeze-v1"
        or private.get("state") != "private_score_frozen_unexecuted"
        or private.get("execution_freeze_sha256") != args.expected_execution_freeze_sha256
    ):
        raise ScoreError("private score freeze closure mismatch")
    if sha256_file(EXECUTION_FREEZE) != args.expected_execution_freeze_sha256:
        raise ScoreError("execution freeze external pin mismatch")
    scorer = private.get("artifacts", {}).get("scorer")
    if (
        type(scorer) is not dict
        or (HERE / scorer.get("path", "")).resolve() != Path(__file__).resolve()
        or scorer.get("sha256") != sha256_file(Path(__file__))
        or scorer.get("size") != Path(__file__).stat().st_size
    ):
        raise ScoreError("private scorer self-closure mismatch")
    gold_path = verify_descriptor(private["artifacts"]["development_gold"], base=HERE)
    if sha256_file(COMPLETION) != args.expected_completion_sha256:
        raise ScoreError("completion external pin mismatch")
    completion = read_json(COMPLETION)
    if (
        completion.get("schema_version") != "generic-v6-v5-theory-failover-dev-completion-v1"
        or completion.get("freeze_sha256") != args.expected_execution_freeze_sha256
        or completion.get("rows") != 185
        or completion.get("gold_accessed") is not False
        or completion.get("final_accessed") is not False
        or completion.get("opaque_identifier_access") is not False
    ):
        raise ScoreError("completion guard mismatch")
    predictions_path = verify_descriptor(completion["predictions"], base=HERE)
    predictions = read_jsonl(predictions_path)
    public = read_jsonl(PUBLIC_SOURCE)
    gold_rows = read_jsonl(gold_path)
    if not (len(predictions) == len(public) == len(gold_rows) == 185):
        raise ScoreError("full denominator closure mismatch")
    gold: dict[str, str] = {}
    for row in gold_rows:
        opaque, answer = row.get("benchmark_id"), row.get("answer")
        if type(opaque) is not str or opaque in gold or answer not in tuple("ABCDE"):
            raise ScoreError("private gold contract mismatch")
        gold[opaque] = answer
    public_ids = [row.get("benchmark_id") for row in public]
    if any(type(value) is not str for value in public_ids) or set(public_ids) != set(gold):
        raise ScoreError("private gold/public ID set mismatch")
    correct = 0
    for prediction, opaque in zip(predictions, public_ids):
        answer = prediction.get("prediction")
        if answer not in tuple("ABCDE") or type(answer) is not str:
            raise ScoreError("failover must preserve a valid full denominator")
        correct += int(answer == gold[opaque])
    result = {
        "schema_version": "generic-failover-private-dev-result-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution_freeze_sha256": args.expected_execution_freeze_sha256,
        "private_score_freeze_sha256": args.expected_private_freeze_sha256,
        "completion_sha256": args.expected_completion_sha256,
        "correct": correct,
        "total": 185,
        "accuracy": correct / 185,
        "success_correct_at_least": 148,
        "threshold_met": correct >= 148,
        "selection_counts": completion["selection_counts"],
        "task_ids_present": False,
        "gold_answers_present": False,
        "per_row_outcomes_present": False,
        "final_data_opened": False,
    }
    exclusive_json(RESULT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-private-freeze-sha256", required=True)
    parser.add_argument("--expected-execution-freeze-sha256", required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
