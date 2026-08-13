"""Private aggregate-only scorer for the frozen 185-row DEV candidate.

This module is never imported by the runtime runner.  It emits no task IDs,
gold answers, per-row outcomes, fixes, or regressions.  Missing/malformed/
transport-error predictions count as wrong on the full denominator of 185.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from nonstream_protocol import exclusive_json, read_json, sha256_bytes, sha256_file, stable_bytes, utc_now


HERE = Path(__file__).resolve().parent
PRIVATE_FREEZE = HERE / "DEV_PRIVATE_SCORE_FREEZE.json"
PRIVATE_FREEZE_SHA = HERE / "DEV_PRIVATE_SCORE_FREEZE_SHA256.txt"
EXECUTION_FREEZE = HERE / "DEV_EXECUTION_FREEZE.json"
COMPLETION = HERE / "DEV_WAVE_COMPLETION.json"
QUEUE = HERE / "frozen" / "queue_public_content_only.jsonl"
PUBLIC_SOURCE = HERE.parent / "maxim_9b_ykslop_generic_content_pipeline_v5_20260811" / "frozen" / "benchmark_public_dev.jsonl"
RESULT = HERE / "DEV_RESULT_PRIVATE.json"


class ScoreError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def verify_descriptor(value: Any) -> Path:
    if type(value) is not dict or set(value) != {"path", "rows", "sha256", "size"}:
        raise ScoreError("private artifact descriptor mismatch")
    path = (HERE / value["path"]).resolve()
    data = stable_bytes(path)
    if len(data) != value["size"] or sha256_bytes(data) != value["sha256"]:
        raise ScoreError("private artifact hash/size mismatch")
    if len([line for line in data.splitlines() if line.strip()]) != value["rows"]:
        raise ScoreError("private artifact row mismatch")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    if RESULT.exists():
        raise ScoreError("private score output already exists")
    if sha256_file(PRIVATE_FREEZE) != args.expected_private_freeze_sha256:
        raise ScoreError("private score freeze external pin mismatch")
    if PRIVATE_FREEZE_SHA.read_text(encoding="ascii").strip() != args.expected_private_freeze_sha256:
        raise ScoreError("private score freeze sidecar mismatch")
    private = read_json(PRIVATE_FREEZE)
    if (
        private.get("schema_version") != "generic-medium-nonstream-private-score-freeze-v6"
        or private.get("state") != "private_score_frozen_unexecuted"
        or private.get("execution_freeze_sha256") != args.expected_execution_freeze_sha256
        or set(private.get("artifacts", {})) != {"development_gold", "scorer"}
    ):
        raise ScoreError("private score freeze closure mismatch")
    gold_path = verify_descriptor(private["artifacts"]["development_gold"])
    scorer_path = Path(private["artifacts"]["scorer"]["path"])
    scorer_absolute = (HERE / scorer_path).resolve()
    if (
        scorer_absolute != Path(__file__).resolve()
        or sha256_file(scorer_absolute) != private["artifacts"]["scorer"]["sha256"]
        or scorer_absolute.stat().st_size != private["artifacts"]["scorer"]["size"]
    ):
        raise ScoreError("private scorer self-closure mismatch")
    if sha256_file(EXECUTION_FREEZE) != args.expected_execution_freeze_sha256:
        raise ScoreError("execution freeze external pin mismatch")
    execution = read_json(EXECUTION_FREEZE)
    if execution.get("source_public_sha256") != sha256_file(PUBLIC_SOURCE):
        raise ScoreError("public ancestry source mismatch")
    if sha256_file(COMPLETION) != args.expected_completion_sha256:
        raise ScoreError("completion external pin mismatch")
    completion = read_json(COMPLETION)
    if (
        completion.get("freeze_sha256") != args.expected_execution_freeze_sha256
        or completion.get("rows") != 185
        or completion.get("gold_accessed") is not False
        or completion.get("final_accessed") is not False
        or completion.get("runtime_opaque_ids") is not False
    ):
        raise ScoreError("completion guard mismatch")
    predictions_path = (HERE / completion["predictions"]["path"]).resolve()
    if (
        sha256_file(predictions_path) != completion["predictions"]["sha256"]
        or predictions_path.stat().st_size != completion["predictions"]["size"]
    ):
        raise ScoreError("prediction artifact mismatch")
    predictions = read_jsonl(predictions_path)
    queue = read_jsonl(QUEUE)
    public = read_jsonl(PUBLIC_SOURCE)
    gold_rows = read_jsonl(gold_path)
    if not (len(predictions) == len(queue) == len(public) == len(gold_rows) == 185):
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

    correct = malformed = transport_errors = 0
    subjects: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for prediction, queue_row, public_row, opaque in zip(predictions, queue, public, public_ids):
        if prediction.get("content_sha256") != queue_row.get("content_sha256"):
            raise ScoreError("prediction/public content order mismatch")
        answer = prediction.get("prediction")
        valid = answer in "ABCDE" if type(answer) is str else False
        is_correct = valid and answer == gold[opaque]
        correct += int(is_correct)
        malformed += int(not valid)
        transport_errors += int(prediction.get("terminal_success") is not True)
        subject = public_row.get("subject")
        if type(subject) is not str:
            raise ScoreError("public subject contract mismatch")
        subjects[subject]["total"] += 1
        subjects[subject]["correct"] += int(is_correct)
    result = {
        "schema_version": "generic-medium-nonstream-private-dev-result-v6",
        "created_utc": utc_now(),
        "execution_freeze_sha256": args.expected_execution_freeze_sha256,
        "private_score_freeze_sha256": args.expected_private_freeze_sha256,
        "completion_sha256": args.expected_completion_sha256,
        "correct": correct,
        "total": 185,
        "accuracy": correct / 185,
        "success_correct_at_least": 148,
        "threshold_met": correct >= 148,
        "malformed_or_missing": malformed,
        "terminal_transport_errors": transport_errors,
        "subject_metrics": {
            subject: {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": counts["correct"] / counts["total"],
            }
            for subject, counts in sorted(subjects.items())
        },
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
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
