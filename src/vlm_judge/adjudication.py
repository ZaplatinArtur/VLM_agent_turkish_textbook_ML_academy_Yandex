from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .schema import JudgeVerdict


_LABEL_BY_SCORE = {
    0: "incorrect",
    1: "partially_correct",
    2: "partially_correct",
    3: "mostly_correct",
    4: "fully_correct",
}


def record_key(value: dict[str, Any]) -> str:
    """Return the stable pointwise key shared by tasks, humans, and judge runs."""

    explicit = value.get("annotation_id") or value.get("pair_id")
    if explicit:
        return str(explicit)
    task_id = str(value.get("task_id") or "").strip()
    setup = str(value.get("setup") or "unknown").strip()
    return f"{task_id}::{setup}"


def adjudication_id(value: dict[str, Any]) -> str:
    return f"adj::{record_key(value)}"


def _stable_fraction(key: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _is_pointwise(task: dict[str, Any]) -> bool:
    return task.get("candidate_a") is None and task.get("candidate_b") is None


def build_adjudication_context(
    tasks: Iterable[dict[str, Any]],
    judge_results: Iterable[dict[str, Any]],
    human_annotations: Iterable[dict[str, Any]],
    adjudications: Iterable[dict[str, Any]] = (),
    *,
    low_confidence_threshold: float = 0.75,
    agreement_sample_rate: float = 0.10,
    agreement_sample_seed: str = "adjudication-control-v1",
) -> dict[str, Any]:
    """Build a reproducible queue of judge/human disagreements and QC controls.

    Only completed pointwise human annotations are eligible. Every disagreement,
    judge failure, reference issue, and low-confidence verdict is queued. A stable
    hash sample of exact agreements is included as a control against confirmation
    bias during adjudication.
    """

    if not 0.0 <= low_confidence_threshold <= 1.0:
        raise ValueError("low_confidence_threshold must be between 0 and 1")
    if not 0.0 <= agreement_sample_rate <= 1.0:
        raise ValueError("agreement_sample_rate must be between 0 and 1")
    if not str(agreement_sample_seed).strip():
        raise ValueError("agreement_sample_seed must not be empty")

    task_list = [dict(value) for value in tasks]
    judge_list = [dict(value) for value in judge_results]
    human_list = [dict(value) for value in human_annotations]
    adjudication_list = [dict(value) for value in adjudications]

    judge_by_key: dict[str, dict[str, Any]] = {}
    duplicate_judge_keys: set[str] = set()
    for value in judge_list:
        key = record_key(value)
        if key in judge_by_key:
            duplicate_judge_keys.add(key)
        judge_by_key[key] = value

    human_by_key = {
        record_key(value): value
        for value in human_list
        if str(value.get("status") or "") == "complete"
    }
    adjudication_by_id = {
        str(value.get("adjudication_id") or ""): value
        for value in adjudication_list
        if value.get("adjudication_id")
    }

    stats: dict[str, Any] = {
        "total_tasks": len(task_list),
        "pointwise_tasks": 0,
        "matched_judge": 0,
        "completed_human": 0,
        "eligible": 0,
        "queue_items": 0,
        "exact_score_disagreements": 0,
        "strict_disagreements": 0,
        "judge_errors": 0,
        "low_confidence": 0,
        "reference_issues": 0,
        "agreement_controls": 0,
        "resolved": 0,
        "duplicate_judge_keys": len(duplicate_judge_keys),
    }
    queue: list[dict[str, Any]] = []

    for task in task_list:
        if not _is_pointwise(task):
            continue
        stats["pointwise_tasks"] += 1
        key = record_key(task)
        judge_result = judge_by_key.get(key)
        human = human_by_key.get(key)
        if judge_result is not None:
            stats["matched_judge"] += 1
        if human is not None:
            stats["completed_human"] += 1
        if judge_result is None or human is None:
            continue
        stats["eligible"] += 1

        reasons: list[str] = []
        priority = 0
        raw_verdict = judge_result.get("verdict") if isinstance(judge_result.get("verdict"), dict) else None
        verdict_error: str | None = None
        if raw_verdict is not None:
            try:
                verdict = JudgeVerdict.from_dict(raw_verdict).to_dict()
            except (KeyError, TypeError, ValueError) as exc:
                verdict = None
                verdict_error = f"invalid_verdict: {exc}"
        else:
            verdict = None
        judge_meta = judge_result.get("judge") if isinstance(judge_result.get("judge"), dict) else {}
        judge_error = judge_meta.get("error") or verdict_error or (None if verdict is not None else "missing_verdict")
        if judge_error:
            reasons.append("judge_error")
            priority += 100
            stats["judge_errors"] += 1

        human_score = human.get("score")
        judge_score = verdict.get("score") if verdict else None
        score_gap: int | None = None
        if isinstance(human_score, int) and isinstance(judge_score, int):
            score_gap = abs(human_score - judge_score)
            if score_gap:
                reasons.append("score_disagreement")
                priority += 20 + score_gap * 10
                stats["exact_score_disagreements"] += 1
            human_strict = human.get("strict_correct")
            if not isinstance(human_strict, bool):
                human_strict = human_score == 4
            judge_strict = verdict.get("strict_correct")
            if isinstance(judge_strict, bool) and human_strict != judge_strict:
                reasons.append("strict_disagreement")
                priority += 80
                stats["strict_disagreements"] += 1

        confidence = verdict.get("confidence") if verdict else None
        if isinstance(confidence, (int, float)) and confidence < low_confidence_threshold:
            reasons.append("low_judge_confidence")
            priority += 30
            stats["low_confidence"] += 1

        reference_issue = bool(human.get("reference_quality_issue")) or bool(
            verdict and verdict.get("reference_quality_issue")
        )
        if reference_issue:
            reasons.append("reference_quality_issue")
            priority += 90
            stats["reference_issues"] += 1

        is_exact_agreement = (
            not judge_error
            and isinstance(human_score, int)
            and isinstance(judge_score, int)
            and human_score == judge_score
            and not reference_issue
        )
        if (
            not reasons
            and is_exact_agreement
            and _stable_fraction(key, agreement_sample_seed) < agreement_sample_rate
        ):
            reasons.append("agreement_control")
            priority += 10
            stats["agreement_controls"] += 1

        if not reasons:
            continue

        current_adjudication_id = adjudication_id(task)
        existing = adjudication_by_id.get(current_adjudication_id)
        if existing and existing.get("status") == "resolved":
            stats["resolved"] += 1

        display_judge_result = dict(judge_result)
        display_judge_result["verdict"] = verdict
        display_judge_meta = dict(judge_meta)
        if judge_error:
            display_judge_meta["error"] = str(judge_error)
        display_judge_result["judge"] = display_judge_meta
        enriched = dict(task)
        enriched["_adjudication"] = {
            "adjudication_id": current_adjudication_id,
            "record_key": key,
            "priority": priority,
            "reasons": reasons,
            "score_gap": score_gap,
            "human": human,
            "judge_result": display_judge_result,
        }
        queue.append(enriched)

    queue.sort(
        key=lambda value: (
            -int(value["_adjudication"]["priority"]),
            str(value.get("task_id") or ""),
            str(value.get("setup") or ""),
        )
    )
    stats["queue_items"] = len(queue)
    return {
        "enabled": bool(judge_list),
        "items": queue,
        "stats": stats,
        "config": {
            "low_confidence_threshold": low_confidence_threshold,
            "agreement_sample_rate": agreement_sample_rate,
            "agreement_sample_seed": agreement_sample_seed,
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} of {path} is not an object")
            values.append(value)
    return values


class AdjudicationStore:
    STATUSES = {"draft", "resolved", "skipped"}
    DECISIONS = {"human", "judge", "custom", "exclude"}
    ISSUE_SOURCES = {"none", "human", "judge", "reference", "candidate", "pipeline"}

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        for record in _load_jsonl(path):
            key = str(record.get("adjudication_id") or "").strip()
            if key:
                self._records[key] = record

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._records.values()]

    def upsert(self, value: dict[str, Any]) -> dict[str, Any]:
        task_id = str(value.get("task_id") or "").strip()
        identifier = str(value.get("adjudication_id") or "").strip()
        if not task_id or not identifier:
            raise ValueError("task_id and adjudication_id are required")
        status = str(value.get("status") or "draft")
        decision = str(value.get("decision") or "").strip() or None
        issue_source = str(value.get("issue_source") or "none")
        if status not in self.STATUSES:
            raise ValueError(f"invalid adjudication status: {status}")
        if decision is not None and decision not in self.DECISIONS:
            raise ValueError(f"invalid adjudication decision: {decision}")
        if issue_source not in self.ISSUE_SOURCES:
            raise ValueError(f"invalid issue_source: {issue_source}")
        if status == "resolved" and decision is None:
            raise ValueError("resolved adjudication requires a decision")

        final_score = value.get("final_score")
        if final_score in ("", None):
            final_score = None
        elif isinstance(final_score, bool) or not isinstance(final_score, int) or not 0 <= final_score <= 4:
            raise ValueError("final_score must be an integer between 0 and 4")
        if status == "resolved" and decision != "exclude" and final_score is None:
            raise ValueError("resolved non-excluded adjudication requires final_score")
        if decision == "exclude":
            final_score = None

        record = dict(value)
        record.update(
            {
                "task_id": task_id,
                "adjudication_id": identifier,
                "status": status,
                "decision": decision,
                "issue_source": issue_source,
                "final_score": final_score,
                "final_label": _LABEL_BY_SCORE.get(final_score),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        with self._lock:
            self._records[identifier] = record
            self._write_atomic()
        return dict(record)

    def _write_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for key in sorted(self._records):
                handle.write(json.dumps(self._records[key], ensure_ascii=False) + "\n")
        os.replace(temporary, self.path)

    def export_csv(self) -> bytes:
        fields = [
            "adjudication_id",
            "task_id",
            "setup",
            "status",
            "decision",
            "final_score",
            "final_label",
            "issue_source",
            "queue_reasons",
            "human_score",
            "judge_score",
            "rationale",
            "adjudicator",
            "updated_at",
        ]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for value in self.list():
            record = dict(value)
            if isinstance(record.get("queue_reasons"), list):
                record["queue_reasons"] = ";".join(record["queue_reasons"])
            writer.writerow(record)
        return ("\ufeff" + output.getvalue()).encode("utf-8")
