from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class HoldoutIntegrityError(RuntimeError):
    """Raised when the public Holdout80 summary is modified or inconsistent."""


SUMMARY_FILE = Path(__file__).with_name("holdout80_verified_summary.json")
# Canonical JSON projection: stable across Git LF/CRLF checkout policies.
# Updated only together with an independently reviewed public aggregate.
EXPECTED_SUMMARY_PROJECTION_SHA256 = (
    "b0cd611d25c649059ae93ad7d4e5cff2c95ac8d30fbf0ef7f1acefd0639d02ef"
)


@dataclass(frozen=True)
class Score:
    correct: int
    total: int
    label: str

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True)
class SubjectScore:
    subject: str
    raw_correct: int
    raw_total: int
    valid_correct: int
    valid_total: int
    measurement: str


@dataclass(frozen=True)
class Erratum:
    kind: str
    affected_rows: int
    subject: str
    finding: str
    treatment: str


@dataclass(frozen=True)
class Holdout80Summary:
    raw: Score
    erratum_inclusive: Score
    valid: Score
    subjects: tuple[SubjectScore, ...]
    errata: tuple[Erratum, ...]
    chronology: tuple[dict[str, Any], ...]
    integrity: dict[str, str]
    scope: dict[str, Any]
    mcq: dict[str, int]
    v7_reference: dict[str, Any]
    claim_limits: tuple[str, ...]
    projection_sha256: str

    def validation_report(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "schema": "holdout80-public-source-evidence-summary-v1",
            "summary_projection_sha256": self.projection_sha256,
            "metric_scope": self.scope["metric_kind"],
            "raw": _score_report(self.raw),
            "erratum_inclusive": _score_report(self.erratum_inclusive),
            "valid_tasks": _score_report(self.valid),
            "audit_status": self.integrity["audit_status"],
            "private_rows_embedded": self.scope["private_rows_embedded"],
        }


def _score_report(score: Score) -> dict[str, Any]:
    return {
        "correct": score.correct,
        "total": score.total,
        "accuracy": score.accuracy,
    }


def _score(value: Any, name: str) -> Score:
    if not isinstance(value, dict):
        raise HoldoutIntegrityError(f"{name} must be an object")
    try:
        score = Score(
            correct=int(value["correct"]),
            total=int(value["total"]),
            label=str(value["label"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HoldoutIntegrityError(f"invalid {name}: {exc}") from exc
    if score.total <= 0 or not 0 <= score.correct <= score.total:
        raise HoldoutIntegrityError(f"invalid count in {name}")
    return score


def _require_hashes(integrity: dict[str, Any]) -> dict[str, str]:
    required = (
        "selection_manifest_sha256",
        "sealed_gold_sha256",
        "mcq_prediction_sha256",
        "mcq_run_projection_sha256",
        "math_output_seal_sha256",
        "math_clean_evaluation_projection_sha256",
    )
    for key in required:
        value = str(integrity.get(key) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise HoldoutIntegrityError(f"invalid SHA-256 in {key}")
    return {str(key): str(value) for key, value in integrity.items()}


def _validate(summary: Holdout80Summary) -> None:
    if (summary.raw.correct, summary.raw.total) != (71, 80):
        raise HoldoutIntegrityError("unexpected raw Holdout80 aggregate")
    if (summary.erratum_inclusive.correct, summary.erratum_inclusive.total) != (79, 80):
        raise HoldoutIntegrityError("unexpected erratum-inclusive aggregate")
    if (summary.valid.correct, summary.valid.total) != (79, 79):
        raise HoldoutIntegrityError("unexpected valid-task aggregate")

    if sum(row.raw_correct for row in summary.subjects) != summary.raw.correct:
        raise HoldoutIntegrityError("subject raw correct does not sum to aggregate")
    if sum(row.raw_total for row in summary.subjects) != summary.raw.total:
        raise HoldoutIntegrityError("subject raw total does not sum to aggregate")
    if sum(row.valid_correct for row in summary.subjects) != summary.valid.correct:
        raise HoldoutIntegrityError("subject valid correct does not sum to aggregate")
    if sum(row.valid_total for row in summary.subjects) != summary.valid.total:
        raise HoldoutIntegrityError("subject valid total does not sum to aggregate")

    by_subject = {row.subject: row for row in summary.subjects}
    expected_subjects = {
        "Math 12": (20, 20, 20, 20),
        "Biology 9": (30, 30, 30, 30),
        "Physics 12": (21, 30, 29, 29),
    }
    if {
        name: (row.raw_correct, row.raw_total, row.valid_correct, row.valid_total)
        for name, row in by_subject.items()
    } != expected_subjects:
        raise HoldoutIntegrityError("unexpected subject aggregates")

    errata = {row.kind: row.affected_rows for row in summary.errata}
    if errata != {"swapped_gold_sections": 8, "invalid_task_type": 1}:
        raise HoldoutIntegrityError("unexpected protocol errata")
    if summary.mcq != {
        "raw_correct": 51,
        "raw_total": 60,
        "erratum_correct": 59,
        "erratum_total": 60,
        "valid_correct": 59,
        "valid_total": 59,
    }:
        raise HoldoutIntegrityError("unexpected MCQ aggregate")
    if summary.scope.get("metric_kind") != "source lookup and source binding":
        raise HoldoutIntegrityError("Holdout80 metric scope changed")
    if summary.scope.get("private_rows_embedded") is not False:
        raise HoldoutIntegrityError("public summary must not embed private rows")
    if summary.scope.get("book_disjoint") is not False:
        raise HoldoutIntegrityError("same-book split must not be labelled book-disjoint")
    if summary.integrity.get("audit_status") != "PASS":
        raise HoldoutIntegrityError("independent audit is not PASS")
    if (
        int(summary.v7_reference.get("correct") or 0),
        int(summary.v7_reference.get("total") or 0),
        summary.v7_reference.get("metric_kind"),
        summary.v7_reference.get("separate_from_holdout80"),
    ) != (242, 274, "QA development replay", True):
        raise HoldoutIntegrityError("V7 comparison scope is inconsistent")
    if tuple(row.get("step") for row in summary.chronology) != tuple(range(1, 7)):
        raise HoldoutIntegrityError("Holdout80 chronology is incomplete")


def load_holdout80_summary(path: Path | None = None) -> Holdout80Summary:
    source = (path or SUMMARY_FILE).resolve()
    if not source.is_file():
        raise HoldoutIntegrityError(f"public Holdout80 summary is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutIntegrityError(f"cannot read public Holdout80 summary: {exc}") from exc
    if not isinstance(payload, dict):
        raise HoldoutIntegrityError("public Holdout80 summary must be an object")
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual_sha = hashlib.sha256(canonical).hexdigest()
    if actual_sha != EXPECTED_SUMMARY_PROJECTION_SHA256:
        raise HoldoutIntegrityError(
            "public Holdout80 summary projection mismatch: "
            f"expected {EXPECTED_SUMMARY_PROJECTION_SHA256}, got {actual_sha}"
        )
    if payload.get("schema_version") != "holdout80-public-source-evidence-summary-v1":
        raise HoldoutIntegrityError("unsupported Holdout80 summary schema")

    scores = payload.get("scores") or {}
    try:
        subjects = tuple(SubjectScore(**row) for row in payload.get("subjects") or [])
        errata = tuple(Erratum(**row) for row in payload.get("errata") or [])
    except (TypeError, ValueError) as exc:
        raise HoldoutIntegrityError(f"invalid Holdout80 aggregate row: {exc}") from exc
    integrity = _require_hashes(payload.get("integrity") or {})
    summary = Holdout80Summary(
        raw=_score(scores.get("raw_protocol"), "raw_protocol"),
        erratum_inclusive=_score(
            scores.get("erratum_inclusive"), "erratum_inclusive"
        ),
        valid=_score(scores.get("valid_tasks"), "valid_tasks"),
        subjects=subjects,
        errata=errata,
        chronology=tuple(payload.get("chronology") or []),
        integrity=integrity,
        scope=dict(payload.get("scope") or {}),
        mcq={str(key): int(value) for key, value in (payload.get("mcq") or {}).items()},
        v7_reference=dict(payload.get("v7_reference") or {}),
        claim_limits=tuple(str(value) for value in payload.get("claim_limits") or []),
        projection_sha256=actual_sha,
    )
    _validate(summary)
    return summary
