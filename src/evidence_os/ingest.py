"""Leakage-resistant ingestion and alignment for Evidence OS policies.

The important boundary in this module is structural: ``task_id`` is used while
candidate runs are joined, but policy code receives only :class:`PolicyCase`
objects, which contain no task identifier.  Gold, judge, score and outcome
fields are rejected recursively before a narrow public projection is made.

This is a guardrail, not a proof that arbitrary prose is clean.  Production
callers must still use solver-only artifacts with trustworthy provenance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any


class EvidenceIngestionError(ValueError):
    """Base error for a rejected Evidence OS input."""


class ForbiddenEvidenceError(EvidenceIngestionError):
    """Raised when an input can expose evaluation-only evidence."""


class LineageRejectedError(EvidenceIngestionError):
    """Raised when an artifact's path or declared lineage is unsafe."""


class AlignmentError(EvidenceIngestionError):
    """Raised when candidate runs cannot be joined one-to-one."""


# Exact normalized key components.  Matching components instead of arbitrary
# substrings avoids false positives such as ``targeting_mode`` while still
# rejecting ``judge_score`` and ``referenceAnswer``.
FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "accuracy",
        "adjudication",
        "correct",
        "correctness",
        "evaluation",
        "gold",
        "groundtruth",
        "judge",
        "metric",
        "oracle",
        "outcome",
        "reference",
        "reward",
        "score",
        "verdict",
    }
)

LINEAGE_DENYLIST = frozenset(
    {
        "adjudication",
        "adjudications",
        "eval",
        "evaluation",
        "gold",
        "judge",
        "judged",
        "judges",
        "labels",
        "oracle",
        "outcome",
        "outcomes",
        "score",
        "scored",
        "scores",
        "verdict",
        "verdicts",
    }
)

# Unknown top-level fields are deliberately not made visible to the policy.
# A caller may pass an even narrower allowlist to ``load_candidate_jsonl``.
DEFAULT_POLICY_FIELDS = frozenset(
    {
        "answer",
        "answer_type",
        "condition",
        "error",
        "final_answer",
        "forced_answer",
        "generation",
        "grade",
        "model",
        "prediction",
        "prompt_version",
        "question",
        "question_images",
        "raw_response",
        "reasoning",
        "response",
        "solution_steps",
        "subject",
        "tool_calls",
        "usage",
    }
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LINEAGE_SPLIT = re.compile(r"[^a-z0-9]+")


def _key_components(key: str) -> tuple[str, ...]:
    split_camel = _CAMEL_BOUNDARY.sub("_", key).casefold()
    return tuple(part for part in _NON_ALNUM.split(split_camel) if part)


def _normalized_key(key: str) -> str:
    return "_".join(_key_components(key))


def _forbidden_key_reason(key: str) -> str | None:
    components = _key_components(key)
    compact = "".join(components)
    if any(component in FORBIDDEN_KEY_TOKENS for component in components):
        return "forbidden evaluation token"
    if compact in {"answerkey", "groundtruth", "iscorrect", "taskscore"}:
        return "forbidden evaluation key"
    # ``scoreboard`` and ``golden_answer`` do not always tokenize cleanly.
    if compact.startswith(("score", "golden")):
        return "forbidden evaluation prefix"
    return None


def _display_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<root>"


def _sanitize_tree(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ForbiddenEvidenceError(
                    f"non-string key at {_display_path(path)}"
                )
            key = _normalized_key(raw_key)

            # Existing solver rows carry this one negative attestation.  It is
            # accepted only at the exact ``generation.gold_access`` path, only
            # as boolean False, and is always projected away.  Allowing the
            # same spelling deeper in a trace would create a general-purpose
            # escape hatch for evaluation data.
            if key == "gold_access" and path == ("generation",):
                if child is not False:
                    raise ForbiddenEvidenceError(
                        f"{_display_path(path + (raw_key,))} must be the boolean false"
                    )
                continue

            if key == "task_id":
                raise ForbiddenEvidenceError(
                    f"nested task_id at {_display_path(path + (raw_key,))}"
                )
            reason = _forbidden_key_reason(raw_key)
            if reason:
                raise ForbiddenEvidenceError(
                    f"{reason} at {_display_path(path + (raw_key,))}"
                )
            clean[raw_key] = _sanitize_tree(child, path + (key,))
        return clean
    if isinstance(value, (list, tuple)):
        return [_sanitize_tree(child, path + (f"[{index}]",)) for index, child in enumerate(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ForbiddenEvidenceError(
        f"unsupported value type {type(value).__name__} at {_display_path(path)}"
    )


def validate_lineage(
    source: str | Path,
    *,
    declared_lineage: str | None = None,
    extra_denied_tokens: Iterable[str] = (),
) -> None:
    """Reject paths or lineage labels associated with evaluator artifacts."""

    denied = set(LINEAGE_DENYLIST)
    for token in extra_denied_tokens:
        denied.update(_LINEAGE_SPLIT.split(str(token).casefold()))
    denied.discard("")

    candidates = [str(source)]
    if declared_lineage is not None:
        candidates.append(declared_lineage)
    source_path = Path(source)
    try:
        candidates.append(str(source_path.resolve(strict=False)))
    except OSError:
        pass

    for candidate in candidates:
        tokens = {part for part in _LINEAGE_SPLIT.split(candidate.casefold()) if part}
        overlap = sorted(tokens & denied)
        if overlap:
            raise LineageRejectedError(
                f"rejected lineage {candidate!r}: denied token(s) {', '.join(overlap)}"
            )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateRun:
    """Staged solver artifact; identifiers remain private to alignment code."""

    name: str
    source: str
    sha256: str
    _ordered_task_ids: tuple[str, ...] = field(repr=False)
    _payload_by_task_id: Mapping[str, Mapping[str, Any]] = field(repr=False)

    @property
    def size(self) -> int:
        return len(self._ordered_task_ids)


@dataclass(frozen=True, slots=True)
class PolicyCase:
    """The only per-example object that should be passed to policy code."""

    candidates: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class AlignedBatch:
    """Aligned policy cases plus a private positional reattachment map."""

    candidate_names: tuple[str, ...]
    cases: tuple[PolicyCase, ...]
    _task_ids: tuple[str, ...] = field(repr=False)

    def attach_task_ids(
        self,
        policy_outputs: Sequence[Mapping[str, Any] | Any],
        *,
        scalar_key: str = "policy_output",
    ) -> list[dict[str, Any]]:
        """Reattach IDs after decisions, preserving only positional alignment."""

        if len(policy_outputs) != len(self._task_ids):
            raise AlignmentError(
                f"policy returned {len(policy_outputs)} rows for {len(self._task_ids)} cases"
            )
        attached: list[dict[str, Any]] = []
        for task_id, output in zip(self._task_ids, policy_outputs, strict=True):
            if isinstance(output, Mapping):
                if "task_id" in output:
                    raise AlignmentError("policy output must not contain task_id")
                row = {"task_id": task_id, **deepcopy(dict(output))}
            else:
                row = {"task_id": task_id, scalar_key: deepcopy(output)}
            attached.append(row)
        return attached


def load_candidate_jsonl(
    path: str | Path,
    *,
    name: str | None = None,
    declared_lineage: str | None = None,
    policy_fields: Iterable[str] = DEFAULT_POLICY_FIELDS,
    extra_denied_lineage_tokens: Iterable[str] = (),
) -> CandidateRun:
    """Load a solver-only JSONL artifact through the production boundary.

    Every raw key is scanned before the top-level allowlist is applied, so a
    hidden ``judge_score`` field cannot become harmless merely by being
    omitted from the public projection.
    """

    source = Path(path)
    validate_lineage(
        source,
        declared_lineage=declared_lineage,
        extra_denied_tokens=extra_denied_lineage_tokens,
    )
    if not source.is_file():
        raise EvidenceIngestionError(f"candidate artifact does not exist: {source}")

    allowed = frozenset(str(key) for key in policy_fields)
    if "task_id" in allowed:
        raise ForbiddenEvidenceError("task_id cannot be a policy field")
    for key in allowed:
        reason = _forbidden_key_reason(key)
        if reason:
            raise ForbiddenEvidenceError(f"unsafe policy field {key!r}: {reason}")

    run_name = (name or source.stem).strip()
    if not run_name:
        raise EvidenceIngestionError("candidate run name must not be empty")

    ordered_ids: list[str] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceIngestionError(
                    f"invalid JSON in {source}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(raw_record, dict):
                raise EvidenceIngestionError(
                    f"expected an object in {source}:{line_number}"
                )

            task_id = str(raw_record.get("task_id") or "").strip()
            if not task_id:
                raise AlignmentError(f"missing task_id in {source}:{line_number}")
            if task_id in payloads:
                raise AlignmentError(f"duplicate task_id {task_id!r} in {source}")

            without_id = {key: value for key, value in raw_record.items() if key != "task_id"}
            sanitized = _sanitize_tree(without_id)
            projected = {key: sanitized[key] for key in allowed if key in sanitized}
            ordered_ids.append(task_id)
            payloads[task_id] = _freeze(projected)

    if not ordered_ids:
        raise EvidenceIngestionError(f"candidate artifact is empty: {source}")

    return CandidateRun(
        name=run_name,
        source=str(source.resolve()),
        sha256=_file_sha256(source),
        _ordered_task_ids=tuple(ordered_ids),
        _payload_by_task_id=MappingProxyType(payloads),
    )


def align_candidate_runs(
    runs: Sequence[CandidateRun],
    *,
    require_identical_task_sets: bool = True,
) -> AlignedBatch:
    """Align runs by ID, then produce ID-free, immutable policy cases."""

    if not runs:
        raise AlignmentError("at least one candidate run is required")
    names = tuple(run.name for run in runs)
    if len(set(names)) != len(names):
        raise AlignmentError("candidate run names must be unique")

    anchor = runs[0]
    anchor_ids = set(anchor._ordered_task_ids)
    if require_identical_task_sets:
        for run in runs[1:]:
            run_ids = set(run._ordered_task_ids)
            missing = anchor_ids - run_ids
            extra = run_ids - anchor_ids
            if missing or extra:
                raise AlignmentError(
                    f"task-set mismatch for {run.name!r}: "
                    f"missing={len(missing)}, extra={len(extra)}"
                )
        ordered_ids = anchor._ordered_task_ids
    else:
        common_ids = set.intersection(*(set(run._ordered_task_ids) for run in runs))
        if not common_ids:
            raise AlignmentError("candidate runs have no common task IDs")
        ordered_ids = tuple(task_id for task_id in anchor._ordered_task_ids if task_id in common_ids)

    cases: list[PolicyCase] = []
    for task_id in ordered_ids:
        candidates = {
            run.name: run._payload_by_task_id[task_id]
            for run in runs
        }
        cases.append(PolicyCase(candidates=MappingProxyType(candidates)))

    return AlignedBatch(
        candidate_names=names,
        cases=tuple(cases),
        _task_ids=tuple(ordered_ids),
    )
