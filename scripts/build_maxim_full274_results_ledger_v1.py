#!/usr/bin/env python3
"""Build a strict, read-only ledger of finalized frozen-full274 scores.

The source reports are never modified.  A branch is promoted to ``final`` only
when one report proves all of the following:

* it is a ``matched_score.json`` artifact, or a ``score.json`` with a
  hash-valid finalization manifest (canonical post-generation scores also
  require a hash-valid, exact-lineage orchestration record);
* benchmark SHA-256 and denominator match the frozen 274-task benchmark;
* the exact matched judge lineage is either declared or independently derived
  from its image-judge JSONL;
* all 274 unique task outcomes are present and their boolean correctness count
  agrees with the reported score;
* the artifact is not a bound, partial/interim result, in-sample score, or a
  generation run with gold access.

Branches registered before scoring remain ``pending`` when no valid final
artifact exists. The only terminal no-score exception is a hash-valid
``SUPERSEDED_BEFORE_CALLS`` attestation that independently records exactly zero
source calls; it is emitted as ``non_final`` without a metric and its score
globs are never inspected. Invalid candidates are recorded as rejections,
never coerced into final scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "maxim-full274-results-ledger-v1"
REGISTRY_SCHEMA_VERSION = "maxim-full274-results-registry-v1"
DEFAULT_BENCHMARK_ROWS = 274
DEFAULT_BENCHMARK_SHA256 = (
    "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
)
DEFAULT_JUDGE_LINEAGE = "frozen-judge-v2-qwen3.5-9b-seed20260714"
EXPECTED_JUDGE_PROMPT = "judge-v2"
EXPECTED_JUDGE_MODEL = "Qwen/Qwen3.5-9B"
EXPECTED_JUDGE_SEED = 20260714
EXPECTED_IMAGE_JUDGE_ROWS = 97
ACCURACY_TOLERANCE = 1e-6
FROZEN_BASELINE_SOLVER_SHA256 = (
    "62bc952c3802308bc0fbf8d8dc1f82ec523a3ab1e3264bae87a5f8828021d75d"
)
FROZEN_BASELINE_JUDGE_SHA256 = (
    "59dcc93454b29dfc65b0a9b1243a177d472b6c0a13cbe46fb5c98079810a73f4"
)
FROZEN_IMAGE_TEMPLATE_SHA256 = (
    "41f35172092f67bc14368d7312cb91ad50256166dc42d0f630ed5e7a9965aa46"
)
FROZEN_JUDGE_BACKEND_CONFIG = {
    "backend": "openai-compatible",
    "model": EXPECTED_JUDGE_MODEL,
    "endpoint": "http://127.0.0.1:18005/v1/chat/completions",
    "temperature": 0.0,
    "max_tokens": 900,
    "seed": EXPECTED_JUDGE_SEED,
    "use_response_format": True,
    "enable_thinking": False,
    "image_mode": "data_url",
}
FROZEN_JUDGE_BACKEND_CONFIG_SHA256 = (
    "e3f71b4af7fa8ad8a6db755d43bdf4a895d087b701436c105da8c5416804fbd9"
)
FROZEN_JUDGE_ADAPTER_SHA256 = {
    "pipeline.py": "d61a248dd5d078924475c819ef824c5e2dc1e587683b2cd84844bb01f72f2f73",
    "prompts.py": "84a1342612ca1fce9db9b7d4adfc4215b3d24e77f60d49a47635a47cc190b8d5",
    "schema.py": "d86fed78c297d0479d00ceb2d54fad502dfbfe8b49cd7be08fed2d25fc7a4e7c",
}
FROZEN_ORCHESTRATION_DELEGATE_SHA256 = {
    "orchestrator": "b8b21519e7a2f55101eb58a5dedbc5192cd585c1bf87679a18474feebe665a5b",
    "preparation": "f5d3acafcc92e80609c0b98f4e8d69a7edc616abdb8de90936952b9aa470a9fc",
    "finalizer": "bc2480cc1ae6c13ab2c4737f6f7811af169c9a7d830b76d51216359aff1f2b59",
    "judge_cli": "439390146c733d476cd03c7bb417cf463b579db14bfa39fa010b824bd2b88cf8",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NON_FINAL_TOKEN_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:bound|bounds|partial|interim|in[ _-]?sample|"
    r"estimated|estimate|projection|projected)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)


class LedgerError(ValueError):
    """Raised for an invalid registry or an invalid final-score candidate."""


@dataclass(frozen=True)
class Pins:
    benchmark_sha256: str
    benchmark_rows: int
    judge_lineage: str
    image_judge_rows: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError(f"cannot read JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise LedgerError("JSON root must be an object")
    return value


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerError(
                        f"invalid JSONL line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise LedgerError(f"JSONL line {line_number} is not an object")
                rows.append(value)
    except OSError as exc:
        raise LedgerError(f"cannot read JSONL: {exc}") from exc
    return rows


def nested_get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def first_present(value: Mapping[str, Any], paths: Iterable[Sequence[str]]) -> Any:
    for path in paths:
        found = nested_get(value, path)
        if found is not None:
            return found
    return None


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerError(f"{label} must be an integer")
    return value


def _strict_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise LedgerError(f"{label} must be finite")
    return number


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _ensure_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise LedgerError(f"{label} escapes repository root: {resolved}") from exc
    return resolved


def _validate_reports_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LedgerError(f"{label} must be a non-empty path")
    pure = Path(value)
    normalized = value.replace("\\", "/")
    if pure.is_absolute() or ".." in pure.parts:
        raise LedgerError(f"unsafe {label}: {value}")
    if not normalized.startswith("reports/"):
        raise LedgerError(f"{label} must stay under reports/")
    return value


def validate_supersession_attestation(
    branch: Mapping[str, Any], repo_root: Path, reports_root: Path
) -> Mapping[str, Any]:
    """Verify a terminal superseded-before-calls registry claim.

    This validation deliberately opens only the explicitly hash-pinned
    attestation. Candidate score globs are neither discovered nor read for a
    superseded branch.
    """

    branch_id = str(branch.get("id"))
    source_call_count = _strict_int(
        branch.get("superseded_source_call_count"),
        f"branch {branch_id!r}.superseded_source_call_count",
    )
    if source_call_count != 0:
        raise LedgerError(
            f"branch {branch_id!r}.superseded_source_call_count must be exactly 0"
        )

    raw_path = _validate_reports_relative_path(
        branch.get("supersession_attestation"),
        f"branch {branch_id!r}.supersession_attestation",
    )
    expected_sha = branch.get("supersession_attestation_sha256")
    if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(
        expected_sha.lower()
    ):
        raise LedgerError(
            f"branch {branch_id!r}.supersession_attestation_sha256 is invalid"
        )

    path = _ensure_inside(
        repo_root / raw_path, reports_root, "supersession attestation"
    )
    if not path.is_file():
        raise LedgerError(
            f"branch {branch_id!r} supersession attestation is missing: {raw_path}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha.lower():
        raise LedgerError(
            f"branch {branch_id!r} supersession attestation SHA256 mismatch: "
            f"expected {expected_sha.lower()}, got {actual_sha}"
        )

    attestation = read_json(path)
    if attestation.get("status") != "SUPERSEDED_BEFORE_CALLS":
        raise LedgerError(
            f"branch {branch_id!r} attestation status is not SUPERSEDED_BEFORE_CALLS"
        )
    attested_call_count = nested_get(
        attestation, ("server_evidence", "source_call_count")
    )
    if (
        isinstance(attested_call_count, bool)
        or not isinstance(attested_call_count, int)
        or attested_call_count != 0
    ):
        raise LedgerError(
            f"branch {branch_id!r} attestation server_evidence.source_call_count "
            "must be exactly 0"
        )

    return {
        "path": _relative(path, repo_root),
        "sha256": actual_sha,
        "schema_version": attestation.get("schema_version"),
        "status": "SUPERSEDED_BEFORE_CALLS",
    }


def load_registry(path: Path) -> tuple[Mapping[str, Any], Pins]:
    registry = read_json(path)
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise LedgerError(
            f"registry schema_version must be {REGISTRY_SCHEMA_VERSION!r}"
        )

    benchmark = registry.get("benchmark")
    judge = registry.get("judge")
    branches = registry.get("branches")
    if not isinstance(benchmark, Mapping):
        raise LedgerError("registry.benchmark must be an object")
    if not isinstance(judge, Mapping):
        raise LedgerError("registry.judge must be an object")
    if not isinstance(branches, list) or not branches:
        raise LedgerError("registry.branches must be a non-empty array")

    benchmark_sha = benchmark.get("sha256")
    benchmark_rows = benchmark.get("rows")
    judge_lineage = judge.get("lineage")
    image_rows = judge.get("image_judge_rows", EXPECTED_IMAGE_JUDGE_ROWS)
    if benchmark_sha != DEFAULT_BENCHMARK_SHA256:
        raise LedgerError("registry benchmark SHA256 is not the frozen pin")
    if benchmark_rows != DEFAULT_BENCHMARK_ROWS:
        raise LedgerError("registry benchmark row count is not the frozen pin")
    if judge_lineage != DEFAULT_JUDGE_LINEAGE:
        raise LedgerError("registry judge lineage is not the frozen pin")
    if image_rows != EXPECTED_IMAGE_JUDGE_ROWS:
        raise LedgerError("registry image-judge row count is not the frozen pin")

    seen: set[str] = set()
    for index, branch in enumerate(branches):
        if not isinstance(branch, Mapping):
            raise LedgerError(f"registry.branches[{index}] must be an object")
        branch_id = branch.get("id")
        if not isinstance(branch_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", branch_id):
            raise LedgerError(f"invalid branch id at index {index}")
        if branch_id in seen:
            raise LedgerError(f"duplicate branch id: {branch_id}")
        seen.add(branch_id)
        if branch.get("preregistered") is not True:
            raise LedgerError(f"branch {branch_id!r} must be preregistered")
        globs = branch.get("report_globs")
        if not isinstance(globs, list) or not globs or not all(
            isinstance(item, str) and item for item in globs
        ):
            raise LedgerError(f"branch {branch_id!r} needs report_globs")
        for pattern in globs:
            pure = Path(pattern)
            if pure.is_absolute() or ".." in pure.parts:
                raise LedgerError(f"unsafe report glob for branch {branch_id!r}: {pattern}")
            if not pattern.replace("\\", "/").startswith("reports/"):
                raise LedgerError(
                    f"report glob for branch {branch_id!r} must stay under reports/"
                )
        result_kind = branch.get("result_kind", "final_candidate")
        if result_kind not in {"final_candidate", "oof_estimate"}:
            raise LedgerError(
                f"branch {branch_id!r} has invalid result_kind {result_kind!r}"
            )
        superseded_before_calls = branch.get("superseded_before_calls", False)
        if not isinstance(superseded_before_calls, bool):
            raise LedgerError(
                f"branch {branch_id!r}.superseded_before_calls must be boolean"
            )
        if superseded_before_calls:
            if result_kind != "final_candidate":
                raise LedgerError(
                    f"branch {branch_id!r} cannot be both superseded and {result_kind!r}"
                )
            source_call_count = _strict_int(
                branch.get("superseded_source_call_count"),
                f"branch {branch_id!r}.superseded_source_call_count",
            )
            if source_call_count != 0:
                raise LedgerError(
                    f"branch {branch_id!r}.superseded_source_call_count must be exactly 0"
                )
            _validate_reports_relative_path(
                branch.get("supersession_attestation"),
                f"branch {branch_id!r}.supersession_attestation",
            )
            attestation_sha = branch.get("supersession_attestation_sha256")
            if not isinstance(attestation_sha, str) or not _SHA256_RE.fullmatch(
                attestation_sha.lower()
            ):
                raise LedgerError(
                    f"branch {branch_id!r}.supersession_attestation_sha256 is invalid"
                )
        legacy_without_orchestration = branch.get(
            "allow_legacy_finalizer_without_orchestration", False
        )
        if not isinstance(legacy_without_orchestration, bool):
            raise LedgerError(
                "branch "
                f"{branch_id!r}.allow_legacy_finalizer_without_orchestration "
                "must be boolean"
            )

    return registry, Pins(
        benchmark_sha256=benchmark_sha,
        benchmark_rows=benchmark_rows,
        judge_lineage=judge_lineage,
        image_judge_rows=image_rows,
    )


def candidate_has_final_path(path: Path) -> bool:
    name = path.name.lower()
    return name in {"matched_score.json", "score.json"}


def reject_non_final_markers(path: Path, report: Mapping[str, Any]) -> None:
    # Only inspect the artifact's reports-relative path.  Parent directories
    # used by test runners (or users) may legitimately contain words such as
    # "partial" without describing the experiment artifact.
    lowered_parts = [part.lower() for part in path.parts]
    try:
        reports_index = lowered_parts.index("reports")
    except ValueError:
        reports_index = max(0, len(path.parts) - 4)
    normalized_path = Path(*path.parts[reports_index:]).as_posix().lower()
    if _NON_FINAL_TOKEN_RE.search(normalized_path):
        raise LedgerError("candidate path is marked partial/bounds/interim/in-sample")

    marker_paths = (
        ("status",),
        ("evaluation_mode",),
        ("scope",),
        ("split",),
        ("finality",),
        ("score_type",),
        ("protocol", "mode"),
        ("protocol", "scope"),
        ("evaluation", "mode"),
        ("evaluation", "scope"),
    )
    for marker_path in marker_paths:
        marker = nested_get(report, marker_path)
        if isinstance(marker, str) and _NON_FINAL_TOKEN_RE.search(marker):
            raise LedgerError(
                f"{'.'.join(marker_path)} marks a non-final result: {marker!r}"
            )

    for flag_path in (
        ("partial",),
        ("is_partial",),
        ("in_sample",),
        ("is_in_sample",),
        ("evaluation", "in_sample"),
    ):
        if nested_get(report, flag_path) is True:
            raise LedgerError(f"{'.'.join(flag_path)} is true")
    for flag_path in (("is_final",), ("finalized",)):
        if nested_get(report, flag_path) is False:
            raise LedgerError(f"{'.'.join(flag_path)} is false")

    overall = report.get("overall")
    if isinstance(overall, Mapping) and any(
        key in overall
        for key in ("lower_correct", "upper_correct", "lower_bound", "upper_bound")
    ):
        raise LedgerError("overall contains bounds instead of an exact final score")

    generation_gold = first_present(
        report,
        (
            ("generation_gold_access",),
            ("guardrails", "generation_gold_access"),
            ("protocol", "generation_gold_access"),
        ),
    )
    if generation_gold is True:
        raise LedgerError("generation_gold_access is true")


def extract_benchmark_sha(report: Mapping[str, Any]) -> str:
    value = first_present(
        report,
        (
            ("provenance", "benchmark", "sha256"),
            ("sources", "benchmark", "sha256"),
            ("benchmark", "sha256"),
            ("benchmark_sha256",),
        ),
    )
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.lower()):
        raise LedgerError("report does not contain a valid benchmark SHA256")
    return value.lower()


def extract_score(report: Mapping[str, Any], pins: Pins) -> tuple[int, float, list[Mapping[str, Any]]]:
    overall = report.get("overall")
    if not isinstance(overall, Mapping):
        raise LedgerError("report.overall must be an object")
    denominator = _strict_int(overall.get("n"), "overall.n")
    if denominator != pins.benchmark_rows:
        raise LedgerError(f"overall.n must be {pins.benchmark_rows}, got {denominator}")

    correct_value = first_present(
        overall,
        (("new_correct",), ("correct",)),
    )
    correct = _strict_int(correct_value, "overall correct")
    if not 0 <= correct <= denominator:
        raise LedgerError("overall correct is outside [0, n]")

    accuracy_value = first_present(
        overall,
        (("new_accuracy",), ("accuracy",)),
    )
    reported_accuracy = _strict_number(accuracy_value, "overall accuracy")
    exact_accuracy = correct / denominator
    if abs(reported_accuracy - exact_accuracy) > ACCURACY_TOLERANCE:
        raise LedgerError(
            f"reported accuracy {reported_accuracy} disagrees with {correct}/{denominator}"
        )

    outcomes = report.get("task_outcomes")
    if not isinstance(outcomes, list):
        raise LedgerError("task_outcomes must be an array")
    if len(outcomes) != denominator:
        raise LedgerError(
            f"task_outcomes must contain {denominator} rows, got {len(outcomes)}"
        )
    task_ids: set[str] = set()
    counted = 0
    typed_outcomes: list[Mapping[str, Any]] = []
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, Mapping):
            raise LedgerError(f"task_outcomes[{index}] is not an object")
        task_id = outcome.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise LedgerError(f"task_outcomes[{index}].task_id is invalid")
        if task_id in task_ids:
            raise LedgerError(f"duplicate task_id in task_outcomes: {task_id}")
        task_ids.add(task_id)
        verdict = outcome.get("new_correct")
        if not isinstance(verdict, bool):
            raise LedgerError(f"task_outcomes[{index}].new_correct is not boolean")
        counted += int(verdict)
        typed_outcomes.append(outcome)
    if counted != correct:
        raise LedgerError(
            f"task outcome count {counted} disagrees with reported correct {correct}"
        )
    return correct, exact_accuracy, typed_outcomes


def _resolve_source_path(raw_path: str, report_path: Path, repo_root: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    local = (report_path.parent / candidate).resolve()
    if local.exists():
        return local
    return (repo_root / candidate).resolve()


def _image_judge_descriptor(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = first_present(
        report,
        (
            ("provenance", "image_judge"),
            ("provenance", "matched_image97_judge"),
            ("sources", "matched_image97_judge"),
            ("matched_image97_judge",),
        ),
    )
    return value if isinstance(value, Mapping) else None


def verify_frozen_baseline_judge_lineage(
    judge_path: Path,
    repo_root: Path,
    pins: Pins,
    outcomes: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Verify the one legacy baseline-replay score without weakening lineage.

    Answer canonicalization is an exact replay of the frozen page-RAG output,
    so its score points at the frozen 274-row hybrid judge rather than a
    separately extracted 97-row image sidecar.  The immutable file hash, full
    task-id set, and every verdict are checked here.
    """

    rows = read_jsonl(judge_path)
    if len(rows) != pins.benchmark_rows:
        raise LedgerError(
            f"frozen baseline judge must contain {pins.benchmark_rows} rows, "
            f"got {len(rows)}"
        )
    verdicts: dict[str, bool] = {}
    image_rows = 0
    for index, row in enumerate(rows):
        task_id = row.get("task_id")
        verdict = nested_get(row, ("verdict", "strict_correct"))
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in verdicts
            or not isinstance(verdict, bool)
        ):
            raise LedgerError(
                f"invalid/duplicate frozen baseline judge row at index {index}"
            )
        verdicts[task_id] = verdict
        if row.get("prompt_version") == EXPECTED_JUDGE_PROMPT:
            image_rows += 1
    if image_rows != pins.image_judge_rows:
        raise LedgerError(
            f"frozen baseline judge must contain {pins.image_judge_rows} "
            f"judge-v2 rows, got {image_rows}"
        )
    outcome_verdicts = {
        str(row["task_id"]): row["new_correct"] for row in outcomes
    }
    if outcome_verdicts != verdicts:
        raise LedgerError(
            "frozen baseline judge verdicts do not match all task outcomes"
        )
    return {
        "method": "derived_from_exact_frozen_baseline_judge",
        "lineage": pins.judge_lineage,
        "artifact": {
            "path": _relative(judge_path, repo_root),
            "sha256": FROZEN_BASELINE_JUDGE_SHA256,
            "rows": len(rows),
            "image_judge_rows": image_rows,
        },
    }


def verify_image_judge_lineage(
    descriptor: Mapping[str, Any],
    report_path: Path,
    repo_root: Path,
    pins: Pins,
    outcomes: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise LedgerError("image-judge descriptor has no path")
    judge_path = _resolve_source_path(raw_path, report_path, repo_root)
    if not judge_path.is_file():
        raise LedgerError(f"image-judge artifact is missing: {judge_path}")
    actual_sha = sha256_file(judge_path)
    pinned_sha = descriptor.get("sha256")
    if pinned_sha is not None:
        if not isinstance(pinned_sha, str) or pinned_sha.lower() != actual_sha:
            raise LedgerError("image-judge artifact SHA256 mismatch")

    if actual_sha == FROZEN_BASELINE_JUDGE_SHA256:
        return verify_frozen_baseline_judge_lineage(
            judge_path, repo_root, pins, outcomes
        )

    rows = read_jsonl(judge_path)
    if len(rows) != pins.image_judge_rows:
        raise LedgerError(
            f"image-judge artifact must contain {pins.image_judge_rows} rows, got {len(rows)}"
        )
    judge_ids: set[str] = set()
    judge_correct: dict[str, bool] = {}
    for index, row in enumerate(rows):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in judge_ids:
            raise LedgerError(f"invalid/duplicate image-judge task_id at row {index}")
        judge_ids.add(task_id)
        if row.get("prompt_version") != EXPECTED_JUDGE_PROMPT:
            raise LedgerError(f"image-judge row {index} is not prompt judge-v2")
        judge = row.get("judge")
        if not isinstance(judge, Mapping):
            raise LedgerError(f"image-judge row {index} has no judge object")
        backend = judge.get("backend_config")
        if not isinstance(backend, Mapping):
            raise LedgerError(f"image-judge row {index} has no backend_config")
        model = judge.get("model", backend.get("model"))
        if model != EXPECTED_JUDGE_MODEL or backend.get("model") != EXPECTED_JUDGE_MODEL:
            raise LedgerError(f"image-judge row {index} has the wrong model")
        if backend.get("seed") != EXPECTED_JUDGE_SEED:
            raise LedgerError(f"image-judge row {index} has the wrong seed")
        if _strict_number(backend.get("temperature"), "judge temperature") != 0.0:
            raise LedgerError(f"image-judge row {index} has nonzero temperature")
        if backend.get("enable_thinking") is not False:
            raise LedgerError(f"image-judge row {index} does not pin thinking=false")
        verdict = nested_get(row, ("verdict", "strict_correct"))
        if not isinstance(verdict, bool):
            raise LedgerError(f"image-judge row {index} has no boolean strict_correct")
        judge_correct[task_id] = verdict

    outcome_image = {
        str(row["task_id"]): row["new_correct"]
        for row in outcomes
        if row.get("score_source") == "image_judge"
    }
    if outcome_image and outcome_image != judge_correct:
        raise LedgerError("image-judge verdicts do not match image-judged task outcomes")
    if not outcome_image:
        # Custom finalized reports may use inherited verdicts and another source
        # label.  They still need exactly the same task-id subset if declared.
        declared_ids = {
            str(row["task_id"])
            for row in outcomes
            if row.get("score_source") in {"matched_image_judge", "matched_reuse"}
        }
        if declared_ids and declared_ids != judge_ids:
            raise LedgerError("declared matched-judge task IDs do not match sidecar")

    return {
        "method": "derived_from_image_judge_jsonl",
        "lineage": pins.judge_lineage,
        "artifact": {
            "path": _relative(judge_path, repo_root),
            "sha256": actual_sha,
            "rows": len(rows),
        },
    }


def verify_lineage(
    report: Mapping[str, Any],
    report_path: Path,
    repo_root: Path,
    pins: Pins,
    outcomes: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    explicit = first_present(
        report,
        (
            ("judge", "lineage"),
            ("matched_judge_lineage",),
            ("protocol", "matched_judge_lineage"),
            ("provenance", "judge", "lineage"),
        ),
    )
    if explicit is not None and explicit != pins.judge_lineage:
        raise LedgerError(
            f"judge lineage mismatch: expected {pins.judge_lineage!r}, got {explicit!r}"
        )
    matched_flag = nested_get(report, ("judge", "matched"))
    if explicit is not None and matched_flag is False:
        raise LedgerError("judge.matched is false")

    descriptor = _image_judge_descriptor(report)
    derived: Mapping[str, Any] | None = None
    if descriptor is not None:
        derived = verify_image_judge_lineage(
            descriptor, report_path, repo_root, pins, outcomes
        )

    if explicit == pins.judge_lineage:
        return {
            "method": "explicit_and_sidecar" if derived else "explicit",
            "lineage": pins.judge_lineage,
            **({"sidecar": derived["artifact"]} if derived else {}),
        }
    if derived is not None:
        return derived
    raise LedgerError("exact matched judge lineage is neither declared nor derivable")


def _verify_frozen_judge_manifest(manifest: Mapping[str, Any]) -> None:
    frozen = manifest.get("frozen_judge_v2")
    if not isinstance(frozen, Mapping):
        raise LedgerError("canonical finalization manifest has no frozen_judge_v2")
    if frozen.get("prompt_version") != EXPECTED_JUDGE_PROMPT:
        raise LedgerError("finalization manifest has the wrong judge prompt")
    backend = frozen.get("backend_config")
    if backend != FROZEN_JUDGE_BACKEND_CONFIG:
        raise LedgerError("finalization manifest has the wrong frozen judge backend")
    if canonical_sha256(backend) != FROZEN_JUDGE_BACKEND_CONFIG_SHA256:
        raise LedgerError("finalization manifest judge backend canonical hash mismatch")
    if frozen.get("backend_config_sha256") != FROZEN_JUDGE_BACKEND_CONFIG_SHA256:
        raise LedgerError("finalization manifest judge backend SHA256 mismatch")
    if frozen.get("adapter_sha256") != FROZEN_JUDGE_ADAPTER_SHA256:
        raise LedgerError("finalization manifest judge adapter hashes mismatch")


def _descriptor_sha(descriptor: Mapping[str, Any], label: str) -> str:
    value = descriptor.get("sha256")
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.lower()):
        raise LedgerError(f"{label} has no valid SHA256")
    return value.lower()


def _verify_orchestration_checksum(state_path: Path) -> str:
    checksum_path = state_path.with_suffix(".sha256")
    if not checksum_path.is_file():
        raise LedgerError("canonical orchestration has no SHA256 sidecar")
    try:
        tokens = checksum_path.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise LedgerError(f"cannot read orchestration SHA256 sidecar: {exc}") from exc
    if len(tokens) != 2 or tokens[1] != state_path.name:
        raise LedgerError("canonical orchestration SHA256 sidecar is malformed")
    actual = sha256_file(state_path)
    if tokens[0].lower() != actual:
        raise LedgerError("canonical orchestration SHA256 mismatch")
    return actual


def verify_postgeneration_orchestration(
    score_path: Path,
    manifest_path: Path,
    repo_root: Path,
    *,
    allow_legacy_without_orchestration: bool,
) -> Mapping[str, Any] | None:
    state_path = score_path.parent / "postgeneration_orchestration_v1.json"
    if not state_path.is_file():
        # The first eight runs used the same hash-bound finalizer directly,
        # before the orchestration wrapper existed.  Their frozen-judge
        # manifest remains admissible; helper-produced runs always carry this
        # sibling and enter the stricter path below.
        if allow_legacy_without_orchestration:
            return None
        raise LedgerError(
            "canonical score.json has no postgeneration_orchestration_v1.json"
        )
    state_sha = _verify_orchestration_checksum(state_path)
    state = read_json(state_path)
    if state.get("schema_version") != "full274-postgeneration-orchestration-v1":
        raise LedgerError("canonical orchestration schema_version mismatch")

    sources = state.get("sources")
    if not isinstance(sources, Mapping):
        raise LedgerError("canonical orchestration has no sources")
    expected_sources = {
        "benchmark": (DEFAULT_BENCHMARK_SHA256, DEFAULT_BENCHMARK_ROWS),
        "baseline_solver": (FROZEN_BASELINE_SOLVER_SHA256, DEFAULT_BENCHMARK_ROWS),
        "baseline_judge": (FROZEN_BASELINE_JUDGE_SHA256, DEFAULT_BENCHMARK_ROWS),
        "image_template": (FROZEN_IMAGE_TEMPLATE_SHA256, EXPECTED_IMAGE_JUDGE_ROWS),
    }
    for name, (expected_sha, expected_rows) in expected_sources.items():
        descriptor = sources.get(name)
        if not isinstance(descriptor, Mapping):
            raise LedgerError(f"canonical orchestration source {name!r} is missing")
        if _descriptor_sha(descriptor, f"orchestration source {name}") != expected_sha:
            raise LedgerError(f"canonical orchestration source {name!r} SHA256 mismatch")
        if descriptor.get("rows") != expected_rows:
            raise LedgerError(f"canonical orchestration source {name!r} row mismatch")

    judge = state.get("judge")
    if not isinstance(judge, Mapping):
        raise LedgerError("canonical orchestration has no judge configuration")
    if (
        judge.get("model") != EXPECTED_JUDGE_MODEL
        or judge.get("prompt_version") != EXPECTED_JUDGE_PROMPT
        or judge.get("backend_config") != FROZEN_JUDGE_BACKEND_CONFIG
        or judge.get("backend_config_sha256") != FROZEN_JUDGE_BACKEND_CONFIG_SHA256
    ):
        raise LedgerError("canonical orchestration frozen judge lineage mismatch")

    delegates = state.get("delegates")
    if not isinstance(delegates, Mapping):
        raise LedgerError("canonical orchestration has no delegates")
    for name, expected_sha in FROZEN_ORCHESTRATION_DELEGATE_SHA256.items():
        descriptor = delegates.get(name)
        if not isinstance(descriptor, Mapping) or descriptor.get("sha256") != expected_sha:
            raise LedgerError(
                f"canonical orchestration delegate {name!r} SHA256 mismatch"
            )

    stages = state.get("stages")
    if not isinstance(stages, Mapping):
        raise LedgerError("canonical orchestration has no stages")
    for name in ("prepare", "judge", "finalize"):
        stage = stages.get(name)
        if not isinstance(stage, Mapping) or stage.get("status") != "complete":
            raise LedgerError(f"canonical orchestration stage {name!r} is not complete")
    artifacts = nested_get(state, ("stages", "finalize", "artifacts"))
    if not isinstance(artifacts, Mapping):
        raise LedgerError("canonical orchestration has no finalized artifacts")
    score_descriptor = artifacts.get("score_json")
    manifest_descriptor = artifacts.get("finalization_manifest")
    if not isinstance(score_descriptor, Mapping) or not isinstance(
        manifest_descriptor, Mapping
    ):
        raise LedgerError("canonical orchestration final artifacts are incomplete")
    if _descriptor_sha(score_descriptor, "orchestration score") != sha256_file(score_path):
        raise LedgerError("canonical orchestration score SHA256 mismatch")
    if _descriptor_sha(
        manifest_descriptor, "orchestration finalization manifest"
    ) != sha256_file(manifest_path):
        raise LedgerError("canonical orchestration manifest SHA256 mismatch")

    return {
        "path": _relative(state_path, repo_root),
        "sha256": state_sha,
        "schema_version": state["schema_version"],
        "frozen_lineage_verified": True,
    }


def verify_finalization_manifest(
    path: Path,
    repo_root: Path,
    *,
    allow_legacy_without_orchestration: bool = False,
) -> Mapping[str, Any] | None:
    if path.name.lower() == "matched_score.json":
        return None
    manifest_path = path.parent / "finalization_manifest.json"
    if not manifest_path.is_file():
        raise LedgerError("score.json has no finalization_manifest.json")
    manifest = read_json(manifest_path)
    descriptor = manifest.get("score")
    if not isinstance(descriptor, Mapping):
        raise LedgerError("finalization manifest has no score descriptor")
    expected_sha = descriptor.get("sha256")
    if expected_sha is None:
        expected_sha = nested_get(descriptor, ("output_hashes", "json"))
    if not isinstance(expected_sha, str) or expected_sha.lower() != sha256_file(path):
        raise LedgerError("finalization manifest score SHA256 mismatch")
    canonical = manifest.get("schema_version") == "maxim-online-finalization-v1"
    if canonical:
        _verify_frozen_judge_manifest(manifest)
    orchestration = (
        verify_postgeneration_orchestration(
            path,
            manifest_path,
            repo_root,
            allow_legacy_without_orchestration=allow_legacy_without_orchestration,
        )
        if canonical
        else None
    )
    result = {
        "path": _relative(manifest_path, repo_root),
        "sha256": sha256_file(manifest_path),
        "schema_version": manifest.get("schema_version"),
    }
    if orchestration is not None:
        result["orchestration"] = orchestration
    return result


def validate_candidate(
    path: Path,
    repo_root: Path,
    pins: Pins,
    *,
    allow_legacy_without_orchestration: bool = False,
) -> Mapping[str, Any]:
    if not candidate_has_final_path(path):
        raise LedgerError("artifact is not matched_score.json or score.json")
    report = read_json(path)
    reject_non_final_markers(path, report)
    benchmark_sha = extract_benchmark_sha(report)
    if benchmark_sha != pins.benchmark_sha256:
        raise LedgerError(
            f"benchmark SHA256 mismatch: expected {pins.benchmark_sha256}, got {benchmark_sha}"
        )
    correct, accuracy, outcomes = extract_score(report, pins)
    lineage = verify_lineage(report, path, repo_root, pins, outcomes)
    is_exact_baseline_replay = (
        lineage.get("method") == "derived_from_exact_frozen_baseline_judge"
    )
    finalization = (
        None
        if path.name.lower() == "score.json" and is_exact_baseline_replay
        else verify_finalization_manifest(
            path,
            repo_root,
            allow_legacy_without_orchestration=allow_legacy_without_orchestration,
        )
    )
    return {
        "correct": correct,
        "denominator": pins.benchmark_rows,
        "accuracy": accuracy,
        "report": {
            "path": _relative(path, repo_root),
            "sha256": sha256_file(path),
        },
        "lineage_verification": lineage,
        **({"finalization_manifest": finalization} if finalization else {}),
    }


def validate_oof_estimate(path: Path, repo_root: Path, pins: Pins) -> Mapping[str, Any]:
    """Audit the OOF router estimate while keeping it outside final scores."""

    report = read_json(path)
    if report.get("status") != "exact_crossfit":
        raise LedgerError("OOF artifact status is not exact_crossfit")
    if report.get("primary_metric_validity") != (
        "out_of_fold_crossfit_same_benchmark_not_external_holdout"
    ):
        raise LedgerError("OOF artifact validity marker mismatch")
    if report.get("generation_gold_access") is not False:
        raise LedgerError("OOF artifact does not pin generation_gold_access=false")
    if report.get("evaluation_gold_access") is not True:
        raise LedgerError("OOF artifact does not disclose evaluation_gold_access=true")
    benchmark = report.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise LedgerError("OOF artifact has no benchmark descriptor")
    if benchmark.get("sha256") != pins.benchmark_sha256 or benchmark.get("n") != pins.benchmark_rows:
        raise LedgerError("OOF artifact benchmark pin mismatch")
    judge = report.get("judge")
    if (
        not isinstance(judge, Mapping)
        or judge.get("lineage") != pins.judge_lineage
        or judge.get("matched") is not True
    ):
        raise LedgerError("OOF artifact matched-judge lineage mismatch")
    metric = report.get("crossfit_subject_plus_disagreement")
    if not isinstance(metric, Mapping):
        raise LedgerError("OOF artifact has no primary crossfit metric")
    correct = _strict_int(metric.get("correct"), "OOF correct")
    denominator = _strict_int(metric.get("n"), "OOF n")
    accuracy = _strict_number(metric.get("accuracy"), "OOF accuracy")
    if denominator != pins.benchmark_rows or abs(accuracy - correct / denominator) > ACCURACY_TOLERANCE:
        raise LedgerError("OOF metric is internally inconsistent")
    outcomes = report.get("task_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != denominator:
        raise LedgerError("OOF task_outcomes must contain exactly 274 rows")
    task_ids: set[str] = set()
    counted = 0
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, Mapping):
            raise LedgerError(f"OOF task_outcomes[{index}] is not an object")
        task_id = outcome.get("task_id")
        verdict = outcome.get("new_correct")
        if not isinstance(task_id, str) or not task_id or task_id in task_ids:
            raise LedgerError(f"invalid/duplicate OOF task_id at row {index}")
        if not isinstance(verdict, bool):
            raise LedgerError(f"OOF task_outcomes[{index}].new_correct is not boolean")
        task_ids.add(task_id)
        counted += int(verdict)
    if counted != correct:
        raise LedgerError("OOF task outcome count disagrees with the OOF metric")
    return {
        "result_kind": "oof_estimate",
        "correct": correct,
        "denominator": denominator,
        "accuracy": correct / denominator,
        "report": {
            "path": _relative(path, repo_root),
            "sha256": sha256_file(path),
        },
        "validity": report["primary_metric_validity"],
        "evaluation_gold_access": True,
        "promotion_eligible": False,
    }


def discover_branch_candidates(
    branch: Mapping[str, Any], repo_root: Path, reports_root: Path
) -> list[Path]:
    found: dict[str, Path] = {}
    for pattern in branch["report_globs"]:
        normalized = pattern.replace("\\", "/")
        for candidate in repo_root.glob(normalized):
            if not candidate.is_file():
                continue
            resolved = _ensure_inside(candidate, reports_root, "candidate report")
            if candidate_has_final_path(resolved):
                found[str(resolved).casefold()] = resolved
    return sorted(found.values(), key=lambda item: _relative(item, repo_root))


def build_ledger(
    repo_root: Path, registry_path: Path
) -> Mapping[str, Any]:
    repo_root = repo_root.resolve()
    registry_path = registry_path.resolve()
    _ensure_inside(registry_path, repo_root, "registry")
    reports_root = (repo_root / "reports").resolve()
    registry, pins = load_registry(registry_path)

    rows: list[Mapping[str, Any]] = []
    for branch in registry["branches"]:
        if branch.get("superseded_before_calls") is True:
            attestation = validate_supersession_attestation(
                branch, repo_root, reports_root
            )
            rows.append(
                {
                    "id": branch["id"],
                    "label": branch.get("label", branch["id"]),
                    "preregistered": True,
                    "status": "non_final",
                    "result_kind": "superseded_before_calls",
                    "terminal_state": "SUPERSEDED_BEFORE_CALLS",
                    "reason": (
                        "superseded before any source/model calls; excluded from "
                        "ranking and never eligible for a final score"
                    ),
                    "source_call_count": 0,
                    "validity": {
                        "source_call_count_verified_zero": True,
                        "attestation_path_and_sha256_verified": True,
                    },
                    "supersession_attestation": attestation,
                    "rejected_candidates": [],
                }
            )
            continue

        candidates = discover_branch_candidates(branch, repo_root, reports_root)
        result_kind = branch.get("result_kind", "final_candidate")
        if result_kind == "oof_estimate":
            audited: list[Mapping[str, Any]] = []
            rejected: list[Mapping[str, Any]] = []
            for candidate in candidates:
                try:
                    audited.append(validate_oof_estimate(candidate, repo_root, pins))
                except LedgerError as exc:
                    rejected.append(
                        {
                            "path": _relative(candidate, repo_root),
                            "sha256": sha256_file(candidate),
                            "reason": str(exc),
                        }
                    )
            common = {
                "id": branch["id"],
                "label": branch.get("label", branch["id"]),
                "preregistered": True,
                "result_kind": "oof_estimate",
            }
            if len(audited) == 1:
                rows.append(
                    {
                        **common,
                        "status": "non_final",
                        "reason": "OOF same-benchmark estimate; not an external holdout final",
                        **audited[0],
                        "rejected_candidates": rejected,
                    }
                )
            else:
                rows.append(
                    {
                        **common,
                        "status": "pending" if not audited else "conflict",
                        "reason": (
                            "no valid OOF estimate artifact"
                            if not audited
                            else "multiple valid OOF estimate artifacts"
                        ),
                        **({"audited_candidates": audited} if audited else {}),
                        "rejected_candidates": rejected,
                    }
                )
            continue
        accepted: list[Mapping[str, Any]] = []
        rejected: list[Mapping[str, Any]] = []
        for candidate in candidates:
            try:
                accepted.append(
                    validate_candidate(
                        candidate,
                        repo_root,
                        pins,
                        allow_legacy_without_orchestration=branch.get(
                            "allow_legacy_finalizer_without_orchestration", False
                        ),
                    )
                )
            except LedgerError as exc:
                rejected.append(
                    {
                        "path": _relative(candidate, repo_root),
                        "sha256": sha256_file(candidate),
                        "reason": str(exc),
                    }
                )

        common = {
            "id": branch["id"],
            "label": branch.get("label", branch["id"]),
            "preregistered": True,
        }
        if len(accepted) == 1:
            rows.append(
                {
                    **common,
                    "status": "final",
                    **accepted[0],
                    "rejected_candidates": rejected,
                }
            )
        elif len(accepted) > 1:
            rows.append(
                {
                    **common,
                    "status": "conflict",
                    "reason": "multiple valid final artifacts match this branch",
                    "accepted_candidates": accepted,
                    "rejected_candidates": rejected,
                }
            )
        else:
            rows.append(
                {
                    **common,
                    "status": "pending",
                    "reason": (
                        "no valid finalized matched-judge full274 report"
                        if candidates
                        else "no finalized matched-judge report found"
                    ),
                    "rejected_candidates": rejected,
                }
            )

    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("final", "pending", "conflict", "non_final")
    }
    finals = [row for row in rows if row["status"] == "final"]
    best = max(finals, key=lambda row: (row["correct"], row["id"])) if finals else None
    registry_sha = sha256_file(registry_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": {
            "rows": pins.benchmark_rows,
            "sha256": pins.benchmark_sha256,
        },
        "judge": {
            "lineage": pins.judge_lineage,
            "image_judge_rows": pins.image_judge_rows,
        },
        "policy": {
            "accepted_artifacts": [
                "matched_score.json",
                "score.json with hash-valid finalization manifest",
                "canonical postgeneration score.json with exact hash-bound orchestration",
                "exact frozen-baseline replay score.json",
            ],
            "requires_exact_274_task_outcomes": True,
            "rejects": ["bounds", "partial", "interim", "in_sample"],
            "missing_preregistered_branch_status": "pending",
            "superseded_before_calls_status": "non_final",
            "superseded_requires_zero_calls_and_hash_valid_attestation": True,
        },
        "registry": {
            "path": _relative(registry_path, repo_root),
            "sha256": registry_sha,
        },
        "summary": {
            "branches": len(rows),
            **status_counts,
            "rejected_candidates": sum(len(row["rejected_candidates"]) for row in rows),
            "best_final": (
                {
                    "id": best["id"],
                    "correct": best["correct"],
                    "denominator": best["denominator"],
                    "accuracy": best["accuracy"],
                }
                if best
                else None
            ),
        },
        "branches": rows,
    }


def render_markdown(ledger: Mapping[str, Any]) -> str:
    benchmark = ledger["benchmark"]
    judge = ledger["judge"]
    summary = ledger["summary"]
    lines = [
        "# Frozen full274 results ledger",
        "",
        f"Benchmark: `{benchmark['sha256']}` ({benchmark['rows']} tasks).  ",
        f"Matched judge lineage: `{judge['lineage']}`.",
        "",
        (
            f"Final: **{summary['final']}**; pending: **{summary['pending']}**; "
            f"non-final estimates: **{summary['non_final']}**; "
            f"conflicts: **{summary['conflict']}**."
        ),
        "",
        "| Branch | Status | Correct | Accuracy | Accepted artifact |",
        "|---|---:|---:|---:|---|",
    ]
    for row in ledger["branches"]:
        if row["status"] == "final" or (
            row["status"] == "non_final"
            and all(
                key in row
                for key in ("correct", "denominator", "accuracy", "report")
            )
        ):
            correct = f"{row['correct']}/{row['denominator']}"
            accuracy = f"{100.0 * row['accuracy']:.3f}%"
            artifact = f"`{row['report']['path']}`"
        else:
            correct = "\u2014"
            accuracy = "\u2014"
            artifact = "\u2014"
        lines.append(
            f"| {row['label']} | {row['status']} | {correct} | {accuracy} | {artifact} |"
        )

    rejections = [
        (row, rejected)
        for row in ledger["branches"]
        for rejected in row.get("rejected_candidates", [])
    ]
    if rejections:
        lines.extend(["", "## Rejected candidates", ""])
        for row, rejected in rejections:
            lines.append(
                f"- **{row['label']}** \u2014 `{rejected['path']}`: {rejected['reason']}"
            )
    lines.extend(
        [
            "",
            "> Pending entries are preregistered branches without an admissible final report; "
            "bounds, partial/interim, in-sample, and OOF estimates are never shown as final.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    ledger: Mapping[str, Any], out_json: Path, out_markdown: Path
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    out_markdown.write_text(render_markdown(ledger), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    registry = args.registry
    if not registry.is_absolute():
        registry = repo_root / registry
    out_json = args.out_json
    if not out_json.is_absolute():
        out_json = repo_root / out_json
    out_md = args.out_md
    if not out_md.is_absolute():
        out_md = repo_root / out_md
    ledger = build_ledger(repo_root, registry)
    write_outputs(ledger, out_json, out_md)
    print(
        json.dumps(
            {
                "out_json": str(out_json.resolve()),
                "out_md": str(out_md.resolve()),
                "summary": ledger["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
