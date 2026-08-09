from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ReplayAggregateError(RuntimeError):
    """Raised when a replay bundle cannot support its stated provenance."""


EXPECTED_MODEL = "Qwen/Qwen3.5-9B"
COMPARISON_SCHEMA = "vlm-9b-milestone-comparison-v2"
AGGREGATE_SCHEMA = "vlm-9b-milestone-aggregate-v2"
SCORE_SCHEMA = "vlm-9b-milestone-score-v2"
NORMALIZED_V2_ADAPTER = "normalized_v2"
SOURCE_REPLAY_V1_ADAPTER = "maxim_9b_source_replay_aggregate_v1"
SOURCE_REPLAY_V1_SCHEMA = "maxim-9b-source-replay-aggregate-v1"
SOURCE_REPLAY_SCORE_V1_SCHEMA = "maxim-full274-score-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PORTABLE_REBASE_TOP_LEVELS = frozenset({"reports", "configs", "artifacts", "scripts"})
PORTABLE_REBASE_MAX_ANCESTORS = 8

MILESTONE_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("page_rag_9b", "Page RAG 9B", "page_rag", "historical_output_control"),
    (
        "no_tools_9b",
        "No-tools 9B",
        "model_only",
        "matched_judge_replay_partial_generation_provenance",
    ),
    (
        "query_active_crop_v2_9b",
        "Query Active Crop V2 9B · final anchor",
        "query_active_crop_v2",
        "preregistered_gold_blind",
    ),
    (
        "source_v1_rebase_9b",
        "Source V1 · 9B rebase",
        "source_rebase_v1",
        "new_profile_bound_replay",
    ),
    (
        "source_v3_rebase_9b",
        "Source V3 · 9B rebase",
        "source_rebase_v3",
        "new_profile_bound_replay",
    ),
    (
        "source_v6_rebase_9b",
        "Source V6 · 9B rebase",
        "source_rebase_v6",
        "new_profile_bound_replay",
    ),
    (
        "source_v7_rebase_9b",
        "Source-adjudicated V7 · 9B rebase",
        "source_rebase_v7",
        "new_profile_bound_replay",
    ),
)
MILESTONE_PIPELINES = {item[0]: item[2] for item in MILESTONE_SPECS}
MILESTONE_LABELS = {item[0]: item[1] for item in MILESTONE_SPECS}
MILESTONE_PROVENANCE = {item[0]: item[3] for item in MILESTONE_SPECS}
PROVENANCE_STATUSES = {
    "historical_output_control",
    "matched_judge_replay_partial_generation_provenance",
    "preregistered_gold_blind",
    "new_profile_bound_replay",
}
INTERMEDIATE_TIMELINE: tuple[tuple[str, str], ...] = (
    ("source_v2", "Source V2 · intermediate provenance step"),
    ("source_v4", "Source V4 · intermediate provenance step"),
    ("source_v5", "Source V5 · intermediate provenance step"),
)
EMPTY_UNION_SHA256 = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class FrozenReplayAggregate:
    milestone_id: str
    native_adapter: str
    model: str
    pipeline: str
    provenance_status: str
    bound_before_score: bool | None
    caveats: tuple[str, ...]
    rows: int
    correct: int
    accuracy: float
    slices: dict[str, dict[str, Any]]
    model_closure: dict[str, Any]
    source_union: dict[str, Any]
    comparisons: tuple[dict[str, Any], ...]
    evaluator: dict[str, Any]
    final_origin_counts: dict[str, int]
    aggregate_path: Path
    aggregate_sha256: str
    benchmark_path: Path
    benchmark_sha256: str
    solver_path: Path
    solver_sha256: str
    raw_solver_path: Path | None
    raw_solver_sha256: str | None
    score_path: Path
    score_sha256: str
    judge_path: Path
    judge_sha256: str
    anchor_path: Path | None
    anchor_sha256: str | None
    certificate_paths: tuple[Path, ...]
    certificate_sha256s: tuple[str, ...]
    provenance_manifest_paths: tuple[Path, ...]
    provenance_manifest_sha256s: tuple[str, ...]

    def validation_report(self) -> dict[str, Any]:
        return {
            "status": "validated_9b_milestone",
            "milestone_id": self.milestone_id,
            "native_adapter": self.native_adapter,
            "milestone_label": MILESTONE_LABELS[self.milestone_id],
            "model": self.model,
            "pipeline": self.pipeline,
            "provenance_status": self.provenance_status,
            "bound_before_score": self.bound_before_score,
            "caveats": list(self.caveats),
            "rows": self.rows,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "slices": self.slices,
            "model_closure": self.model_closure,
            "source_union": self.source_union,
            "comparisons": list(self.comparisons),
            "evaluator": self.evaluator,
            "final_origin_counts": self.final_origin_counts,
            "hashes": {
                "aggregate": self.aggregate_sha256,
                "benchmark": self.benchmark_sha256,
                "solver": self.solver_sha256,
                "raw_solver": self.raw_solver_sha256,
                "score": self.score_sha256,
                "judge": self.judge_sha256,
                "anchor": self.anchor_sha256,
                "certificates": list(self.certificate_sha256s),
                "provenance_manifests": list(self.provenance_manifest_sha256s),
            },
        }


@dataclass(frozen=True)
class FrozenReplayComparison:
    manifest_path: Path
    manifest_sha256: str
    benchmark_path: Path
    benchmark_sha256: str
    milestones: tuple[FrozenReplayAggregate, ...]

    @property
    def final(self) -> FrozenReplayAggregate:
        return self.milestones[-1]

    def validation_report(self) -> dict[str, Any]:
        return {
            "status": "validated_heterogeneous_9b_comparison",
            "claim": "seven 9B-closed milestones with explicit provenance strength",
            "model": EXPECTED_MODEL,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "benchmark_sha256": self.benchmark_sha256,
            "milestones": [item.validation_report() for item in self.milestones],
        }


def unloaded_replay_report() -> dict[str, Any]:
    return {
        "status": "not_loaded",
        "claim": None,
        "reason": (
            "no explicit seven-stage manifest was supplied; no 9B score is shown "
            "without its artifact hashes, model closure and provenance status"
        ),
    }


def empty_milestone_schema() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "milestone_id": milestone_id,
            "label": label,
            "pipeline": pipeline,
            "provenance_status": provenance,
            "status": "awaiting_validated_aggregate",
        }
        for milestone_id, label, pipeline, provenance in MILESTONE_SPECS
    )


def intermediate_timeline_schema() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "stage_id": stage_id,
            "label": label,
            "status": "timeline_only_not_a_primary_comparison_card",
        }
        for stage_id, label in INTERMEDIATE_TIMELINE
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReplayAggregateError(f"cannot hash replay artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReplayAggregateError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayAggregateError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayAggregateError(f"{label} must be a JSON object: {path}")
    return value


def _strict_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    missing = allowed - set(value)
    unknown = set(value) - allowed
    if missing or unknown:
        raise ReplayAggregateError(
            f"{label} schema mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _validate_sha(value: Any, label: str) -> str:
    parsed = str(value or "").casefold()
    if not SHA256_RE.fullmatch(parsed):
        raise ReplayAggregateError(f"{label} SHA-256 is malformed")
    return parsed


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _portable_absolute_rebase(base: Path, unresolved: Path, label: str) -> Path:
    """Rebase one missing absolute artifact path into the current repository.

    Frozen native reports predate portable locators.  We retain their exact SHA
    authority while allowing a teammate clone to move: only one allowlisted
    top-level suffix may be extracted, and only one bounded ancestor candidate
    may exist inside the current repository root.
    """

    parts = unresolved.parts
    top_level_positions = [
        index
        for index, part in enumerate(parts)
        if part.casefold() in PORTABLE_REBASE_TOP_LEVELS
    ]
    if len(top_level_positions) != 1:
        raise ReplayAggregateError(
            f"{label} missing absolute path has an unknown or ambiguous portable prefix"
        )
    suffix_parts = parts[top_level_positions[0] :]
    if any(part in {"", ".", ".."} for part in suffix_parts):
        raise ReplayAggregateError(f"{label} portable path traversal is forbidden")
    suffix = Path(*suffix_parts)

    resolved_base = base.expanduser().resolve()
    repo_roots: list[Path] = []
    cursor = resolved_base
    for _ in range(PORTABLE_REBASE_MAX_ANCESTORS + 1):
        if any((cursor / name).is_dir() for name in PORTABLE_REBASE_TOP_LEVELS):
            repo_roots.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent

    candidates: list[Path] = []
    for repo_root in repo_roots:
        candidate = (repo_root / suffix).resolve()
        if not _is_within(candidate, repo_root):
            raise ReplayAggregateError(f"{label} portable rebase escaped repository root")
        if candidate.is_file() and candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        raise ReplayAggregateError(
            f"{label} is missing and no bounded portable rebase candidate exists: "
            f"{unresolved}"
        )
    if len(candidates) != 1:
        raise ReplayAggregateError(
            f"{label} portable rebase is ambiguous: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    return candidates[0]


def _resolve_artifact_locator(base: Path, locator: str, label: str) -> Path:
    unresolved = Path(locator).expanduser()
    if not unresolved.is_absolute():
        return (base / unresolved).resolve()
    original = unresolved.resolve()
    if original.is_file():
        return original
    return _portable_absolute_rebase(base, unresolved, label)


def _resolve_descriptor(base: Path, value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise ReplayAggregateError(f"{label} descriptor must be an object")
    _strict_keys(value, {"path", "sha256"}, f"{label} descriptor")
    locator = value["path"]
    if not isinstance(locator, str) or not locator.strip():
        raise ReplayAggregateError(f"{label} path must be a non-empty string")
    path = _resolve_artifact_locator(base, locator, label)
    expected = _validate_sha(value["sha256"], label)
    if _sha256(path) != expected:
        raise ReplayAggregateError(f"{label} SHA-256 mismatch: {path}")
    return path, expected


def _resolve_native_descriptor(
    aggregate_dir: Path,
    value: Any,
    label: str,
    *,
    rows_required: bool = False,
) -> tuple[Path, str, int | None]:
    """Resolve a descriptor emitted by the native replay pipeline.

    Native reports predate the portable comparison wrapper and therefore contain
    absolute paths.  The wrapper still pins the aggregate itself, and every path
    imported from it is independently SHA-checked here.  Relative paths are
    resolved against the native aggregate directory.
    """

    if not isinstance(value, dict):
        raise ReplayAggregateError(f"{label} descriptor must be an object")
    allowed = {"path", "sha256", "rows"} if rows_required else {"path", "sha256"}
    _strict_keys(value, allowed, f"{label} descriptor")
    locator = value.get("path")
    if not isinstance(locator, str) or not locator.strip():
        raise ReplayAggregateError(f"{label} path must be a non-empty string")
    path = _resolve_artifact_locator(aggregate_dir, locator, label)
    expected = _validate_sha(value.get("sha256"), label)
    if _sha256(path) != expected:
        raise ReplayAggregateError(f"{label} SHA-256 mismatch: {path}")
    rows: int | None = None
    if rows_required:
        rows = _positive_int(value.get("rows"), f"{label}.rows")
        if len(_read_jsonl(path, label)) != rows:
            raise ReplayAggregateError(f"{label} declared row count differs from file")
    return path, expected, rows


def _canonical_json_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_role_descriptors(
    base: Path, values: Any, label: str
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    if not isinstance(values, list):
        raise ReplayAggregateError(f"{label} must be an array")
    paths: list[Path] = []
    hashes: list[str] = []
    roles: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ReplayAggregateError(f"{label}[{index}] must be an object")
        _strict_keys(value, {"role", "path", "sha256"}, f"{label}[{index}]")
        role = str(value["role"] or "")
        if not role or role in roles:
            raise ReplayAggregateError(f"{label} roles must be non-empty and unique")
        roles.add(role)
        path, digest = _resolve_descriptor(
            base,
            {"path": value["path"], "sha256": value["sha256"]},
            f"{label}[{index}]",
        )
        paths.append(path)
        hashes.append(digest)
    return tuple(paths), tuple(hashes)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReplayAggregateError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReplayAggregateError(f"{label} must be a non-negative integer")
    return value


def _metric(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReplayAggregateError(f"{label} must be an object")
    _strict_keys(value, {"rows", "correct", "accuracy"}, label)
    rows = _positive_int(value["rows"], f"{label}.rows")
    correct = _nonnegative_int(value["correct"], f"{label}.correct")
    if correct > rows:
        raise ReplayAggregateError(f"{label}.correct exceeds rows")
    accuracy = value["accuracy"]
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        raise ReplayAggregateError(f"{label}.accuracy must be numeric")
    accuracy = float(accuracy)
    if not math.isfinite(accuracy) or not math.isclose(
        accuracy, correct / rows, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ReplayAggregateError(f"{label}.accuracy disagrees with correct/rows")
    return {"rows": rows, "correct": correct, "accuracy": accuracy}


def _metrics(value: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not isinstance(value, dict):
        raise ReplayAggregateError("metrics must be an object")
    _strict_keys(value, {"rows", "correct", "accuracy", "slices"}, "metrics")
    overall = _metric({key: value[key] for key in ("rows", "correct", "accuracy")}, "metrics")
    slices_value = value["slices"]
    if not isinstance(slices_value, dict):
        raise ReplayAggregateError("metrics.slices must be an object")
    expected = {"deterministic", "image", "math", "non_math"}
    _strict_keys(slices_value, expected, "metrics.slices")
    slices = {key: _metric(slices_value[key], f"metrics.slices.{key}") for key in expected}
    for left, right in (("deterministic", "image"), ("math", "non_math")):
        if (
            slices[left]["rows"] + slices[right]["rows"] != overall["rows"]
            or slices[left]["correct"] + slices[right]["correct"] != overall["correct"]
        ):
            raise ReplayAggregateError(f"{left}+{right} slices do not close to overall")
    return overall, slices


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ReplayAggregateError(f"{label}:{line_number} is not an object")
                task_id = str(row.get("task_id") or "")
                if not task_id or task_id in seen:
                    raise ReplayAggregateError(
                        f"{label}:{line_number} has missing or duplicate task_id"
                    )
                seen.add(task_id)
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayAggregateError(f"cannot validate {label} {path}: {exc}") from exc
    return rows


def _read_jsonl_bytes_by_task(path: Path, label: str) -> dict[str, bytes]:
    """Index raw JSONL records without treating newline style as row content."""

    result: dict[str, bytes] = {}
    try:
        for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line.decode("utf-8"))
            if not isinstance(row, dict):
                raise ReplayAggregateError(f"{label}:{line_number} is not an object")
            task_id = str(row.get("task_id") or "")
            if not task_id or task_id in result:
                raise ReplayAggregateError(
                    f"{label}:{line_number} has missing or duplicate task_id"
                )
            result[task_id] = raw_line
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayAggregateError(f"cannot validate raw rows for {label} {path}: {exc}") from exc
    return result


def _task_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row["task_id"]) for row in rows}


def _validate_raw_solver_projection(
    raw_rows: list[dict[str, Any]],
    normalized_rows: list[dict[str, Any]],
) -> None:
    """Prove that a normalized legacy solver retains one exact raw solver.

    The normalized v2 adapter is allowed to add only ``final_origin``.  Keeping
    this check here prevents an arbitrary SHA-pinned JSONL from being presented
    as the raw identity used by native source-replay anchor comparisons.
    """

    if _task_ids(raw_rows) != _task_ids(normalized_rows):
        raise ReplayAggregateError(
            "raw_solver task set differs from the normalized solver"
        )
    normalized_by_task = {str(row["task_id"]): row for row in normalized_rows}
    for raw_row in raw_rows:
        task_id = str(raw_row["task_id"])
        projected = dict(raw_row)
        normalized = normalized_by_task[task_id]
        if "final_origin" not in projected:
            projected["final_origin"] = normalized.get("final_origin")
        if projected != normalized:
            raise ReplayAggregateError(
                f"raw_solver row {task_id} is not the exact final_origin-only "
                "source projection of the normalized solver"
            )


def _accepted_certificate_ids(paths: Iterable[Path]) -> set[str]:
    accepted: set[str] = set()
    for index, path in enumerate(paths):
        for row in _read_jsonl(path, f"certificates[{index}]"):
            generation = row.get("generation") or {}
            is_accepted = (
                (row.get("status") == "pass" and row.get("strength") == "strong")
                or row.get("source_certificate") is True
                or generation.get("source_certificate") is True
            )
            if is_accepted:
                accepted.add(str(row["task_id"]))
    return accepted


def _union_sha(task_ids: set[str]) -> str:
    if not task_ids:
        return EMPTY_UNION_SHA256
    return hashlib.sha256(("\n".join(sorted(task_ids)) + "\n").encode("utf-8")).hexdigest()


def _validate_provenance(
    aggregate: dict[str, Any], milestone_id: str, manifest_count: int
) -> tuple[str, bool | None, tuple[str, ...]]:
    status = str(aggregate["provenance_status"])
    if status not in PROVENANCE_STATUSES or status != MILESTONE_PROVENANCE[milestone_id]:
        raise ReplayAggregateError(f"{milestone_id}: provenance status is invalid")
    bound = aggregate["bound_before_score"]
    if bound not in (True, False, None):
        raise ReplayAggregateError("bound_before_score must be true, false or null")
    caveats_value = aggregate["caveats"]
    if not isinstance(caveats_value, list) or any(
        not isinstance(item, str) or not item.strip() for item in caveats_value
    ):
        raise ReplayAggregateError("caveats must be an array of non-empty strings")
    caveats = tuple(caveats_value)
    if status in {
        "historical_output_control",
        "matched_judge_replay_partial_generation_provenance",
    } and not caveats:
        raise ReplayAggregateError(f"{status} requires an explicit caveat")
    if status in {"preregistered_gold_blind", "new_profile_bound_replay"}:
        if bound is not True or manifest_count == 0:
            raise ReplayAggregateError(
                f"{status} requires bound_before_score=true and a hashed provenance manifest"
            )
    return status, bound, caveats


def _validate_model_and_origins(
    rows: list[dict[str, Any]], declared: Any
) -> tuple[dict[str, Any], dict[str, int]]:
    if not isinstance(declared, dict):
        raise ReplayAggregateError("model_closure must be an object")
    _strict_keys(
        declared,
        {"expected_model", "checked_rows", "matching_rows", "foreign_models"},
        "model_closure",
    )
    foreign: set[str] = set()
    origin_counts = {
        "model_anchor": 0,
        "deterministic_source_replacement": 0,
        "unknown": 0,
    }
    for line_number, row in enumerate(rows, start=1):
        model = str(row.get("base_row_model") or row.get("model") or "")
        if model != EXPECTED_MODEL:
            foreign.add(model or "<missing>")
        origin = str(row.get("final_origin") or "unknown")
        if origin not in origin_counts:
            raise ReplayAggregateError(f"solver:{line_number} has unsupported final_origin")
        origin_counts[origin] += 1
    expected_declared = {
        "expected_model": EXPECTED_MODEL,
        "checked_rows": len(rows),
        "matching_rows": len(rows) - sum(
            1
            for row in rows
            if str(row.get("base_row_model") or row.get("model") or "") != EXPECTED_MODEL
        ),
        "foreign_models": sorted(foreign),
    }
    if declared != expected_declared or foreign:
        raise ReplayAggregateError("solver rows do not satisfy exact Qwen3.5-9B model closure")
    return expected_declared, origin_counts


def _validate_source_union(
    value: Any,
    certificate_paths: tuple[Path, ...],
    absence_reason: Any,
    origin_counts: dict[str, int],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReplayAggregateError("source_union must be an object")
    _strict_keys(
        value,
        {"sha256", "size", "replacements", "confirmations", "stage_counts"},
        "source_union",
    )
    stage_counts = value["stage_counts"]
    if not isinstance(stage_counts, dict) or any(
        not isinstance(key, str)
        or not key
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for key, count in stage_counts.items()
    ):
        raise ReplayAggregateError("source_union.stage_counts is invalid")
    accepted = _accepted_certificate_ids(certificate_paths)
    parsed = {
        "sha256": _validate_sha(value["sha256"], "source union"),
        "size": _nonnegative_int(value["size"], "source_union.size"),
        "replacements": _nonnegative_int(value["replacements"], "source_union.replacements"),
        "confirmations": _nonnegative_int(value["confirmations"], "source_union.confirmations"),
        "stage_counts": stage_counts,
    }
    if not certificate_paths:
        if not isinstance(absence_reason, str) or not absence_reason.strip():
            raise ReplayAggregateError("empty certificates require certificate_absence_reason")
        if parsed != {
            "sha256": EMPTY_UNION_SHA256,
            "size": 0,
            "replacements": 0,
            "confirmations": 0,
            "stage_counts": {},
        }:
            raise ReplayAggregateError("non-source milestone must have an empty source union")
    else:
        if absence_reason is not None:
            raise ReplayAggregateError("certificate_absence_reason must be null when certificates exist")
        if parsed["sha256"] != _union_sha(accepted) or parsed["size"] != len(accepted):
            raise ReplayAggregateError("source union hash/size differs from certificate artifacts")
        if parsed["replacements"] + parsed["confirmations"] != parsed["size"]:
            raise ReplayAggregateError("source replacement+confirmation counts do not close")
    if parsed["replacements"] != origin_counts["deterministic_source_replacement"]:
        raise ReplayAggregateError("source replacements disagree with solver final origins")
    return parsed


def _validate_evaluator(value: Any, rows: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReplayAggregateError("evaluator must be an object")
    _strict_keys(
        value,
        {
            "semantics",
            "deterministic_rows",
            "image_rows",
            "source_certified_image_rows",
            "model_judged_image_rows",
            "judge_model",
        },
        "evaluator",
    )
    parsed = {
        "semantics": str(value["semantics"] or ""),
        "deterministic_rows": _nonnegative_int(value["deterministic_rows"], "evaluator.deterministic_rows"),
        "image_rows": _nonnegative_int(value["image_rows"], "evaluator.image_rows"),
        "source_certified_image_rows": _nonnegative_int(
            value["source_certified_image_rows"], "evaluator.source_certified_image_rows"
        ),
        "model_judged_image_rows": _nonnegative_int(
            value["model_judged_image_rows"], "evaluator.model_judged_image_rows"
        ),
        "judge_model": value["judge_model"],
    }
    if not parsed["semantics"]:
        raise ReplayAggregateError("evaluator.semantics must be explicit")
    if parsed["judge_model"] not in (None, EXPECTED_MODEL):
        raise ReplayAggregateError("evaluator judge_model breaks the 9B closure")
    if parsed["deterministic_rows"] + parsed["image_rows"] != rows:
        raise ReplayAggregateError("evaluator deterministic+image rows do not close")
    if (
        parsed["source_certified_image_rows"] + parsed["model_judged_image_rows"]
        != parsed["image_rows"]
    ):
        raise ReplayAggregateError("evaluator image split does not close")
    return parsed


def _validate_comparisons(value: Any, rows: int) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ReplayAggregateError("comparisons must be an array")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ReplayAggregateError(f"comparisons[{index}] must be an object")
        _strict_keys(
            item,
            {
                "baseline_milestone_id",
                "baseline_solver_sha256",
                "fixes",
                "regressions",
                "unchanged",
            },
            f"comparisons[{index}]",
        )
        baseline = str(item["baseline_milestone_id"])
        if baseline not in MILESTONE_PIPELINES or baseline in seen:
            raise ReplayAggregateError("comparison baseline must be a unique primary milestone")
        seen.add(baseline)
        parsed = {
            "baseline_milestone_id": baseline,
            "baseline_solver_sha256": _validate_sha(
                item["baseline_solver_sha256"], f"comparisons[{index}].baseline_solver"
            ),
            "fixes": _nonnegative_int(item["fixes"], f"comparisons[{index}].fixes"),
            "regressions": _nonnegative_int(
                item["regressions"], f"comparisons[{index}].regressions"
            ),
            "unchanged": _nonnegative_int(item["unchanged"], f"comparisons[{index}].unchanged"),
        }
        if parsed["fixes"] + parsed["regressions"] + parsed["unchanged"] != rows:
            raise ReplayAggregateError("comparison transition counts do not close")
        result.append(parsed)
    return tuple(result)


def _load_aggregate(
    path: Path,
    *,
    base: Path,
    benchmark_path: Path,
    benchmark_hash: str,
    expected_milestone_id: str,
) -> FrozenReplayAggregate:
    aggregate = _read_object(path, "9B milestone aggregate")
    _strict_keys(
        aggregate,
        {
            "schema_version",
            "milestone_id",
            "model",
            "pipeline",
            "provenance_status",
            "bound_before_score",
            "caveats",
            "provenance_manifests",
            "artifacts",
            "certificate_absence_reason",
            "benchmark_sha256",
            "metrics",
            "model_closure",
            "source_union",
            "comparisons",
            "evaluator",
            "final_origin_counts",
        },
        "9B milestone aggregate",
    )
    milestone_id = str(aggregate["milestone_id"])
    if aggregate["schema_version"] != AGGREGATE_SCHEMA or milestone_id != expected_milestone_id:
        raise ReplayAggregateError("aggregate schema or milestone id mismatch")
    if aggregate["model"] != EXPECTED_MODEL:
        raise ReplayAggregateError("aggregate model is not Qwen3.5-9B")
    pipeline = str(aggregate["pipeline"])
    if pipeline != MILESTONE_PIPELINES[milestone_id]:
        raise ReplayAggregateError("aggregate pipeline does not match its milestone")
    if _validate_sha(aggregate["benchmark_sha256"], "benchmark") != benchmark_hash:
        raise ReplayAggregateError("aggregate benchmark hash differs from comparison benchmark")

    provenance_paths, provenance_hashes = _resolve_role_descriptors(
        base, aggregate["provenance_manifests"], "provenance_manifests"
    )
    provenance_status, bound, caveats = _validate_provenance(
        aggregate, milestone_id, len(provenance_paths)
    )
    artifacts = aggregate["artifacts"]
    if not isinstance(artifacts, dict):
        raise ReplayAggregateError("artifacts must be an object")
    _strict_keys(
        artifacts,
        {"solver", "raw_solver", "score", "judge", "certificates"},
        "artifacts",
    )
    solver_path, solver_hash = _resolve_descriptor(base, artifacts["solver"], "solver")
    raw_solver_path, raw_solver_hash = _resolve_descriptor(
        base, artifacts["raw_solver"], "raw_solver"
    )
    score_path, score_hash = _resolve_descriptor(base, artifacts["score"], "score")
    judge_path, judge_hash = _resolve_descriptor(base, artifacts["judge"], "judge")
    certificate_paths, certificate_hashes = _resolve_role_descriptors(
        base, artifacts["certificates"], "certificates"
    )

    benchmark_rows = _read_jsonl(benchmark_path, "benchmark")
    solver_rows = _read_jsonl(solver_path, "solver")
    raw_solver_rows = _read_jsonl(raw_solver_path, "raw_solver")
    judge_rows = _read_jsonl(judge_path, "judge")
    benchmark_ids = _task_ids(benchmark_rows)
    if _task_ids(solver_rows) != benchmark_ids or _task_ids(judge_rows) != benchmark_ids:
        raise ReplayAggregateError("solver/judge task sets differ from the exact benchmark")
    _validate_raw_solver_projection(raw_solver_rows, solver_rows)
    overall, slices = _metrics(aggregate["metrics"])
    if len(benchmark_rows) != overall["rows"]:
        raise ReplayAggregateError("metrics rows differ from the benchmark row count")
    model_closure, scanned_origins = _validate_model_and_origins(
        solver_rows, aggregate["model_closure"]
    )
    declared_origins = aggregate["final_origin_counts"]
    if not isinstance(declared_origins, dict):
        raise ReplayAggregateError("final_origin_counts must be an object")
    _strict_keys(
        declared_origins,
        {"model_anchor", "deterministic_source_replacement", "unknown"},
        "final_origin_counts",
    )
    declared_origins = {
        key: _nonnegative_int(value, f"final_origin_counts.{key}")
        for key, value in declared_origins.items()
    }
    if declared_origins != scanned_origins or sum(declared_origins.values()) != overall["rows"]:
        raise ReplayAggregateError("declared final origins differ from solver rows")
    if declared_origins["unknown"] != 0:
        raise ReplayAggregateError("unknown final origins prevent a 9B-closed claim")
    source_union = _validate_source_union(
        aggregate["source_union"],
        certificate_paths,
        aggregate["certificate_absence_reason"],
        declared_origins,
    )
    evaluator = _validate_evaluator(aggregate["evaluator"], overall["rows"])
    comparisons = _validate_comparisons(aggregate["comparisons"], overall["rows"])

    score = _read_object(score_path, "9B milestone score")
    _strict_keys(
        score,
        {
            "schema_version",
            "milestone_id",
            "model",
            "pipeline",
            "benchmark_sha256",
            "solver_sha256",
            "judge_sha256",
            "certificate_sha256s",
            "metrics",
            "source_union",
            "comparisons",
            "evaluator",
            "final_origin_counts",
        },
        "9B milestone score",
    )
    expected_score = {
        "schema_version": SCORE_SCHEMA,
        "milestone_id": milestone_id,
        "model": EXPECTED_MODEL,
        "pipeline": pipeline,
        "benchmark_sha256": benchmark_hash,
        "solver_sha256": solver_hash,
        "judge_sha256": judge_hash,
        "certificate_sha256s": list(certificate_hashes),
        "metrics": aggregate["metrics"],
        "source_union": aggregate["source_union"],
        "comparisons": aggregate["comparisons"],
        "evaluator": aggregate["evaluator"],
        "final_origin_counts": aggregate["final_origin_counts"],
    }
    if score != expected_score:
        raise ReplayAggregateError("score projection differs from the hashed aggregate contract")

    return FrozenReplayAggregate(
        milestone_id=milestone_id,
        native_adapter=NORMALIZED_V2_ADAPTER,
        model=EXPECTED_MODEL,
        pipeline=pipeline,
        provenance_status=provenance_status,
        bound_before_score=bound,
        caveats=caveats,
        rows=overall["rows"],
        correct=overall["correct"],
        accuracy=overall["accuracy"],
        slices=slices,
        model_closure=model_closure,
        source_union=source_union,
        comparisons=comparisons,
        evaluator=evaluator,
        final_origin_counts=declared_origins,
        aggregate_path=path,
        aggregate_sha256=_sha256(path),
        benchmark_path=benchmark_path,
        benchmark_sha256=benchmark_hash,
        solver_path=solver_path,
        solver_sha256=solver_hash,
        raw_solver_path=raw_solver_path,
        raw_solver_sha256=raw_solver_hash,
        score_path=score_path,
        score_sha256=score_hash,
        judge_path=judge_path,
        judge_sha256=judge_hash,
        anchor_path=None,
        anchor_sha256=None,
        certificate_paths=certificate_paths,
        certificate_sha256s=certificate_hashes,
        provenance_manifest_paths=provenance_paths,
        provenance_manifest_sha256s=provenance_hashes,
    )


def _native_metric(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReplayAggregateError(f"{label} must be an object")
    rows = _positive_int(value.get("n"), f"{label}.n")
    correct = _nonnegative_int(value.get("new_correct"), f"{label}.new_correct")
    if correct > rows:
        raise ReplayAggregateError(f"{label}.new_correct exceeds n")
    accuracy = value.get("new_accuracy")
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        raise ReplayAggregateError(f"{label}.new_accuracy must be numeric")
    if not math.isclose(float(accuracy), correct / rows, rel_tol=0.0, abs_tol=1e-6):
        raise ReplayAggregateError(f"{label}.new_accuracy disagrees with new_correct/n")
    return {"rows": rows, "correct": correct, "accuracy": correct / rows}


def _validate_native_closures(aggregate: dict[str, Any]) -> tuple[list[str], list[str]]:
    generation_closure = aggregate.get("upstream_generation_model_closure")
    if generation_closure != [EXPECTED_MODEL]:
        raise ReplayAggregateError("native upstream generation model closure is not exactly 9B")
    answer_origin_closure = aggregate.get("answer_origin_closure")
    final_origins = aggregate.get("final_origin_counts")
    allowed_origins = {
        "qwen35_9b_anchor_passthrough",
        "official_source_confirmation_of_9b_anchor",
        "deterministic_official_source_replacement",
    }
    if not isinstance(final_origins, dict) or set(final_origins) != allowed_origins:
        raise ReplayAggregateError("native final origin count keys are invalid")
    expected_origins: list[str] = []
    for origin, count in final_origins.items():
        parsed_count = _nonnegative_int(count, f"final_origin_counts.{origin}")
        if parsed_count:
            expected_origins.append(origin)
    expected_origins.sort()
    if answer_origin_closure != expected_origins:
        raise ReplayAggregateError(
            "native answer-origin closure must be the sorted nonzero final-origin keys"
        )
    if "model_closure" in aggregate and aggregate["model_closure"] != generation_closure:
        raise ReplayAggregateError("legacy native model_closure is not a generation-only alias")
    return list(generation_closure), list(answer_origin_closure)


def _native_page_rag_comparison(
    value: Any,
    *,
    task_ids: set[str],
    overall: dict[str, Any],
) -> None:
    """Validate, but deliberately do not expose, the native Page RAG delta."""

    if not isinstance(value, dict):
        raise ReplayAggregateError("native comparisons must be an object")
    _strict_keys(
        value,
        {
            "authority",
            "fixed_count",
            "fixed_task_ids",
            "regressed_count",
            "regressed_task_ids",
            "net_correct_change",
        },
        "native Page RAG comparison",
    )
    if value["authority"] != "score_maxim_full274.changes_vs_frozen_page_rag":
        raise ReplayAggregateError("native Page RAG comparison authority is unknown")
    fixed_ids = value["fixed_task_ids"]
    regressed_ids = value["regressed_task_ids"]
    if not isinstance(fixed_ids, list) or not isinstance(regressed_ids, list):
        raise ReplayAggregateError("native Page RAG task-id lists are malformed")
    fixed = {str(item) for item in fixed_ids}
    regressed = {str(item) for item in regressed_ids}
    if (
        len(fixed) != len(fixed_ids)
        or len(regressed) != len(regressed_ids)
        or fixed & regressed
        or not (fixed | regressed) <= task_ids
    ):
        raise ReplayAggregateError("native Page RAG transition task sets are invalid")
    fixed_count = _nonnegative_int(value["fixed_count"], "native fixed_count")
    regressed_count = _nonnegative_int(value["regressed_count"], "native regressed_count")
    net = int(value["net_correct_change"])
    if fixed_count != len(fixed) or regressed_count != len(regressed):
        raise ReplayAggregateError("native Page RAG transition counts disagree with task ids")
    if fixed_count - regressed_count != net:
        raise ReplayAggregateError("native Page RAG net change does not close")
    if overall["correct"] - net < 0:
        raise ReplayAggregateError("native Page RAG comparison disagrees with overall score")


def _native_explicit_comparison(
    aggregate_dir: Path,
    value: Any,
    *,
    role: str,
    rows: int,
    current_correct: int,
) -> tuple[dict[str, Any] | None, tuple[Path, ...], tuple[str, ...]]:
    """Validate a native, explicitly labelled comparison and normalize it for the UI."""

    if value is None:
        return None, (), ()
    if not isinstance(value, dict):
        raise ReplayAggregateError(f"native comparison_vs_{role} must be an object or null")
    _strict_keys(
        value,
        {
            "role",
            "label",
            "before_correct",
            "after_correct",
            "before_accuracy",
            "after_accuracy",
            "fixed",
            "regressed",
            "both_correct",
            "both_wrong",
            "delta_correct",
            "solver",
            "score",
        },
        f"native comparison_vs_{role}",
    )
    if value["role"] != role:
        raise ReplayAggregateError(f"native comparison_vs_{role} has the wrong role")
    label = str(value["label"] or "")
    primary_by_label = {
        "ActiveCrop9B": "query_active_crop_v2_9b",
        "SourceV1": "source_v1_rebase_9b",
        "SourceV3": "source_v3_rebase_9b",
        "SourceV6": "source_v6_rebase_9b",
        "SourceV7": "source_v7_rebase_9b",
    }
    before = _nonnegative_int(value["before_correct"], f"comparison_vs_{role}.before_correct")
    after = _nonnegative_int(value["after_correct"], f"comparison_vs_{role}.after_correct")
    fixed = _nonnegative_int(value["fixed"], f"comparison_vs_{role}.fixed")
    regressed = _nonnegative_int(value["regressed"], f"comparison_vs_{role}.regressed")
    both_correct = _nonnegative_int(
        value["both_correct"], f"comparison_vs_{role}.both_correct"
    )
    both_wrong = _nonnegative_int(value["both_wrong"], f"comparison_vs_{role}.both_wrong")
    delta = int(value["delta_correct"])
    if (
        fixed + regressed + both_correct + both_wrong != rows
        or before != both_correct + regressed
        or after != both_correct + fixed
        or delta != fixed - regressed
        or after - before != delta
        or after != current_correct
    ):
        raise ReplayAggregateError(f"native comparison_vs_{role} transition counts do not close")
    for key, numerator in (("before_accuracy", before), ("after_accuracy", after)):
        accuracy = value[key]
        if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not math.isclose(
            float(accuracy), numerator / rows, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ReplayAggregateError(f"native comparison_vs_{role}.{key} is inconsistent")
    solver_path, solver_hash, solver_rows = _resolve_native_descriptor(
        aggregate_dir, value["solver"], f"comparison_vs_{role}.solver", rows_required=True
    )
    score_path, score_hash, _ = _resolve_native_descriptor(
        aggregate_dir, value["score"], f"comparison_vs_{role}.score"
    )
    if solver_rows != rows:
        raise ReplayAggregateError(f"native comparison_vs_{role} solver denominator differs")
    baseline_id = primary_by_label.get(label)
    normalized = None
    if baseline_id is not None:
        normalized = {
            "baseline_milestone_id": baseline_id,
            "baseline_solver_sha256": solver_hash,
            "fixes": fixed,
            "regressions": regressed,
            "unchanged": both_correct + both_wrong,
        }
    return normalized, (solver_path, score_path), (solver_hash, score_hash)


def _validate_native_judge_manifest(
    aggregate_dir: Path,
    manifest_path: Path,
    *,
    previous_output: tuple[Path, str] | None,
    original_9b_rows: dict[str, bytes] | None,
    image_rows: int,
    stage_index: int,
) -> tuple[tuple[Path, str], dict[str, bytes], dict[str, int]]:
    manifest = _read_object(manifest_path, f"stages[{stage_index}].judge_manifest")
    if "copied_9b_judge_rows_byte_identical" in manifest:
        raise ReplayAggregateError(
            "judge manifest uses misleading copied_9b_judge_rows_byte_identical field"
        )
    required = {
        "base_image_judge",
        "base_solver",
        "composition",
        "output",
        "source_adjudicated_image_rows",
        "stage_source_adjudicated_image_rows_count",
        "copied_base_judge_rows_byte_identical",
        "cumulative_source_adjudicated_image_rows_count",
        "cumulative_original_9b_judge_rows_count",
        "gold_access",
        "benchmark_candidate_or_outcome_access",
        "inherited_27b_outputs",
        "upstream_generation_model_closure",
    }
    missing = required - set(manifest)
    if missing:
        raise ReplayAggregateError(
            f"judge manifest lacks cumulative evaluator lineage fields: {sorted(missing)}"
        )
    if (
        manifest["gold_access"] is not False
        or manifest["benchmark_candidate_or_outcome_access"] is not False
        or manifest["inherited_27b_outputs"] is not False
        or manifest["upstream_generation_model_closure"] != [EXPECTED_MODEL]
    ):
        raise ReplayAggregateError("judge manifest breaks gold/model provenance closure")

    base_solver_path, _, _ = _resolve_native_descriptor(
        aggregate_dir,
        manifest["base_solver"],
        f"stages[{stage_index}].judge_manifest.base_solver",
    )
    composition = manifest["composition"]
    if not isinstance(composition, dict) or not isinstance(composition.get("solver"), dict):
        raise ReplayAggregateError("judge manifest composition solver descriptor is missing")
    composed_solver_path, _, composed_solver_rows = _resolve_native_descriptor(
        aggregate_dir,
        composition["solver"],
        f"stages[{stage_index}].judge_manifest.composition.solver",
        rows_required=True,
    )
    if composed_solver_rows is None:
        raise ReplayAggregateError("judge manifest composition solver row count is missing")
    base_solver = {
        str(row["task_id"]): row
        for row in _read_jsonl(base_solver_path, "judge manifest immediate-base solver")
    }
    composed_solver = {
        str(row["task_id"]): row
        for row in _read_jsonl(composed_solver_path, "judge manifest composed solver")
    }
    if set(base_solver) != set(composed_solver) or len(composed_solver) != composed_solver_rows:
        raise ReplayAggregateError("judge manifest base/composed solver task sets differ")

    base_path, base_hash, _ = _resolve_native_descriptor(
        aggregate_dir,
        manifest["base_image_judge"],
        f"stages[{stage_index}].judge_manifest.base_image_judge",
    )
    output_path, output_hash, output_count = _resolve_native_descriptor(
        aggregate_dir,
        manifest["output"],
        f"stages[{stage_index}].judge_manifest.output",
        rows_required=True,
    )
    if output_count != image_rows:
        raise ReplayAggregateError("judge manifest output does not use the image denominator")
    if previous_output is not None and (base_path, base_hash) != previous_output:
        raise ReplayAggregateError("judge manifest base is not the preceding stage output")

    base_raw = _read_jsonl_bytes_by_task(base_path, "judge immediate base")
    output_raw = _read_jsonl_bytes_by_task(output_path, "judge stage output")
    if len(base_raw) != image_rows or set(base_raw) != set(output_raw):
        raise ReplayAggregateError("judge stage base/output task sets do not close")
    if original_9b_rows is None:
        original_9b_rows = dict(base_raw)
    if set(original_9b_rows) != set(output_raw):
        raise ReplayAggregateError("judge output differs from original 9B image task set")

    copied_base = sum(output_raw[task_id] == base_raw[task_id] for task_id in output_raw)
    original_9b = sum(
        output_raw[task_id] == original_9b_rows[task_id] for task_id in output_raw
    )
    output_rows_by_id = {
        str(row["task_id"]): row for row in _read_jsonl(output_path, "judge stage output")
    }
    source_adjudicated_ids: set[str] = set()
    for task_id, row in output_rows_by_id.items():
        judge = row.get("judge")
        if not isinstance(judge, dict):
            raise ReplayAggregateError("judge output row lacks judge metadata")
        if judge.get("backend") == "deterministic-official-source-certificate":
            if judge.get("model") is not None:
                raise ReplayAggregateError("deterministic source verdict unexpectedly names a model")
            source_adjudicated_ids.add(task_id)
        elif judge.get("model") != EXPECTED_MODEL:
            raise ReplayAggregateError("non-source judge output is not an original 9B verdict")
    cumulative_source = len(source_adjudicated_ids)
    if cumulative_source + original_9b != image_rows:
        raise ReplayAggregateError(
            "cumulative source-adjudicated/original-9B evaluator split does not close"
        )

    stage_records = manifest["source_adjudicated_image_rows"]
    if not isinstance(stage_records, list):
        raise ReplayAggregateError("source_adjudicated_image_rows must remain a stage-local array")
    stage_ids: set[str] = set()
    for row_number, record in enumerate(stage_records, start=1):
        if not isinstance(record, dict):
            raise ReplayAggregateError("source adjudication record must be an object")
        _strict_keys(
            record,
            {"task_id", "verdict_origin", "stage_answer_action", "trace_fingerprint"},
            f"source_adjudicated_image_rows[{row_number}]",
        )
        task_id = str(record["task_id"] or "")
        if not task_id or task_id in stage_ids or task_id not in source_adjudicated_ids:
            raise ReplayAggregateError("stage-local source adjudication task set is invalid")
        if record["verdict_origin"] != "deterministic_official_source_adjudication":
            raise ReplayAggregateError("stage-local source verdict origin is invalid")
        action = record["stage_answer_action"]
        if action not in {
            "keep_immediate_base_confirmed_by_source",
            "replace_immediate_base_with_source",
        }:
            raise ReplayAggregateError("stage-local source answer action is invalid")
        answers_equal = str(base_solver[task_id].get("final_answer") or "") == str(
            composed_solver[task_id].get("final_answer") or ""
        )
        if answers_equal != (action == "keep_immediate_base_confirmed_by_source"):
            raise ReplayAggregateError("stage-local source answer action fails solver recount")
        _validate_sha(record["trace_fingerprint"], "source adjudication trace fingerprint")
        stage_ids.add(task_id)

    declared = {
        "stage_source_adjudicated_image_rows_count": _nonnegative_int(
            manifest["stage_source_adjudicated_image_rows_count"],
            "stage source-adjudicated image rows",
        ),
        "copied_base_judge_rows_byte_identical": _nonnegative_int(
            manifest["copied_base_judge_rows_byte_identical"],
            "copied immediate-base judge rows",
        ),
        "cumulative_source_adjudicated_image_rows_count": _nonnegative_int(
            manifest["cumulative_source_adjudicated_image_rows_count"],
            "cumulative source-adjudicated image rows",
        ),
        "cumulative_original_9b_judge_rows_count": _nonnegative_int(
            manifest["cumulative_original_9b_judge_rows_count"],
            "cumulative original-9B judge rows",
        ),
    }
    expected = {
        "stage_source_adjudicated_image_rows_count": len(stage_records),
        "copied_base_judge_rows_byte_identical": copied_base,
        "cumulative_source_adjudicated_image_rows_count": cumulative_source,
        "cumulative_original_9b_judge_rows_count": original_9b,
    }
    if declared != expected:
        raise ReplayAggregateError("judge manifest evaluator lineage counts fail row recount")
    return (output_path, output_hash), original_9b_rows, declared


def _native_stage_artifacts(
    aggregate_dir: Path,
    stages_value: Any,
    stage_counts_value: Any,
    certificate_bundle_value: Any,
    *,
    rows: int,
    image_rows: int,
) -> tuple[
    tuple[Path, ...],
    tuple[str, ...],
    tuple[Path, ...],
    tuple[str, ...],
    list[str],
    Path,
    str,
    Path,
    str,
    dict[str, int],
]:
    if not isinstance(stages_value, list) or not stages_value:
        raise ReplayAggregateError("native stages must be a non-empty array")
    if not isinstance(stage_counts_value, list) or len(stage_counts_value) != len(stages_value):
        raise ReplayAggregateError("native stage_counts must align one-to-one with stages")
    if not isinstance(certificate_bundle_value, list) or len(certificate_bundle_value) != len(
        stages_value
    ):
        raise ReplayAggregateError("native certificate_bundle must align one-to-one with stages")

    certificate_paths: list[Path] = []
    certificate_hashes: list[str] = []
    provenance_paths: list[Path] = []
    provenance_hashes: list[str] = []
    stage_names: list[str] = []
    last_solver_path: Path | None = None
    last_solver_hash: str | None = None
    previous_judge_output: tuple[Path, str] | None = None
    original_9b_judge_rows: dict[str, bytes] | None = None
    final_judge_lineage: dict[str, int] | None = None
    expected_stage_keys = {
        "name",
        "profile",
        "resolver_manifest",
        "candidates",
        "certificates",
        "composition_manifest",
        "decisions",
        "solver",
        "judge_manifest",
        "stage_answer_confirmations",
        "stage_answer_replacements",
        "stage_certificate_count",
    }
    for index, (stage, counts, bundle) in enumerate(
        zip(stages_value, stage_counts_value, certificate_bundle_value, strict=True)
    ):
        if not isinstance(stage, dict) or not isinstance(counts, dict) or not isinstance(bundle, dict):
            raise ReplayAggregateError(f"native stage {index} metadata is malformed")
        _strict_keys(stage, expected_stage_keys, f"native stages[{index}]")
        _strict_keys(
            counts,
            {
                "name",
                "stage_answer_confirmations",
                "stage_answer_replacements",
                "stage_certificate_count",
            },
            f"native stage_counts[{index}]",
        )
        _strict_keys(
            bundle,
            {
                "stage",
                "path",
                "rows",
                "sha256",
                "candidate_path",
                "candidate_sha256",
            },
            f"native certificate_bundle[{index}]",
        )
        name = str(stage["name"] or "")
        if not name or name in stage_names or counts["name"] != name or bundle["stage"] != name:
            raise ReplayAggregateError("native stage names are missing, duplicated or unbound")
        stage_names.append(name)
        for key in (
            "stage_answer_confirmations",
            "stage_answer_replacements",
            "stage_certificate_count",
        ):
            scalar = _nonnegative_int(stage[key], f"stages[{index}].{key}")
            if counts[key] != scalar:
                raise ReplayAggregateError("native stage_counts differs from stages")
        if (
            stage["stage_answer_confirmations"] + stage["stage_answer_replacements"]
            != stage["stage_certificate_count"]
        ):
            raise ReplayAggregateError("native stage certificate outcomes do not close")

        metadata_artifacts: dict[str, tuple[Path, str]] = {}
        for key in ("profile", "resolver_manifest", "composition_manifest", "judge_manifest"):
            artifact_path, artifact_hash, _ = _resolve_native_descriptor(
                aggregate_dir, stage[key], f"stages[{index}].{key}"
            )
            metadata_artifacts[key] = (artifact_path, artifact_hash)
            provenance_paths.append(artifact_path)
            provenance_hashes.append(artifact_hash)
            if key == "resolver_manifest":
                resolver_manifest = _read_object(artifact_path, f"stages[{index}].resolver_manifest")
                if (
                    resolver_manifest.get("gold_access") is not False
                    or resolver_manifest.get("benchmark_candidate_or_outcome_access") is not False
                ):
                    raise ReplayAggregateError(
                        "native resolver manifest does not attest gold/outcome access=false"
                    )
            elif key == "composition_manifest":
                composition_manifest = _read_object(
                    artifact_path, f"stages[{index}].composition_manifest"
                )
                outcome_attestation = composition_manifest.get("score_or_outcome_access")
                if outcome_attestation is None:
                    outcome_attestation = composition_manifest.get(
                        "benchmark_candidate_or_outcome_access"
                    )
                if (
                    composition_manifest.get("gold_access") is not False
                    or outcome_attestation is not False
                ):
                    raise ReplayAggregateError(
                        "native composition manifest does not attest gold/outcome access=false"
                    )
        (
            previous_judge_output,
            original_9b_judge_rows,
            final_judge_lineage,
        ) = _validate_native_judge_manifest(
            aggregate_dir,
            metadata_artifacts["judge_manifest"][0],
            previous_output=previous_judge_output,
            original_9b_rows=original_9b_judge_rows,
            image_rows=image_rows,
            stage_index=index,
        )
        row_artifacts: dict[str, tuple[Path, str, int | None]] = {}
        for key in ("candidates", "certificates", "decisions", "solver"):
            row_artifacts[key] = _resolve_native_descriptor(
                aggregate_dir,
                stage[key],
                f"stages[{index}].{key}",
                rows_required=True,
            )
        candidate_path, candidate_hash, candidate_rows = row_artifacts["candidates"]
        certificate_path, certificate_hash, certificate_rows = row_artifacts["certificates"]
        decision_path, decision_hash, decision_rows = row_artifacts["decisions"]
        solver_path, solver_hash, solver_rows = row_artifacts["solver"]
        if candidate_rows != rows or decision_rows != rows or solver_rows != rows:
            raise ReplayAggregateError("native stage row artifacts do not use the fixed denominator")
        for row_number, solver_row in enumerate(
            _read_jsonl(solver_path, f"stages[{index}].solver"), start=1
        ):
            if solver_row.get("model") != EXPECTED_MODEL:
                raise ReplayAggregateError(
                    f"native stages[{index}].solver:{row_number} breaks 9B model closure"
                )
            if {"gold", "gold_answer", "reference_answer", "correct_answer"} & set(
                solver_row
            ):
                raise ReplayAggregateError(
                    f"native stages[{index}].solver:{row_number} contains a forbidden gold field"
                )
            generation = solver_row.get("generation")
            if generation is not None and not isinstance(generation, dict):
                raise ReplayAggregateError("native stage solver generation metadata is malformed")
            if isinstance(generation, dict) and "gold_access" in generation:
                if generation["gold_access"] is not False:
                    raise ReplayAggregateError("native stage solver has non-false gold_access")
        if certificate_rows != stage["stage_certificate_count"]:
            raise ReplayAggregateError("native stage certificate row count is inconsistent")
        bundle_certificate_path, bundle_certificate_hash, bundle_rows = _resolve_native_descriptor(
            aggregate_dir,
            {"path": bundle["path"], "sha256": bundle["sha256"], "rows": bundle["rows"]},
            f"certificate_bundle[{index}].certificate",
            rows_required=True,
        )
        bundle_candidate_path, bundle_candidate_hash, _ = _resolve_native_descriptor(
            aggregate_dir,
            {"path": bundle["candidate_path"], "sha256": bundle["candidate_sha256"]},
            f"certificate_bundle[{index}].candidate",
        )
        if (
            bundle_certificate_path != certificate_path
            or bundle_certificate_hash != certificate_hash
            or bundle_rows != certificate_rows
            or bundle_candidate_path != candidate_path
            or bundle_candidate_hash != candidate_hash
        ):
            raise ReplayAggregateError("native certificate bundle is not bound to its stage")
        certificate_paths.append(certificate_path)
        certificate_hashes.append(certificate_hash)
        provenance_paths.extend((candidate_path, decision_path, solver_path))
        provenance_hashes.extend((candidate_hash, decision_hash, solver_hash))
        last_solver_path, last_solver_hash = solver_path, solver_hash
    assert (
        last_solver_path is not None
        and last_solver_hash is not None
        and previous_judge_output is not None
        and final_judge_lineage is not None
    )
    return (
        tuple(certificate_paths),
        tuple(certificate_hashes),
        tuple(provenance_paths),
        tuple(provenance_hashes),
        stage_names,
        last_solver_path,
        last_solver_hash,
        previous_judge_output[0],
        previous_judge_output[1],
        final_judge_lineage,
    )


def _load_native_source_replay_v1(
    path: Path,
    *,
    benchmark_path: Path,
    benchmark_hash: str,
    expected_milestone_id: str,
) -> FrozenReplayAggregate:
    if not expected_milestone_id.startswith("source_"):
        raise ReplayAggregateError("native source replay aggregate cannot back a non-source milestone")
    aggregate = _read_object(path, "native 9B source replay aggregate")
    required = {
        "schema_version",
        "created_at_utc",
        "label",
        "reporting_status",
        "model_selection_status",
        "anchor",
        "benchmark",
        "final_solver",
        "final_image_judge",
        "score",
        "scorer",
        "certificate_bundle",
        "stages",
        "stage_counts",
        "upstream_generation_model_closure",
        "answer_origin_closure",
        "inherited_27b_outputs",
        "gold_access_during_generation",
        "gold_access_during_postgeneration_score",
        "protocol",
        "overall",
        "slices",
        "evaluator_split",
        "comparisons",
        "comparison_vs_page_rag",
        "comparison_vs_anchor",
        "comparison_vs_adjacent",
        "content_projection",
        "content_projection_contract",
        "content_projection_sha256",
        "source_union",
        "final_origin_counts",
    }
    optional: set[str] = {"model_closure"}
    missing = required - set(aggregate)
    unknown = set(aggregate) - required - optional
    if missing or unknown:
        raise ReplayAggregateError(
            f"native source aggregate schema mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if aggregate["schema_version"] != SOURCE_REPLAY_V1_SCHEMA:
        raise ReplayAggregateError("native source replay schema is not supported")
    if aggregate["content_projection_contract"] != {
        "canonicalization": "utf8-json-sort-keys-compact",
        "recursively_excluded_keys": ["created_at_utc", "path"],
        "other_fields_excluded": [],
    }:
        raise ReplayAggregateError("native content projection contract is unknown")
    expected_explicit_comparisons = [aggregate["comparison_vs_anchor"]]
    if aggregate["comparison_vs_adjacent"] is not None:
        expected_explicit_comparisons.append(aggregate["comparison_vs_adjacent"])
    if aggregate["comparisons"] != expected_explicit_comparisons:
        raise ReplayAggregateError("native explicit comparison list differs from role fields")
    content_projection = aggregate["content_projection"]
    if not isinstance(content_projection, dict) or _validate_sha(
        aggregate["content_projection_sha256"], "native content projection"
    ) != _canonical_json_sha(content_projection):
        raise ReplayAggregateError("native stable content projection hash mismatch")
    version_match = re.fullmatch(r"source_(v\d+)_rebase_9b", expected_milestone_id)
    if version_match is None:
        raise ReplayAggregateError("native source replay milestone id is malformed")
    version_token = f"source_{version_match.group(1)}"
    label = str(aggregate["label"] or "")
    if version_token not in label:
        raise ReplayAggregateError("native aggregate label is not bound to the wrapper milestone")
    if aggregate["reporting_status"] != "materialized_profile_bound_development_replay":
        raise ReplayAggregateError("native replay is not a materialized profile-bound replay")
    if aggregate["inherited_27b_outputs"] is not False:
        raise ReplayAggregateError("native replay inherits 27B outputs")
    if aggregate["gold_access_during_generation"] is not False:
        raise ReplayAggregateError("native replay does not prove gold-blind generation")
    if aggregate["gold_access_during_postgeneration_score"] is not True:
        raise ReplayAggregateError("native replay does not identify post-generation scoring access")
    generation_closure, answer_origin_closure = _validate_native_closures(aggregate)

    aggregate_dir = path.parent.resolve()
    native_benchmark_path, native_benchmark_hash, _ = _resolve_native_descriptor(
        aggregate_dir, aggregate["benchmark"], "native benchmark"
    )
    if native_benchmark_hash != benchmark_hash:
        raise ReplayAggregateError("native aggregate benchmark differs from comparison benchmark")
    benchmark_rows = _read_jsonl(benchmark_path, "comparison benchmark")
    native_benchmark_rows = _read_jsonl(native_benchmark_path, "native benchmark")
    if _task_ids(benchmark_rows) != _task_ids(native_benchmark_rows):
        raise ReplayAggregateError("native and comparison benchmark task sets differ")
    benchmark_ids = _task_ids(benchmark_rows)
    overall = _native_metric(aggregate["overall"], "native overall")
    if overall["rows"] != len(benchmark_rows):
        raise ReplayAggregateError("native score denominator differs from benchmark")
    protocol = aggregate["protocol"]
    if not isinstance(protocol, dict):
        raise ReplayAggregateError("native protocol must be an object")
    image_rows_from_protocol = _positive_int(
        protocol.get("image_judge_rows"), "native protocol.image_judge_rows"
    )
    if protocol.get("fixed_denominator") != overall["rows"]:
        raise ReplayAggregateError("native protocol fixed denominator is inconsistent")

    anchor_path, anchor_hash, anchor_rows_declared = _resolve_native_descriptor(
        aggregate_dir, aggregate["anchor"], "native 9B anchor", rows_required=True
    )
    solver_path, solver_hash, solver_rows_declared = _resolve_native_descriptor(
        aggregate_dir, aggregate["final_solver"], "native final solver", rows_required=True
    )
    judge_path, judge_hash, judge_rows_declared = _resolve_native_descriptor(
        aggregate_dir, aggregate["final_image_judge"], "native final image judge", rows_required=True
    )
    score_path, score_hash, _ = _resolve_native_descriptor(
        aggregate_dir, aggregate["score"], "native score"
    )
    scorer_path, scorer_hash, _ = _resolve_native_descriptor(
        aggregate_dir, aggregate["scorer"], "native scorer"
    )
    if anchor_rows_declared != overall["rows"] or solver_rows_declared != overall["rows"]:
        raise ReplayAggregateError("native anchor/final solver do not use the fixed denominator")

    (
        certificate_paths,
        certificate_hashes,
        provenance_paths,
        provenance_hashes,
        stage_names,
        final_stage_solver_path,
        final_stage_solver_hash,
        final_stage_judge_path,
        final_stage_judge_hash,
        final_judge_lineage,
    ) = _native_stage_artifacts(
        aggregate_dir,
        aggregate["stages"],
        aggregate["stage_counts"],
        aggregate["certificate_bundle"],
        rows=overall["rows"],
        image_rows=image_rows_from_protocol,
    )
    if version_token not in stage_names[-1]:
        raise ReplayAggregateError("native final stage is not bound to the wrapper source version")
    if final_stage_solver_path != solver_path or final_stage_solver_hash != solver_hash:
        raise ReplayAggregateError("native final_solver is not the final stage solver")
    if final_stage_judge_path != judge_path or final_stage_judge_hash != judge_hash:
        raise ReplayAggregateError("native final_image_judge is not the final stage judge output")
    provenance_paths = (*provenance_paths, scorer_path)
    provenance_hashes = (*provenance_hashes, scorer_hash)

    anchor_rows = _read_jsonl(anchor_path, "native 9B anchor")
    solver_rows = _read_jsonl(solver_path, "native final solver")
    if _task_ids(anchor_rows) != benchmark_ids or _task_ids(solver_rows) != benchmark_ids:
        raise ReplayAggregateError("native anchor/final solver task set differs from benchmark")
    anchor_by_id = {str(row["task_id"]): row for row in anchor_rows}
    solver_by_id = {str(row["task_id"]): row for row in solver_rows}
    forbidden_gold_fields = {"gold", "gold_answer", "reference_answer", "correct_answer"}
    missing_final_gold_attestation: set[str] = set()
    for role, rows_value in (("anchor", anchor_rows), ("final solver", solver_rows)):
        for index, row in enumerate(rows_value, start=1):
            if row.get("model") != EXPECTED_MODEL:
                raise ReplayAggregateError(f"native {role}:{index} breaks exact 9B model closure")
            if forbidden_gold_fields & set(row):
                raise ReplayAggregateError(f"native {role}:{index} contains a forbidden gold field")
            generation = row.get("generation")
            if generation is not None and not isinstance(generation, dict):
                raise ReplayAggregateError(f"native {role}:{index} has malformed generation metadata")
            generation = generation or {}
            if "gold_access" in generation and generation["gold_access"] is not False:
                raise ReplayAggregateError(f"native {role}:{index} has non-false gold_access")
            if role == "final solver" and "gold_access" not in generation:
                missing_final_gold_attestation.add(str(row["task_id"]))

    source_union_value = aggregate["source_union"]
    if not isinstance(source_union_value, dict):
        raise ReplayAggregateError("native source_union must be an object")
    _strict_keys(
        source_union_value,
        {"sha256", "size", "answer_conflicts", "latest_stage_owner_projection"},
        "native source_union",
    )
    projection = source_union_value["latest_stage_owner_projection"]
    if not isinstance(projection, list):
        raise ReplayAggregateError("native source union projection must be an array")
    projection_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(projection):
        if not isinstance(item, dict):
            raise ReplayAggregateError(f"source union projection[{index}] must be an object")
        _strict_keys(item, {"task_id", "owner_stage", "answer_sha256"}, f"source union[{index}]")
        task_id = str(item["task_id"] or "")
        if not task_id or task_id in projection_by_id or task_id not in benchmark_ids:
            raise ReplayAggregateError("native source union has an invalid task id")
        if item["owner_stage"] not in stage_names:
            raise ReplayAggregateError("native source union references an unknown owner stage")
        expected_answer_hash = hashlib.sha256(
            str(solver_by_id[task_id].get("final_answer") or "").encode("utf-8")
        ).hexdigest()
        if _validate_sha(item["answer_sha256"], f"source union[{index}].answer") != expected_answer_hash:
            raise ReplayAggregateError("native source union answer hash differs from final solver")
        projection_by_id[task_id] = item
    union_size = _nonnegative_int(source_union_value["size"], "native source_union.size")
    if union_size != len(projection_by_id):
        raise ReplayAggregateError("native source union size differs from its projection")
    if _nonnegative_int(source_union_value["answer_conflicts"], "source_union.answer_conflicts") != 0:
        raise ReplayAggregateError("native source union contains unresolved answer conflicts")
    if _validate_sha(source_union_value["sha256"], "native source union") != _canonical_json_sha(
        projection
    ):
        raise ReplayAggregateError("native source union projection hash mismatch")
    accepted_ids = _accepted_certificate_ids(certificate_paths)
    if accepted_ids != set(projection_by_id):
        raise ReplayAggregateError("native source union differs from accepted certificate artifacts")

    changed_ids = {
        task_id
        for task_id in benchmark_ids
        if str(solver_by_id[task_id].get("final_answer") or "")
        != str(anchor_by_id[task_id].get("final_answer") or "")
    }
    override_ids = {
        task_id
        for task_id, row in solver_by_id.items()
        if isinstance(row.get("generation"), dict)
        and (
            isinstance(row["generation"].get("official_source_override"), dict)
            or isinstance(row["generation"].get("fill_blank_page_activity_override"), dict)
        )
    }
    if changed_ids != override_ids or not changed_ids <= set(projection_by_id):
        raise ReplayAggregateError("native source replacements are not fail-closed against the anchor")
    for task_id in missing_final_gold_attestation:
        if task_id in changed_ids:
            if task_id not in accepted_ids:
                raise ReplayAggregateError(
                    "final row without row-level gold attestation lacks a certified source decision"
                )
        elif str(solver_by_id[task_id].get("final_answer") or "") != str(
            anchor_by_id[task_id].get("final_answer") or ""
        ):
            raise ReplayAggregateError(
                "final row without row-level gold attestation is not byte-preserving passthrough"
            )
    confirmations = set(projection_by_id) - changed_ids
    passthrough = benchmark_ids - set(projection_by_id)
    declared_origins = aggregate["final_origin_counts"]
    if not isinstance(declared_origins, dict):
        raise ReplayAggregateError("native final_origin_counts must be an object")
    expected_native_origins = {
        "deterministic_official_source_replacement": len(changed_ids),
        "official_source_confirmation_of_9b_anchor": len(confirmations),
        "qwen35_9b_anchor_passthrough": len(passthrough),
    }
    if declared_origins != expected_native_origins:
        raise ReplayAggregateError("native final origin counts differ from anchor/source artifacts")
    normalized_origins = {
        "model_anchor": len(confirmations) + len(passthrough),
        "deterministic_source_replacement": len(changed_ids),
        "unknown": 0,
    }
    source_union = {
        "sha256": str(source_union_value["sha256"]),
        "size": union_size,
        "replacements": len(changed_ids),
        "confirmations": len(confirmations),
        "stage_counts": {
            str(item["name"]): int(item["stage_certificate_count"])
            for item in aggregate["stage_counts"]
        },
    }

    score = _read_object(score_path, "native 9B score")
    if score.get("schema_version") != SOURCE_REPLAY_SCORE_V1_SCHEMA:
        raise ReplayAggregateError("native score schema is not supported")
    if score.get("models") != [EXPECTED_MODEL]:
        raise ReplayAggregateError("native score model list breaks the 9B closure")
    if score.get("overall") != aggregate["overall"]:
        raise ReplayAggregateError("native score overall differs from aggregate")
    if score.get("by_source") != aggregate["evaluator_split"]:
        raise ReplayAggregateError("native score evaluator split differs from aggregate")
    if score.get("by_subject") != aggregate["slices"]:
        raise ReplayAggregateError("native score subject slices differ from aggregate")
    page_rag_score_projection = dict(aggregate["comparison_vs_page_rag"])
    page_rag_score_projection.pop("authority", None)
    if score.get("changes_vs_frozen_page_rag") != page_rag_score_projection:
        raise ReplayAggregateError("native Page RAG comparison differs between score and aggregate")
    score_provenance = score.get("provenance")
    if not isinstance(score_provenance, dict):
        raise ReplayAggregateError("native score provenance is missing")
    expected_score_hashes = {
        "benchmark": benchmark_hash,
        "solver_results": solver_hash,
        "image_judge": judge_hash,
        "scorer": scorer_hash,
    }
    for role, expected_hash in expected_score_hashes.items():
        descriptor = score_provenance.get(role)
        if not isinstance(descriptor, dict) or _validate_sha(
            descriptor.get("sha256"), f"score.provenance.{role}"
        ) != expected_hash:
            raise ReplayAggregateError(f"native score {role} hash link is broken")
    guardrails = score.get("guardrails")
    if not isinstance(guardrails, dict) or any(
        guardrails.get(key) != expected
        for key, expected in {
            "frozen_sha_pins_checked": True,
            "task_id_sets_match": True,
            "forbidden_gold_fields_in_solver": 0,
            "explicit_nonfalse_generation_gold_access": 0,
            "benchmark_rows_verified": overall["rows"],
            "solver_rows_verified": overall["rows"],
            "image_judge_rows_supplied": judge_rows_declared,
        }.items()
    ):
        raise ReplayAggregateError("native score guardrails do not close")
    task_outcomes = score.get("task_outcomes")
    if not isinstance(task_outcomes, list):
        raise ReplayAggregateError("native score task_outcomes must be an array")
    outcome_by_id: dict[str, dict[str, Any]] = {}
    for index, outcome in enumerate(task_outcomes):
        if not isinstance(outcome, dict):
            raise ReplayAggregateError(f"task_outcomes[{index}] is not an object")
        task_id = str(outcome.get("task_id") or "")
        if task_id not in benchmark_ids or task_id in outcome_by_id:
            raise ReplayAggregateError("native score task outcomes have an invalid task id")
        if not isinstance(outcome.get("new_correct"), bool):
            raise ReplayAggregateError("native score task outcome lacks boolean new_correct")
        outcome_by_id[task_id] = outcome
    if set(outcome_by_id) != benchmark_ids:
        raise ReplayAggregateError("native score task outcomes differ from benchmark")
    if sum(int(row["new_correct"]) for row in task_outcomes) != overall["correct"]:
        raise ReplayAggregateError("native task outcomes do not close to overall correct")

    subject_slices = aggregate["slices"]
    if not isinstance(subject_slices, dict) or "Math" not in subject_slices:
        raise ReplayAggregateError("native score has no Math subject slice")
    parsed_subjects = {
        str(subject): _native_metric(value, f"native slices.{subject}")
        for subject, value in subject_slices.items()
    }
    if (
        sum(item["rows"] for item in parsed_subjects.values()) != overall["rows"]
        or sum(item["correct"] for item in parsed_subjects.values()) != overall["correct"]
    ):
        raise ReplayAggregateError("native subject slices do not close to overall")
    evaluator_split = aggregate["evaluator_split"]
    if not isinstance(evaluator_split, dict):
        raise ReplayAggregateError("native evaluator_split must be an object")
    _strict_keys(evaluator_split, {"deterministic", "image_judge"}, "native evaluator_split")
    deterministic_metric = _native_metric(evaluator_split["deterministic"], "evaluator deterministic")
    image_metric = _native_metric(evaluator_split["image_judge"], "evaluator image")
    if (
        deterministic_metric["rows"] + image_metric["rows"] != overall["rows"]
        or deterministic_metric["correct"] + image_metric["correct"] != overall["correct"]
        or image_metric["rows"] != judge_rows_declared
    ):
        raise ReplayAggregateError("native evaluator split does not close")

    image_judge_rows = _read_jsonl(judge_path, "native final image judge")
    image_ids = {
        task_id
        for task_id, outcome in outcome_by_id.items()
        if outcome.get("score_source") == "image_judge"
    }
    if _task_ids(image_judge_rows) != image_ids:
        raise ReplayAggregateError("native image judge task set differs from score outcomes")
    source_certified_image = 0
    model_judged_image = 0
    for row in image_judge_rows:
        task_id = str(row["task_id"])
        verdict = row.get("verdict")
        if not isinstance(verdict, dict) or not isinstance(verdict.get("strict_correct"), bool):
            raise ReplayAggregateError("native image judge row lacks strict_correct")
        if verdict["strict_correct"] != outcome_by_id[task_id]["new_correct"]:
            raise ReplayAggregateError("native image judge verdict differs from score outcome")
        judge_metadata = row.get("judge")
        if not isinstance(judge_metadata, dict):
            raise ReplayAggregateError("native image judge metadata is missing")
        judge_model = judge_metadata.get("model")
        if judge_model is None:
            if judge_metadata.get("backend") != "deterministic-official-source-certificate":
                raise ReplayAggregateError("model-free image verdict is not source-certified")
            source_certified_image += 1
        elif judge_model == EXPECTED_MODEL:
            model_judged_image += 1
        else:
            raise ReplayAggregateError("native image judge breaks the 9B model closure")
    if (
        source_certified_image
        != final_judge_lineage["cumulative_source_adjudicated_image_rows_count"]
        or model_judged_image
        != final_judge_lineage["cumulative_original_9b_judge_rows_count"]
    ):
        raise ReplayAggregateError("final image judge disagrees with cumulative lineage recount")
    evaluator = {
        "semantics": (
            "frozen deterministic matcher + image strict_correct; cumulative source-adjudicated "
            "verdicts are deterministic and remaining rows are byte-identical to the original "
            "ActiveCrop Qwen3.5-9B judge, not merely copied from the immediate base"
        ),
        "deterministic_rows": deterministic_metric["rows"],
        "image_rows": image_metric["rows"],
        "source_certified_image_rows": source_certified_image,
        "model_judged_image_rows": model_judged_image,
        "source_adjudicated_image_rows": final_judge_lineage[
            "cumulative_source_adjudicated_image_rows_count"
        ],
        "original_9b_judge_rows": final_judge_lineage[
            "cumulative_original_9b_judge_rows_count"
        ],
        "copied_base_judge_rows_byte_identical": final_judge_lineage[
            "copied_base_judge_rows_byte_identical"
        ],
        "stage_source_adjudicated_image_rows": final_judge_lineage[
            "stage_source_adjudicated_image_rows_count"
        ],
        "judge_model": EXPECTED_MODEL if model_judged_image else None,
    }
    math_metric = parsed_subjects["Math"]
    non_math_rows = overall["rows"] - math_metric["rows"]
    non_math_correct = overall["correct"] - math_metric["correct"]
    slices = {
        "deterministic": deterministic_metric,
        "image": image_metric,
        "math": math_metric,
        "non_math": {
            "rows": non_math_rows,
            "correct": non_math_correct,
            "accuracy": non_math_correct / non_math_rows,
        },
    }
    _native_page_rag_comparison(
        aggregate["comparison_vs_page_rag"], task_ids=benchmark_ids, overall=overall
    )
    baseline_correct = _nonnegative_int(
        aggregate["overall"].get("baseline_correct"), "native overall.baseline_correct"
    )
    if (
        overall["correct"] - baseline_correct
        != aggregate["comparison_vs_page_rag"]["net_correct_change"]
    ):
        raise ReplayAggregateError("native Page RAG delta differs from overall baseline")
    anchor_comparison, anchor_comparison_paths, anchor_comparison_hashes = (
        _native_explicit_comparison(
            aggregate_dir,
            aggregate["comparison_vs_anchor"],
            role="anchor",
            rows=overall["rows"],
            current_correct=overall["correct"],
        )
    )
    if (
        anchor_comparison is None
        or anchor_comparison["baseline_milestone_id"] != "query_active_crop_v2_9b"
        or anchor_comparison["baseline_solver_sha256"] != anchor_hash
    ):
        raise ReplayAggregateError("native comparison_vs_anchor is not bound to anchor")
    adjacent_comparison, adjacent_comparison_paths, adjacent_comparison_hashes = (
        _native_explicit_comparison(
            aggregate_dir,
            aggregate["comparison_vs_adjacent"],
            role="adjacent",
            rows=overall["rows"],
            current_correct=overall["correct"],
        )
    )
    comparisons = tuple(
        item for item in (anchor_comparison, adjacent_comparison) if item is not None
    )
    provenance_paths = (
        *provenance_paths,
        *anchor_comparison_paths,
        *adjacent_comparison_paths,
    )
    provenance_hashes = (
        *provenance_hashes,
        *anchor_comparison_hashes,
        *adjacent_comparison_hashes,
    )
    caveats = (
        "development replay on a fixed benchmark; external generalization is not implied",
        "Page RAG, ActiveCrop anchor and adjacent source deltas have separate native authorities",
    )
    model_closure = {
        "expected_model": EXPECTED_MODEL,
        "checked_rows": overall["rows"],
        "matching_rows": overall["rows"],
        "foreign_models": [],
        "upstream_generation_model_closure": list(generation_closure),
        "answer_origin_closure": list(answer_origin_closure),
        "inherited_27b_outputs": False,
    }
    return FrozenReplayAggregate(
        milestone_id=expected_milestone_id,
        native_adapter=SOURCE_REPLAY_V1_ADAPTER,
        model=EXPECTED_MODEL,
        pipeline=MILESTONE_PIPELINES[expected_milestone_id],
        provenance_status="new_profile_bound_replay",
        bound_before_score=True,
        caveats=caveats,
        rows=overall["rows"],
        correct=overall["correct"],
        accuracy=overall["accuracy"],
        slices=slices,
        model_closure=model_closure,
        source_union=source_union,
        comparisons=comparisons,
        evaluator=evaluator,
        final_origin_counts=normalized_origins,
        aggregate_path=path,
        aggregate_sha256=_sha256(path),
        benchmark_path=benchmark_path,
        benchmark_sha256=benchmark_hash,
        solver_path=solver_path,
        solver_sha256=solver_hash,
        raw_solver_path=None,
        raw_solver_sha256=None,
        score_path=score_path,
        score_sha256=score_hash,
        judge_path=judge_path,
        judge_sha256=judge_hash,
        anchor_path=anchor_path,
        anchor_sha256=anchor_hash,
        certificate_paths=certificate_paths,
        certificate_sha256s=certificate_hashes,
        provenance_manifest_paths=provenance_paths,
        provenance_manifest_sha256s=provenance_hashes,
    )


def load_frozen_9b_replay_aggregate(path: Path | str) -> FrozenReplayAggregate:
    raise ReplayAggregateError(
        "standalone milestone loading is intentionally disabled: use the seven-stage "
        "comparison manifest so benchmark hash/model closure can be verified"
    )


def load_frozen_9b_comparison(path: Path | str) -> FrozenReplayComparison:
    manifest_path = Path(path).expanduser().resolve()
    manifest = _read_object(manifest_path, "9B comparison manifest")
    _strict_keys(
        manifest,
        {"schema_version", "model", "benchmark", "milestones"},
        "9B comparison manifest",
    )
    if manifest["schema_version"] != COMPARISON_SCHEMA or manifest["model"] != EXPECTED_MODEL:
        raise ReplayAggregateError("comparison schema/model is not the expected 9B contract")
    base = manifest_path.parent.resolve()
    benchmark_path, benchmark_hash = _resolve_descriptor(base, manifest["benchmark"], "benchmark")
    descriptors = manifest["milestones"]
    if not isinstance(descriptors, list) or len(descriptors) != len(MILESTONE_SPECS):
        raise ReplayAggregateError("comparison must contain exactly seven milestones")
    expected_ids = [item[0] for item in MILESTONE_SPECS]
    actual_ids: list[str] = []
    loaded: list[FrozenReplayAggregate] = []
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict):
            raise ReplayAggregateError(f"milestones[{index}] must be an object")
        _strict_keys(
            descriptor,
            {"milestone_id", "adapter", "aggregate"},
            f"milestones[{index}]",
        )
        milestone_id = str(descriptor["milestone_id"])
        actual_ids.append(milestone_id)
        adapter = str(descriptor["adapter"] or "")
        aggregate_path, aggregate_hash = _resolve_descriptor(
            base, descriptor["aggregate"], f"{milestone_id} aggregate"
        )
        if adapter == NORMALIZED_V2_ADAPTER:
            item = _load_aggregate(
                aggregate_path,
                base=base,
                benchmark_path=benchmark_path,
                benchmark_hash=benchmark_hash,
                expected_milestone_id=milestone_id,
            )
        elif adapter == SOURCE_REPLAY_V1_ADAPTER:
            item = _load_native_source_replay_v1(
                aggregate_path,
                benchmark_path=benchmark_path,
                benchmark_hash=benchmark_hash,
                expected_milestone_id=milestone_id,
            )
        else:
            raise ReplayAggregateError(
                f"{milestone_id}: unsupported native aggregate adapter {adapter!r}"
            )
        if item.aggregate_sha256 != aggregate_hash:
            raise ReplayAggregateError(f"{milestone_id}: aggregate hash closure failed")
        loaded.append(item)
    if actual_ids != expected_ids:
        raise ReplayAggregateError(
            f"milestone order/set mismatch: expected={expected_ids}, got={actual_ids}"
        )
    if len({item.rows for item in loaded}) != 1:
        raise ReplayAggregateError("milestones do not share one benchmark row count")

    by_id = {item.milestone_id: item for item in loaded}
    milestone_positions = {item.milestone_id: index for index, item in enumerate(loaded)}
    for item in loaded:
        for comparison in item.comparisons:
            if (
                milestone_positions[comparison["baseline_milestone_id"]]
                >= milestone_positions[item.milestone_id]
            ):
                raise ReplayAggregateError("comparison baseline must precede the current milestone")
            baseline = by_id[comparison["baseline_milestone_id"]]
            expected_baseline_sha = baseline.solver_sha256
            identity_label = "normalized solver"
            if (
                item.native_adapter == SOURCE_REPLAY_V1_ADAPTER
                and baseline.native_adapter == NORMALIZED_V2_ADAPTER
            ):
                if baseline.raw_solver_sha256 is None:
                    raise ReplayAggregateError(
                        f"{item.milestone_id}: native comparison baseline "
                        f"{baseline.milestone_id} has no verified raw_solver identity"
                    )
                expected_baseline_sha = baseline.raw_solver_sha256
                identity_label = "verified raw_solver"
            if comparison["baseline_solver_sha256"] != expected_baseline_sha:
                raise ReplayAggregateError(
                    f"{item.milestone_id}: comparison baseline solver hash link to "
                    f"{baseline.milestone_id} is broken; expected {identity_label} "
                    f"{expected_baseline_sha}, got "
                    f"{comparison['baseline_solver_sha256']}"
                )
            if item.correct - baseline.correct != comparison["fixes"] - comparison["regressions"]:
                raise ReplayAggregateError("comparison fixes/regressions disagree with score delta")

    return FrozenReplayComparison(
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        benchmark_path=benchmark_path,
        benchmark_sha256=benchmark_hash,
        milestones=tuple(loaded),
    )
