"""Build a provenance-safe final report for Maksim's eight experiments.

The builder is deliberately configuration-driven: it does not know experiment
paths or scores.  A config points at the frozen benchmark/scorer, one baseline,
and exactly eight ideas.  Matched-judge scores, historical replay scores, and
pending bounds are represented separately so a replay can never silently become
an exact matched result.

Example source entry::

    {
      "id": "subject_router",
      "name": "Router by task type",
      "matched_score": {
        "path": "subject_router/score.json",
        "judge_lineage": "judge-v2-qwen35-9b",
        "matched": true,
        "judge_manifest": {"path": "subject_router/finalization_manifest.json"},
        "judge_artifact": {"path": "subject_router/matched_image97_judge.jsonl"}
      },
      "replay_score": {
        "path": "subject_router/mixed_replay_score.json",
        "judge_lineage": "mixed-historical-replay",
        "matched": false
      },
      "manifest": {"path": "subject_router/evaluation_manifest.json"}
    }

Relative paths are resolved from the config directory.  Missing idea artifacts
are allowed and produce an honest ``pending`` row (or ``replay`` when only a
replay score exists).  The baseline matched score must exist.

An entry can expose measured, incomplete work with an explicit ``progress``
object either directly in the entry or at the root of its manifest::

    {"completed": 41, "total": 97, "unit": "judge rows", "stage": "judge-v2"}

Progress is never interpreted as accuracy.  It produces ``partial`` only while
``0 < completed < total`` and only when neither a matched nor replay score is
available.

For stronger matched-score provenance, a config may also provide a structured
``matched_judge_profile`` at the root and optional ``judge_manifest`` and/or
``judge_artifact`` source specs inside ``matched_score``.  When such paths are
configured and the score exists, the builder validates the actual evidence
against the profile and the image-judge hash embedded in the score.  Existing
configs without these optional fields remain supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "maxim8-final-report-v1"
CONFIG_SCHEMA_VERSION = "maxim8-final-report-config-v1"
EXPECTED_TOTAL_N = 274
EXPECTED_BASELINE_CORRECT = 141
STATUS_EXACT = "exact"
STATUS_REPLAY = "replay"
STATUS_PARTIAL = "partial"
STATUS_PENDING = "pending"


class ReportConfigError(ValueError):
    """Raised when config or an input artifact violates report invariants."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportConfigError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportConfigError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(config_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def _required_str(mapping: Mapping[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReportConfigError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def _int(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReportConfigError(f"{where} must be an integer >= {minimum}")
    return value


def _optional_nonnegative_int(value: Any, where: str) -> int | None:
    if value is None:
        return None
    return _int(value, where)


def _check_accuracy(value: Any, correct: int, n: int, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportConfigError(f"{where} must be numeric")
    accuracy = float(value)
    expected = correct / n
    if not math.isfinite(accuracy) or abs(accuracy - expected) > 1.0e-6:
        raise ReportConfigError(
            f"{where}={accuracy!r} is inconsistent with {correct}/{n}"
        )
    return round(expected, 6)


def _source_spec(value: Any, where: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ReportConfigError(f"{where} must be an object")
    _required_str(value, "path", where)
    return dict(value)


def _source_record(
    spec: Mapping[str, Any] | None,
    *,
    config_dir: Path,
    where: str,
) -> tuple[dict[str, Any] | None, Path | None]:
    if spec is None:
        return None, None
    path = _resolve(config_dir, _required_str(spec, "path", where))
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        actual_sha = _sha256(path)
        declared_sha = spec.get("sha256")
        if declared_sha is not None:
            if not isinstance(declared_sha, str) or actual_sha != declared_sha.lower():
                raise ReportConfigError(
                    f"{where} SHA256 mismatch: expected {declared_sha}, got {actual_sha}"
                )
        record["sha256"] = actual_sha
    else:
        record["sha256"] = None
    return record, path


def _parse_progress(value: Any, where: str) -> dict[str, Any] | None:
    """Parse explicit measured progress without treating it as a score.

    ``completed``/``total`` is the canonical spelling.  The ``*_rows`` aliases
    are accepted because evaluation manifests commonly describe row counts.
    Empty progress is represented by ``None``; bounds alone never imply
    progress.
    """

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ReportConfigError(f"{where} must be an object")
    has_plain = "completed" in value or "total" in value
    has_rows = "completed_rows" in value or "total_rows" in value
    if has_plain and has_rows:
        raise ReportConfigError(
            f"{where} must use either completed/total or completed_rows/total_rows"
        )
    completed_key, total_key = (
        ("completed_rows", "total_rows") if has_rows else ("completed", "total")
    )
    if completed_key not in value or total_key not in value:
        raise ReportConfigError(
            f"{where} must contain both {completed_key} and {total_key}"
        )
    completed = _int(value.get(completed_key), f"{where}.{completed_key}")
    total = _int(value.get(total_key), f"{where}.{total_key}", minimum=1)
    if completed > total:
        raise ReportConfigError(f"{where}.completed exceeds total")
    unit = _required_str(value, "unit", where)
    stage = value.get("stage")
    if stage is not None and (not isinstance(stage, str) or not stage.strip()):
        raise ReportConfigError(f"{where}.stage must be a non-empty string")
    measured_at_utc = value.get("measured_at_utc")
    if measured_at_utc is not None and (
        not isinstance(measured_at_utc, str) or not measured_at_utc.strip()
    ):
        raise ReportConfigError(
            f"{where}.measured_at_utc must be a non-empty string"
        )
    return {
        "completed": completed,
        "total": total,
        "unit": unit,
        "stage": stage.strip() if isinstance(stage, str) else None,
        "measured_at_utc": (
            measured_at_utc.strip() if isinstance(measured_at_utc, str) else None
        ),
    }


def _progress_from_entry_and_manifest(
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    where: str,
) -> dict[str, Any] | None:
    entry_progress = _parse_progress(entry.get("progress"), f"{where}.progress")
    manifest_progress = _parse_progress(
        manifest.get("progress") if isinstance(manifest, dict) else None,
        f"{where}.manifest.progress",
    )
    if (
        entry_progress is not None
        and manifest_progress is not None
        and entry_progress != manifest_progress
    ):
        raise ReportConfigError(
            f"{where} entry and manifest progress disagree"
        )
    return entry_progress or manifest_progress


def _load_benchmark(
    spec: Mapping[str, Any], config_dir: Path
) -> tuple[dict[str, Any], dict[str, int]]:
    path = _resolve(config_dir, _required_str(spec, "path", "benchmark"))
    if not path.is_file():
        raise ReportConfigError(f"benchmark does not exist: {path}")
    expected_sha = _required_str(spec, "sha256", "benchmark").lower()
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ReportConfigError(
            f"benchmark SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    expected_n = _int(spec.get("n"), "benchmark.n", minimum=1)
    if expected_n != EXPECTED_TOTAL_N:
        raise ReportConfigError(
            f"benchmark.n must be the frozen common-bench size {EXPECTED_TOTAL_N}"
        )
    math_n = _int(spec.get("math_n"), "benchmark.math_n")
    non_math_n = _int(spec.get("non_math_n"), "benchmark.non_math_n")
    if math_n + non_math_n != expected_n:
        raise ReportConfigError("benchmark math/non-Math sizes do not sum to n")

    rows = 0
    actual_math = 0
    ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ReportConfigError(
                        f"benchmark line {line_number} is not a JSON object"
                    )
                task_id = row.get("task_id")
                if not isinstance(task_id, str) or not task_id:
                    raise ReportConfigError(
                        f"benchmark line {line_number} has no task_id"
                    )
                if task_id in ids:
                    raise ReportConfigError(f"duplicate benchmark task_id: {task_id}")
                ids.add(task_id)
                rows += 1
                actual_math += row.get("subject") == "Math"
    except json.JSONDecodeError as exc:
        raise ReportConfigError(f"invalid benchmark JSONL {path}: {exc}") from exc
    if rows != expected_n:
        raise ReportConfigError(f"benchmark has {rows} rows, expected {expected_n}")
    if actual_math != math_n or rows - actual_math != non_math_n:
        raise ReportConfigError(
            "benchmark subject sizes disagree with configured math_n/non_math_n"
        )
    return (
        {"path": str(path), "sha256": actual_sha, "rows": rows},
        {"overall": rows, "math": actual_math, "non_math": rows - actual_math},
    )


def _load_scorer(spec: Mapping[str, Any], config_dir: Path) -> dict[str, Any]:
    path = _resolve(config_dir, _required_str(spec, "path", "scorer"))
    if not path.is_file():
        raise ReportConfigError(f"scorer does not exist: {path}")
    expected_sha = _required_str(spec, "sha256", "scorer").lower()
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ReportConfigError(
            f"scorer SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    return {"path": str(path), "sha256": actual_sha}


def _provenance_sha(report: Mapping[str, Any], key: str, where: str) -> str:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ReportConfigError(f"{where}.provenance is missing")
    source = provenance.get(key)
    if not isinstance(source, dict):
        raise ReportConfigError(f"{where}.provenance.{key} is missing")
    return _required_str(source, "sha256", f"{where}.provenance.{key}").lower()


def _summary_counts(summary: Mapping[str, Any], where: str) -> dict[str, Any]:
    n = _int(summary.get("n"), f"{where}.n", minimum=1)
    correct = _int(summary.get("new_correct"), f"{where}.new_correct")
    if correct > n:
        raise ReportConfigError(f"{where}.new_correct exceeds n")
    accuracy = _check_accuracy(summary.get("new_accuracy"), correct, n, f"{where}.new_accuracy")
    return {"correct": correct, "n": n, "accuracy": accuracy}


def _auto_operational(document: Mapping[str, Any]) -> dict[str, int | None]:
    operational = document.get("operational")
    calls: int | None = None
    tokens: int | None = None
    if isinstance(operational, dict):
        calls_block = operational.get("model_calls")
        if isinstance(calls_block, dict):
            raw_calls = calls_block.get("call_count_total")
            reported_rows = calls_block.get("reported_rows")
            # The frozen scorer uses 0/0 to mean "not reported", not zero calls.
            if not (raw_calls == 0 and reported_rows == 0):
                calls = _optional_nonnegative_int(
                    raw_calls, "operational.model_calls.call_count_total"
                )
        tokens_block = operational.get("tokens")
        if isinstance(tokens_block, dict):
            tokens = _optional_nonnegative_int(
                tokens_block.get("combined_tokens_total"),
                "operational.tokens.combined_tokens_total",
            )
    if calls is None:
        for key in ("model_calls", "call_count_total", "audit_calls_added"):
            if key in document and isinstance(document.get(key), int):
                calls = _optional_nonnegative_int(document.get(key), key)
                break
    if tokens is None:
        for key in ("combined_tokens", "combined_tokens_total", "tokens"):
            if key in document and isinstance(document.get(key), int):
                tokens = _optional_nonnegative_int(document.get(key), key)
                break
    return {"model_calls": calls, "combined_tokens": tokens}


def _json_pointer(document: Any, pointer: str, where: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ReportConfigError(f"{where} must be a JSON pointer starting with /")
    current = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ReportConfigError(f"{where} does not resolve at {part!r}")
    return current


def _operational_from_source(
    spec: Mapping[str, Any] | None,
    *,
    config_dir: Path,
    fallback_document: Mapping[str, Any] | None,
    where: str,
) -> tuple[dict[str, int | None], dict[str, Any] | None]:
    if spec is None:
        return _auto_operational(fallback_document or {}), None
    source, path = _source_record(spec, config_dir=config_dir, where=where)
    if path is None or not path.is_file():
        return _auto_operational(fallback_document or {}), source
    document = _load_json(path)
    metrics = _auto_operational(document)
    pointer_fields = {
        "model_calls": "model_calls_pointer",
        "combined_tokens": "combined_tokens_pointer",
    }
    for metric, pointer_key in pointer_fields.items():
        pointer = spec.get(pointer_key)
        if pointer is not None:
            if not isinstance(pointer, str):
                raise ReportConfigError(f"{where}.{pointer_key} must be a string")
            metrics[metric] = _optional_nonnegative_int(
                _json_pointer(document, pointer, f"{where}.{pointer_key}"),
                f"{where}.{pointer_key}",
            )
    return metrics, source


def _parse_score(
    report: Mapping[str, Any],
    *,
    where: str,
    benchmark_sha256: str,
    scorer_sha256: str,
    expected_sizes: Mapping[str, int],
    baseline_correct: int,
) -> dict[str, Any]:
    if _provenance_sha(report, "benchmark", where) != benchmark_sha256:
        raise ReportConfigError(f"{where} benchmark provenance SHA256 mismatch")
    if _provenance_sha(report, "scorer", where) != scorer_sha256:
        raise ReportConfigError(f"{where} scorer provenance SHA256 mismatch")
    overall_raw = report.get("overall")
    by_subject = report.get("by_subject")
    if not isinstance(overall_raw, dict) or not isinstance(by_subject, dict):
        raise ReportConfigError(f"{where} is missing overall/by_subject")
    overall = _summary_counts(overall_raw, f"{where}.overall")
    if overall["n"] != expected_sizes["overall"]:
        raise ReportConfigError(f"{where} does not cover the full 274 benchmark")

    subject_n = 0
    subject_correct = 0
    for subject, raw_summary in by_subject.items():
        if not isinstance(raw_summary, dict):
            raise ReportConfigError(f"{where}.by_subject.{subject} must be an object")
        parsed = _summary_counts(raw_summary, f"{where}.by_subject.{subject}")
        subject_n += parsed["n"]
        subject_correct += parsed["correct"]
    if subject_n != overall["n"] or subject_correct != overall["correct"]:
        raise ReportConfigError(f"{where} subject counts do not reconstruct overall")
    math_raw = by_subject.get("Math")
    if not isinstance(math_raw, dict):
        raise ReportConfigError(f"{where}.by_subject.Math is missing")
    math_slice = _summary_counts(math_raw, f"{where}.by_subject.Math")
    if math_slice["n"] != expected_sizes["math"]:
        raise ReportConfigError(f"{where} Math n is inconsistent with benchmark")
    non_math_correct = overall["correct"] - math_slice["correct"]
    non_math_n = overall["n"] - math_slice["n"]
    if non_math_n != expected_sizes["non_math"] or non_math_correct < 0:
        raise ReportConfigError(f"{where} non-Math slice is inconsistent")
    non_math = {
        "correct": non_math_correct,
        "n": non_math_n,
        "accuracy": round(non_math_correct / non_math_n, 6),
    }

    artifact_baseline = _int(
        overall_raw.get("baseline_correct"), f"{where}.overall.baseline_correct"
    )
    if artifact_baseline != baseline_correct:
        raise ReportConfigError(
            f"{where} uses baseline {artifact_baseline}, expected {baseline_correct}"
        )
    delta = overall["correct"] - baseline_correct
    if "delta_correct" in overall_raw and overall_raw.get("delta_correct") != delta:
        raise ReportConfigError(f"{where}.overall.delta_correct is inconsistent")
    fixed = _int(overall_raw.get("fixed"), f"{where}.overall.fixed")
    regressed = _int(overall_raw.get("regressed"), f"{where}.overall.regressed")
    if fixed - regressed != delta:
        raise ReportConfigError(
            f"{where} fixed-regressed does not equal delta to the frozen baseline"
        )
    return {
        "overall": overall,
        "math": math_slice,
        "non_math": non_math,
        "vs_baseline": {
            "delta_correct": delta,
            "fixed": fixed,
            "regressed": regressed,
        },
    }


def _find_bounds(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = document.get("score_bounds")
    if isinstance(direct, dict):
        return direct
    direct = document.get("current_score_bounds_before_fresh_judge")
    if isinstance(direct, dict):
        return direct
    judge = document.get("image_judge_v2")
    if isinstance(judge, dict):
        nested = judge.get("current_score_bounds_before_fresh_judge")
        if isinstance(nested, dict):
            return nested
    if "current_lower_correct" in document or "current_upper_correct" in document:
        return {
            "lower_correct": document.get("current_lower_correct"),
            "upper_correct": document.get("current_upper_correct"),
            "n": document.get("n", EXPECTED_TOTAL_N),
        }
    return None


def _parse_bound_slice(
    raw: Mapping[str, Any], where: str, expected_n: int
) -> dict[str, Any]:
    lower = _int(raw.get("lower_correct"), f"{where}.lower_correct")
    upper = _int(raw.get("upper_correct"), f"{where}.upper_correct")
    n = _int(raw.get("n", expected_n), f"{where}.n", minimum=1)
    if n != expected_n or lower > upper or upper > n:
        raise ReportConfigError(f"{where} has invalid bounds")
    lower_accuracy = round(lower / n, 6)
    upper_accuracy = round(upper / n, 6)
    if raw.get("lower_accuracy") is not None:
        _check_accuracy(raw.get("lower_accuracy"), lower, n, f"{where}.lower_accuracy")
    if raw.get("upper_accuracy") is not None:
        _check_accuracy(raw.get("upper_accuracy"), upper, n, f"{where}.upper_accuracy")
    return {
        "lower_correct": lower,
        "upper_correct": upper,
        "n": n,
        "lower_accuracy": lower_accuracy,
        "upper_accuracy": upper_accuracy,
    }


def _bounds_from_manifest(
    manifest: Mapping[str, Any] | None,
    expected_sizes: Mapping[str, int],
    where: str,
) -> dict[str, Any] | None:
    if manifest is None:
        return None
    raw = _find_bounds(manifest)
    if raw is None:
        return None
    result = {
        "overall": _parse_bound_slice(raw, f"{where}.bounds.overall", expected_sizes["overall"]),
        "math": None,
        "non_math": None,
    }
    for key in ("math", "non_math"):
        candidate = raw.get(key)
        if isinstance(candidate, dict):
            result[key] = _parse_bound_slice(
                candidate, f"{where}.bounds.{key}", expected_sizes[key]
            )
    return result


def _validate_manifest_benchmark(
    manifest: Mapping[str, Any], benchmark_sha256: str, where: str
) -> None:
    candidates: list[Any] = [manifest.get("benchmark")]
    sources = manifest.get("sources")
    if isinstance(sources, dict):
        candidates.append(sources.get("benchmark"))
    provenance = manifest.get("provenance")
    if isinstance(provenance, dict):
        candidates.append(provenance.get("benchmark"))
    hashes = [
        candidate.get("sha256").lower()
        for candidate in candidates
        if isinstance(candidate, dict) and isinstance(candidate.get("sha256"), str)
    ]
    if hashes and any(value != benchmark_sha256 for value in hashes):
        raise ReportConfigError(f"{where} benchmark provenance SHA256 mismatch")


def _load_optional_document(
    spec: Mapping[str, Any] | None,
    *,
    config_dir: Path,
    where: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    record, path = _source_record(spec, config_dir=config_dir, where=where)
    if path is None or not path.is_file():
        return None, record
    return _load_json(path), record


def _lineage(
    spec: Mapping[str, Any], *, expected_matched: bool, matched_lineage: str, where: str
) -> dict[str, Any]:
    matched = spec.get("matched")
    if not isinstance(matched, bool) or matched is not expected_matched:
        raise ReportConfigError(f"{where}.matched must be {str(expected_matched).lower()}")
    lineage = _required_str(spec, "judge_lineage", where)
    if expected_matched and lineage != matched_lineage:
        raise ReportConfigError(
            f"{where}.judge_lineage={lineage!r} does not match {matched_lineage!r}"
        )
    return {"lineage": lineage, "matched": matched}


JUDGE_PROFILE_REQUIRED_FIELDS = ("prompt_version", "model", "seed")
JUDGE_PROFILE_OPTIONAL_FIELDS = (
    "temperature",
    "max_tokens",
    "enable_thinking",
    "use_response_format",
    "image_mode",
    "backend",
    "backend_config_hash",
)


def _matched_judge_profile(
    value: Any, matched_lineage: str
) -> dict[str, Any] | None:
    """Validate the optional structured definition of a matched judge lineage."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ReportConfigError("config.matched_judge_profile must be an object")
    lineage = _required_str(value, "lineage", "config.matched_judge_profile")
    if lineage != matched_lineage:
        raise ReportConfigError(
            "config.matched_judge_profile.lineage does not match "
            "config.matched_judge_lineage"
        )
    result: dict[str, Any] = {"lineage": lineage}
    for key in JUDGE_PROFILE_REQUIRED_FIELDS:
        if key in ("prompt_version", "model"):
            result[key] = _required_str(
                value, key, "config.matched_judge_profile"
            )
        else:
            result[key] = _int(
                value.get(key), f"config.matched_judge_profile.{key}"
            )
    for key in JUDGE_PROFILE_OPTIONAL_FIELDS:
        if key not in value:
            continue
        raw = value.get(key)
        where = f"config.matched_judge_profile.{key}"
        if key == "temperature":
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ReportConfigError(f"{where} must be numeric")
            numeric = float(raw)
            if not math.isfinite(numeric):
                raise ReportConfigError(f"{where} must be finite")
            result[key] = numeric
        elif key == "max_tokens":
            result[key] = _int(raw, where, minimum=1)
        elif key in ("enable_thinking", "use_response_format"):
            if not isinstance(raw, bool):
                raise ReportConfigError(f"{where} must be boolean")
            result[key] = raw
        else:
            result[key] = _required_str(value, key, "config.matched_judge_profile")
    return result


def _manifest_lineage_profile(
    manifest: Mapping[str, Any], where: str
) -> tuple[str | None, Mapping[str, Any] | None]:
    lineage: str | None = None
    for key in ("matched_judge_lineage", "judge_lineage"):
        raw = manifest.get(key)
        if raw is not None:
            if not isinstance(raw, str) or not raw.strip():
                raise ReportConfigError(f"{where}.{key} must be a non-empty string")
            lineage = raw.strip()
            break
    profile: Mapping[str, Any] | None = None
    for key in ("matched_judge_profile", "judge_profile"):
        raw = manifest.get(key)
        if raw is not None:
            if not isinstance(raw, dict):
                raise ReportConfigError(f"{where}.{key} must be an object")
            profile = raw
            break
    return lineage, profile


def _manifest_judge_artifact_spec(
    manifest: Mapping[str, Any], where: str
) -> Mapping[str, Any] | None:
    candidates: list[Any] = [manifest.get("judge_artifact")]
    matched = manifest.get("matched_judge")
    if isinstance(matched, dict):
        candidates.append(matched.get("output"))
    candidates.append(manifest.get("output"))
    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, dict):
            raise ReportConfigError(f"{where} judge artifact reference must be an object")
        if isinstance(candidate.get("path"), str) and candidate.get("path").strip():
            return candidate
    return None


def _score_image_judge_sha(report: Mapping[str, Any], where: str) -> str:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ReportConfigError(f"{where}.provenance is missing")
    image_judge = provenance.get("image_judge")
    if not isinstance(image_judge, dict):
        raise ReportConfigError(f"{where}.provenance.image_judge is missing")
    return _required_str(
        image_judge, "sha256", f"{where}.provenance.image_judge"
    ).lower()


def _judge_row_profile(row: Mapping[str, Any], where: str) -> dict[str, Any]:
    judge = row.get("judge")
    if not isinstance(judge, dict):
        raise ReportConfigError(f"{where}.judge is missing")
    backend_config = judge.get("backend_config")
    if not isinstance(backend_config, dict):
        raise ReportConfigError(f"{where}.judge.backend_config is missing")
    error = judge.get("error")
    if error not in (None, ""):
        raise ReportConfigError(f"{where} has judge.error={error!r}")
    return {
        "prompt_version": row.get("prompt_version"),
        "model": backend_config.get("model", judge.get("model")),
        "seed": backend_config.get("seed"),
        "temperature": backend_config.get("temperature"),
        "max_tokens": backend_config.get("max_tokens"),
        "enable_thinking": backend_config.get("enable_thinking"),
        "use_response_format": backend_config.get("use_response_format"),
        "image_mode": backend_config.get("image_mode"),
        "backend": judge.get("backend", backend_config.get("backend")),
        "backend_config_hash": judge.get("backend_config_hash"),
    }


def _profile_value_matches(expected: Any, actual: Any) -> bool:
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        return math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= 1e-12
    return actual == expected


def _validate_profile_values(
    actual: Mapping[str, Any], expected: Mapping[str, Any], where: str
) -> None:
    for key, expected_value in expected.items():
        if key == "lineage":
            continue
        if key not in actual or not _profile_value_matches(expected_value, actual.get(key)):
            raise ReportConfigError(
                f"{where}.{key}={actual.get(key)!r} does not match "
                f"configured matched judge value {expected_value!r}"
            )


def _validate_judge_artifact(
    path: Path,
    *,
    expected_profile: Mapping[str, Any],
    where: str,
) -> dict[str, Any]:
    rows = 0
    deterministic_rows = 0
    profiled_judge_rows = 0
    task_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ReportConfigError(
                        f"{where} line {line_number} is not a JSON object"
                    )
                task_id = row.get("task_id")
                if not isinstance(task_id, str) or not task_id:
                    raise ReportConfigError(
                        f"{where} line {line_number} has no task_id"
                    )
                if task_id in task_ids:
                    raise ReportConfigError(f"{where} duplicate task_id: {task_id}")
                task_ids.add(task_id)
                if row.get("error") not in (None, ""):
                    raise ReportConfigError(
                        f"{where} line {line_number} has error={row.get('error')!r}"
                    )
                verdict = row.get("verdict")
                if not isinstance(verdict, dict) or not isinstance(
                    verdict.get("strict_correct"), bool
                ):
                    raise ReportConfigError(
                        f"{where} line {line_number} has no boolean "
                        "verdict.strict_correct"
                    )
                strict_correct = verdict["strict_correct"]
                deterministic = row.get("deterministic")
                if (
                    isinstance(deterministic, dict)
                    and deterministic.get("applicable") is True
                ):
                    metadata = row.get("metadata")
                    score_source = (
                        metadata.get("score_source")
                        if isinstance(metadata, dict)
                        else None
                    )
                    if score_source not in {
                        "exact",
                        "deterministic",
                        "reference_text",
                    }:
                        raise ReportConfigError(
                            f"{where} line {line_number} has unrecognized "
                            f"deterministic score_source={score_source!r}"
                        )
                    deterministic_match = deterministic.get("matched")
                    if (
                        not isinstance(deterministic_match, bool)
                        or deterministic_match is not strict_correct
                    ):
                        raise ReportConfigError(
                            f"{where} line {line_number} deterministic.matched "
                            "must equal verdict.strict_correct"
                        )
                    judge = row.get("judge")
                    if isinstance(judge, dict) and judge.get("error") not in (None, ""):
                        raise ReportConfigError(
                            f"{where} line {line_number} has judge.error="
                            f"{judge.get('error')!r}"
                        )
                    deterministic_rows += 1
                else:
                    actual = _judge_row_profile(row, f"{where} line {line_number}")
                    _validate_profile_values(
                        actual, expected_profile, f"{where} line {line_number}"
                    )
                    profiled_judge_rows += 1
                rows += 1
    except json.JSONDecodeError as exc:
        raise ReportConfigError(f"invalid judge JSONL {path}: {exc}") from exc
    if rows == 0:
        raise ReportConfigError(f"{where} is empty")
    if profiled_judge_rows == 0:
        raise ReportConfigError(f"{where} contains no rows from the matched judge")
    return {
        "rows": rows,
        "unique_task_ids": len(task_ids),
        "deterministic_rows": deterministic_rows,
        "profiled_judge_rows": profiled_judge_rows,
    }


def _validate_matched_judge_evidence(
    *,
    score_report: Mapping[str, Any],
    score_where: str,
    expected_profile: Mapping[str, Any] | None,
    matched_lineage: str,
    config_dir: Path,
    manifest_spec: Mapping[str, Any] | None,
    manifest_document: Mapping[str, Any] | None,
    manifest_source: Mapping[str, Any] | None,
    artifact_spec: Mapping[str, Any] | None,
    artifact_source: Mapping[str, Any] | None,
    artifact_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    configured = manifest_spec is not None or artifact_spec is not None
    if not configured:
        return {"validated": False, "method": None}, dict(artifact_source) if artifact_source else None
    if expected_profile is None:
        raise ReportConfigError(
            f"{score_where} configures judge evidence paths but "
            "config.matched_judge_profile is missing"
        )
    if manifest_spec is not None and manifest_document is None:
        raise ReportConfigError(f"{score_where} configured judge manifest is missing")
    if artifact_spec is not None and (artifact_path is None or not artifact_path.is_file()):
        raise ReportConfigError(f"{score_where} configured judge artifact is missing")

    manifest_profile_validated = False
    manifest_artifact: Mapping[str, Any] | None = None
    if manifest_document is not None:
        manifest_lineage, manifest_profile = _manifest_lineage_profile(
            manifest_document, f"{score_where}.judge_manifest"
        )
        if manifest_lineage is not None and manifest_lineage != matched_lineage:
            raise ReportConfigError(
                f"{score_where}.judge_manifest lineage {manifest_lineage!r} "
                f"does not match {matched_lineage!r}"
            )
        if manifest_profile is not None:
            _validate_profile_values(
                manifest_profile,
                expected_profile,
                f"{score_where}.judge_manifest.profile",
            )
            manifest_profile_validated = True
        manifest_artifact = _manifest_judge_artifact_spec(
            manifest_document, f"{score_where}.judge_manifest"
        )

    resolved_artifact_source = dict(artifact_source) if artifact_source else None
    resolved_artifact_path = artifact_path
    if resolved_artifact_path is None and manifest_artifact is not None:
        manifest_path = (
            Path(str(manifest_source["path"]))
            if isinstance(manifest_source, dict) and manifest_source.get("path")
            else config_dir
        )
        resolved_artifact_source, resolved_artifact_path = _source_record(
            manifest_artifact,
            config_dir=manifest_path.parent if manifest_path.is_file() else config_dir,
            where=f"{score_where}.judge_manifest.output",
        )

    score_judge_sha = _score_image_judge_sha(score_report, score_where)
    artifact_validated = False
    artifact_details: dict[str, Any] | None = None
    if resolved_artifact_path is not None and resolved_artifact_path.is_file():
        assert resolved_artifact_source is not None
        actual_sha = _required_str(
            resolved_artifact_source, "sha256", f"{score_where}.judge_artifact"
        )
        if actual_sha != score_judge_sha:
            raise ReportConfigError(
                f"{score_where} score image-judge SHA256 {score_judge_sha} "
                f"does not match configured artifact {actual_sha}"
            )
        if manifest_artifact is not None and isinstance(
            manifest_artifact.get("sha256"), str
        ):
            manifest_sha = str(manifest_artifact["sha256"]).lower()
            if manifest_sha != actual_sha:
                raise ReportConfigError(
                    f"{score_where} judge manifest output SHA256 does not match artifact"
                )
        artifact_details = _validate_judge_artifact(
            resolved_artifact_path,
            expected_profile=expected_profile,
            where=f"{score_where}.judge_artifact",
        )
        artifact_validated = True
    elif manifest_artifact is not None and isinstance(
        manifest_artifact.get("sha256"), str
    ):
        manifest_sha = str(manifest_artifact["sha256"]).lower()
        if manifest_sha != score_judge_sha:
            raise ReportConfigError(
                f"{score_where} judge manifest output SHA256 does not match score"
            )

    if not artifact_validated and not manifest_profile_validated:
        raise ReportConfigError(
            f"{score_where} judge evidence does not contain an accessible artifact "
            "or a structured matched judge profile"
        )
    return (
        {
            "validated": True,
            "method": (
                "artifact+manifest"
                if artifact_validated and manifest_document is not None
                else "artifact"
                if artifact_validated
                else "manifest-profile"
            ),
            "lineage": matched_lineage,
            "image_judge_sha256": score_judge_sha,
            "artifact": artifact_details,
        },
        resolved_artifact_source,
    )


def _build_entry(
    entry: Mapping[str, Any],
    *,
    config_dir: Path,
    expected_sizes: Mapping[str, int],
    benchmark_sha256: str,
    scorer_sha256: str,
    baseline_correct: int,
    matched_lineage: str,
    matched_profile: Mapping[str, Any] | None,
    baseline: bool,
) -> dict[str, Any]:
    entry_id = _required_str(entry, "id", "entry")
    name = _required_str(entry, "name", f"entry[{entry_id}]")
    matched_spec = _source_spec(entry.get("matched_score"), f"entry[{entry_id}].matched_score")
    replay_spec = _source_spec(entry.get("replay_score"), f"entry[{entry_id}].replay_score")
    manifest_spec = _source_spec(entry.get("manifest"), f"entry[{entry_id}].manifest")
    operational_spec = _source_spec(
        entry.get("operational_manifest"), f"entry[{entry_id}].operational_manifest"
    )
    judge_manifest_spec = _source_spec(
        matched_spec.get("judge_manifest") if matched_spec is not None else None,
        f"entry[{entry_id}].matched_score.judge_manifest",
    )
    judge_artifact_spec = _source_spec(
        matched_spec.get("judge_artifact") if matched_spec is not None else None,
        f"entry[{entry_id}].matched_score.judge_artifact",
    )

    # Validate source semantics even while a future score file is still missing.
    # This keeps a pending config from changing meaning merely because a file
    # later appears at the configured path.
    matched_judge = (
        _lineage(
            matched_spec,
            expected_matched=True,
            matched_lineage=matched_lineage,
            where=f"entry[{entry_id}].matched_score",
        )
        if matched_spec is not None
        else None
    )
    replay_judge = (
        _lineage(
            replay_spec,
            expected_matched=False,
            matched_lineage=matched_lineage,
            where=f"entry[{entry_id}].replay_score",
        )
        if replay_spec is not None
        else None
    )

    matched_document, matched_source = _load_optional_document(
        matched_spec, config_dir=config_dir, where=f"entry[{entry_id}].matched_score"
    )
    replay_document, replay_source = _load_optional_document(
        replay_spec, config_dir=config_dir, where=f"entry[{entry_id}].replay_score"
    )
    manifest_document, manifest_source = _load_optional_document(
        manifest_spec, config_dir=config_dir, where=f"entry[{entry_id}].manifest"
    )
    if manifest_document is not None:
        _validate_manifest_benchmark(
            manifest_document, benchmark_sha256, f"entry[{entry_id}].manifest"
        )
    progress = _progress_from_entry_and_manifest(
        entry, manifest_document, f"entry[{entry_id}]"
    )
    judge_manifest_document, judge_manifest_source = _load_optional_document(
        judge_manifest_spec,
        config_dir=config_dir,
        where=f"entry[{entry_id}].matched_score.judge_manifest",
    )
    judge_artifact_source, judge_artifact_path = _source_record(
        judge_artifact_spec,
        config_dir=config_dir,
        where=f"entry[{entry_id}].matched_score.judge_artifact",
    )

    if matched_document is not None:
        assert matched_spec is not None
        assert matched_judge is not None
        status = STATUS_EXACT
        selected_document = matched_document
        selected_source = matched_source
        judge = matched_judge
    elif replay_document is not None:
        assert replay_spec is not None
        assert replay_judge is not None
        status = STATUS_REPLAY
        selected_document = replay_document
        selected_source = replay_source
        judge = replay_judge
    elif progress is not None and 0 < progress["completed"] < progress["total"]:
        status = STATUS_PARTIAL
        selected_document = None
        selected_source = None
        judge = {"lineage": matched_lineage, "matched": False}
    else:
        status = STATUS_PENDING
        selected_document = None
        selected_source = None
        judge = {"lineage": matched_lineage, "matched": False}

    if baseline and status != STATUS_EXACT:
        raise ReportConfigError("baseline must have an existing matched_score")
    metrics = None
    if selected_document is not None:
        metrics = _parse_score(
            selected_document,
            where=f"entry[{entry_id}].{status}_score",
            benchmark_sha256=benchmark_sha256,
            scorer_sha256=scorer_sha256,
            expected_sizes=expected_sizes,
            baseline_correct=baseline_correct,
        )
    judge_evidence: dict[str, Any] = {"validated": False, "method": None}
    if status == STATUS_EXACT:
        assert selected_document is not None
        judge_evidence, judge_artifact_source = _validate_matched_judge_evidence(
            score_report=selected_document,
            score_where=f"entry[{entry_id}].exact_score",
            expected_profile=matched_profile,
            matched_lineage=matched_lineage,
            config_dir=config_dir,
            manifest_spec=judge_manifest_spec,
            manifest_document=judge_manifest_document,
            manifest_source=judge_manifest_source,
            artifact_spec=judge_artifact_spec,
            artifact_source=judge_artifact_source,
            artifact_path=judge_artifact_path,
        )
    judge = {**judge, "evidence": judge_evidence}
    operational, operational_source = _operational_from_source(
        operational_spec,
        config_dir=config_dir,
        fallback_document=selected_document,
        where=f"entry[{entry_id}].operational_manifest",
    )
    pre_judge_bounds = _bounds_from_manifest(
        manifest_document, expected_sizes, f"entry[{entry_id}].manifest"
    )
    if status not in (STATUS_PENDING, STATUS_PARTIAL):
        pre_judge_bounds = None

    return {
        "id": entry_id,
        "name": name,
        "kind": "baseline" if baseline else "idea",
        "status": status,
        "judge": judge,
        "overall": metrics["overall"] if metrics else None,
        "math": metrics["math"] if metrics else None,
        "non_math": metrics["non_math"] if metrics else None,
        "vs_baseline": metrics["vs_baseline"] if metrics else None,
        # ``matched_bounds`` is retained as a compatibility alias.  Both keys
        # now contain only explicitly pre-judge bounds and are always null once
        # an exact or replay score exists.
        "matched_bounds": pre_judge_bounds,
        "pre_judge_bounds": pre_judge_bounds,
        "progress": (
            progress if status in (STATUS_PENDING, STATUS_PARTIAL) else None
        ),
        "operational": operational,
        "sources": {
            "selected_score": selected_source,
            "matched_score": matched_source,
            "replay_score": replay_source,
            "manifest": manifest_source,
            "operational_manifest": operational_source,
            "judge_manifest": judge_manifest_source,
            "judge_artifact": judge_artifact_source,
        },
    }


def build_report(config_path: Path | str) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ReportConfigError(
            f"config.schema_version must be {CONFIG_SCHEMA_VERSION!r}"
        )
    benchmark_spec = config.get("benchmark")
    scorer_spec = config.get("scorer")
    baseline_spec = config.get("baseline")
    ideas = config.get("ideas")
    if not isinstance(benchmark_spec, dict) or not isinstance(scorer_spec, dict):
        raise ReportConfigError("config benchmark/scorer objects are required")
    if not isinstance(baseline_spec, dict):
        raise ReportConfigError("config.baseline must be an object")
    if not isinstance(ideas, list) or len(ideas) != 8:
        raise ReportConfigError("config.ideas must contain exactly 8 ideas")
    if any(not isinstance(idea, dict) for idea in ideas):
        raise ReportConfigError("every config.ideas item must be an object")

    config_dir = config_path.parent
    benchmark, expected_sizes = _load_benchmark(benchmark_spec, config_dir)
    scorer = _load_scorer(scorer_spec, config_dir)
    matched_lineage = _required_str(config, "matched_judge_lineage", "config")
    matched_profile = _matched_judge_profile(
        config.get("matched_judge_profile"), matched_lineage
    )
    baseline_reference = config.get("baseline_reference")
    if not isinstance(baseline_reference, dict):
        raise ReportConfigError("config.baseline_reference must be an object")
    baseline_correct = _int(
        baseline_reference.get("correct"), "baseline_reference.correct"
    )
    baseline_n = _int(baseline_reference.get("n"), "baseline_reference.n", minimum=1)
    if baseline_n != EXPECTED_TOTAL_N or baseline_correct != EXPECTED_BASELINE_CORRECT:
        raise ReportConfigError(
            "baseline_reference must be the frozen page-RAG result "
            f"{EXPECTED_BASELINE_CORRECT}/{EXPECTED_TOTAL_N}"
        )

    all_specs: Sequence[Mapping[str, Any]] = [baseline_spec, *ideas]
    ids = [_required_str(spec, "id", "entry") for spec in all_specs]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise ReportConfigError(f"duplicate idea/baseline IDs: {duplicates}")

    baseline_row = _build_entry(
        baseline_spec,
        config_dir=config_dir,
        expected_sizes=expected_sizes,
        benchmark_sha256=benchmark["sha256"],
        scorer_sha256=scorer["sha256"],
        baseline_correct=baseline_correct,
        matched_lineage=matched_lineage,
        matched_profile=matched_profile,
        baseline=True,
    )
    assert baseline_row["overall"] is not None
    if baseline_row["overall"]["correct"] != baseline_correct:
        raise ReportConfigError(
            "baseline matched score disagrees with baseline_reference.correct"
        )
    idea_rows = [
        _build_entry(
            idea,
            config_dir=config_dir,
            expected_sizes=expected_sizes,
            benchmark_sha256=benchmark["sha256"],
            scorer_sha256=scorer["sha256"],
            baseline_correct=baseline_correct,
            matched_lineage=matched_lineage,
            matched_profile=matched_profile,
            baseline=False,
        )
        for idea in ideas
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "benchmark": benchmark,
        "scorer": scorer,
        "matched_judge_lineage": matched_lineage,
        "matched_judge_profile": matched_profile,
        "baseline_reference": {
            "correct": baseline_correct,
            "n": baseline_n,
            "accuracy": round(baseline_correct / baseline_n, 6),
        },
        "baseline": baseline_row,
        "ideas": idea_rows,
        "status_counts": {
            status: sum(row["status"] == status for row in idea_rows)
            for status in (
                STATUS_EXACT,
                STATUS_REPLAY,
                STATUS_PARTIAL,
                STATUS_PENDING,
            )
        },
    }


def _format_slice(value: Mapping[str, Any] | None, status: str) -> str:
    if value is None:
        return "partial; no full-bench score" if status == STATUS_PARTIAL else "pending"
    suffix = " (replay)" if status == STATUS_REPLAY else ""
    return f"{value['correct']}/{value['n']} ({100 * value['accuracy']:.3f}%){suffix}"


def _format_bounds(bounds: Mapping[str, Any] | None) -> str:
    if not bounds or not isinstance(bounds.get("overall"), dict):
        return "—"
    overall = bounds["overall"]
    return (
        f"{overall['lower_correct']}–{overall['upper_correct']}/{overall['n']} "
        f"({100 * overall['lower_accuracy']:.3f}%–"
        f"{100 * overall['upper_accuracy']:.3f}%)"
    )


def _format_int(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}".replace(",", " ")


def _format_progress(progress: Mapping[str, Any] | None) -> str:
    if not progress:
        return "—"
    stage = f"; {progress['stage']}" if progress.get("stage") else ""
    return (
        f"{progress['completed']}/{progress['total']} "
        f"{progress['unit']}{stage}"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    baseline = report["baseline"]
    rows: Iterable[Mapping[str, Any]] = [baseline, *report["ideas"]]
    lines = [
        "# Maksim: 8 ideas on the frozen common benchmark",
        "",
        (
            f"Benchmark: **{report['benchmark']['rows']}** tasks; frozen baseline: "
            f"**{report['baseline_reference']['correct']}/{report['baseline_reference']['n']} "
            f"({100 * report['baseline_reference']['accuracy']:.3f}%)**. "
            f"Matched judge lineage: `{report['matched_judge_lineage']}`."
        ),
        "",
        "`exact` means a score from the configured matched judge lineage; "
        "`replay` is historical/non-matched evidence and is not an exact matched result; "
        "`partial` reports measured incomplete progress only, never accuracy; "
        "`pending` has no full score yet. Pre-judge bounds are shown only for "
        "`partial`/`pending` and are not scores.",
        "",
        "| Variant | Status | Progress | Overall | Math | Non-Math | Δ correct | Fixed / regressed | Pre-judge bounds | Model calls | Tokens | Judge lineage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        comparison = row.get("vs_baseline")
        delta = "—" if comparison is None else f"{comparison['delta_correct']:+d}"
        switches = (
            "—"
            if comparison is None
            else f"{comparison['fixed']} / {comparison['regressed']}"
        )
        judge = row["judge"]
        lineage = f"{judge['lineage']} ({'matched' if judge['matched'] else 'not matched'})"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    str(row["status"]),
                    _format_progress(row.get("progress")),
                    _format_slice(row.get("overall"), str(row["status"])),
                    _format_slice(row.get("math"), str(row["status"])),
                    _format_slice(row.get("non_math"), str(row["status"])),
                    delta,
                    switches,
                    _format_bounds(row.get("pre_judge_bounds")),
                    _format_int(row["operational"]["model_calls"]),
                    _format_int(row["operational"]["combined_tokens"]),
                    lineage,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Benchmark SHA256: `{report['benchmark']['sha256']}`",
            f"- Scorer SHA256: `{report['scorer']['sha256']}`",
            f"- Config SHA256: `{report['config']['sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    report: Mapping[str, Any], output_json: Path, output_md: Path, overwrite: bool
) -> None:
    targets = [output_json.resolve(), output_md.resolve()]
    if targets[0] == targets[1]:
        raise ReportConfigError("JSON and Markdown outputs must be different files")
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing report(s): "
            + ", ".join(str(path) for path in existing)
        )
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    targets[1].write_text(render_markdown(report), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.config)
    _write_outputs(report, args.output_json, args.output_md, args.overwrite)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "output_md": str(args.output_md.resolve()),
                "status_counts": report["status_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
