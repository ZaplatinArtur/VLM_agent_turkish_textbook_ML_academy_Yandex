#!/usr/bin/env python3
"""Materialize the frozen staged+truncation selector and exact-web overlay.

This is an explicitly exploratory/post-hoc, local-CPU-only composer.  It
replays the selector from the frozen source outcomes and public task fields;
it never opens the benchmark or a reference-answer/solution artifact.  The
solver JSONL is written and hashed before the selected frozen image-judge rows
are opened.  The standard scorer must be run separately after materialization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_maxim_local_tool_vote_oof_v1 import (  # noqa: E402
    Config,
    DEFAULT_ID,
    Source,
    evaluate_config,
    normalize_answer,
    stable_fold,
)
from analyze_maxim_pairwise_tool_selector_oof_v1 import (  # noqa: E402
    ORACLE_080_POOL,
    build_pairs,
    crossfit_probabilities,
    select_with_threshold,
)
from compose_maxim_truncation_repair_v1 import capped_incomplete  # noqa: E402


SCHEMA_VERSION = "maxim-staged-truncation-exact-web-overlay-composition-v1"
PROFILE_SCHEMA_VERSION = "maxim-staged-truncation-exact-web-overlay-profile-v1"
FINAL_REPORT_SCHEMA_VERSION = "maxim-staged-truncation-exact-web-overlay-report-v1"
EXPECTED_ROWS = 274
EXPECTED_IMAGE_ROWS = 97
EXPECTED_DETERMINISTIC_ROWS = 177
TRUNCATION_SOURCE = "final_meta_verifier_v3_choice_token_compat_v31"
EXACT_WEB_CONDITION = "maxim_exact_official_web_deterministic_v1"

REFERENCE_FIELD_NAMES = {
    "reference_answer",
    "reference_solution",
    "gold_answer",
    "gold_solution",
}


class CompositionError(RuntimeError):
    """Raised when frozen inputs or composition invariants do not match."""


@dataclass(frozen=True)
class FrozenSource:
    source: Source
    rows: tuple[dict[str, Any], ...]
    score_sources: tuple[str, ...]
    judge_path: Path | None
    judge_sha256: str | None
    score_sha256: str
    solver_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_path(repo_root: Path, raw: object) -> Path:
    path = Path(str(raw or ""))
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompositionError(f"{label}: cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompositionError(f"{label}: expected JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise CompositionError(f"{label}: cannot read JSONL {path}: {exc}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompositionError(f"{label}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise CompositionError(f"{label}:{number}: expected object")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen:
            raise CompositionError(f"{label}:{number}: missing/duplicate task_id")
        seen.add(task_id)
        rows.append(row)
    if not rows:
        raise CompositionError(f"{label}: no rows: {path}")
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as target:
            json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def assert_no_reference_fields(value: Any, label: str, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key in REFERENCE_FIELD_NAMES:
                raise CompositionError(f"{label}: forbidden reference field at {path}.{raw_key}")
            assert_no_reference_fields(child, label, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_reference_fields(child, label, f"{path}[{index}]")


def verify_record(repo_root: Path, record: Mapping[str, Any], label: str) -> Path:
    path = resolve_path(repo_root, record.get("path"))
    expected = str(record.get("sha256") or "").lower()
    if not path.is_file():
        raise CompositionError(f"{label}: missing file: {path}")
    actual = sha256_file(path)
    if expected and actual != expected:
        raise CompositionError(f"{label}: SHA256 mismatch: expected={expected}, actual={actual}")
    return path


def load_profile(repo_root: Path, path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = resolve_path(repo_root, path)
    profile = read_json(resolved, "profile")
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise CompositionError(f"profile schema mismatch: {resolved}")
    if profile.get("status") != "frozen_after_prior_target_outcome_exposure_posthoc_exploratory":
        raise CompositionError(f"profile status mismatch: {resolved}")
    inputs = profile.get("inputs")
    if not isinstance(inputs, Mapping):
        raise CompositionError(f"profile inputs missing: {resolved}")
    for name in ("ledger", "public_tasks", "frozen_analyzer_report", "exact_web_solver"):
        record = inputs.get(name)
        if not isinstance(record, Mapping):
            raise CompositionError(f"profile input missing: {name}")
        verify_record(repo_root, record, f"profile.{name}")
    return resolved, profile


def source_clean(row: Mapping[str, Any], normalized: str) -> bool:
    return (
        bool(normalized)
        and not bool(row.get("error"))
        and row.get("forced_answer") is not True
        and len(str(row.get("final_answer") or "")) <= 256
    )


def load_frozen_sources(
    repo_root: Path,
    ledger_path: Path,
    public_rows: list[dict[str, Any]],
) -> tuple[dict[str, Source], dict[str, FrozenSource], list[str]]:
    ledger = read_json(ledger_path, "ledger")
    task_order = [str(row["task_id"]) for row in public_rows]
    if len(task_order) != EXPECTED_ROWS or len(set(task_order)) != EXPECTED_ROWS:
        raise CompositionError("public task order must contain 274 unique rows")
    frozen: dict[str, FrozenSource] = {}
    for branch in ledger.get("branches", []):
        if not isinstance(branch, Mapping) or branch.get("status") != "final":
            continue
        report_record = branch.get("report")
        if not isinstance(report_record, Mapping) or not report_record.get("path"):
            continue
        score_path = resolve_path(repo_root, report_record.get("path"))
        if not score_path.is_file():
            continue
        expected_score_sha = str(report_record.get("sha256") or "").lower()
        score_sha = sha256_file(score_path)
        if expected_score_sha and score_sha != expected_score_sha:
            raise CompositionError(f"ledger score SHA mismatch: {score_path}")
        score = read_json(score_path, f"score:{branch.get('id')}")
        provenance = score.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        solver_record = provenance.get("solver_results")
        if not isinstance(solver_record, Mapping) or not solver_record.get("path"):
            continue
        solver_path = resolve_path(repo_root, solver_record.get("path"))
        if not solver_path.is_file():
            continue
        solver_sha = sha256_file(solver_path)
        expected_solver_sha = str(solver_record.get("sha256") or "").lower()
        if expected_solver_sha and solver_sha != expected_solver_sha:
            raise CompositionError(f"score solver SHA mismatch: {solver_path}")
        rows = read_jsonl(solver_path, f"solver:{branch.get('id')}")
        if len(rows) != EXPECTED_ROWS:
            continue
        assert_no_reference_fields(rows, f"solver:{branch.get('id')}")
        row_order = [str(row["task_id"]) for row in rows]
        if row_order != task_order:
            continue
        outcomes = score.get("task_outcomes")
        if not isinstance(outcomes, list) or len(outcomes) != EXPECTED_ROWS:
            continue
        outcome_order = [str(row.get("task_id") or "") for row in outcomes if isinstance(row, Mapping)]
        if outcome_order != task_order:
            continue
        source_id = str(branch.get("id") or "")
        if not source_id:
            continue
        answers = tuple(normalize_answer(row.get("final_answer")) for row in rows)
        source = Source(
            source_id=source_id,
            solver_path=solver_path,
            score_path=score_path,
            answers=answers,
            clean=tuple(source_clean(row, answer) for row, answer in zip(rows, answers)),
            outcomes=tuple(bool(row.get("new_correct")) for row in outcomes),
        )
        judge_record = provenance.get("image_judge")
        judge_path: Path | None = None
        judge_sha: str | None = None
        if isinstance(judge_record, Mapping) and judge_record.get("path"):
            candidate_path = resolve_path(repo_root, judge_record.get("path"))
            if candidate_path.is_file():
                candidate_sha = sha256_file(candidate_path)
                expected_judge_sha = str(judge_record.get("sha256") or "").lower()
                if expected_judge_sha and candidate_sha != expected_judge_sha:
                    raise CompositionError(f"score judge SHA mismatch: {candidate_path}")
                judge_path = candidate_path
                judge_sha = candidate_sha
        frozen[source_id] = FrozenSource(
            source=source,
            rows=tuple(rows),
            score_sources=tuple(str(row.get("score_source") or "") for row in outcomes),
            judge_path=judge_path,
            judge_sha256=judge_sha,
            score_sha256=score_sha,
            solver_sha256=solver_sha,
        )
    if DEFAULT_ID not in frozen:
        raise CompositionError(f"default source missing from frozen ledger: {DEFAULT_ID}")
    if TRUNCATION_SOURCE not in frozen:
        raise CompositionError(f"truncation source missing: {TRUNCATION_SOURCE}")
    return {source_id: item.source for source_id, item in frozen.items()}, frozen, task_order


def confidence(row: Mapping[str, Any]) -> float:
    generation = row.get("generation")
    value = generation.get("confidence") if isinstance(generation, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 1.0


def replay_staged_selector(
    public_rows: list[dict[str, Any]],
    sources: Mapping[str, Source],
    frozen: Mapping[str, FrozenSource],
    selector_config: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    vote_record = selector_config.get("vote")
    pairwise_record = selector_config.get("pairwise")
    if not isinstance(vote_record, Mapping) or not isinstance(pairwise_record, Mapping):
        raise CompositionError("selector vote/pairwise config missing")
    vote_config = Config(
        pool=str(vote_record["pool"]),
        minimum_accuracy=float(vote_record["minimum_accuracy"]),
        context=str(vote_record["context"]),
        shrinkage=float(vote_record["shrinkage"]),
        weight=str(vote_record["weight"]),
        default_bonus=float(vote_record["default_bonus"]),
        minimum_support=int(vote_record["minimum_support"]),
        minimum_margin=float(vote_record["minimum_margin"]),
    )
    folds = [stable_fold(str(row["task_id"])) for row in public_rows]
    vote = evaluate_config(vote_config, public_rows, sources, folds)
    solver_rows = {source_id: list(item.rows) for source_id, item in frozen.items()}
    pairs, _ = build_pairs(public_rows, sources, solver_rows)
    probabilities = crossfit_probabilities(
        pairs,
        folds,
        epochs=int(pairwise_record["epochs"]),
        learning_rate=float(pairwise_record["learning_rate"]),
        l2=float(pairwise_record["l2"]),
    )
    pairwise_pool = tuple(
        source_id for source_id in ORACLE_080_POOL if source_id in sources
    )
    pairwise = select_with_threshold(
        probabilities,
        sources,
        len(public_rows),
        pairwise_pool,
        threshold=float(pairwise_record["threshold"]),
        minimum_probability_gap=float(pairwise_record["minimum_probability_gap"]),
    )
    max_confidence = float(selector_config["anchor_confidence_max"])
    cap = int(selector_config["truncation"]["exact_character_cap"])
    default_rows = frozen[DEFAULT_ID].rows
    selected: list[str] = []
    audits: list[dict[str, Any]] = []
    for index, task in enumerate(public_rows):
        selected_source = DEFAULT_ID
        row_confidence = confidence(default_rows[index])
        reason = "anchor_confidence_above_hurdle"
        if row_confidence <= max_confidence:
            if vote["selected"][index] != DEFAULT_ID:
                selected_source = str(vote["selected"][index])
                reason = "crossfit_vote_override"
            elif pairwise["selected"][index] != DEFAULT_ID:
                selected_source = str(pairwise["selected"][index])
                reason = "crossfit_pairwise_fallback_override"
            else:
                reason = "hurdle_open_no_override"
        triggered, truncation_reason = capped_incomplete(
            str(default_rows[index].get("final_answer") or ""),
            cap,
        )
        if triggered:
            selected_source = TRUNCATION_SOURCE
            reason = "truncation_repair_last:" + truncation_reason
        selected.append(selected_source)
        audits.append(
            {
                "task_id": str(task["task_id"]),
                "fold": folds[index],
                "anchor_confidence": row_confidence,
                "vote_selected_source": str(vote["selected"][index]),
                "vote_support": int(vote["supports"][index]),
                "pairwise_selected_source": str(pairwise["selected"][index]),
                "staged_selected_source": selected_source,
                "staged_reason": reason,
                "reference_text_accessed": False,
                "held_out_outcome_used_as_feature": False,
            }
        )
    correctness = [sources[source_id].outcomes[index] for index, source_id in enumerate(selected)]
    replay = {
        "correct": sum(correctness),
        "denominator": len(correctness),
        "accuracy": sum(correctness) / len(correctness),
        "overrides": sum(source_id != DEFAULT_ID for source_id in selected),
        "selected_source_counts": dict(sorted(Counter(selected).items())),
        "selection_sha256": canonical_sha256(
            [[str(public_rows[index]["task_id"]), source_id] for index, source_id in enumerate(selected)]
        ),
        "vote_config": vote_config.key(),
        "pairwise_config": {
            "pool": list(pairwise_pool),
            "epochs": int(pairwise_record["epochs"]),
            "learning_rate": float(pairwise_record["learning_rate"]),
            "l2": float(pairwise_record["l2"]),
            "threshold": float(pairwise_record["threshold"]),
            "minimum_probability_gap": float(pairwise_record["minimum_probability_gap"]),
        },
    }
    return selected, audits, replay


def verify_analyzer_replay(
    analyzer_report: Mapping[str, Any],
    task_order: list[str],
    selected: list[str],
    replay: Mapping[str, Any],
) -> None:
    section = analyzer_report.get("posthoc_staged_plus_truncation_repair")
    score = section.get("score") if isinstance(section, Mapping) else None
    if not isinstance(score, Mapping):
        raise CompositionError("frozen analyzer staged+truncation score missing")
    expected_counts = score.get("selected_source_counts")
    actual_override_ids = [
        task_order[index]
        for index, source_id in enumerate(selected)
        if source_id != DEFAULT_ID
    ]
    checks = {
        "correct": replay["correct"] == score.get("correct"),
        "denominator": replay["denominator"] == score.get("denominator"),
        "overrides": replay["overrides"] == score.get("overrides"),
        "selected_source_counts": replay["selected_source_counts"] == expected_counts,
        "override_task_ids": actual_override_ids == score.get("override_task_ids"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise CompositionError(f"staged+truncation replay mismatch: {failures}")


def is_exact_web_row(row: Mapping[str, Any]) -> bool:
    generation = row.get("generation")
    return (
        row.get("condition") == EXACT_WEB_CONDITION
        and row.get("model") == "exact-official-web-key"
        and isinstance(generation, Mapping)
        and generation.get("exact_question_match") is True
        and generation.get("explicit_official_answer_key") is True
    )


def exact_web_hosts(row: Mapping[str, Any]) -> set[str]:
    hosts: set[str] = set()
    calls = row.get("tool_calls")
    if not isinstance(calls, list):
        return hosts
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        for key in ("url", "key_url"):
            value = str(call.get(key) or "")
            if value:
                host = (urlsplit(value).hostname or "").casefold()
                if host:
                    hosts.add(host)
    return hosts


def overlay_allowed(row: Mapping[str, Any], overlay: Mapping[str, Any]) -> bool:
    if not is_exact_web_row(row):
        return False
    mode = str(overlay.get("mode") or "")
    if mode == "official_hosts_only":
        hosts = exact_web_hosts(row)
        allowed = {str(value).casefold() for value in overlay.get("allowed_hosts", [])}
        return bool(hosts) and hosts <= allowed
    if mode == "all_frozen_exact_web_rows_including_third_party_copy":
        return True
    raise CompositionError(f"unknown overlay mode: {mode}")


def solver_failed(row: Mapping[str, Any]) -> bool:
    return bool(row.get("error")) or not bool(str(row.get("final_answer") or "").strip())


def strict_judge_value(row: Mapping[str, Any]) -> bool:
    verdict = row.get("verdict")
    if not isinstance(verdict, Mapping) or not isinstance(verdict.get("strict_correct"), bool):
        raise CompositionError(f"judge row missing verdict.strict_correct: {row.get('task_id')}")
    judge = row.get("judge")
    if isinstance(judge, Mapping) and judge.get("error"):
        raise CompositionError(f"judge row has error: {row.get('task_id')}: {judge.get('error')}")
    return bool(verdict["strict_correct"])


def materialize_profile(
    *,
    repo_root: Path,
    output_root: Path,
    profile_path: Path,
    profile: Mapping[str, Any],
    public_rows: list[dict[str, Any]],
    task_order: list[str],
    sources: Mapping[str, Source],
    frozen: Mapping[str, FrozenSource],
    staged_selected: list[str],
    staged_audits: list[dict[str, Any]],
    replay: Mapping[str, Any],
    exact_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    profile_id = str(profile.get("profile_id") or "")
    if not profile_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in profile_id):
        raise CompositionError(f"invalid profile_id: {profile_id!r}")
    profile_dir = output_root / profile_id
    if profile_dir.exists():
        raise CompositionError(f"profile output already exists: {profile_dir}")
    exact_by_id = {str(row["task_id"]): row for row in exact_rows}
    overlay = profile.get("overlay")
    if not isinstance(overlay, Mapping):
        raise CompositionError(f"{profile_id}: overlay config missing")

    output_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    final_sources: list[str] = []
    overlay_ids: list[str] = []
    default_partition = frozen[DEFAULT_ID].score_sources
    for index, task_id in enumerate(task_order):
        staged_source = staged_selected[index]
        final_source = staged_source
        row = copy.deepcopy(frozen[staged_source].rows[index])
        exact_row = exact_by_id.get(task_id)
        web_applied = exact_row is not None and overlay_allowed(exact_row, overlay)
        if web_applied:
            if default_partition[index] != "deterministic":
                raise CompositionError(f"{profile_id}: exact-web row is not deterministic: {task_id}")
            row = copy.deepcopy(exact_row)
            final_source = "exact_web"
            overlay_ids.append(task_id)
        assert_no_reference_fields(row, f"composed solver:{profile_id}:{task_id}")
        output_rows.append(row)
        final_sources.append(final_source)
        decision = copy.deepcopy(staged_audits[index])
        decision.update(
            {
                "final_selected_source": final_source,
                "exact_web_overlay_applied_last": web_applied,
                "exact_web_hosts": sorted(exact_web_hosts(exact_row)) if exact_row else [],
                "final_answer_sha256": canonical_sha256(str(row.get("final_answer") or "")),
            }
        )
        decision_rows.append(decision)
    expected_overlay_rows = int(overlay["expected_rows"])
    if len(overlay_ids) != expected_overlay_rows:
        raise CompositionError(
            f"{profile_id}: expected {expected_overlay_rows} exact-web rows, got {len(overlay_ids)}"
        )

    run_dir = profile_dir / "run"
    solver_path = run_dir / "solver.jsonl"
    decisions_path = run_dir / "selection_audit.jsonl"
    atomic_write_jsonl(solver_path, output_rows)
    atomic_write_jsonl(decisions_path, decision_rows)
    solver_freeze_path = run_dir / "solver_freeze.json"
    solver_freeze = {
        "schema_version": "maxim-solver-freeze-before-judge-v1",
        "profile_id": profile_id,
        "rows": len(output_rows),
        "solver": {"path": str(solver_path.resolve()), "sha256": sha256_file(solver_path)},
        "selection_audit": {
            "path": str(decisions_path.resolve()),
            "sha256": sha256_file(decisions_path),
        },
        "judge_artifacts_opened_before_this_freeze": False,
        "benchmark_or_reference_artifacts_opened": False,
    }
    atomic_write_json(solver_freeze_path, solver_freeze)

    # Deliberate phase boundary: judge files are opened only after solver_freeze.
    judge_cache: dict[str, dict[str, dict[str, Any]]] = {}
    judge_rows: list[dict[str, Any]] = []
    judge_lineage: list[dict[str, Any]] = []
    image_indices = [
        index for index, source_name in enumerate(default_partition) if source_name == "image_judge"
    ]
    if len(image_indices) != EXPECTED_IMAGE_ROWS:
        raise CompositionError(f"frozen image partition mismatch: {len(image_indices)}")
    if len(default_partition) - len(image_indices) != EXPECTED_DETERMINISTIC_ROWS:
        raise CompositionError("frozen deterministic partition mismatch")
    for index in image_indices:
        task_id = task_order[index]
        source_id = staged_selected[index]
        item = frozen[source_id]
        if item.judge_path is None or item.judge_sha256 is None:
            raise CompositionError(f"selected source has no frozen image judge: {source_id}")
        if source_id not in judge_cache:
            rows = read_jsonl(item.judge_path, f"judge:{source_id}")
            judge_cache[source_id] = {str(row["task_id"]): row for row in rows}
        judge_row = judge_cache[source_id].get(task_id)
        if judge_row is None:
            raise CompositionError(f"selected judge row missing: {source_id}:{task_id}")
        expected_outcome = strict_judge_value(judge_row) and not solver_failed(output_rows[index])
        if expected_outcome != sources[source_id].outcomes[index]:
            raise CompositionError(f"selected judge/outcome mismatch: {source_id}:{task_id}")
        judge_rows.append(copy.deepcopy(judge_row))
        judge_lineage.append(
            {
                "task_id": task_id,
                "selected_source": source_id,
                "source_solver_sha256": item.solver_sha256,
                "source_judge_path": str(item.judge_path),
                "source_judge_sha256": item.judge_sha256,
                "selected_judge_row_sha256": canonical_sha256(judge_row),
                "selected_solver_answer_sha256": canonical_sha256(
                    str(output_rows[index].get("final_answer") or "")
                ),
                "outcome_parity_verified": True,
            }
        )
    evaluation_dir = profile_dir / "evaluation"
    judge_path = evaluation_dir / "matched_image97_judge.jsonl"
    judge_lineage_path = evaluation_dir / "matched_image97_judge_lineage.jsonl"
    atomic_write_jsonl(judge_path, judge_rows)
    atomic_write_jsonl(judge_lineage_path, judge_lineage)

    used_sources = sorted(set(staged_selected))
    source_records = {
        source_id: {
            "solver": {
                "path": str(frozen[source_id].source.solver_path),
                "sha256": frozen[source_id].solver_sha256,
            },
            "score_outcomes": {
                "path": str(frozen[source_id].source.score_path),
                "sha256": frozen[source_id].score_sha256,
            },
            "image_judge": (
                {
                    "path": str(frozen[source_id].judge_path),
                    "sha256": frozen[source_id].judge_sha256,
                }
                if frozen[source_id].judge_path is not None
                else None
            ),
        }
        for source_id in used_sources
    }
    manifest_path = profile_dir / "composition_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id,
        "reporting_status": "exploratory_posthoc_not_independent_holdout_not_strict_final",
        "profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
        "selector_replay": dict(replay),
        "web_overlay": {
            "applied_last": True,
            "mode": overlay["mode"],
            "rows": len(overlay_ids),
            "task_ids": overlay_ids,
            "final_source_counts": dict(sorted(Counter(final_sources).items())),
        },
        "outputs": {
            "solver": {"path": str(solver_path.resolve()), "sha256": sha256_file(solver_path), "rows": len(output_rows)},
            "solver_freeze": {"path": str(solver_freeze_path.resolve()), "sha256": sha256_file(solver_freeze_path)},
            "selection_audit": {"path": str(decisions_path.resolve()), "sha256": sha256_file(decisions_path), "rows": len(decision_rows)},
            "matched_image97_judge": {"path": str(judge_path.resolve()), "sha256": sha256_file(judge_path), "rows": len(judge_rows)},
            "matched_image97_judge_lineage": {"path": str(judge_lineage_path.resolve()), "sha256": sha256_file(judge_lineage_path), "rows": len(judge_lineage)},
        },
        "used_frozen_sources": source_records,
        "no_gold_composer_audit": {
            "benchmark_opened": False,
            "reference_answer_accessed": False,
            "reference_solution_accessed": False,
            "public_task_fields_recursively_checked": True,
            "solver_rows_recursively_checked": True,
            "binary_outcome_labels_read_from_frozen_score_reports": True,
            "binary_outcome_use": "cross-fitted source reliability/pairwise training and replay verification",
            "held_out_outcome_used_as_selector_feature": False,
            "posthoc_hyperparameters_and_targeting": True,
            "solver_frozen_before_selected_judge_rows_opened": True,
            "standard_scorer_invoked_during_composition": False,
        },
        "resource_policy": {
            "network_calls": 0,
            "model_calls": 0,
            "gpu_calls": 0,
            "shared_compute_calls": 0,
            "local_cpu_only": True,
        },
        "limitations": [
            "The selector hyperparameters and web target set were chosen after outcome exposure.",
            "Binary correctness labels from frozen score reports are used for cross-fitted training.",
            "This materialization is exploratory and is not an independent holdout result.",
        ],
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "profile_id": profile_id,
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "solver": manifest["outputs"]["solver"],
        "matched_image97_judge": manifest["outputs"]["matched_image97_judge"],
        "selector_replay_correct": replay["correct"],
        "web_overlay_rows": len(overlay_ids),
    }


def materialize(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    output_root = resolve_path(repo_root, args.output_dir)
    if output_root.exists():
        raise CompositionError(f"output already exists: {output_root}")
    loaded_profiles = [load_profile(repo_root, path) for path in args.profile]
    if not loaded_profiles:
        raise CompositionError("at least one profile is required")
    first_inputs = loaded_profiles[0][1]["inputs"]
    for _, profile in loaded_profiles[1:]:
        if profile["inputs"] != first_inputs or profile["selector"] != loaded_profiles[0][1]["selector"]:
            raise CompositionError("profiles must share frozen inputs and selector")
    ledger_path = verify_record(repo_root, first_inputs["ledger"], "ledger")
    public_path = verify_record(repo_root, first_inputs["public_tasks"], "public_tasks")
    analyzer_path = verify_record(repo_root, first_inputs["frozen_analyzer_report"], "analyzer")
    exact_path = verify_record(repo_root, first_inputs["exact_web_solver"], "exact_web_solver")
    public_rows = read_jsonl(public_path, "public_tasks")
    assert_no_reference_fields(public_rows, "public_tasks")
    sources, frozen, task_order = load_frozen_sources(repo_root, ledger_path, public_rows)
    staged_selected, staged_audits, replay = replay_staged_selector(
        public_rows,
        sources,
        frozen,
        loaded_profiles[0][1]["selector"],
    )
    analyzer_report = read_json(analyzer_path, "frozen_analyzer_report")
    verify_analyzer_replay(analyzer_report, task_order, staged_selected, replay)
    exact_rows = read_jsonl(exact_path, "exact_web_solver")
    assert_no_reference_fields(exact_rows, "exact_web_solver")
    if [str(row["task_id"]) for row in exact_rows] != task_order:
        raise CompositionError("exact-web solver task order mismatch")

    output_root.mkdir(parents=True)
    profile_results = []
    for profile_path, profile in loaded_profiles:
        profile_results.append(
            materialize_profile(
                repo_root=repo_root,
                output_root=output_root,
                profile_path=profile_path,
                profile=profile,
                public_rows=public_rows,
                task_order=task_order,
                sources=sources,
                frozen=frozen,
                staged_selected=staged_selected,
                staged_audits=staged_audits,
                replay=replay,
                exact_rows=exact_rows,
            )
        )
    top_manifest = {
        "schema_version": SCHEMA_VERSION,
        "reporting_status": "exploratory_posthoc_not_independent_holdout_not_strict_final",
        "composer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": first_inputs,
        "staged_truncation_replay": replay,
        "profiles": profile_results,
        "score_phase": "not_run; invoke scripts/score_maxim_full274.py only after this manifest exists",
        "no_gold_composer": True,
        "network_model_gpu_calls": 0,
    }
    manifest_path = output_root / "materialization_manifest.json"
    atomic_write_json(manifest_path, top_manifest)
    print(json.dumps(top_manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Staged + truncation + exact-web overlay v1",
        "",
        "> Exploratory/post-hoc result, not an independent holdout and not a strict-final claim.",
        "",
        "The frozen staged+truncation selector was reproduced at "
        f"{report['staged_truncation_replay']['correct']}/{report['staged_truncation_replay']['denominator']} "
        "before either web profile was overlaid.",
        "",
        "| Profile | Correct | Accuracy | Delta vs staged 211 | Web rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["profiles"]:
        lines.append(
            f"| `{item['profile_id']}` | {item['correct']}/{item['denominator']} | "
            f"{item['accuracy']:.6f} | {item['delta_vs_staged']:+d} | {item['web_overlay_rows']} |"
        )
    lines += [
        "",
        "Each solver was frozen before the corresponding 97-row judge file was assembled. "
        "Every image row was copied from the frozen judge artifact of the row's selected source. "
        "The standard `score_maxim_full274.py` scorer was then run against the frozen outputs.",
        "",
        "The official-only policy permits only OSYM/MEB hosts. The exploratory profile also "
        "includes the frozen `val_0194` third-party identical-copy evidence row.",
        "",
        "No model, network, GPU, or shared-compute calls were made by this experiment.",
        "",
    ]
    return "\n".join(lines)


def finalize(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    output_root = resolve_path(repo_root, args.output_dir)
    materialization_path = output_root / "materialization_manifest.json"
    materialization = read_json(materialization_path, "materialization_manifest")
    profiles: list[dict[str, Any]] = []
    for record in materialization.get("profiles", []):
        if not isinstance(record, Mapping):
            raise CompositionError("invalid materialized profile record")
        profile_id = str(record["profile_id"])
        composition_path = Path(str(record["manifest"]["path"]))
        if sha256_file(composition_path) != record["manifest"]["sha256"]:
            raise CompositionError(f"composition manifest changed before scoring: {profile_id}")
        composition = read_json(composition_path, f"composition:{profile_id}")
        score_path = output_root / profile_id / "evaluation" / "score.json"
        score = read_json(score_path, f"score:{profile_id}")
        provenance = score.get("provenance")
        if not isinstance(provenance, Mapping):
            raise CompositionError(f"score provenance missing: {profile_id}")
        if provenance.get("solver_results", {}).get("sha256") != composition["outputs"]["solver"]["sha256"]:
            raise CompositionError(f"scored solver is not frozen solver: {profile_id}")
        if provenance.get("image_judge", {}).get("sha256") != composition["outputs"]["matched_image97_judge"]["sha256"]:
            raise CompositionError(f"scored judge is not matched frozen judge: {profile_id}")
        overall = score.get("overall")
        if not isinstance(overall, Mapping):
            raise CompositionError(f"score overall missing: {profile_id}")
        correct = int(overall["new_correct"])
        denominator = int(overall["n"])
        profiles.append(
            {
                "profile_id": profile_id,
                "reporting_status": "exploratory_posthoc_not_independent_holdout",
                "correct": correct,
                "denominator": denominator,
                "accuracy": correct / denominator,
                "delta_vs_staged": correct - int(materialization["staged_truncation_replay"]["correct"]),
                "web_overlay_rows": int(record["web_overlay_rows"]),
                "composition_manifest": artifact_record(composition_path),
                "score": artifact_record(score_path),
                "solver": composition["outputs"]["solver"],
                "matched_image97_judge": composition["outputs"]["matched_image97_judge"],
            }
        )
    report = {
        "schema_version": FINAL_REPORT_SCHEMA_VERSION,
        "created_on": "2026-08-04",
        "reporting_status": "exploratory_posthoc_not_independent_holdout_not_strict_final",
        "staged_truncation_replay": materialization["staged_truncation_replay"],
        "profiles": profiles,
        "protocol": {
            "composer_reference_access": False,
            "solver_frozen_before_judge_assembly": True,
            "judge_rows": EXPECTED_IMAGE_ROWS,
            "judge_selection": "per task from the frozen artifact of the staged-selected source",
            "exact_web_overlay_order": "last, deterministic rows only",
            "standard_scorer": "scripts/score_maxim_full274.py",
        },
        "resource_policy": {"network_calls": 0, "model_calls": 0, "gpu_calls": 0, "shared_compute_calls": 0},
        "limitations": [
            "All selector and web-routing claims are post-hoc and exploratory.",
            "The 274-row benchmark is not an untouched holdout for these profiles.",
            "The profile including val_0194 uses a third-party identical copy because the official endpoint was unavailable.",
        ],
    }
    report_json = output_root / "REPORT.json"
    report_md = output_root / "REPORT.md"
    atomic_write_json(report_json, report)
    report_md.write_text(markdown_report(report), encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": FINAL_REPORT_SCHEMA_VERSION,
        "reporting_status": report["reporting_status"],
        "composer": artifact_record(Path(__file__).resolve()),
        "materialization_manifest": artifact_record(materialization_path),
        "report_json": artifact_record(report_json),
        "report_md": artifact_record(report_md),
        "profiles": profiles,
        "old_reports_modified": False,
        "closure_or_target_reports_modified": False,
    }
    manifest_path = output_root / "MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    checksum_paths = [
        Path(__file__).resolve(),
        materialization_path,
        report_json,
        report_md,
        manifest_path,
    ]
    for profile in profiles:
        profile_dir = output_root / profile["profile_id"]
        checksum_paths.extend(sorted(path for path in profile_dir.rglob("*") if path.is_file()))
    unique_paths = sorted(set(path.resolve() for path in checksum_paths), key=str)
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(repo_root).as_posix()}\n" for path in unique_paths
    )
    (output_root / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--profile", type=Path, action="append", required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    materialize_parser.set_defaults(handler=materialize)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    finalize_parser.set_defaults(handler=finalize)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except CompositionError as exc:
        print(f"COMPOSITION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
