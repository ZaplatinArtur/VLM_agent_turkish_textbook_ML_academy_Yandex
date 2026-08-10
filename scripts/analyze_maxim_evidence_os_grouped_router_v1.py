#!/usr/bin/env python3
"""Exploratory, evaluator-only grouped cross-fit routing diagnostic.

This script estimates whether *observable* properties of already-frozen solver
branches can route between their answers.  It is deliberately not a production
composer and it is not a benchmark score:

* correctness labels are consumed only by this evaluator;
* every reported task prediction is out-of-fold by canonical source family;
* regularization and the override margin are selected in an inner grouped CV;
* source locators are used only to form splits and are never model features;
* task ids are used only to align rows and to make the audit output readable.

The pure-Python logistic regression keeps the diagnostic runnable in the
minimal repository environment (which does not require numpy or sklearn).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, unquote, urlsplit


SCHEMA_VERSION = "maxim-evidence-os-grouped-router-v1"
LABEL = "EXPLORATORY_CROSSFIT_DIAGNOSTIC_NOT_HOLDOUT"
BRANCH_ORDER = (
    "anchor",
    "active_crop",
    "structural_rag",
    "calculator",
    "parser",
    "visual_sketchpad",
)
L2_GRID = (0.003, 0.03, 0.15)
MARGIN_GRID = (0.0, 0.03, 0.07, 0.12, 0.20)
OUTER_FOLDS = 5
INNER_FOLDS = 3
EPOCHS = 160


class DiagnosticError(RuntimeError):
    """Raised when a frozen-input or leakage invariant is violated."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DiagnosticError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise DiagnosticError(f"non-object JSONL row at {path}:{line_no}")
                rows.append(row)
    except OSError as exc:
        raise DiagnosticError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DiagnosticError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_mapping(value: Any, source: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DiagnosticError(f"{source}: expected object")
    return value


def _require_sha256(value: Any, source: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise DiagnosticError(f"{source}: expected a 64-character SHA-256")
    return value.casefold()


def _require_bool(value: Any, source: str) -> bool:
    if type(value) is not bool:
        raise DiagnosticError(f"{source}: expected strict JSON boolean")
    return value


def _config_path(repo_root: Path, value: Any, source: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DiagnosticError(f"{source}: expected non-empty relative path")
    raw = Path(value)
    if raw.is_absolute():
        raise DiagnosticError(f"{source}: absolute paths are not allowed")
    resolved = (repo_root / raw).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise DiagnosticError(f"{source}: path escapes repository root") from exc
    return resolved


def _verify_file(
    repo_root: Path,
    spec: Mapping[str, Any],
    source: str,
) -> tuple[Path, dict[str, str]]:
    path = _config_path(repo_root, spec.get("path"), f"{source}.path")
    expected = _require_sha256(spec.get("sha256"), f"{source}.sha256")
    actual = _sha256_file(path)
    if actual != expected:
        raise DiagnosticError(
            f"{source}: SHA-256 mismatch for {path}; expected {expected}, got {actual}"
        )
    return path, {"path": str(path.relative_to(repo_root)), "sha256": actual}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _index_unique(rows: Iterable[Mapping[str, Any]], source: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise DiagnosticError(f"{source}: row lacks task_id")
        if task_id in result:
            raise DiagnosticError(f"{source}: duplicate task_id {task_id}")
        result[task_id] = row
    return result


def _public_feature_guard(value: Any, path: str = "row") -> None:
    """Reject evaluator labels hidden in the public candidate projection."""

    forbidden_exact = {
        "reference_answer",
        "reference_solution",
        "new_correct",
        "baseline_correct",
        "strict_correct",
        "judge_correct",
        "score_source",
        "transition",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).casefold()
            if key_text in forbidden_exact:
                raise DiagnosticError(f"forbidden evaluator field in public input: {path}.{key}")
            if key_text == "gold_access":
                if child is not False:
                    raise DiagnosticError(f"non-false gold_access in public input: {path}.{key}")
                continue
            _public_feature_guard(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _public_feature_guard(child, f"{path}[{index}]")


def normalize_answer(value: Any, answer_type: str) -> str:
    """Normalize a candidate answer without reference-answer access."""

    if value is None:
        return "<missing>"
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    if answer_type == "choice":
        match = re.search(r"(?<![a-z])([a-e])(?![a-z])", text)
        if match:
            return match.group(1).upper()
    if answer_type == "numeric":
        text = re.sub(r"(?<=\d),(?=\d)", ".", text).replace(" ", "")
    text = re.sub(r"^[\s\[\]{}()'\"`]+|[\s\[\]{}()'\"`.,;:!?]+$", "", text)
    return text or "<empty>"


def _format_valid(answer: str, answer_type: str) -> bool:
    if answer in {"<missing>", "<empty>"}:
        return False
    if answer_type == "choice":
        return bool(re.fullmatch(r"[A-E]", answer))
    if answer_type == "numeric":
        return bool(re.search(r"[-+]?\d+(?:\.\d+)?", answer))
    return True


def _canonical_source_family(raw_source: Any) -> str:
    """Make a source-document family key used exclusively by the splitter."""

    source = unicodedata.normalize("NFKC", str(raw_source or "")).strip()
    if not source:
        raise DiagnosticError("validation.meta row has an empty source")
    parts = urlsplit(source)
    host = parts.netloc.casefold().removeprefix("www.")
    path = re.sub(r"/+", "/", unquote(parts.path)).rstrip("/").casefold()
    query = parse_qs(parts.query, keep_blank_values=True)
    if host == "docs.yandex.ru":
        embedded = unquote((query.get("url") or [""])[0]).strip().casefold()
        return f"docs.yandex.ru|{embedded or path}"
    if host in {"youtube.com", "youtu.be"}:
        video = (query.get("v") or [parts.path.strip("/")])[0].strip().casefold()
        return f"youtube|{video}"
    # Paths identify a PDF, book, or provider collection for the remaining
    # benchmark sources.  Tracking/query parameters are intentionally ignored.
    return f"{host}|{path or '/'}"


def _balanced_group_folds(
    task_ids: Sequence[str],
    families: Mapping[str, str],
    folds: int,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for task_id in task_ids:
        grouped[families[task_id]].append(task_id)
    folds = min(folds, len(grouped))
    if folds < 2:
        raise DiagnosticError("at least two source families are required")
    loads = [0] * folds
    family_counts = [0] * folds
    assignment: dict[str, int] = {}
    for family, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        fold = min(range(folds), key=lambda index: (loads[index], family_counts[index], index))
        loads[fold] += len(members)
        family_counts[fold] += 1
        for task_id in members:
            assignment[task_id] = fold
    audit = [
        {"fold": index, "tasks": loads[index], "families": family_counts[index]}
        for index in range(folds)
    ]
    return assignment, audit


def _clip_count(value: Any, denominator: float = 8.0) -> float:
    if not isinstance(value, list):
        return 0.0
    return min(len(value) / denominator, 1.0)


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _candidate_observations(row: Mapping[str, Any]) -> dict[str, float]:
    generation = row.get("generation")
    generation = generation if isinstance(generation, dict) else {}
    evidence = generation.get("selection_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    calculator = generation.get("calculator_sympy")
    calculator = calculator if isinstance(calculator, dict) else {}
    confidence_raw = generation.get("confidence", evidence.get("confidence"))
    confidence_present = isinstance(confidence_raw, (int, float)) and not isinstance(confidence_raw, bool)
    confidence = min(max(_as_float(confidence_raw, 0.0), 0.0), 1.0)
    calls = calculator.get("call_traces")
    return {
        "forced": float(bool(row.get("forced_answer"))),
        "error": float(bool(row.get("error"))),
        "confidence_present": float(confidence_present),
        "confidence": confidence,
        "selection_certificate": float(bool(evidence)),
        "format_certificate": float(evidence.get("answer_format_verified") is True),
        "evidence_visible": float(evidence.get("all_required_evidence_visible") is True),
        "original_consistent": float(evidence.get("original_crop_consistent") is True),
        "baseline_supported": float(evidence.get("baseline_supported") is True),
        "verification_checks": _clip_count(evidence.get("verification_checks")),
        "visible_facts": _clip_count(evidence.get("visible_facts")),
        "active_crops": _clip_count(generation.get("active_crops"), denominator=3.0),
        "evidence_citations": _clip_count(generation.get("evidence_citations")),
        "calculator_certificate": float(bool(calculator)),
        "calculator_audit": float(calculator.get("audit_triggered") is True),
        "calculator_switch": float(calculator.get("switch_applied") is True),
        "calculator_calls": _clip_count(calls, denominator=4.0),
    }


def _feature_rows_for_task(
    task_id: str,
    *,
    task_meta: Mapping[str, Mapping[str, str]],
    public_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    branches: Sequence[str],
) -> dict[str, dict[str, float]]:
    subject = task_meta[task_id]["subject"]
    answer_type = task_meta[task_id]["answer_type"]
    answers = {
        branch: normalize_answer(public_rows[branch][task_id].get("final_answer"), answer_type)
        for branch in branches
    }
    support = Counter(answers.values())
    distinct_fraction = len(support) / len(branches)
    anchor_answer = answers["anchor"]
    result: dict[str, dict[str, float]] = {}
    for branch in branches:
        row = public_rows[branch][task_id]
        answer = answers[branch]
        observations = _candidate_observations(row)
        valid = _format_valid(answer, answer_type)
        agreement = support[answer] / len(branches)
        changed = float(answer != anchor_answer)
        features: dict[str, float] = {
            f"subject={subject}": 1.0,
            f"answer_type={answer_type}": 1.0,
            f"branch={branch}": 1.0,
            f"branch_subject={branch}|{subject}": 1.0,
            f"branch_answer_type={branch}|{answer_type}": 1.0,
            f"branch_anchor_agreement={branch}|{int(not changed)}": 1.0,
            "support_fraction": agreement,
            "distinct_fraction": distinct_fraction,
            "agrees_anchor": 1.0 - changed,
            "changes_anchor": changed,
            "is_anchor": float(branch == "anchor"),
            "format_valid": float(valid),
            "answer_length": min(len(answer) / 80.0, 1.0),
        }
        for key, value in observations.items():
            features[key] = value
            # Branch interactions let a certificate mean something different
            # for a calculator versus a crop verifier without source leakage.
            if value:
                features[f"branch_observation={branch}|{key}"] = value
        result[branch] = features
    return result


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-min(value, 40.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -40.0))
    return z / (1.0 + z)


def _fit_logistic(
    samples: Sequence[tuple[Mapping[str, float], int]],
    *,
    l2: float,
    epochs: int = EPOCHS,
) -> tuple[dict[str, float], float]:
    if not samples:
        raise DiagnosticError("cannot fit logistic regression without samples")
    positive_rate = (sum(label for _, label in samples) + 0.5) / (len(samples) + 1.0)
    bias = math.log(positive_rate / (1.0 - positive_rate))
    weights: dict[str, float] = {}
    inverse_n = 1.0 / len(samples)
    for epoch in range(epochs):
        gradient: dict[str, float] = defaultdict(float)
        bias_gradient = 0.0
        for features, label in samples:
            score = bias + sum(weights.get(key, 0.0) * value for key, value in features.items())
            error = _sigmoid(score) - label
            bias_gradient += error
            for key, value in features.items():
                gradient[key] += error * value
        learning_rate = 0.8 / math.sqrt(1.0 + epoch / 30.0)
        bias -= learning_rate * bias_gradient * inverse_n
        all_keys = set(weights) | set(gradient)
        max_gradient = 0.0
        for key in all_keys:
            old = weights.get(key, 0.0)
            value = gradient.get(key, 0.0) * inverse_n + l2 * old
            max_gradient = max(max_gradient, abs(value))
            updated = old - learning_rate * value
            if abs(updated) < 1e-12:
                weights.pop(key, None)
            else:
                weights[key] = updated
        if epoch >= 40 and max_gradient < 2e-5 and abs(bias_gradient * inverse_n) < 2e-5:
            break
    return weights, bias


def _predict_logistic(model: tuple[Mapping[str, float], float], features: Mapping[str, float]) -> float:
    weights, bias = model
    return _sigmoid(bias + sum(weights.get(key, 0.0) * value for key, value in features.items()))


def _choose_candidate(
    probabilities: Mapping[str, float],
    *,
    task_features: Mapping[str, Mapping[str, float]],
    public_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    task_id: str,
    task_meta: Mapping[str, Mapping[str, str]],
    branches: Sequence[str],
    margin: float,
) -> str:
    answer_type = task_meta[task_id]["answer_type"]
    anchor_answer = normalize_answer(public_rows["anchor"][task_id].get("final_answer"), answer_type)
    anchor_invalid = not bool(task_features["anchor"]["format_valid"])
    alternatives: list[str] = []
    for branch in branches:
        if branch == "anchor":
            continue
        row_features = task_features[branch]
        if not row_features["format_valid"] or row_features["error"] or row_features["forced"]:
            continue
        answer = normalize_answer(public_rows[branch][task_id].get("final_answer"), answer_type)
        if answer != anchor_answer or anchor_invalid:
            alternatives.append(branch)
    if not alternatives:
        return "anchor"
    best = max(alternatives, key=lambda branch: (probabilities[branch], -branches.index(branch)))
    if anchor_invalid or probabilities[best] >= probabilities["anchor"] + margin:
        return best
    return "anchor"


def _make_samples(
    task_ids: Sequence[str],
    *,
    features: Mapping[str, Mapping[str, Mapping[str, float]]],
    outcomes: Mapping[str, Mapping[str, bool]],
    branches: Sequence[str],
) -> list[tuple[Mapping[str, float], int]]:
    return [
        (features[task_id][branch], int(outcomes[branch][task_id]))
        for task_id in task_ids
        for branch in branches
    ]


def _metrics(
    task_ids: Sequence[str],
    selected: Mapping[str, str],
    *,
    outcomes: Mapping[str, Mapping[str, bool]],
    families: Mapping[str, str],
    public_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    task_meta: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    correct = sum(outcomes[selected[task_id]][task_id] for task_id in task_ids)
    anchor_correct = sum(outcomes["anchor"][task_id] for task_id in task_ids)
    fixes = sum(
        outcomes[selected[task_id]][task_id] and not outcomes["anchor"][task_id]
        for task_id in task_ids
    )
    regressions = sum(
        outcomes["anchor"][task_id] and not outcomes[selected[task_id]][task_id]
        for task_id in task_ids
    )
    branch_overrides = sum(selected[task_id] != "anchor" for task_id in task_ids)
    answer_changes = 0
    for task_id in task_ids:
        answer_type = task_meta[task_id]["answer_type"]
        chosen = normalize_answer(public_rows[selected[task_id]][task_id].get("final_answer"), answer_type)
        anchor = normalize_answer(public_rows["anchor"][task_id].get("final_answer"), answer_type)
        answer_changes += chosen != anchor
    family_rows: dict[str, list[str]] = defaultdict(list)
    for task_id in task_ids:
        family_rows[families[task_id]].append(task_id)
    selected_family_accuracies = [
        sum(outcomes[selected[task_id]][task_id] for task_id in members) / len(members)
        for members in family_rows.values()
    ]
    anchor_family_accuracies = [
        sum(outcomes["anchor"][task_id] for task_id in members) / len(members)
        for members in family_rows.values()
    ]
    n = len(task_ids)
    return {
        "correct": correct,
        "n": n,
        "accuracy": round(correct / n, 6) if n else None,
        "anchor_correct": anchor_correct,
        "anchor_accuracy": round(anchor_correct / n, 6) if n else None,
        "delta_correct": correct - anchor_correct,
        "delta_pp": round(100.0 * (correct - anchor_correct) / n, 3) if n else None,
        "family_count": len(family_rows),
        "family_macro_accuracy": round(sum(selected_family_accuracies) / len(selected_family_accuracies), 6),
        "anchor_family_macro_accuracy": round(sum(anchor_family_accuracies) / len(anchor_family_accuracies), 6),
        "family_macro_delta_pp": round(
            100.0
            * (
                sum(selected_family_accuracies) / len(selected_family_accuracies)
                - sum(anchor_family_accuracies) / len(anchor_family_accuracies)
            ),
            3,
        ),
        "fixes": fixes,
        "regressions": regressions,
        "branch_overrides": branch_overrides,
        "answer_changes": answer_changes,
        "selected_branch_counts": dict(sorted(Counter(selected.values()).items())),
    }


def _inner_select(
    train_ids: Sequence[str],
    *,
    families: Mapping[str, str],
    features: Mapping[str, Mapping[str, Mapping[str, float]]],
    outcomes: Mapping[str, Mapping[str, bool]],
    public_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    task_meta: Mapping[str, Mapping[str, str]],
    branches: Sequence[str],
) -> tuple[float, float, list[dict[str, Any]]]:
    assignment, _ = _balanced_group_folds(train_ids, families, INNER_FOLDS)
    probability_cache: dict[float, dict[str, dict[str, float]]] = {}
    for l2 in L2_GRID:
        probability_cache[l2] = {}
        for fold in sorted(set(assignment.values())):
            fit_ids = [task_id for task_id in train_ids if assignment[task_id] != fold]
            test_ids = [task_id for task_id in train_ids if assignment[task_id] == fold]
            model = _fit_logistic(
                _make_samples(fit_ids, features=features, outcomes=outcomes, branches=branches),
                l2=l2,
            )
            for task_id in test_ids:
                probability_cache[l2][task_id] = {
                    branch: _predict_logistic(model, features[task_id][branch])
                    for branch in branches
                }
    sweep: list[dict[str, Any]] = []
    best_key: tuple[Any, ...] | None = None
    best_pair = (L2_GRID[0], MARGIN_GRID[-1])
    for l2 in L2_GRID:
        for margin in MARGIN_GRID:
            selected = {
                task_id: _choose_candidate(
                    probability_cache[l2][task_id],
                    task_features=features[task_id],
                    public_rows=public_rows,
                    task_id=task_id,
                    task_meta=task_meta,
                    branches=branches,
                    margin=margin,
                )
                for task_id in train_ids
            }
            metric = _metrics(
                train_ids,
                selected,
                outcomes=outcomes,
                families=families,
                public_rows=public_rows,
                task_meta=task_meta,
            )
            row = {"l2": l2, "margin": margin, **metric}
            sweep.append(row)
            # All terms here are inner-OOF only.  Conservative tie breaking
            # prefers fewer regressions/overrides, then a wider margin.
            key = (
                metric["correct"],
                metric["family_macro_accuracy"],
                -metric["regressions"],
                -metric["branch_overrides"],
                margin,
                l2,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_pair = (l2, margin)
    return best_pair[0], best_pair[1], sweep


def _load_frozen_inputs(
    repo_root: Path,
    config_path: Path,
) -> tuple[
    dict[str, Path],
    dict[str, Path],
    Path,
    Path,
    int,
    dict[str, Any],
]:
    config = _require_mapping(_load_json(config_path), "frozen profile")
    expected_rows = config.get("expected_rows")
    if type(expected_rows) is not int or expected_rows <= 0:
        raise DiagnosticError("frozen profile expected_rows must be a positive integer")

    anchor_spec = _require_mapping(config.get("anchor"), "frozen profile.anchor")
    legacy_specs = _require_mapping(
        config.get("legacy_modules"), "frozen profile.legacy_modules"
    )
    expected_legacy = set(BRANCH_ORDER[1:])
    if set(legacy_specs) != expected_legacy:
        missing = sorted(expected_legacy - set(legacy_specs))
        extra = sorted(set(legacy_specs) - expected_legacy)
        raise DiagnosticError(
            f"legacy_modules must exactly match preregistered BRANCH_ORDER; "
            f"missing={missing}, extra={extra}"
        )

    evaluation = _require_mapping(config.get("evaluation"), "frozen profile.evaluation")
    score_specs = _require_mapping(
        evaluation.get("grouped_router_score_artifacts"),
        "frozen profile.evaluation.grouped_router_score_artifacts",
    )
    expected_branches = set(BRANCH_ORDER)
    if set(score_specs) != expected_branches:
        missing = sorted(expected_branches - set(score_specs))
        extra = sorted(set(score_specs) - expected_branches)
        raise DiagnosticError(
            f"grouped_router_score_artifacts must exactly match preregistered "
            f"BRANCH_ORDER; missing={missing}, extra={extra}"
        )

    public_task_spec = _require_mapping(
        config.get("public_tasks"), "frozen profile.public_tasks"
    )
    public_tasks_path, public_tasks_audit = _verify_file(
        repo_root, public_task_spec, "frozen profile.public_tasks"
    )
    metadata_spec = _require_mapping(
        evaluation.get("source_family_metadata"),
        "frozen profile.evaluation.source_family_metadata",
    )
    metadata_path, metadata_audit = _verify_file(
        repo_root,
        metadata_spec,
        "frozen profile.evaluation.source_family_metadata",
    )

    stage = repo_root / "tmp" / "maxim_evidence_os_v1_stage"
    public_paths: dict[str, Path] = {}
    score_paths: dict[str, Path] = {}
    branch_audit: dict[str, dict[str, Any]] = {}
    for branch in BRANCH_ORDER:
        raw_spec = anchor_spec if branch == "anchor" else _require_mapping(
            legacy_specs[branch], f"frozen profile.legacy_modules.{branch}"
        )
        raw_path, raw_audit = _verify_file(
            repo_root, raw_spec, f"frozen profile raw solver {branch}"
        )
        del raw_path  # The raw solver is provenance-only and never a model input.

        projection_path = (stage / f"{branch}.public.jsonl").resolve()
        projection_expected = _require_sha256(
            raw_spec.get("public_projection_sha256"),
            f"frozen profile public projection {branch}.sha256",
        )
        projection_actual = _sha256_file(projection_path)
        if projection_actual != projection_expected:
            raise DiagnosticError(
                f"public projection {branch}: SHA-256 mismatch for {projection_path}; "
                f"expected {projection_expected}, got {projection_actual}"
            )

        score_spec = _require_mapping(
            score_specs[branch],
            f"frozen profile.evaluation.grouped_router_score_artifacts.{branch}",
        )
        score_path, score_audit = _verify_file(
            repo_root, score_spec, f"frozen profile score artifact {branch}"
        )
        score = _require_mapping(_load_json(score_path), f"{branch} score artifact")
        provenance = _require_mapping(score.get("provenance"), f"{branch} score.provenance")
        solver_provenance = _require_mapping(
            provenance.get("solver_results"), f"{branch} score.provenance.solver_results"
        )
        provenance_sha = _require_sha256(
            solver_provenance.get("sha256"),
            f"{branch} score.provenance.solver_results.sha256",
        )
        raw_sha = raw_audit["sha256"]
        if provenance_sha != raw_sha:
            raise DiagnosticError(
                f"{branch}: score solver provenance SHA {provenance_sha} does not "
                f"match frozen raw solver SHA {raw_sha}"
            )

        public_paths[branch] = projection_path
        score_paths[branch] = score_path
        branch_audit[branch] = {
            "public_projection": {
                "path": str(projection_path.relative_to(repo_root)),
                "sha256": projection_actual,
            },
            "score_artifact": score_audit,
            "raw_solver": raw_audit,
            "score_provenance_solver_sha256": provenance_sha,
        }

    input_audit: dict[str, Any] = {
        "config": {
            "path": str(config_path.relative_to(repo_root)),
            "sha256": _sha256_file(config_path),
        },
        "public_tasks": public_tasks_audit,
        "validation_metadata": metadata_audit,
        "branches": branch_audit,
    }
    return (
        public_paths,
        score_paths,
        public_tasks_path,
        metadata_path,
        expected_rows,
        input_audit,
    )


def run(
    repo_root: Path,
    output_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = (
        config_path.resolve()
        if config_path is not None
        else (repo_root / "configs/maxim_evidence_os_v1.json").resolve()
    )
    try:
        config_path.relative_to(repo_root)
    except ValueError as exc:
        raise DiagnosticError("frozen profile config must be inside repository root") from exc
    (
        public_paths,
        score_paths,
        public_tasks_path,
        meta_path,
        expected_rows,
        input_audit,
    ) = _load_frozen_inputs(repo_root, config_path)
    public_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    score_outcomes: dict[str, dict[str, Mapping[str, Any]]] = {}
    anchor_ids: set[str] | None = None
    for branch in BRANCH_ORDER:
        public_path = public_paths[branch]
        score_path = score_paths[branch]
        public_index = _index_unique(_load_jsonl(public_path), f"{branch} public")
        for task_id, row in public_index.items():
            _public_feature_guard(row, f"{branch}[{task_id}]")
        score = _load_json(score_path)
        if not isinstance(score, dict) or not isinstance(score.get("task_outcomes"), list):
            raise DiagnosticError(f"{branch}: score artifact lacks task_outcomes")
        outcome_index = _index_unique(score["task_outcomes"], f"{branch} outcomes")
        if (
            len(public_index) != expected_rows
            or len(outcome_index) != expected_rows
            or set(public_index) != set(outcome_index)
        ):
            raise DiagnosticError(
                f"{branch}: expected a complete matched {expected_rows}-row public/score artifact"
            )
        for task_id, row in outcome_index.items():
            _require_bool(row.get("new_correct"), f"{branch} outcomes[{task_id}].new_correct")
        if anchor_ids is None:
            if branch != "anchor":
                raise DiagnosticError("anchor must be loaded first")
            anchor_ids = set(public_index)
        elif set(public_index) != anchor_ids:
            raise DiagnosticError(f"{branch}: task-id set differs from anchor")
        public_rows[branch] = public_index
        score_outcomes[branch] = outcome_index
    branches = list(BRANCH_ORDER)

    # Preserve the anchor artifact's order only for readable output.  It never
    # enters a feature or a fold assignment.
    task_ids = [str(row["task_id"]) for row in _load_jsonl(public_paths["anchor"])]
    public_task_rows = _index_unique(_load_jsonl(public_tasks_path), "public tasks")
    if set(public_task_rows) != set(task_ids):
        raise DiagnosticError("gold-free public task set differs from anchor")
    for task_id, row in public_task_rows.items():
        _public_feature_guard(row, f"public_tasks[{task_id}]")
    task_meta = {
        task_id: {
            "subject": str(public_task_rows[task_id].get("subject", "<unknown>")),
            "answer_type": str(public_task_rows[task_id].get("answer_type", "<unknown>")),
        }
        for task_id in task_ids
    }
    labels = {
        branch: {
            task_id: _require_bool(
                score_outcomes[branch][task_id].get("new_correct"),
                f"{branch} outcomes[{task_id}].new_correct",
            )
            for task_id in task_ids
        }
        for branch in branches
    }

    meta_index = _index_unique(_load_jsonl(meta_path), "validation.meta")
    missing_meta = [task_id for task_id in task_ids if task_id not in meta_index]
    if missing_meta:
        raise DiagnosticError(f"validation.meta misses {len(missing_meta)} benchmark task ids")
    family_keys = {task_id: _canonical_source_family(meta_index[task_id].get("source")) for task_id in task_ids}
    # Opaque ids are emitted; raw source locators never reach model features or outputs.
    family_id_by_key = {
        family: f"family_{index:03d}"
        for index, family in enumerate(sorted(set(family_keys.values())), 1)
    }
    families = {task_id: family_id_by_key[family_keys[task_id]] for task_id in task_ids}
    outer_assignment, outer_balance = _balanced_group_folds(task_ids, families, OUTER_FOLDS)

    feature_rows = {
        task_id: _feature_rows_for_task(
            task_id,
            task_meta=task_meta,
            public_rows=public_rows,
            branches=branches,
        )
        for task_id in task_ids
    }
    oof_selected: dict[str, str] = {}
    oof_probabilities: dict[str, dict[str, float]] = {}
    fold_reports: list[dict[str, Any]] = []
    fold_hyperparameters: dict[int, tuple[float, float]] = {}
    for fold in sorted(set(outer_assignment.values())):
        train_ids = [task_id for task_id in task_ids if outer_assignment[task_id] != fold]
        test_ids = [task_id for task_id in task_ids if outer_assignment[task_id] == fold]
        train_families = {families[task_id] for task_id in train_ids}
        test_families = {families[task_id] for task_id in test_ids}
        if train_families & test_families:
            raise DiagnosticError(f"outer fold {fold} has source-family leakage")
        l2, margin, inner_sweep = _inner_select(
            train_ids,
            families=families,
            features=feature_rows,
            outcomes=labels,
            public_rows=public_rows,
            task_meta=task_meta,
            branches=branches,
        )
        fold_hyperparameters[fold] = (l2, margin)
        model = _fit_logistic(
            _make_samples(train_ids, features=feature_rows, outcomes=labels, branches=branches),
            l2=l2,
        )
        fold_selected: dict[str, str] = {}
        for task_id in test_ids:
            probabilities = {
                branch: _predict_logistic(model, feature_rows[task_id][branch])
                for branch in branches
            }
            selected = _choose_candidate(
                probabilities,
                task_features=feature_rows[task_id],
                public_rows=public_rows,
                task_id=task_id,
                task_meta=task_meta,
                branches=branches,
                margin=margin,
            )
            oof_probabilities[task_id] = probabilities
            oof_selected[task_id] = selected
            fold_selected[task_id] = selected
        fold_metric = _metrics(
            test_ids,
            fold_selected,
            outcomes=labels,
            families=families,
            public_rows=public_rows,
            task_meta=task_meta,
        )
        best_inner = max(
            (row for row in inner_sweep if row["l2"] == l2 and row["margin"] == margin),
            key=lambda row: row["correct"],
        )
        fold_reports.append(
            {
                "outer_fold": fold,
                "train_tasks": len(train_ids),
                "train_families": len(train_families),
                "test_tasks": len(test_ids),
                "test_families": len(test_families),
                "selected_l2": l2,
                "selected_margin": margin,
                "inner_oof_selected_metric": best_inner,
                "outer_metric": fold_metric,
            }
        )

    overall = _metrics(
        task_ids,
        oof_selected,
        outcomes=labels,
        families=families,
        public_rows=public_rows,
        task_meta=task_meta,
    )
    family_sizes = Counter(families.values())
    individual = {
        branch: {
            "correct": sum(labels[branch].values()),
            "n": len(task_ids),
            "accuracy": round(sum(labels[branch].values()) / len(task_ids), 6),
        }
        for branch in branches
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "label": LABEL,
        "claim_boundary": (
            "Exploratory nested source-family cross-fit diagnostic on the same frozen development "
            "benchmark. It is neither an untouched holdout nor a strict benchmark submission, and "
            "it does not compose a production solver."
        ),
        "protocol": {
            "outer_folds": len(set(outer_assignment.values())),
            "inner_folds": INNER_FOLDS,
            "model": "pure-Python L2-regularized binary logistic regression over candidate rows",
            "epochs_fixed_before_outer_evaluation": EPOCHS,
            "l2_grid_inner_only": list(L2_GRID),
            "override_margin_grid_inner_only": list(MARGIN_GRID),
            "selection": "best valid non-anchor candidate with a distinct normalized answer; fail closed to anchor",
            "source_family": "canonical source-document locator from validation.meta.jsonl; split-only; opaque in output",
            "task_metadata": "subject and answer_type from the gold-free public task queue",
            "task_id_use": "alignment and audit output only",
            "forbidden_features": ["task_id", "row order", "SHA", "source URL/family", "gold/reference", "judge/score outcome"],
            "allowed_features": [
                "subject",
                "answer_type",
                "branch kind",
                "normalized answer agreement/support",
                "forced/error/format observations",
                "model self-confidence and certificate observations",
            ],
        },
        "branches": {
            "included": branches,
            "preregistered_complete": branches == list(BRANCH_ORDER),
            "individual_frozen_scores": individual,
        },
        "inputs": input_audit,
        "source_family_audit": {
            "families": len(family_sizes),
            "largest_family_tasks": max(family_sizes.values()),
            "median_family_tasks": sorted(family_sizes.values())[(len(family_sizes) - 1) // 2],
            "outer_balance": outer_balance,
            "raw_source_locators_emitted": False,
        },
        "oof": overall,
        "folds": fold_reports,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    with (output_dir / "oof_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for task_id in task_ids:
            selected = oof_selected[task_id]
            answer_type = task_meta[task_id]["answer_type"]
            selected_answer = normalize_answer(public_rows[selected][task_id].get("final_answer"), answer_type)
            anchor_answer = normalize_answer(public_rows["anchor"][task_id].get("final_answer"), answer_type)
            l2, margin = fold_hyperparameters[outer_assignment[task_id]]
            row = {
                "task_id": task_id,
                "source_family_id": families[task_id],
                "outer_fold": outer_assignment[task_id],
                "subject": task_meta[task_id]["subject"],
                "answer_type": answer_type,
                "selected_branch": selected,
                "selected_correct": labels[selected][task_id],
                "anchor_correct": labels["anchor"][task_id],
                "fix": labels[selected][task_id] and not labels["anchor"][task_id],
                "regression": labels["anchor"][task_id] and not labels[selected][task_id],
                "branch_override": selected != "anchor",
                "answer_changed": selected_answer != anchor_answer,
                "selected_probability": round(oof_probabilities[task_id][selected], 6),
                "anchor_probability": round(oof_probabilities[task_id]["anchor"], 6),
                "inner_selected_l2": l2,
                "inner_selected_margin": margin,
                "label": LABEL,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    lines = [
        "# Maksim Evidence OS grouped router v1",
        "",
        f"**{LABEL}**",
        "",
        summary["claim_boundary"],
        "",
        "## OOF result",
        "",
        f"- Router: {overall['correct']}/{overall['n']} = {overall['accuracy']:.6f}",
        f"- Frozen anchor: {overall['anchor_correct']}/{overall['n']} = {overall['anchor_accuracy']:.6f}",
        f"- Delta: {overall['delta_correct']:+d} answers ({overall['delta_pp']:+.3f} pp)",
        f"- Family macro: {overall['family_macro_accuracy']:.6f} vs anchor {overall['anchor_family_macro_accuracy']:.6f} ({overall['family_macro_delta_pp']:+.3f} pp)",
        f"- Fixes / regressions: {overall['fixes']} / {overall['regressions']}",
        f"- Branch overrides / answer changes: {overall['branch_overrides']} / {overall['answer_changes']}",
        "",
        "## Split audit",
        "",
        f"- Canonical source-document families: {len(family_sizes)}; largest family: {max(family_sizes.values())} tasks.",
        "- Every outer fold holds out complete families. L2 and override margin are chosen only by inner grouped OOF.",
        "- Source locators/family ids, task ids, row order, hashes, references, and judge outcomes are not model features.",
        "",
        "## Outer folds",
        "",
        "| Fold | Test tasks | Test families | L2 | Margin | Router | Anchor | Delta | Fix/regress | Overrides |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_reports:
        metric = row["outer_metric"]
        lines.append(
            f"| {row['outer_fold']} | {row['test_tasks']} | {row['test_families']} | "
            f"{row['selected_l2']:.3f} | {row['selected_margin']:.2f} | "
            f"{metric['accuracy']:.6f} | {metric['anchor_accuracy']:.6f} | "
            f"{metric['delta_correct']:+d} | {metric['fixes']}/{metric['regressions']} | "
            f"{metric['branch_overrides']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is an honest OOF diagnostic of transfer across the source families represented in the dev set. "
            "It can reject an unhelpful router, but it cannot establish a production or untouched-book score. "
            "A final policy still needs preregistration and evaluation on newly generated, book-disjoint data.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_root / "configs/maxim_evidence_os_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "tmp/maxim_evidence_os_grouped_router_v1",
    )
    args = parser.parse_args()
    summary = run(
        args.repo_root.resolve(),
        args.output_dir.resolve(),
        args.config.resolve(),
    )
    print(json.dumps({"label": summary["label"], "oof": summary["oof"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
