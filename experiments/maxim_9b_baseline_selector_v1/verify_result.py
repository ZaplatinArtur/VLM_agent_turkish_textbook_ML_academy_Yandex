from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = HERE / "RESULT.json"
SCHEMA = "maxim-9b-baseline-selector-final-result-v1"


class ResultError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ResultError(f"JSON object required: {path}")
    return value


def descriptors(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for item in value.values():
            yield from descriptors(item)
    elif isinstance(value, list):
        for item in value:
            yield from descriptors(item)


def pinned_path(descriptor: dict[str, Any]) -> Path:
    relative = Path(descriptor["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ResultError(f"unsafe artifact path: {relative}")
    path = REPO_ROOT / relative
    if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
        raise ResultError(f"artifact hash mismatch: {relative}")
    return path


def task_map(score: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = score.get("task_outcomes")
    if not isinstance(rows, list) or len(rows) != 274:
        raise ResultError("score does not contain 274 task outcomes")
    mapped = {row.get("task_id"): row for row in rows}
    if len(mapped) != 274 or None in mapped:
        raise ResultError("score task IDs are not unique full274")
    return mapped


def verify(result_path: Path = RESULT_PATH) -> dict[str, Any]:
    result = read_json(result_path)
    if result.get("schema_version") != SCHEMA:
        raise ResultError("result schema mismatch")
    if result.get("status") != "audited_four_arm_wave_complete_posthoc_report":
        raise ResultError("result status mismatch")
    if result.get("model_closure") != ["Qwen/Qwen3.5-9B"]:
        raise ResultError("model closure mismatch")
    if result.get("benchmark") != {
        "rows": 274,
        "sha256": "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9",
    }:
        raise ResultError("benchmark pin mismatch")
    if result.get("protocol") != {"deterministic_rows": 177, "image_judge_rows": 97}:
        raise ResultError("protocol split mismatch")

    for descriptor in descriptors(result):
        pinned_path(descriptor)

    expected_arms = {
        "v1_1_primary": (239, 0.872263),
        "v1_1_secondary": (237, 0.864964),
        "v1_2_exploratory": (239, 0.872263),
        "v1_2_primary": (240, 0.875912),
    }
    arms = result.get("wave", {}).get("arms", {})
    if set(arms) != set(expected_arms):
        raise ResultError("four-arm closure mismatch")
    loaded_scores: dict[str, dict[str, Any]] = {}
    for arm, (correct, accuracy) in expected_arms.items():
        entry = arms[arm]
        if entry.get("correct") != correct or entry.get("n") != 274:
            raise ResultError(f"reported total mismatch: {arm}")
        if not math.isclose(entry.get("accuracy", -1), accuracy, abs_tol=5e-7):
            raise ResultError(f"reported accuracy mismatch: {arm}")
        score = read_json(pinned_path(entry["score"]))
        loaded_scores[arm] = score
        overall = score.get("overall", {})
        if overall.get("new_correct") != correct or overall.get("n") != 274:
            raise ResultError(f"score total mismatch: {arm}")
        if not math.isclose(overall.get("new_accuracy", -1), accuracy, abs_tol=5e-7):
            raise ResultError(f"score accuracy mismatch: {arm}")

    best = result.get("best", {})
    if best.get("arm") != "v1_2_primary" or best.get("correct") != 240:
        raise ResultError("best arm mismatch")
    best_score = loaded_scores["v1_2_primary"]
    if best.get("slices") != {
        "math": {"correct": 109, "n": 139, "accuracy": 0.784173},
        "deterministic": {"correct": 158, "n": 177, "accuracy": 0.892655},
        "image_judge": {"correct": 82, "n": 97, "accuracy": 0.845361},
    }:
        raise ResultError("best slice report mismatch")
    if best_score["by_subject"]["Math"]["new_correct"] != 109:
        raise ResultError("Math score mismatch")
    if best_score["by_source"]["deterministic"]["new_correct"] != 158:
        raise ResultError("deterministic score mismatch")
    if best_score["by_source"]["image_judge"]["new_correct"] != 82:
        raise ResultError("image score mismatch")

    artifacts = result["artifacts"]
    source_score = read_json(pinned_path(artifacts["final_source_score"]))
    if source_score["overall"]["new_correct"] != 238:
        raise ResultError("final source baseline is not 238")
    source_tasks = task_map(source_score)
    best_tasks = task_map(best_score)
    fixed = sorted(
        task_id
        for task_id in source_tasks
        if not source_tasks[task_id]["new_correct"] and best_tasks[task_id]["new_correct"]
    )
    regressed = sorted(
        task_id
        for task_id in source_tasks
        if source_tasks[task_id]["new_correct"] and not best_tasks[task_id]["new_correct"]
    )
    if fixed != ["val_0089", "val_0251"] or regressed:
        raise ResultError("best-vs-source transition set mismatch")
    comparison = best.get("comparison_vs_final_source", {})
    if comparison.get("fixed_task_ids") != fixed or comparison.get("regressed_task_ids") != []:
        raise ResultError("reported best-vs-source transitions mismatch")
    if comparison.get("fixed") != 2 or comparison.get("regressed") != 0:
        raise ResultError("reported transition arithmetic mismatch")

    composition = read_json(pinned_path(artifacts["v1_2_composition_manifest"]))
    preservation = composition.get("preservation", {})
    if (
        preservation.get("source_union_rows") != 156
        or preservation.get("image_judge_rows") != 97
        or preservation.get("source_union_changes_primary") != 0
        or preservation.get("image_judge_changes_primary") != 0
    ):
        raise ResultError("source/image preservation mismatch")

    nulls = result.get("separate_null_controls", {})
    calibrated = read_json(pinned_path(artifacts["source_calibrated_candidate_manifest"]))
    if calibrated.get("full_uncovered_replacements_vs_active_crop") != 0:
        raise ResultError("source-calibrated control is not a null selection")
    if nulls.get("source_calibrated_selector", {}).get("uncovered_replacements") != 0:
        raise ResultError("reported source-calibrated null mismatch")

    canonical = read_json(pinned_path(artifacts["canonicalization_candidate_manifest"]))
    canonical_counts = {
        "choice_nfkc_only": canonical["arms"]["choice_nfkc_only"]["canonicalized_rows"],
        "choice_curated_confusable_exploratory": canonical["arms"]
        ["choice_curated_confusable_exploratory"]["canonicalized_rows"],
        "fraction_percent_explicit_question_contract_exploratory": canonical["arms"]
        ["fraction_percent_explicit_question_contract_exploratory"]
        ["canonicalized_rows"],
    }
    if any(canonical_counts.values()) or nulls.get("answer_canonicalization", {}).get(
        "changed_rows_by_arm"
    ) != canonical_counts:
        raise ResultError("canonicalization 0/0/0 mismatch")

    repair_score = read_json(pinned_path(artifacts["repair_score"]))
    if repair_score["overall"]["new_correct"] != 240:
        raise ResultError("repair score total changed")
    repair_tasks = task_map(repair_score)
    if any(
        repair_tasks[task_id]["new_correct"] != best_tasks[task_id]["new_correct"]
        for task_id in best_tasks
    ):
        raise ResultError("repair changed a correctness outcome")
    if not best_tasks["val_0223"]["new_correct"] or not repair_tasks["val_0223"]["new_correct"]:
        raise ResultError("val_0223 is not correct-to-correct")
    repair_null = nulls.get("answer_contract_repair_v1_1", {})
    if repair_null.get("changed_rows") != 1 or repair_null.get("net_correct_change") != 0:
        raise ResultError("repair null report mismatch")

    caveat = result.get("known_development_caveat", {})
    if caveat.get("independent_heldout_claim") is not False:
        raise ResultError("known-development caveat missing")
    return {
        "status": "PASS",
        "arms": {arm: expected_arms[arm][0] for arm in expected_arms},
        "best": 240,
        "fixed_vs_source": fixed,
        "regressed_vs_source": regressed,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, sort_keys=True, indent=2))
