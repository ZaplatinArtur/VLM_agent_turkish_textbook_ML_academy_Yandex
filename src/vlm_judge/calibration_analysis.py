from __future__ import annotations

import itertools
import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from .schema import JudgeVerdict


SCORE_VALUES = tuple(range(5))


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _annotation_key(record: dict[str, Any]) -> tuple[str, str | None]:
    task_id = str(record.get("task_id") or "").strip()
    setup_value = record.get("setup")
    setup = str(setup_value).strip() if setup_value not in (None, "") else None
    return task_id, setup


def _valid_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if score in SCORE_VALUES else None


def _human_score(record: dict[str, Any]) -> int | None:
    if record.get("status") not in (None, "", "complete"):
        return None
    return _valid_score(record.get("score"))


def _judge_score(record: dict[str, Any]) -> tuple[int | None, str]:
    verdict = record.get("verdict")
    if isinstance(verdict, dict):
        try:
            parsed = JudgeVerdict.from_dict(verdict)
        except (KeyError, TypeError, ValueError):
            return None, "invalid"
        if parsed.label == "unjudgeable":
            return None, "unjudgeable"
        score = _valid_score(parsed.score)
        return score, "scored" if score is not None else "invalid"
    score = _valid_score(record.get("score"))
    if score is not None:
        return score, "scored"
    return None, "missing"


def _cohen_kappa(
    left: list[int],
    right: list[int],
    *,
    quadratic: bool,
) -> float | None:
    if not left or len(left) != len(right):
        return None
    n = len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    max_distance = (len(SCORE_VALUES) - 1) ** 2

    observed = 0.0
    expected = 0.0
    for first in SCORE_VALUES:
        for second in SCORE_VALUES:
            if quadratic:
                weight = ((first - second) ** 2) / max_distance
            else:
                weight = 0.0 if first == second else 1.0
            observed_count = sum(
                1 for value_a, value_b in zip(left, right) if value_a == first and value_b == second
            )
            observed += weight * observed_count / n
            expected += weight * (left_counts[first] / n) * (right_counts[second] / n)
    if math.isclose(expected, 0.0):
        # Kappa is undefined when both raters use only one category, even if
        # their raw agreement is perfect. Returning 1 would overstate evidence.
        return None
    return 1.0 - observed / expected


def _comparison_summary(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    if not pairs:
        return {
            "comparisons": 0,
            "exact_score_agreement": None,
            "within_one_score": None,
            "mean_absolute_error": None,
            "strict_binary_agreement": None,
            "exact_score_agreement_ci95": None,
            "strict_binary_agreement_ci95": None,
            "strict_precision": None,
            "strict_recall": None,
            "strict_f1": None,
            "cohen_kappa": None,
            "quadratic_weighted_kappa": None,
            "macro_f1_5_score": None,
            "per_human_score": {},
            "confusion_matrix_human_rows_judge_columns": [[0] * 5 for _ in range(5)],
        }

    human = [pair[0] for pair in pairs]
    judge = [pair[1] for pair in pairs]
    differences = [abs(first - second) for first, second in pairs]
    matrix = [[0] * 5 for _ in range(5)]
    for first, second in pairs:
        matrix[first][second] += 1

    true_positive = sum(first == 4 and second == 4 for first, second in pairs)
    false_positive = sum(first != 4 and second == 4 for first, second in pairs)
    false_negative = sum(first == 4 and second != 4 for first, second in pairs)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    per_score: dict[str, dict[str, Any]] = {}
    score_f1_values: list[float] = []
    for score in SCORE_VALUES:
        score_true_positive = sum(first == score and second == score for first, second in pairs)
        score_false_positive = sum(first != score and second == score for first, second in pairs)
        score_false_negative = sum(first == score and second != score for first, second in pairs)
        score_precision = (
            score_true_positive / (score_true_positive + score_false_positive)
            if score_true_positive + score_false_positive
            else None
        )
        score_recall = (
            score_true_positive / (score_true_positive + score_false_negative)
            if score_true_positive + score_false_negative
            else None
        )
        score_f1 = (
            2 * score_precision * score_recall / (score_precision + score_recall)
            if score_precision is not None and score_recall is not None and score_precision + score_recall
            else None
        )
        if score_f1 is not None:
            score_f1_values.append(score_f1)
        per_score[str(score)] = {
            "human_support": sum(first == score for first, _ in pairs),
            "judge_predictions": sum(second == score for _, second in pairs),
            "precision": score_precision,
            "recall": score_recall,
            "f1": score_f1,
        }
    exact_successes = sum(first == second for first, second in pairs)
    strict_successes = sum((first == 4) == (second == 4) for first, second in pairs)
    return {
        "comparisons": len(pairs),
        "exact_score_agreement": exact_successes / len(pairs),
        "exact_score_agreement_ci95": _wilson_interval(exact_successes, len(pairs)),
        "within_one_score": statistics.mean(distance <= 1 for distance in differences),
        "mean_absolute_error": statistics.mean(differences),
        "strict_binary_agreement": strict_successes / len(pairs),
        "strict_binary_agreement_ci95": _wilson_interval(strict_successes, len(pairs)),
        "strict_precision": precision,
        "strict_recall": recall,
        "strict_f1": f1,
        "cohen_kappa": _cohen_kappa(human, judge, quadratic=False),
        "quadratic_weighted_kappa": _cohen_kappa(human, judge, quadratic=True),
        "macro_f1_5_score": statistics.mean(score_f1_values) if score_f1_values else None,
        "per_human_score": per_score,
        "confusion_matrix_human_rows_judge_columns": matrix,
    }


def _inter_annotator_summary(
    grouped_scores: dict[tuple[str, str | None], list[int]],
) -> dict[str, Any]:
    pairs = [
        pair
        for scores in grouped_scores.values()
        for pair in itertools.combinations(scores, 2)
    ]
    return {
        "tasks_with_multiple_annotations": sum(len(scores) > 1 for scores in grouped_scores.values()),
        **_comparison_summary(pairs),
    }


def analyze_calibration(
    human_records: Iterable[dict[str, Any]],
    judge_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    humans = [record for record in human_records if _human_score(record) is not None]
    judges = list(judge_records)
    judges_by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
    judges_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_judge_keys: set[str] = set()
    for record in judges:
        key = _annotation_key(record)
        if not key[0]:
            continue
        if key in judges_by_key:
            duplicate_judge_keys.add("::".join(value or "" for value in key))
        judges_by_key[key] = record
        judges_by_task[key[0]].append(record)

    pairs: list[tuple[int, int]] = []
    grouped_human_scores: dict[tuple[str, str | None], list[int]] = defaultdict(list)
    by_subject: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_answer_type: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_setup: dict[str, list[tuple[int, int]]] = defaultdict(list)
    missing_judge: set[str] = set()
    ambiguous_setup: set[str] = set()
    judge_statuses = Counter()
    confidence_bins: dict[str, list[bool]] = defaultdict(list)
    confidence_pairs: list[tuple[float, bool, bool]] = []

    for human_record in humans:
        human_score = _human_score(human_record)
        assert human_score is not None
        key = _annotation_key(human_record)
        grouped_human_scores[key].append(human_score)
        judge_record = judges_by_key.get(key)
        if judge_record is None and key[1] is None:
            candidates = judges_by_task.get(key[0], [])
            if len(candidates) == 1:
                judge_record = candidates[0]
            elif len(candidates) > 1:
                ambiguous_setup.add(key[0])
        if judge_record is None:
            missing_judge.add("::".join(value or "" for value in key))
            continue

        judge_score, status = _judge_score(judge_record)
        judge_statuses[status] += 1
        if judge_score is None:
            continue
        pair = (human_score, judge_score)
        pairs.append(pair)
        subject = str(human_record.get("subject") or judge_record.get("subject") or "unknown")
        answer_type = str(human_record.get("answer_type") or judge_record.get("answer_type") or "unknown")
        by_subject[subject].append(pair)
        by_answer_type[answer_type].append(pair)
        by_setup[str(judge_record.get("setup") or human_record.get("setup") or "unknown")].append(pair)

        verdict = judge_record.get("verdict")
        if isinstance(verdict, dict) and isinstance(verdict.get("confidence"), (int, float)):
            confidence = max(0.0, min(1.0, float(verdict["confidence"])))
            lower = min(0.8, math.floor(confidence * 5) / 5)
            label = f"{lower:.1f}-{lower + 0.2:.1f}"
            confidence_bins[label].append(human_score == judge_score)
            confidence_pairs.append(
                (confidence, human_score == judge_score, (human_score == 4) == (judge_score == 4))
            )

    unique_human_keys = set(grouped_human_scores)
    report = {
        "human_annotations": len(humans),
        "human_tasks": len(unique_human_keys),
        "judge_records": len(judges),
        "matched_comparisons": len(pairs),
        "coverage_over_human_annotations": len(pairs) / len(humans) if humans else None,
        "judge_record_statuses": dict(judge_statuses),
        "missing_judge_keys": sorted(missing_judge),
        "ambiguous_task_ids_without_setup": sorted(ambiguous_setup),
        "duplicate_judge_keys": sorted(duplicate_judge_keys),
        "overall": _comparison_summary(pairs),
        "human_inter_annotator": _inter_annotator_summary(grouped_human_scores),
        "by_subject": {key: _comparison_summary(value) for key, value in sorted(by_subject.items())},
        "by_answer_type": {key: _comparison_summary(value) for key, value in sorted(by_answer_type.items())},
        "by_setup": {key: _comparison_summary(value) for key, value in sorted(by_setup.items())},
        "agreement_by_judge_confidence": {
            key: {"comparisons": len(values), "exact_score_agreement": statistics.mean(values)}
            for key, values in sorted(confidence_bins.items())
        },
        "selective_agreement_by_minimum_confidence": {
            f"{threshold:.2f}": {
                "comparisons": len(selected),
                "coverage_over_confidence_scored": len(selected) / len(confidence_pairs) if confidence_pairs else None,
                "exact_score_agreement": statistics.mean(value[1] for value in selected) if selected else None,
                "strict_binary_agreement": statistics.mean(value[2] for value in selected) if selected else None,
            }
            for threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
            if (selected := [value for value in confidence_pairs if value[0] >= threshold]) or confidence_pairs
        },
    }
    return report


def _normalized_arena_winner(record: dict[str, Any]) -> str | None:
    winner = record.get("winner")
    if winner == "A":
        setup = record.get("candidate_a_setup")
        return f"setup:{setup}" if setup else None
    if winner == "B":
        setup = record.get("candidate_b_setup")
        return f"setup:{setup}" if setup else None
    if winner in {"tie", "unjudgeable"}:
        return str(winner)
    return None


def analyze_arena_annotations(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    complete = [
        record
        for record in records
        if record.get("status") in (None, "", "complete")
        and record.get("winner") in {"A", "B", "tie", "unjudgeable"}
    ]
    side_counts = Counter(str(record["winner"]) for record in complete)
    normalized_counts = Counter(
        winner for record in complete if (winner := _normalized_arena_winner(record)) is not None
    )
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    missing_setup_mapping = 0
    for record in complete:
        setups = tuple(sorted(filter(None, [record.get("candidate_a_setup"), record.get("candidate_b_setup")])))
        if len(setups) != 2:
            missing_setup_mapping += 1
            continue
        groups[(str(record.get("task_id") or ""), setups)].append(record)

    mirrored_groups = 0
    consistent_groups = 0
    same_side_groups = 0
    opposite_orientations = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        first, second = group[0], group[1]
        mirrored_groups += 1
        normalized_first = _normalized_arena_winner(first)
        normalized_second = _normalized_arena_winner(second)
        if normalized_first is not None and normalized_first == normalized_second:
            consistent_groups += 1
        if first.get("winner") == second.get("winner") and first.get("winner") in {"A", "B"}:
            same_side_groups += 1
        if bool(first.get("side_swapped")) != bool(second.get("side_swapped")):
            opposite_orientations += 1

    decisive = side_counts["A"] + side_counts["B"]
    return {
        "complete_votes": len(complete),
        "side_winner_counts": dict(side_counts),
        "underlying_winner_counts": dict(normalized_counts),
        "side_a_rate_among_decisive": side_counts["A"] / decisive if decisive else None,
        "missing_candidate_setup_mapping": missing_setup_mapping,
        "mirrored_groups": mirrored_groups,
        "groups_with_opposite_orientations": opposite_orientations,
        "underlying_decision_consistency": consistent_groups / mirrored_groups if mirrored_groups else None,
        "same_side_selection_rate": same_side_groups / mirrored_groups if mirrored_groups else None,
    }
