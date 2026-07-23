"""Paired B0 vs textbook-RAG evaluation on one fixed task set.

The report deliberately keeps two metric views separate:

* deterministic accuracy on automatically comparable answers;
* binary LLM-judge accuracy on the full task denominator.

It also reports paired answer flips and strict corpus coverage so that an
overall RAG delta is not confused with retrieval quality on supported grades.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from retrieve.metadata import canonical_subject, infer_textbook_metadata

from .eval import match


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _by_task(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = str(record.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"{label}: record without task_id")
        if task_id in indexed:
            raise ValueError(f"{label}: duplicate task_id {task_id}")
        indexed[task_id] = record
    return indexed


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _is_auto_evaluable(task: dict[str, Any]) -> bool:
    reference = str(task.get("reference_answer") or "")
    return (
        not reference.startswith("http")
        and not reference.startswith("[REFERENCE_IMAGE")
        and task.get("answer_type") != "free_form"
    )


def _automatic_score(
    task: dict[str, Any], result: dict[str, Any] | None
) -> bool:
    if result is None or result.get("error") or not result.get("final_answer"):
        return False
    score = match(
        str(result["final_answer"]),
        str(task.get("reference_answer") or ""),
        str(task.get("answer_type") or "short_text"),
    )
    return bool(score)


def _judge_score(record: dict[str, Any] | None) -> bool | None:
    if record is None:
        return None
    verdict = record.get("verdict")
    if not isinstance(verdict, dict):
        return None
    strict_correct = verdict.get("strict_correct")
    if isinstance(strict_correct, bool):
        return strict_correct
    score = verdict.get("score")
    if isinstance(score, bool) or score not in (0, 1):
        return None
    return score == 1


def _paired_counts(
    task_ids: list[str],
    baseline_scores: dict[str, bool | None],
    rag_scores: dict[str, bool | None],
) -> dict[str, Any]:
    baseline_correct = sum(baseline_scores.get(task_id) is True for task_id in task_ids)
    rag_correct = sum(rag_scores.get(task_id) is True for task_id in task_ids)
    paired_ids = [
        task_id
        for task_id in task_ids
        if baseline_scores.get(task_id) is not None
        and rag_scores.get(task_id) is not None
    ]
    fixed_ids = [
        task_id
        for task_id in paired_ids
        if baseline_scores[task_id] is False and rag_scores[task_id] is True
    ]
    regressed_ids = [
        task_id
        for task_id in paired_ids
        if baseline_scores[task_id] is True and rag_scores[task_id] is False
    ]
    both_correct = sum(
        baseline_scores[task_id] is True and rag_scores[task_id] is True
        for task_id in paired_ids
    )
    fixed = len(fixed_ids)
    regressed = len(regressed_ids)
    both_wrong = sum(
        baseline_scores[task_id] is False and rag_scores[task_id] is False
        for task_id in paired_ids
    )
    return {
        "denominator": len(task_ids),
        "baseline_evaluated": sum(
            baseline_scores.get(task_id) is not None for task_id in task_ids
        ),
        "rag_evaluated": sum(
            rag_scores.get(task_id) is not None for task_id in task_ids
        ),
        "baseline_correct": baseline_correct,
        "rag_correct": rag_correct,
        "baseline_accuracy": _ratio(baseline_correct, len(task_ids)),
        "rag_accuracy": _ratio(rag_correct, len(task_ids)),
        "delta": _ratio(rag_correct - baseline_correct, len(task_ids)),
        "paired_evaluated": len(paired_ids),
        "both_correct": both_correct,
        "fixed_by_rag": fixed,
        "regressed_with_rag": regressed,
        "fixed_task_ids": fixed_ids,
        "regressed_task_ids": regressed_ids,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": _mcnemar_exact_p(fixed, regressed),
        "paired_delta_ci95": _paired_bootstrap_ci(
            [
                int(rag_scores[task_id] is True)
                - int(baseline_scores[task_id] is True)
                for task_id in paired_ids
            ]
        ),
    }


def _mcnemar_exact_p(fixed: int, regressed: int) -> float:
    discordant = fixed + regressed
    if discordant == 0:
        return 1.0
    smaller = min(fixed, regressed)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired_bootstrap_ci(
    deltas: list[int], *, samples: int = 10_000, seed: int = 20260723
) -> list[float] | None:
    if not deltas:
        return None
    generator = random.Random(seed)
    size = len(deltas)
    means = sorted(
        sum(deltas[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]


def _corpus_pairs(chunks_dir: Path | None) -> set[tuple[str, int]]:
    if chunks_dir is None or not chunks_dir.exists():
        return set()
    pairs: set[tuple[str, int]] = set()
    for path in chunks_dir.glob("*.jsonl"):
        metadata = infer_textbook_metadata(path.stem)
        subject = canonical_subject(metadata.get("subject"))
        grade = metadata.get("grade")
        if subject is not None and isinstance(grade, int):
            pairs.add((subject, grade))
    return pairs


def _coverage_label(task: dict[str, Any], corpus_pairs: set[tuple[str, int]]) -> str:
    if not corpus_pairs:
        return "inventory_unavailable"
    subject = canonical_subject(task.get("subject"))
    grade = task.get("grade")
    corpus_subjects = {pair[0] for pair in corpus_pairs}
    if subject not in corpus_subjects:
        return "subject_absent"
    if not isinstance(grade, int):
        return "grade_unknown"
    if (subject, grade) not in corpus_pairs:
        return "grade_absent"
    return "covered"


def _tool_metrics(
    task_ids: list[str], rag_results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    total_calls = 0
    tasks_with_calls = 0
    tasks_with_hits = 0
    tasks_with_errors = 0
    returned_hits = 0
    latencies: list[float] = []
    distinct_chunks: set[str] = set()
    for task_id in task_ids:
        result = rag_results.get(task_id) or {}
        calls = result.get("tool_calls")
        calls = calls if isinstance(calls, list) else []
        if calls:
            tasks_with_calls += 1
        task_has_hits = False
        task_has_error = False
        for call in calls:
            if not isinstance(call, dict):
                continue
            total_calls += 1
            chunk_ids = call.get("returned_chunk_ids")
            chunk_ids = chunk_ids if isinstance(chunk_ids, list) else []
            if chunk_ids:
                task_has_hits = True
                returned_hits += len(chunk_ids)
                distinct_chunks.update(map(str, chunk_ids))
            if call.get("error"):
                task_has_error = True
            latency = call.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
        tasks_with_hits += task_has_hits
        tasks_with_errors += task_has_error
    return {
        "tasks": len(task_ids),
        "tasks_with_tool_calls": tasks_with_calls,
        "tasks_without_tool_calls": len(task_ids) - tasks_with_calls,
        "tool_call_rate": _ratio(tasks_with_calls, len(task_ids)),
        "tasks_with_retrieval_hits": tasks_with_hits,
        "tasks_with_tool_errors": tasks_with_errors,
        "total_tool_calls": total_calls,
        "returned_hits": returned_hits,
        "distinct_returned_chunks": len(distinct_chunks),
        "mean_tool_latency_ms": (
            sum(latencies) / len(latencies) if latencies else None
        ),
    }


def _judge_flip_cases(
    task_ids: list[str],
    tasks: dict[str, dict[str, Any]],
    baseline_results: dict[str, dict[str, Any]],
    rag_results: dict[str, dict[str, Any]],
    baseline_judge: dict[str, dict[str, Any]],
    rag_judge: dict[str, dict[str, Any]],
    baseline_scores: dict[str, bool | None],
    rag_scores: dict[str, bool | None],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for task_id in task_ids:
        baseline_score = baseline_scores.get(task_id)
        rag_score = rag_scores.get(task_id)
        if baseline_score is None or rag_score is None or baseline_score == rag_score:
            continue
        task = tasks[task_id]
        b0_result = baseline_results.get(task_id) or {}
        rag_result = rag_results.get(task_id) or {}
        b0_verdict = (baseline_judge.get(task_id) or {}).get("verdict") or {}
        rag_verdict = (rag_judge.get(task_id) or {}).get("verdict") or {}
        cases.append(
            {
                "task_id": task_id,
                "direction": "fixed_by_rag" if rag_score else "regressed_with_rag",
                "subject": task.get("subject"),
                "grade": task.get("grade"),
                "question": task.get("question"),
                "reference_answer": task.get("reference_answer"),
                "baseline_answer": b0_result.get("final_answer"),
                "rag_answer": rag_result.get("final_answer"),
                "baseline_judge_rationale": b0_verdict.get("rationale"),
                "rag_judge_rationale": rag_verdict.get("rationale"),
                "tool_calls": rag_result.get("tool_calls") or [],
            }
        )
    return cases


def build_report(
    tasks: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
    rag_results: list[dict[str, Any]],
    *,
    baseline_judge: list[dict[str, Any]] | None = None,
    rag_judge: list[dict[str, Any]] | None = None,
    chunks_dir: Path | None = None,
) -> dict[str, Any]:
    tasks_by_id = _by_task(tasks, "tasks")
    baseline_by_id = _by_task(baseline_results, "baseline results")
    rag_by_id = _by_task(rag_results, "RAG results")
    baseline_judge_by_id = _by_task(baseline_judge or [], "baseline judge")
    rag_judge_by_id = _by_task(rag_judge or [], "RAG judge")
    task_ids = list(tasks_by_id)

    auto_ids = [
        task_id for task_id in task_ids if _is_auto_evaluable(tasks_by_id[task_id])
    ]
    baseline_auto = {
        task_id: _automatic_score(tasks_by_id[task_id], baseline_by_id.get(task_id))
        for task_id in auto_ids
    }
    rag_auto = {
        task_id: _automatic_score(tasks_by_id[task_id], rag_by_id.get(task_id))
        for task_id in auto_ids
    }
    report: dict[str, Any] = {
        "task_set": {
            "tasks": len(task_ids),
            "baseline_results": len(baseline_by_id),
            "rag_results": len(rag_by_id),
            "missing_baseline_results": len(set(task_ids) - set(baseline_by_id)),
            "missing_rag_results": len(set(task_ids) - set(rag_by_id)),
        },
        "automatic": _paired_counts(auto_ids, baseline_auto, rag_auto),
        "tool_usage": _tool_metrics(task_ids, rag_by_id),
    }

    corpus_pairs = _corpus_pairs(chunks_dir)
    coverage = {
        task_id: _coverage_label(tasks_by_id[task_id], corpus_pairs)
        for task_id in task_ids
    }
    report["corpus_coverage"] = {
        "inventory_pairs": len(corpus_pairs),
        "counts": dict(sorted(Counter(coverage.values()).items())),
    }

    if baseline_judge is not None and rag_judge is not None:
        baseline_scores = {
            task_id: _judge_score(baseline_judge_by_id.get(task_id))
            for task_id in task_ids
        }
        rag_scores = {
            task_id: _judge_score(rag_judge_by_id.get(task_id))
            for task_id in task_ids
        }
        report["judge_full"] = _paired_counts(task_ids, baseline_scores, rag_scores)
        report["judge_flip_cases"] = _judge_flip_cases(
            task_ids,
            tasks_by_id,
            baseline_by_id,
            rag_by_id,
            baseline_judge_by_id,
            rag_judge_by_id,
            baseline_scores,
            rag_scores,
        )
        report["judge_by_coverage"] = {
            label: _paired_counts(
                [
                    task_id
                    for task_id in task_ids
                    if coverage[task_id] == label
                ],
                baseline_scores,
                rag_scores,
            )
            for label in sorted(set(coverage.values()))
        }
    return report


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    task_set = report["task_set"]
    automatic = report["automatic"]
    tool = report["tool_usage"]
    coverage = report["corpus_coverage"]["counts"]
    lines = [
        "# B0 vs textbook RAG",
        "",
        f"Заданий: **{task_set['tasks']}**. "
        f"Результатов B0: **{task_set['baseline_results']}**, "
        f"RAG: **{task_set['rag_results']}**.",
        "",
        "## Автоматически проверяемая часть",
        "",
        f"- B0: {automatic['baseline_correct']}/{automatic['denominator']} "
        f"= **{_percent(automatic['baseline_accuracy'])}**",
        f"- RAG: {automatic['rag_correct']}/{automatic['denominator']} "
        f"= **{_percent(automatic['rag_accuracy'])}**",
        f"- Разница: **{_percent(automatic['delta'])}**; "
        f"исправлено {automatic['fixed_by_rag']}, "
        f"ухудшено {automatic['regressed_with_rag']}.",
        "",
        "## Покрытие корпуса",
        "",
        *[f"- `{label}`: {count}" for label, count in coverage.items()],
        "",
        "## Использование retrieval",
        "",
        f"- Вызвали tool: {tool['tasks_with_tool_calls']}/{tool['tasks']} "
        f"({_percent(tool['tool_call_rate'])})",
        f"- Получили хотя бы один чанк: {tool['tasks_with_retrieval_hits']}",
        f"- Ошибка tool: {tool['tasks_with_tool_errors']}",
        f"- Всего вызовов: {tool['total_tool_calls']}; "
        f"уникальных возвращённых чанков: {tool['distinct_returned_chunks']}",
    ]
    judge = report.get("judge_full")
    if judge:
        lines.extend(
            [
                "",
                "## LLM-as-a-judge на общем знаменателе",
                "",
                f"- B0: {judge['baseline_correct']}/{judge['denominator']} "
                f"= **{_percent(judge['baseline_accuracy'])}**",
                f"- RAG: {judge['rag_correct']}/{judge['denominator']} "
                f"= **{_percent(judge['rag_accuracy'])}**",
                f"- Разница: **{_percent(judge['delta'])}**",
                f"- RAG исправил {judge['fixed_by_rag']} ответов и ухудшил "
                f"{judge['regressed_with_rag']}.",
                f"- McNemar exact p-value: `{judge['mcnemar_exact_p']:.4f}`; "
                f"парный 95% bootstrap CI: `{judge['paired_delta_ci95']}`.",
                f"- Валидных judge-оценок: B0 {judge['baseline_evaluated']}, "
                f"RAG {judge['rag_evaluated']}; пропуски остаются в знаменателе "
                "и не считаются правильными.",
                f"- ID исправленных: `{', '.join(judge['fixed_task_ids']) or '—'}`",
                f"- ID ухудшенных: `{', '.join(judge['regressed_task_ids']) or '—'}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## LLM-as-a-judge",
                "",
                "Judge-файлы пока не переданы. После их появления повторите "
                "команду с `--baseline-judge` и `--rag-judge`.",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--rag-results", type=Path, required=True)
    parser.add_argument("--baseline-judge", type=Path)
    parser.add_argument("--rag-judge", type=Path)
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks/jsonl"))
    parser.add_argument(
        "--out-json", type=Path, default=Path("reports/rag_eval_summary.json")
    )
    parser.add_argument(
        "--out-md", type=Path, default=Path("reports/rag_eval_summary.md")
    )
    args = parser.parse_args(argv)
    if (args.baseline_judge is None) != (args.rag_judge is None):
        parser.error("judge files must be supplied together")

    report = build_report(
        _read_jsonl(args.tasks),
        _read_jsonl(args.baseline_results),
        _read_jsonl(args.rag_results),
        baseline_judge=(
            _read_jsonl(args.baseline_judge) if args.baseline_judge else None
        ),
        rag_judge=_read_jsonl(args.rag_judge) if args.rag_judge else None,
        chunks_dir=args.chunks_dir,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
