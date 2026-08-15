"""Deep dive into RAG failures: wrong judge verdicts + tool/evidence taxonomy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["task_id"]): r for r in rows if r.get("task_id")}


def _judge_ok(row: dict[str, Any] | None) -> bool | None:
    if not row:
        return None
    verdict = row.get("verdict") or {}
    if isinstance(verdict, dict) and isinstance(verdict.get("strict_correct"), bool):
        return bool(verdict["strict_correct"])
    # multimodal 0..4
    if isinstance(verdict, dict) and "score" in verdict:
        try:
            return int(verdict["score"]) >= 4
        except Exception:
            return None
    return None


def _tool_summary(result: dict[str, Any] | None) -> dict[str, Any]:
    tools = (result or {}).get("tool_calls") or []
    page_ids: list[str] = []
    queries: list[str] = []
    errors: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        args = t.get("args") or {}
        if isinstance(args, dict) and args.get("query"):
            queries.append(str(args["query"]))
        ids = t.get("returned_chunk_ids") or []
        page_ids.extend(str(x) for x in ids)
        if t.get("error"):
            errors.append(str(t["error"]))
    return {
        "n_calls": len(tools),
        "queries": queries,
        "page_ids": page_ids,
        "errors": errors,
        "had_hits": bool(page_ids),
    }


def _bucket(result: dict[str, Any] | None, tool: dict[str, Any]) -> str:
    if result is None:
        return "missing_result"
    if result.get("error"):
        return f"runtime:{result.get('error')}"
    if tool["n_calls"] == 0:
        return "no_tool_used"
    if tool["errors"] and not tool["had_hits"]:
        return "tool_error_no_hits"
    if tool["errors"]:
        return "tool_error_partial"
    if not tool["had_hits"]:
        return "empty_hits"
    return "evidence_present_but_wrong"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--baseline-results", type=Path, required=True)
    p.add_argument("--rag-results", type=Path, required=True)
    p.add_argument("--baseline-judge", type=Path, required=True)
    p.add_argument("--rag-judge", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    tasks = _by_id(_read_jsonl(args.tasks))
    b0 = _by_id(_read_jsonl(args.baseline_results))
    rag = _by_id(_read_jsonl(args.rag_results))
    b0j = _by_id(_read_jsonl(args.baseline_judge))
    ragj = _by_id(_read_jsonl(args.rag_judge))

    wrong: list[dict[str, Any]] = []
    buckets: Counter[str] = Counter()
    flips = {"fixed_by_rag": [], "regressed_with_rag": [], "both_wrong": [], "both_ok": []}

    for tid, task in tasks.items():
        b_ok = _judge_ok(b0j.get(tid))
        r_ok = _judge_ok(ragj.get(tid))
        tool = _tool_summary(rag.get(tid))
        if b_ok is True and r_ok is True:
            flips["both_ok"].append(tid)
        elif b_ok is False and r_ok is True:
            flips["fixed_by_rag"].append(tid)
        elif b_ok is True and r_ok is False:
            flips["regressed_with_rag"].append(tid)
        elif b_ok is False and r_ok is False:
            flips["both_wrong"].append(tid)

        if r_ok is not False:
            continue
        bucket = _bucket(rag.get(tid), tool)
        buckets[bucket] += 1
        verdict = (ragj.get(tid) or {}).get("verdict") or {}
        wrong.append(
            {
                "task_id": tid,
                "subject": task.get("subject"),
                "grade": task.get("grade"),
                "bucket": bucket,
                "b0_ok": b_ok,
                "rag_ok": r_ok,
                "final_answer": (rag.get(tid) or {}).get("final_answer"),
                "reference_answer": task.get("reference_answer"),
                "tool": tool,
                "judge_rationale": verdict.get("rationale") or verdict.get("reason"),
                "judge_score": verdict.get("score") if isinstance(verdict, dict) else None,
            }
        )

    tips: list[str] = []
    n_wrong = max(len(wrong), 1)
    if buckets.get("no_tool_used", 0) / n_wrong > 0.25:
        tips.append(
            "Много wrong без tool: усилить visual_rag_tool_policy (обязательный первый search "
            "для curriculum задач) + few-shot tool call."
        )
    if buckets.get("empty_hits", 0) + buckets.get("tool_error_no_hits", 0) > 0:
        tips.append(
            "Пустые hits / tool_error: проверить grade/subject фильтр (case), query formulation, "
            "и покрытие индекса (high-school chem/phys в MaxSim MEB-индексе почти нет)."
        )
    if buckets.get("evidence_present_but_wrong", 0) / n_wrong > 0.3:
        tips.append(
            "Evidence есть, ответ неверный: grounded-generation баг — запретить слепое копирование "
            "чисел/букв из answer_text; требовать сверку с картинкой вопроса."
        )
    if flips["regressed_with_rag"]:
        tips.append(
            f"Регрессии RAG ({len(flips['regressed_with_rag'])}): смотреть wrong page / misleading "
            "near-miss; возможно поднять min MaxSim margin или требовать subject match hard."
        )
    if not tips:
        tips.append("Явного доминирующего failure mode нет — смотреть per-task JSONL.")

    summary = {
        "n_tasks": len(tasks),
        "n_rag_wrong": len(wrong),
        "buckets": dict(buckets),
        "flips": {k: v for k, v in flips.items()},
        "flip_counts": {k: len(v) for k, v in flips.items()},
        "fix_recommendations": tips,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "wrong_rag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "wrong_rag_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in wrong:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = [
        "# Анализ неправильных ответов visual RAG",
        "",
        f"- задач: {len(tasks)}",
        f"- RAG wrong (judge): {len(wrong)}",
        "",
        "## Buckets",
        "",
        "```json",
        json.dumps(dict(buckets), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Flips B0 ↔ RAG",
        "",
        f"- both_ok: {len(flips['both_ok'])}",
        f"- fixed_by_rag: {len(flips['fixed_by_rag'])} → `{', '.join(flips['fixed_by_rag']) or '—'}`",
        f"- regressed_with_rag: {len(flips['regressed_with_rag'])} → `{', '.join(flips['regressed_with_rag']) or '—'}`",
        f"- both_wrong: {len(flips['both_wrong'])}",
        "",
        "## Рекомендации",
        "",
    ]
    for tip in tips:
        md.append(f"- {tip}")
    md.append("")
    md.append("Детали кейсов: `wrong_rag_cases.jsonl`.")
    md.append("")
    (args.out_dir / "wrong_rag_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
