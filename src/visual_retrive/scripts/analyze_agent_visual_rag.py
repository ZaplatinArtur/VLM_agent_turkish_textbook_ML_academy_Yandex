"""Analyze agent_visual_rag results vs baseline + judge: RAG/agent error taxonomy.

Inputs (JSONL):
  --tasks validation tasks
  --baseline-results b0_no_tools.jsonl
  --rag-results agent_visual_rag.jsonl
  --baseline-judge / --rag-judge optional vlm_judge outputs

Writes a markdown+json report with buckets:
  - no_tool_used / tool_error / empty_hits
  - retrieval_miss (gold page never returned) when gold_page_id present
  - evidence_ignored (tool returned pages but judge says wrong)
  - grounded_wrong_number / parse_error / etc.
and concrete fix recommendations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _by_id(rows: list[dict[str, Any]], key: str = "task_id") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        tid = str(row.get(key) or "").strip()
        if tid:
            out[tid] = row
    return out


def _judge_ok(row: dict[str, Any] | None) -> bool | None:
    if not row:
        return None
    verdict = row.get("verdict")
    if isinstance(verdict, dict) and isinstance(verdict.get("strict_correct"), bool):
        return bool(verdict["strict_correct"])
    return None


def _classify_agent(
    *,
    task: dict[str, Any],
    result: dict[str, Any] | None,
    judge_ok: bool | None,
) -> str:
    if result is None:
        return "missing_result"
    if result.get("error") == "parse_error":
        return "parse_error"
    if result.get("error"):
        return f"runtime_error:{result.get('error')}"

    tools = result.get("tool_calls") or []
    if not tools:
        return "no_tool_used" if judge_ok is False else ("no_tool_ok" if judge_ok else "no_tool_used")

    any_err = any(t.get("error") for t in tools if isinstance(t, dict))
    if any_err:
        return "tool_error"

    page_ids: list[str] = []
    empty = True
    for t in tools:
        if not isinstance(t, dict):
            continue
        ids = t.get("returned_chunk_ids") or []
        if ids:
            empty = False
            page_ids.extend(str(x) for x in ids)
    if empty:
        return "empty_hits"

    gold = task.get("gold_page_id") or task.get("positive_page_id")
    if gold and str(gold) not in page_ids:
        # only if we know gold
        if judge_ok is False:
            return "retrieval_miss_and_wrong"
        return "retrieval_miss"

    if judge_ok is False:
        return "evidence_present_but_wrong"
    if judge_ok is True:
        return "tool_helped_or_correct"
    return "tool_used_unjudged"


def _tips(counts: Counter[str], n: int) -> list[str]:
    tips: list[str] = []
    if n <= 0:
        return ["No tasks analyzed."]
    def rate(k: str) -> float:
        return counts.get(k, 0) / n

    if rate("no_tool_used") > 0.25:
        tips.append(
            "Agent aracı az kullanıyor: VISUAL_RAG_TOOL_POLICY'yi sertleştir "
            "(zorunlu ilk tool call for curriculum topics) veya few-shot tool örnekleri ekle."
        )
    if rate("empty_hits") + rate("tool_error") > 0.1:
        tips.append(
            "Boş/hatalı tool: index yolunu (MLA_VISUAL_INDEX_DIR) ve GPU memory'yi kontrol et; "
            "query min length / filter mismatch logla."
        )
    if rate("retrieval_miss_and_wrong") + rate("retrieval_miss") > 0.2:
        tips.append(
            "Retrieval miss: MaxSim rerank, larger candidate pool, subject/grade filter from task; "
            "val-hard queries üzerinde LoRA devam eğitimi."
        )
    if rate("evidence_present_but_wrong") > 0.15:
        tips.append(
            "Evidence ignored / wrong grounded answer: prompt'a 'sayıları sayfadan değil "
            "sorudan al' kuralı; answer_text'i kısaltıp sadece formül/adım özeti ver; "
            "multi-hit contradiction check."
        )
    if rate("parse_error") > 0.05:
        tips.append(
            "Parse errors: structured decoding / JSON schema zorunlu; max_tokens artır; "
            "tool JSON'unu context'ten ayır."
        )
    if not tips:
        tips.append("Dagılım dengeli; paired flips ve subject slice'larına bakarak ince ayar yap.")
    tips.append(
        "En iyi düzeltme sırası: (1) retrieval hit@1/5, (2) tool-use rate, "
        "(3) grounded generation / copy-paste, (4) judge-disagreed edge cases."
    )
    return tips


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--baseline-results", type=Path, default=None)
    p.add_argument("--rag-results", type=Path, required=True)
    p.add_argument("--baseline-judge", type=Path, default=None)
    p.add_argument("--rag-judge", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    tasks = _by_id(_read_jsonl(args.tasks))
    base = _by_id(_read_jsonl(args.baseline_results))
    rag = _by_id(_read_jsonl(args.rag_results))
    base_j = _by_id(_read_jsonl(args.baseline_judge))
    rag_j = _by_id(_read_jsonl(args.rag_judge))

    buckets: Counter[str] = Counter()
    flips = Counter()
    rows_out: list[dict[str, Any]] = []

    for tid, task in tasks.items():
        r = rag.get(tid)
        b = base.get(tid)
        rj = _judge_ok(rag_j.get(tid))
        bj = _judge_ok(base_j.get(tid))
        bucket = _classify_agent(task=task, result=r, judge_ok=rj)
        buckets[bucket] += 1
        if bj is not None and rj is not None:
            if (not bj) and rj:
                flips["b0_wrong_rag_right"] += 1
            elif bj and (not rj):
                flips["b0_right_rag_wrong"] += 1
            elif bj and rj:
                flips["both_right"] += 1
            else:
                flips["both_wrong"] += 1
        rows_out.append(
            {
                "task_id": tid,
                "bucket": bucket,
                "baseline_judge_ok": bj,
                "rag_judge_ok": rj,
                "tool_calls": (r or {}).get("tool_calls"),
                "final_answer": (r or {}).get("final_answer"),
            }
        )

    n = len(tasks)
    tips = _tips(buckets, n)
    report = {
        "n_tasks": n,
        "buckets": dict(buckets),
        "paired_flips": dict(flips),
        "fix_recommendations": tips,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "agent_error_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "agent_error_per_task.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = ["# Agent visual RAG error analysis", "", "## Buckets", ""]
    for k, v in buckets.most_common():
        md.append(f"- `{k}`: {v} ({v / max(n,1):.1%})")
    md += ["", "## Paired flips", ""]
    for k, v in flips.most_common():
        md.append(f"- `{k}`: {v}")
    md += ["", "## Fix recommendations", ""]
    for tip in tips:
        md.append(f"- {tip}")
    md.append("")
    (args.out_dir / "agent_error_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
