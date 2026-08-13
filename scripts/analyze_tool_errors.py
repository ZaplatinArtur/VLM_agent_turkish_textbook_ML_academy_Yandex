# -*- coding: utf-8 -*-
"""Разбор ошибок в условиях с инструментами (веб-поиск, ретрив по учебникам).

Считает по логам tool_calls: патологии тул-цикла (пустая выдача, дубли,
превышение лимита, галлюцинированные имена инструментов, навигационный мусор
в чанках), связь ошибок с forced-финалом и бюджетом токенов, а также шум
сэмплинга по routed-математике (она идёт идентичным B0-кодом).

Запуск из корня репозитория:
    python scripts/analyze_tool_errors.py [--out reports/tool_errors_stats.txt]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mla_baseline.eval import match  # noqa: E402

# имя -> (файл прогона, файлы вердиктов судьи: base + delta)
RUNS = {
    "B0":     ("reports/b0_full_32k.jsonl",
               ["reports/judge_out_b0.jsonl", "reports/judge_out_b0_delta.jsonl"]),
    "B1snip": ("reports/b1_full_32k_v2.jsonl", []),
    "B1deep": ("reports/b1_deep_32k.jsonl", []),
    "B1dr":   ("reports/b1_deep_routed_32k.jsonl",
               ["reports/judge_out_b1dr.jsonl", "reports/judge_out_b1dr_delta.jsonl"]),
    "ARAG":   ("reports/agent_rag_32k.jsonl",
               ["reports/judge_out_arag.jsonl", "reports/judge_out_arag_delta.jsonl"]),
    "ARAGr":  ("reports/agent_rag_routed_32k.jsonl",
               ["reports/judge_out_aragr.jsonl", "reports/judge_out_aragr_delta.jsonl"]),
}
REAL_TOOLS = ("web_search", "search_textbooks")
EMPTY_PREFIXES = ("Sonuç bulunamadı", "Ders kitaplarında sonuç", "Sayfalar açılamadı",
                  "Arama hatası")
BOILER_MARKERS = ("Öğrenci Çözümleri", "Çözümü Değerlendir", "Sinirlendim", "Bayıldım")
JUNK_ANSWER = re.compile(r'^\s*[{}\]\[>*]|"reasoning"')


def load_jsonl(rel: str) -> dict[str, dict]:
    out = {}
    for line in (ROOT / rel).open(encoding="utf-8"):
        line = line.strip()
        if line:
            row = json.loads(line)
            out[row["task_id"]] = row
    return out


def load_judge(rels: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rel in rels:
        if not (ROOT / rel).exists():
            continue
        for row in load_jsonl(rel).values():
            v = row.get("verdict")
            if v and v.get("score") is not None:
                out[row["task_id"]] = v
    return out


def classify(call: dict) -> str:
    """Категория одного тул-вызова по его результату."""
    prev = call.get("result_preview") or ""
    if call.get("tool") not in REAL_TOOLS:
        return "hallucinated_tool"
    if not str((call.get("args") or {}).get("query") or "").strip():
        return "empty_query"
    if prev.startswith("Bu sorguyu zaten yaptın"):
        return "duplicate"
    if prev.startswith("Arama limitine ulaştın"):
        return "over_limit"
    if prev.startswith(EMPTY_PREFIXES):
        return "no_results"
    if sum(m in prev for m in BOILER_MARKERS) >= 2:
        return "boilerplate_hit"
    return "hit"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/tool_errors_stats.txt")
    args = ap.parse_args()

    tasks = load_jsonl("data/eval/validation.jsonl")
    meta = load_jsonl("data/eval/validation.meta.jsonl")
    transcripts = set(json.load(
        (ROOT / "reports/answer_transcripts.json").open(encoding="utf-8")))
    results = {n: load_jsonl(r) for n, (r, _) in RUNS.items()}
    judges = {n: load_judge(j) for n, (_, j) in RUNS.items()}

    def ok(cond: str, tid: str):
        """Гибридная метрика: exact для авто-задач, судья для остальных."""
        task, m = tasks[tid], meta.get(tid, {})
        if m.get("answer_is_url") and tid not in transcripts:
            return None  # эталон-картинка без транскрипции — судить не по чему
        r = results[cond].get(tid) or {}
        if not m.get("answer_is_url") and task["answer_type"] != "free_form":
            return False if r.get("error") else bool(
                match(r.get("final_answer") or "", task["reference_answer"],
                      task["answer_type"]))
        v = judges[cond].get(tid)
        return bool(v["score"]) if v else None

    def acc(cond: str, ids) -> float:
        return sum(bool(ok(cond, t)) for t in ids) / max(len(ids), 1)

    L: list[str] = []
    L.append("### Патологии тул-цикла")
    for cond in ("B1snip", "B1deep", "B1dr", "ARAG", "ARAGr"):
        cnt = Counter()
        names = Counter()
        for r in results[cond].values():
            for c in (r.get("tool_calls") or []):
                cnt[classify(c)] += 1
                names[c.get("tool")] += 1
        tot = sum(cnt.values())
        if not tot:
            continue
        L.append(f"  {cond}: вызовов {tot}, полезных {cnt['hit']} ({cnt['hit']/tot:.0%})")
        for k, v in cnt.most_common():
            L.append(f"      {k:18s} {v:4d} ({v/tot:4.0%})")
        bad = {k: v for k, v in names.items() if k not in REAL_TOOLS}
        if bad:
            L.append(f"      несуществующие имена инструментов: {bad}")

    L.append("")
    L.append("### forced-финал как драйвер потерь")
    for cond in RUNS:
        ids = [t for t in tasks if ok(cond, t) is not None]
        if not ids:
            continue
        f = [t for t in ids if results[cond][t].get("forced_answer")]
        nf = [t for t in ids if not results[cond][t].get("forced_answer")]
        lost = [t for t in ids if ok("B0", t) and not ok(cond, t)]
        lost_f = [t for t in lost if results[cond][t].get("forced_answer")]
        toks = sorted(((results[cond][t].get("usage") or {}).get("output_tokens") or 0)
                      for t in ids)
        L.append(f"  {cond:7s} forced {len(f):3d}/{len(ids)} (acc {acc(cond, f):.1%}) | "
                 f"без forced acc {acc(cond, nf):.1%} | медиана токенов {toks[len(toks)//2]} | "
                 f"потерь от B0 {len(lost)}, из них forced {len(lost_f)} "
                 f"({len(lost_f)/max(len(lost),1):.0%})")

    L.append("")
    L.append("### Точность в зависимости от того, была ли реальная выдача")
    for cond in ("B1deep", "B1dr", "ARAG", "ARAGr"):
        ids = [t for t in tasks if ok(cond, t) is not None]
        got, empty, never = [], [], []
        for t in ids:
            calls = results[cond][t].get("tool_calls") or []
            if not calls:
                never.append(t)
            elif any(classify(c) in ("hit", "boilerplate_hit") for c in calls):
                got.append(t)
            else:
                empty.append(t)
        L.append(f"  {cond:7s} выдача была n={len(got):3d}: {acc(cond, got):.1%} "
                 f"(B0 {acc('B0', got):.1%}) | пусто n={len(empty):3d}: {acc(cond, empty):.1%} "
                 f"(B0 {acc('B0', empty):.1%}) | не искала n={len(never):3d}: "
                 f"{acc(cond, never):.1%} (B0 {acc('B0', never):.1%})")

    L.append("")
    L.append("### Шум сэмплинга: routed-Math идёт идентичным B0-кодом")
    math_ids = [t for t in tasks if tasks[t]["subject"] == "Math" and ok("B0", t) is not None]
    for cond in ("B1dr", "ARAGr"):
        n_calls = sum(1 for t in math_ids if results[cond][t].get("tool_calls"))
        diff = [t for t in math_ids if bool(ok(cond, t)) != bool(ok("B0", t))]
        L.append(f"  {cond:7s} Math n={len(math_ids)} (тул-вызовов {n_calls}): "
                 f"{acc(cond, math_ids):.1%} vs B0 {acc('B0', math_ids):.1%} "
                 f"→ {100*(acc(cond, math_ids)-acc('B0', math_ids)):+.1f} пп при идентичном "
                 f"коде, разошлись на {len(diff)} задачах")

    L.append("")
    L.append("### Мусор в final_answer (утечка JSON, обрывки)")
    for cond in RUNS:
        ids = [t for t in tasks if ok(cond, t) is not None]
        if not ids:
            continue
        junk = [t for t in ids
                if JUNK_ANSWER.search(str(results[cond][t].get("final_answer") or ""))]
        L.append(f"  {cond:7s} {len(junk):3d} шт., из них ошибочных "
                 f"{sum(1 for t in junk if not ok(cond, t))}")

    text = "\n".join(L)
    (ROOT / args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
