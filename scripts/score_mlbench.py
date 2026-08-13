# -*- coding: utf-8 -*-
"""Сводка мультиязычных прогонов бенч-300: языки × условия, предметы, тул-цикл.

    python scripts/score_mlbench.py [--results reports/mlbench300]

Ожидает файлы t300_<язык>_<b0|b1>.jsonl (TUMLU) и e300_<язык>_<b0|b1>.jsonl
(EXAMS-V); задачи берёт из data/mlbench/bench300/. Выход — текстовая сводка
и reports/mlbench300_summary.md.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mla_baseline.eval import match  # noqa: E402

BENCH_DIR = {"t300": "tumlu", "e300": "exams-v"}


def load(path: Path) -> dict:
    return {r["task_id"]: r for r in
            (json.loads(l) for l in path.open(encoding="utf-8") if l.strip())}


# Буква варианта по ПОЗИЦИИ в национальном алфавите: болгарские экзамены
# нумеруют варианты А/Б/В/Г — модель же отвечает латиницей A/B/C/D.
# Кириллическая В — это третий вариант, то есть латинская C, а не B.
_CHOICE_MAP = str.maketrans({
    "А": "A", "Б": "B", "В": "C", "Г": "D", "Д": "E",
    "а": "A", "б": "B", "в": "C", "г": "D", "д": "E",
    "Γ": "D",  # греческая гамма — так модель иногда рисует Г
})


def _norm_choice(ans: str) -> str:
    return (ans or "").strip().translate(_CHOICE_MAP).upper()[:1]


def good(res: dict, task: dict) -> bool:
    at = task.get("answer_type") or "choice"
    if at == "choice":
        return _norm_choice(res.get("final_answer")) == \
            _norm_choice(task["reference_answer"])
    return match(res.get("final_answer") or "", task["reference_answer"], at)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="reports/mlbench300")
    args = parser.parse_args()
    res_dir = ROOT / args.results

    tasks: dict[str, dict] = {}
    for bench in ("tumlu", "exams-v"):
        for p in (ROOT / "data/mlbench/bench300" / bench).glob("*.jsonl"):
            tasks.update(load(p))

    runs: dict[tuple, dict] = {}   # (bench, lang, cond) -> результаты
    for p in sorted(res_dir.glob("*300_*.jsonl")):
        prefix, rest = p.stem.split("_", 1)
        lang, cond = rest.rsplit("_", 1)
        runs[(BENCH_DIR[prefix], lang, cond)] = load(p)

    lines: list[str] = []
    out = lines.append

    for bench in ("tumlu", "exams-v"):
        langs = sorted({l for (b, l, _) in runs if b == bench})
        if not langs:
            continue
        out(f"\n## {bench.upper()} (канон-300, exact match)\n")
        out("| Язык | n | B0 | b1_search | Δ | b1: искало | пустых выдач |")
        out("|---|---|---|---|---|---|---|")
        for lang in langs:
            b0 = runs.get((bench, lang, "b0"), {})
            b1 = runs.get((bench, lang, "b1"), {})
            common = [t for t in b0 if t in b1
                      and not b0[t].get("error") and not b1[t].get("error")]
            if not common:
                continue
            a0 = sum(good(b0[t], tasks[t]) for t in common) / len(common)
            a1 = sum(good(b1[t], tasks[t]) for t in common) / len(common)
            searched = sum(1 for t in common if b1[t].get("tool_calls"))
            calls = [c for t in common for c in (b1[t].get("tool_calls") or [])]
            empty = sum(1 for c in calls if (c.get("result_preview") or "")
                        .startswith(("Sonuç", "Arama", "Sayfalar")))
            out(f"| {lang} | {len(common)} | {a0:.1%} | {a1:.1%} | "
                f"{(a1 - a0) * 100:+.1f} | {searched}/{len(common)} | "
                f"{empty}/{len(calls) or 1} |")

    # MGSM: открытые числовые ответы (файлы mgsm_<lang>_<b0|b1>.jsonl)
    mgsm_tasks: dict[str, dict] = {}
    for p in (ROOT / "data/mlbench/mgsm").glob("*.jsonl"):
        mgsm_tasks.update(load(p))
    mgsm_runs: dict[tuple, dict] = {}
    for p in sorted(res_dir.glob("mgsm_*_b*.jsonl")):
        _, lang, cond = p.stem.split("_")
        mgsm_runs[(lang, cond)] = load(p)
    if mgsm_runs:
        out("\n## MGSM (открытые числовые ответы, exact match)\n")
        out("| Язык | n | B0 | b1_search | Δ |")
        out("|---|---|---|---|---|")
        for lang in ("en", "de", "fr", "es", "ru", "zh", "ja", "th", "sw",
                     "bn", "te"):
            b0 = mgsm_runs.get((lang, "b0"), {})
            b1 = mgsm_runs.get((lang, "b1"), {})
            common = [t for t in b0 if t in b1
                      and not b0[t].get("error") and not b1[t].get("error")]
            if len(common) < 200:
                continue
            a0 = sum(match(b0[t].get("final_answer") or "",
                           mgsm_tasks[t]["reference_answer"], "numeric")
                     for t in common) / len(common)
            a1 = sum(match(b1[t].get("final_answer") or "",
                           mgsm_tasks[t]["reference_answer"], "numeric")
                     for t in common) / len(common)
            out(f"| {lang} | {len(common)} | {a0:.1%} | {a1:.1%} | "
                f"{(a1 - a0) * 100:+.1f} |")

    # предметы: только TUMLU (у EXAMS-V предметы уже сгруппированы в данных)
    out("\n## TUMLU: предметы × языки (B0)\n")
    subj_acc: dict[str, dict[str, tuple]] = defaultdict(dict)
    for (bench, lang, cond), rows in runs.items():
        if bench != "tumlu" or cond != "b0":
            continue
        per = defaultdict(lambda: [0, 0])
        for t, r in rows.items():
            if r.get("error") or t not in tasks:
                continue
            s = tasks[t].get("subject") or "?"
            per[s][1] += 1
            per[s][0] += good(r, tasks[t])
        for s, (ok, n) in per.items():
            subj_acc[s][lang] = (ok, n)
    langs = sorted({l for (b, l, c) in runs if b == "tumlu" and c == "b0"})
    out("| Предмет | " + " | ".join(langs) + " |")
    out("|---|" + "---|" * len(langs))
    for s in sorted(subj_acc, key=lambda s: -sum(n for _, n in subj_acc[s].values())):
        cells = []
        for l in langs:
            ok, n = subj_acc[s].get(l, (0, 0))
            cells.append(f"{ok/n:.0%}" if n >= 10 else ("–" if n == 0 else f"({n})"))
        out(f"| {s} | " + " | ".join(cells) + " |")

    # качество прогона
    out("\n## Здоровье прогонов\n")
    out("| Прогон | ошибок | forced | без ответа |")
    out("|---|---|---|---|")
    for (bench, lang, cond), rows in sorted(runs.items()):
        err = sum(1 for r in rows.values() if r.get("error"))
        fo = sum(1 for r in rows.values() if r.get("forced_answer"))
        na = sum(1 for r in rows.values()
                 if not r.get("error") and not (r.get("final_answer") or "").strip())
        if err or fo or na:
            out(f"| {bench}/{lang}/{cond} | {err} | {fo} | {na} |")

    text = "\n".join(lines)
    print(text)
    (ROOT / "reports/mlbench300_summary.md").write_text(
        "# Мультиязычный бенч-300: сводка\n" + text + "\n", encoding="utf-8")
    print("\nзаписано: reports/mlbench300_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
