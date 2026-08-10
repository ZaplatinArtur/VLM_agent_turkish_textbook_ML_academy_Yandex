# -*- coding: utf-8 -*-
"""Диагностика пустой выдачи веб-поиска по логам прогонов.

Отвечает на вопрос, который решает, что чинить: пустота — свойство запроса
(модель формулирует так, что не находится) или свойство бэкенда (SearXNG не
отвечает независимо от запроса)?

Разделяющий признак — один и тот же запрос с разным исходом. Если запрос,
давший пустоту, в другом месте вернул результаты, дело не в формулировке.

Запуск из корня репозитория:
    python scripts/diagnose_web_search.py [--out reports/web_search_diag.txt]
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "B1snip": "reports/b1_full_32k_v2.jsonl",
    "B1deep": "reports/b1_deep_32k.jsonl",
    "B1dr": "reports/b1_deep_routed_32k.jsonl",
}
EMPTY_PREFIXES = ("Sonuç bulunamadı", "Sayfalar açılamadı", "Arama hatası")
DUP_PREFIX = "Bu sorguyu zaten yaptın"
LIMIT_PREFIX = "Arama limitine ulaştın"

MATH_CHARS = set("=+×÷^√∫∑πΔ°<>|")
DIGITS = re.compile(r"\d")


def outcome(preview: str) -> str:
    if preview.startswith(DUP_PREFIX):
        return "duplicate"
    if preview.startswith(LIMIT_PREFIX):
        return "over_limit"
    if preview.startswith("Arama hatası"):
        return "error"
    if preview.startswith(("Sonuç bulunamadı", "Sayfalar açılamadı")):
        return "empty"
    return "hit"


def features(query: str) -> dict:
    words = query.split()
    return {
        "chars": len(query),
        "words": len(words),
        "has_question": "?" in query,
        "has_quote": '"' in query or "'" in query,
        "has_digit": bool(DIGITS.search(query)),
        "has_math": any(c in MATH_CHARS for c in query),
        "has_newline": "\n" in query,
    }


def collect() -> tuple[list[dict], dict[str, set]]:
    """Веб-вызовы (прогон, номер задачи, запрос, исход) и охват по задачам."""
    calls = []
    tasks: dict[str, set] = {"any": set(), "hit": set()}
    for run, rel in RUNS.items():
        path = ROOT / rel
        if not path.exists():
            print(f"нет файла: {rel}", file=sys.stderr)
            continue
        for order, line in enumerate(path.open(encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for call in row.get("tool_calls") or []:
                if call.get("tool") != "web_search":
                    continue
                query = str((call.get("args") or {}).get("query") or "").strip()
                result = outcome(call.get("result_preview") or "")
                key = (run, row.get("task_id"))
                tasks["any"].add(key)
                if result == "hit":
                    tasks["hit"].add(key)
                calls.append({
                    "run": run,
                    "task_order": order,
                    "task_id": row.get("task_id"),
                    "query": query,
                    "outcome": result,
                    **features(query),
                })
    return calls, tasks


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "0/0"


def section(title: str) -> str:
    return f"\n{title}\n" + "-" * len(title)


def report(calls: list[dict], tasks: dict[str, set]) -> str:
    out: list[str] = ["ДИАГНОСТИКА ВЕБ-ПОИСКА ПО ЛОГАМ", "=" * 33]

    # 0. Сводка по всем прогонам сразу
    reached = [c for c in calls if c["outcome"] in ("hit", "empty")]
    empty = [c for c in reached if c["outcome"] == "empty"]
    blocked = len(calls) - len(reached)
    out.append(section("0. Итого по трём прогонам"))
    out.append(f"  вызовов web_search: {len(calls)}")
    out.append(f"  дошло до бэкенда:   {len(reached)} "
               f"(ещё {blocked} отбиты дедупом/лимитом, до сети не дошли)")
    out.append(f"  ПУСТЫХ от дошедших: {pct(len(empty), len(reached))}")
    out.append(f"  пустых от всех вызовов: {pct(len(empty), len(calls))}")
    out.append(f"  задач с поиском: {len(tasks['any'])}, "
               f"из них хоть раз получили выдачу: {pct(len(tasks['hit']), len(tasks['any']))}")

    # 1. Исходы по прогонам
    out.append(section("1. Исходы вызовов"))
    by_run = defaultdict(Counter)
    for c in calls:
        by_run[c["run"]][c["outcome"]] += 1
    kinds = ["hit", "empty", "error", "duplicate", "over_limit"]
    out.append("прогон   " + "".join(f"{k:>12}" for k in kinds) + f"{'всего':>10}")
    for run, counter in by_run.items():
        total = sum(counter.values())
        out.append(f"{run:<9}" + "".join(f"{counter[k]:>12}" for k in kinds)
                   + f"{total:>10}")

    # 2. Один и тот же запрос с разным исходом — признак нестабильного бэкенда
    out.append(section("2. Один запрос, разные исходы (нестабильность бэкенда)"))
    real = [c for c in calls if c["outcome"] in ("hit", "empty")]
    per_query = defaultdict(set)
    for c in real:
        per_query[c["query"].casefold()].add(c["outcome"])
    repeated = {q: o for q, o in per_query.items() if len(o) > 1}
    multi = sum(1 for q in per_query
                if sum(1 for c in real if c["query"].casefold() == q) > 1)
    out.append(f"уникальных запросов: {len(per_query)}, "
               f"встречались более одного раза: {multi}")
    out.append(f"из них дали и hit, и empty: {len(repeated)}")
    for q in list(repeated)[:10]:
        out.append(f"  · {q[:90]}")

    # 3. Признаки запроса: различают ли они пустоту и попадание
    out.append(section("3. Признаки запроса при пустоте и при попадании"))
    hits = [c for c in real if c["outcome"] == "hit"]
    empties = [c for c in real if c["outcome"] == "empty"]

    def stat(rows: list[dict], key: str) -> str:
        vals = [r[key] for r in rows]
        if not vals:
            return "—"
        return (f"медиана {statistics.median(vals):.0f}, "
                f"среднее {statistics.mean(vals):.0f}, макс {max(vals)}")

    out.append(f"попаданий {len(hits)}, пустых {len(empties)}")
    for key in ("chars", "words"):
        out.append(f"  {key:<6} hit:   {stat(hits, key)}")
        out.append(f"  {key:<6} empty: {stat(empties, key)}")
    for flag in ("has_question", "has_quote", "has_digit", "has_math", "has_newline"):
        h = sum(1 for c in hits if c[flag])
        e = sum(1 for c in empties if c[flag])
        out.append(f"  {flag:<13} hit: {pct(h, len(hits)):<16} empty: {pct(e, len(empties))}")

    # 4. Длина запроса и доля пустоты по корзинам
    out.append(section("4. Доля пустой выдачи по длине запроса"))
    buckets = [(0, 30), (30, 50), (50, 70), (70, 100), (100, 10_000)]
    for lo, hi in buckets:
        rows = [c for c in real if lo <= c["chars"] < hi]
        empty = sum(1 for c in rows if c["outcome"] == "empty")
        label = f"{lo}–{hi if hi < 10_000 else '∞'} символов"
        out.append(f"  {label:<20} пусто {pct(empty, len(rows))}")

    # 5. Дрейф по ходу прогона — признак троттлинга/бана движков
    out.append(section("5. Доля пустой выдачи по ходу прогона (четверти)"))
    for run in RUNS:
        rows = [c for c in real if c["run"] == run]
        if not rows:
            continue
        top = max(c["task_order"] for c in rows) + 1
        parts = []
        for q in range(4):
            lo, hi = top * q // 4, top * (q + 1) // 4
            sub = [c for c in rows if lo <= c["task_order"] < hi]
            empty = sum(1 for c in sub if c["outcome"] == "empty")
            parts.append(f"Q{q + 1} {pct(empty, len(sub)):<15}")
        out.append(f"  {run:<8}" + "".join(parts))

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/web_search_diag.txt")
    args = parser.parse_args()

    calls, tasks = collect()
    if not calls:
        print("нет данных", file=sys.stderr)
        return 1
    text = report(calls, tasks)
    print(text)
    (ROOT / args.out).write_text(text, encoding="utf-8")
    print(f"записано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
