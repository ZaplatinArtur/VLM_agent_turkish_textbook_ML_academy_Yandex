# -*- coding: utf-8 -*-
"""Гибридная метрика прогона: exact для авто-задач + судья для остальных.

Правило то же, что в прежних отчётах:
  * задача с эталоном-текстом (`answer_is_url != True`) — точное совпадение;
  * задача с эталоном-картинкой, для которой есть транскрипция — вердикт судьи
    из delta-файла;
  * задача с эталоном-картинкой без транскрипции — из метрики исключается.

    python scripts/hybrid_score.py b0_v100_fixed b1_search_v100
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mla_baseline.eval import match  # noqa: E402


def load(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        return {}
    return {r["task_id"]: r for r in
            (json.loads(l) for l in p.open(encoding="utf-8") if l.strip())}


def verdicts(rel: str) -> dict[str, int]:
    out = {}
    for tid, row in load(rel).items():
        v = row.get("verdict") or {}
        if v.get("score") is not None:
            out[tid] = int(v["score"])
    return out


def score(name: str, tasks: dict, meta: dict) -> dict:
    run = load(f"reports/{name}.jsonl")
    delta = verdicts(f"reports/jout_{name}_delta.jsonl")
    ok = n = auto = jud = 0
    for tid, r in run.items():
        is_img = str(meta.get(tid, {}).get("answer_is_url")) == "True"
        if is_img:
            if tid not in delta:      # нерасшифрованный эталон — вне метрики
                continue
            n += 1
            jud += 1
            ok += delta[tid]
        else:
            ref = (tasks.get(tid, {}).get("reference_answer") or "").strip()
            if not ref:
                continue
            n += 1
            auto += 1
            ok += bool(match(r.get("final_answer") or "", ref,
                             tasks[tid].get("answer_type") or "short"))
    return {"name": name, "ok": ok, "n": n, "auto": auto, "judge": jud,
            "errors": sum(1 for r in run.values() if r.get("error"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="имена файлов в reports/ без .jsonl")
    args = ap.parse_args()

    tasks = load("data/eval/validation.jsonl")
    meta = load("data/eval/validation.meta.jsonl")

    print(f"{'прогон':<26}{'гибрид':>18}{'авто':>7}{'судья':>7}{'ошибок':>8}")
    for name in args.runs:
        s = score(name, tasks, meta)
        if not s["n"]:
            print(f"{name}: нет данных")
            continue
        print(f"{s['name']:<26}{s['ok']:>4}/{s['n']} = {s['ok']/s['n']:6.1%}"
              f"{s['auto']:>7}{s['judge']:>7}{s['errors']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
