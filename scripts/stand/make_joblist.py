# -*- coding: utf-8 -*-
"""Генератор joblist.txt для воркеров: бенчмарки × условия.

    python scripts/stand/make_joblist.py --benches tumlu,exams-v,mgsm \
        --conditions b0_no_tools,b1_search > joblist.txt

Формат строки: <выходной файл> <условие> <файл задач>
Воркеры (worker.sh) разбирают список сверху вниз через mkdir-локи, поэтому
порядок строк = приоритет. Готовые работы (строк в выходе >= задач) пропускаются
воркером сами — список можно перегенерировать и перезапускать безопасно.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SHORT = {"b0_no_tools": "b0", "b1_search": "b1", "b1_routed": "b1r",
         "b1_deep_routed": "b1dr", "agent_rag": "arag", "agent_rag_routed": "aragr"}

# каталоги задач: канон-300 для choice-бенчей, полный MGSM (он и так мал)
BENCH_DIRS = {
    "tumlu": "data/mlbench/bench300/tumlu",
    "exams-v": "data/mlbench/bench300/exams-v",
    "mgsm": "data/mlbench/mgsm",
}
PREFIX = {"tumlu": "t300", "exams-v": "e300", "mgsm": "mgsm"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benches", default="tumlu,exams-v,mgsm")
    parser.add_argument("--conditions", default="b0_no_tools,b1_search")
    args = parser.parse_args()

    for bench in args.benches.split(","):
        src = ROOT / BENCH_DIRS[bench]
        for tasks in sorted(src.glob("*.jsonl")):
            lang = tasks.stem
            for cond in args.conditions.split(","):
                short = SHORT.get(cond, cond)
                out = f"results/{PREFIX[bench]}_{lang}_{short}.jsonl"
                rel = tasks.relative_to(ROOT).as_posix()
                print(f"{out} {cond} {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
