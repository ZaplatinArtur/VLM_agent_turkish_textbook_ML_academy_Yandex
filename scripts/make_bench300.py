# -*- coding: utf-8 -*-
"""Фиксированный переиспользуемый срез бенчмарков: 300 задач на язык.

Зачем фиксировать: срез, который каждый раз сэмплится заново, нельзя
сравнивать между прогонами и между собой. Здесь сэмплинг детерминирован
(сид 47, стратификация по предмету), результат коммитится в репозиторий и
становится КАНОНОМ: все условия, все модели, все будущие прогоны меряются
на одних и тех же задачах.

    python scripts/make_bench300.py            # tumlu + exams-v
    python scripts/make_bench300.py --n 300

Выход: data/mlbench/bench300/<бенчмарк>/<язык>.jsonl
Языки, где задач меньше N, берутся целиком. Картинки не копируются —
ссылки указывают на исходные data/mlbench/<бенчмарк>/images/.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "mlbench"
SEED = 47


def strata_key(row: dict) -> tuple:
    """Ключ страты: предмет + прокси сложности + модальность.

    Явных меток сложности в этих бенчмарках нет, единственный доступный
    прокси — класс (EXAMS-V; в TUMLU/MGSM его нет вовсе). Модальность
    (текстовый лист против листа с рисунком) добавлена потому, что она
    сильнее влияет на результат, чем предмет: картинку надо ещё прочитать.
    """
    return (row.get("subject") or "unknown",
            row.get("grade"),
            row.get("_modality") or "")


def sample_lang(path: Path, n: int) -> list[dict]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if len(rows) <= n:
        return rows
    # стратификация: доли страт сохраняются, внутри страты порядок задаётся
    # сидом — воспроизводимо на любой машине
    by_subj = defaultdict(list)
    for r in rows:
        by_subj[strata_key(r)].append(r)
    rng = random.Random(SEED)
    picked: list[dict] = []
    # квоты пропорционально размеру предмета, остаток — крупнейшим
    quotas = {s: (len(v) * n) // len(rows) for s, v in by_subj.items()}
    leftover = n - sum(quotas.values())
    # остаток — самым крупным стратам; при равенстве размера решает ключ,
    # чтобы результат не зависел от порядка обхода словаря
    order = sorted(by_subj, key=lambda s: (-len(by_subj[s]),
                                           tuple(str(x) for x in s)))
    for s in order[:leftover]:
        quotas[s] += 1
    for s in sorted(by_subj, key=lambda k: tuple(str(x) for x in k)):
        pool = sorted(by_subj[s], key=lambda r: r["task_id"])
        rng.shuffle(pool)
        picked.extend(pool[: quotas[s]])
    picked.sort(key=lambda r: r["task_id"])
    return picked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--benches", default="tumlu,exams-v")
    args = parser.parse_args()

    for bench in args.benches.split(","):
        src_dir = SRC / bench
        out_dir = SRC / "bench300" / bench
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for path in sorted(src_dir.glob("*.jsonl")):
            rows = sample_lang(path, args.n)
            out = out_dir / path.name
            with out.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(rows)
            print(f"{bench}/{path.stem}: {len(rows)}")
        print(f"{bench}: итого {total}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
