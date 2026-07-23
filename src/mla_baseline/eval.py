"""Быстрые метрики по результатам прогона (до LLM-as-Judge).

Точный матч со следующей нормализацией:
  - choice: берём первую латинскую букву A-E из ответа (регистр не важен);
  - numeric: парсим число (запятая = десятичная точка), сравниваем как float;
  - short_text: сравнение строк без регистра/пунктуации/лишних пробелов;
  - free_form и ответы-URL: авто-оценке не поддаются -> "needs_judge".

Это нижняя граница качества для смоук-проверок; финальная метрика — судья.

Запуск:
  python -m mla_baseline.eval --results results/b0_no_tools_v2_cot.jsonl \\
      --tasks data/validation.jsonl [--meta data/validation.meta.jsonl]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _norm_choice(s: str) -> str | None:
    m = re.search(r"[A-Ea-e]", s.strip())
    return m.group(0).upper() if m else None


def _norm_number(s: str) -> float | None:
    s = s.strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    try:
        return float(m.group(0)) if m else None
    except ValueError:
        return None


def _norm_text(s: str) -> str:
    return re.sub(r"[\W_]+", " ", s.casefold(), flags=re.UNICODE).strip()


def match(pred: str, ref: str, answer_type: str) -> bool | None:
    """True/False — сравнимо; None — нужен судья."""
    if ref.startswith("http") or answer_type == "free_form":
        return None
    if answer_type == "choice":
        return _norm_choice(pred) == _norm_choice(ref)
    if answer_type == "numeric":
        p, r = _norm_number(pred), _norm_number(ref)
        return p is not None and r is not None and abs(p - r) < 1e-6
    return _norm_text(pred) == _norm_text(ref)


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "    -"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--by", default="subject",
                    help="срез: subject | grade | поле из meta (type, class, "
                         "question_format, question_type)")
    ap.add_argument("--dump-misses", type=Path, default=None,
                    help="выгрузить ошибочные ответы в JSONL для разбора")
    args = ap.parse_args()

    tasks = {t["task_id"]: t for t in _load(args.tasks)}
    meta = {m["task_id"]: m for m in _load(args.meta)} if args.meta else {}
    results = _load(args.results)

    total = len(results)
    errors = sum(1 for r in results if r.get("error") not in (None, "parse_error"))
    parse_errors = sum(1 for r in results if r.get("error") == "parse_error")

    scored: list[tuple[dict, bool | None]] = []
    for r in results:
        t = tasks.get(r["task_id"])
        if t is None or not r.get("final_answer"):
            scored.append((r, False if t else None))
            continue
        # эталон — картинка (a5xxx / URL): авто-сравнение невозможно
        if meta.get(r["task_id"], {}).get("answer_is_url"):
            scored.append((r, None))
            continue
        scored.append((r, match(r["final_answer"], t["reference_answer"], t["answer_type"])))

    judged = [(r, m) for r, m in scored if m is not None]
    correct = sum(1 for _, m in judged if m)
    needs_judge = sum(1 for _, m in scored if m is None)

    print(f"Всего результатов:  {total}")
    print(f"Ошибки вызова:      {errors}")
    print(f"Ошибки парсинга:    {parse_errors}")
    print(f"Нужен судья:        {needs_judge} (free_form / ответ-URL)")
    print(f"Авто-оценено:       {len(judged)}")
    print(f"Верно (exact):      {correct}  ({_pct(correct, len(judged))})")

    if judged:
        def key(r: dict) -> str:
            if args.by in ("subject", "grade"):
                v = tasks.get(r["task_id"], {}).get(args.by)
            else:
                v = meta.get(r["task_id"], {}).get(args.by)
            return str(v) if v not in (None, "") else "(нет)"

        groups: dict[str, list[bool]] = defaultdict(list)
        for r, m in judged:
            groups[key(r)].append(m)
        print(f"\nПо срезу «{args.by}»:")
        for g in sorted(groups, key=lambda g: -len(groups[g])):
            ms = groups[g]
            print(f"  {g:<35} {sum(ms):>4}/{len(ms):<4} {_pct(sum(ms), len(ms))}")

    if args.dump_misses:
        with args.dump_misses.open("w", encoding="utf-8") as f:
            for r, m in scored:
                if m is False:
                    t = tasks.get(r["task_id"], {})
                    f.write(json.dumps({
                        "task_id": r["task_id"],
                        "expected": t.get("reference_answer"),
                        "got": r.get("final_answer"),
                        "answer_type": t.get("answer_type"),
                        "error": r.get("error"),
                    }, ensure_ascii=False) + "\n")
        print(f"\nПромахи выгружены: {args.dump_misses}")


if __name__ == "__main__":
    main()
