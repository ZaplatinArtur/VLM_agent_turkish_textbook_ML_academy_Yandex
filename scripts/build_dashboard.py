# -*- coding: utf-8 -*-
"""Собирает интерактивный HTML-отчёт по прогонам бенч-300.

Читает сырые SolveResult-JSONL, сводит каждую задачу в одну строку
(бенчмарк, язык, предмет, тип вопроса, модальность, верно ли B0, верно ли b1,
искал ли агент), пакует в колонки и подставляет в шаблон.

    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --results reports/mlbench300 --conditions b0,b1

Выход: reports/mlbench300_dashboard.html — один самодостаточный файл,
открывается в браузере без сервера. Шаблон — reports/dashboard_template.html
(вся вёрстка и графики там; данные подставляются вместо __DATA__).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mla_baseline.eval import match  # noqa: E402

# Буква варианта по ПОЗИЦИИ: болгарские экзамены нумеруют варианты А/Б/В/Г,
# модель отвечает латиницей — кириллическая В это третий вариант, то есть C.
_CHOICE_MAP = str.maketrans({
    "А": "A", "Б": "B", "В": "C", "Г": "D", "Д": "E",
    "а": "A", "б": "B", "в": "C", "г": "D", "д": "E", "Γ": "D",
})

BENCH = ["tumlu", "exams-v", "mgsm"]
BENCH_RU = ["TUMLU", "EXAMS-V", "MGSM"]
PREFIX = {"t300": "tumlu", "e300": "exams-v", "mgsm": "mgsm"}
# какой каталог задач соответствует бенчмарку: у TUMLU и EXAMS-V прогоны идут
# по канон-срезу, MGSM берётся целиком (250 задач на язык)
TASK_DIR = {"tumlu": "bench300/tumlu", "exams-v": "bench300/exams-v", "mgsm": "mgsm"}

LANG_RU = {
    "azerbaijani": "азербайджанский", "crimean-tatar": "крымскотатарский",
    "karakalpak": "каракалпакский", "kazakh": "казахский", "kyrgyz": "киргизский",
    "tatar": "татарский", "turkish": "турецкий", "uyghur": "уйгурский",
    "uzbek": "узбекский", "arabic": "арабский", "bulgarian": "болгарский",
    "chinese": "китайский", "croatian": "хорватский", "english": "английский",
    "french": "французский", "german": "немецкий", "hungarian": "венгерский",
    "italian": "итальянский", "polish": "польский", "serbian": "сербский",
    "slovakian": "словацкий", "spanish": "испанский",
    "en": "английский", "de": "немецкий", "fr": "французский", "es": "испанский",
    "ru": "русский", "zh": "китайский", "ja": "японский", "th": "тайский",
    "sw": "суахили", "bn": "бенгальский", "te": "телугу",
}
SUBJ_RU = {
    "Native L&L": "родной язык и лит-ра", "Biology": "биология",
    "Chemistry": "химия", "Maths": "математика", "Physics": "физика",
    "History": "история", "Geography": "география", "Logic": "логика",
    "Human and Society": "человек и общество",
    "Religion and Ethics": "религия и этика", "Philosophy": "философия",
    "Kyrgyz Literature": "киргизская лит-ра", "Kyrgyz language": "киргизский язык",
    "Natural Science": "естественные науки", "Social Sciences": "общественные науки",
    "Other": "прочее",
}


def load(path: Path) -> dict:
    return {r["task_id"]: r for r in
            (json.loads(l) for l in path.open(encoding="utf-8") if l.strip())}


def norm_choice(ans) -> str:
    return (ans or "").strip().translate(_CHOICE_MAP).upper()[:1]


def correct(res: dict, task: dict) -> bool:
    at = task.get("answer_type") or "choice"
    if at == "choice":
        return norm_choice(res.get("final_answer")) == norm_choice(task["reference_answer"])
    return bool(match(res.get("final_answer") or "", task["reference_answer"], at))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="reports/mlbench300")
    parser.add_argument("--conditions", default="b0,b1",
                        help="базовое,сравниваемое — как в именах файлов")
    parser.add_argument("--out", default="reports/mlbench300_dashboard.html")
    args = parser.parse_args()
    base, alt = args.conditions.split(",")

    tasks: dict[str, dict] = {}
    for bench, sub in TASK_DIR.items():
        for p in (ROOT / "data/mlbench" / sub).glob("*.jsonl"):
            for tid, r in load(p).items():
                r["_bench"] = bench
                r["subject"] = (r.get("subject") or "unknown").strip()
                tasks[tid] = r

    runs: dict[tuple, dict] = {}
    for p in sorted((ROOT / args.results).glob("*.jsonl")):
        prefix, rest = p.stem.split("_", 1)
        if prefix not in PREFIX or "_" not in rest:
            continue
        lang, cond = rest.rsplit("_", 1)
        runs[(PREFIX[prefix], lang, cond)] = load(p)

    rows = []
    for (bench, lang, cond) in sorted(runs):
        if cond != base:
            continue
        r_alt = runs.get((bench, lang, alt))
        if not r_alt:
            print(f"пропущен {bench}/{lang}: нет условия {alt}")
            continue
        for tid, r0 in runs[(bench, lang, base)].items():
            r1, t = r_alt.get(tid), tasks.get(tid)
            # задача идёт в сводку, только если оба условия отработали её без
            # ошибки — иначе сравнение шло бы по разным наборам
            if t is None or r1 is None or r0.get("error") or r1.get("error"):
                continue
            rows.append({
                "bench": bench, "lang": lang, "subject": t["subject"],
                "qtype": 0 if (t.get("answer_type") or "choice") == "choice" else 1,
                "mod": 1 if t.get("question_images") else 0,
                "b0": int(correct(r0, t)), "b1": int(correct(r1, t)),
                "sw": 1 if r1.get("tool_calls") else 0,
            })
    if not rows:
        print("нечего сводить: не найдено пар прогонов")
        return 1

    langs, subjs, lang_bench = [], [], {}
    for r in rows:
        if r["lang"] not in lang_bench:
            langs.append(r["lang"])
        lang_bench[r["lang"]] = BENCH_RU[BENCH.index(r["bench"])]
        if r["subject"] not in subjs:
            subjs.append(r["subject"])
    ru_l = lambda l: LANG_RU.get(l, l)  # noqa: E731
    langs.sort(key=ru_l)
    subjs.sort(key=lambda s: SUBJ_RU.get(s, s))
    # один язык встречается в двух бенчах (english в EXAMS-V и en в MGSM):
    # без пометки такие срезы слились бы в одну строку графика
    dupes = {n for n in (ru_l(l) for l in langs)
             if sum(1 for x in langs if ru_l(x) == n) > 1}
    labels = [ru_l(l) + (f" · {lang_bench[l]}" if ru_l(l) in dupes else "")
              for l in langs]
    li = {l: i for i, l in enumerate(langs)}
    si = {s: i for i, s in enumerate(subjs)}

    cols = {"b": [], "l": [], "s": [], "q": [], "m": [], "a": [], "c": [], "w": []}
    for r in rows:
        cols["b"].append(BENCH.index(r["bench"]))
        cols["l"].append(li[r["lang"]])
        cols["s"].append(si[r["subject"]])
        cols["q"].append(r["qtype"])
        cols["m"].append(r["mod"])
        cols["a"].append(r["b0"])
        cols["c"].append(r["b1"])
        cols["w"].append(r["sw"])
    # однозначные колонки едут строкой цифр — файл втрое легче
    packed = {k: ("".join(map(str, v)) if max(v) < 10 else v) for k, v in cols.items()}

    data = {"bench": BENCH_RU, "langs": labels,
            "subj": [SUBJ_RU.get(s, s) for s in subjs], "cols": packed}
    tpl = (ROOT / "reports/dashboard_template.html").read_text(encoding="utf-8")
    out = ROOT / args.out
    out.write_text(
        tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False,
                                           separators=(",", ":"))),
        encoding="utf-8")
    ok0 = sum(r["b0"] for r in rows) / len(rows)
    ok1 = sum(r["b1"] for r in rows) / len(rows)
    print(f"{len(rows)} задач, {len(langs)} языковых срезов, {len(subjs)} предметов")
    print(f"{base} {ok0:.1%}  {alt} {ok1:.1%}  Δ {(ok1 - ok0) * 100:+.1f} пп")
    print(f"записано: {out.relative_to(ROOT)} ({out.stat().st_size // 1024} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
