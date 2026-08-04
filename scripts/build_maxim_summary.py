"""Build the frozen cross-idea summary for Maksim's full-274 experiments.

The builder only aggregates post-generation score artifacts.  It never reads
answers into a generation prompt and never calls a model or a remote service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO / "reports" / "maxim_ideas_full274_20260731"
BENCHMARK = REPO / "artifacts" / "baselines" / "basic_page_rag_v1" / "validation_274.jsonl"

STANDARD_SCORES = {
    "page_rag": REPORT_DIR / "scorer_frozen_replay.json",
    "direct_rf_v2": REPORT_DIR / "direct_reasoning_first_v2" / "score.json",
    "decompose_rf_v2": REPORT_DIR / "decompose_reasoning_first_v2" / "score.json",
    "parallel8_rf_v2": REPORT_DIR / "parallel8_reasoning_first_v2" / "score.json",
    "graph253": REPORT_DIR / "graph253_v1" / "score.json",
    "no_tools_frozen": REPORT_DIR / "no_tools_control_score.json",
}
ELEMENT_RESULT = REPORT_DIR / "element_chunking_proxy_v1" / "result.json"

DISPLAY = {
    "page_rag": "Basic page-RAG (frozen)",
    "no_tools_frozen": "No-tools long reasoning (frozen)",
    "direct_rf_v2": "Direct reasoning-first v2",
    "element_proxy": "Идея 1: element-level proxy",
    "decompose_rf_v2": "Идея 2: decomposition",
    "parallel8_rf_v2": "Идея 3: 8× reasoning + judge",
    "graph253": "Идея 4: Graph253",
}
ORDER = [
    "page_rag",
    "no_tools_frozen",
    "direct_rf_v2",
    "element_proxy",
    "decompose_rf_v2",
    "parallel8_rf_v2",
    "graph253",
]
SUBJECT_ORDER = [
    "ATATÜRKÇÜLÜK",
    "Biology",
    "Chemistry",
    "English",
    "Geography",
    "History",
    "Math",
    "Philosophy",
    "Physics",
    "Science",
    "Sociology",
    "Turkish language and literature",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _accuracy(correct: int, total: int) -> float:
    return round(correct / total, 6)


def _mcnemar_exact(fixed: int, regressed: int) -> float:
    disagreements = fixed + regressed
    if not disagreements:
        return 1.0
    tail = sum(math.comb(disagreements, i) for i in range(min(fixed, regressed) + 1))
    return min(1.0, 2.0 * tail / (2**disagreements))


def _standard_row(key: str, report: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    math_slice = report["by_subject"]["Math"]
    non_math_correct = int(overall["new_correct"]) - int(math_slice["new_correct"])
    non_math_n = int(overall["n"]) - int(math_slice["n"])
    fixed = int(overall.get("fixed", 0))
    regressed = int(overall.get("regressed", 0))
    operational = report.get("operational", {})
    tokens = operational.get("tokens", {}).get("combined_tokens_total")
    calls = operational.get("model_calls", {}).get("call_count_total")
    if not calls:
        calls = None
    return {
        "id": key,
        "display": DISPLAY[key],
        "overall": {
            "correct": int(overall["new_correct"]),
            "n": int(overall["n"]),
            "accuracy": _accuracy(int(overall["new_correct"]), int(overall["n"])),
        },
        "math": {
            "correct": int(math_slice["new_correct"]),
            "n": int(math_slice["n"]),
            "accuracy": _accuracy(int(math_slice["new_correct"]), int(math_slice["n"])),
        },
        "non_math": {
            "correct": non_math_correct,
            "n": non_math_n,
            "accuracy": _accuracy(non_math_correct, non_math_n),
        },
        "vs_page": {
            "delta_correct": int(overall["new_correct"]) - 141,
            "fixed": fixed,
            "regressed": regressed,
            "mcnemar_exact_two_sided_p": _mcnemar_exact(fixed, regressed),
        },
        "combined_solver_tokens": int(tokens) if tokens is not None else None,
        "logical_model_calls": int(calls) if calls is not None else None,
    }


def _element_row(report: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]["treatment"]
    math_slice = report["by_subject"]["Math"]
    non_math_correct = int(overall["correct"]) - int(math_slice["treatment_correct"])
    non_math_n = int(overall["total"]) - int(math_slice["n"])
    return {
        "id": "element_proxy",
        "display": DISPLAY["element_proxy"],
        "overall": {
            "correct": int(overall["correct"]),
            "n": int(overall["total"]),
            "accuracy": _accuracy(int(overall["correct"]), int(overall["total"])),
        },
        "math": {
            "correct": int(math_slice["treatment_correct"]),
            "n": int(math_slice["n"]),
            "accuracy": _accuracy(int(math_slice["treatment_correct"]), int(math_slice["n"])),
        },
        "non_math": {
            "correct": non_math_correct,
            "n": non_math_n,
            "accuracy": _accuracy(non_math_correct, non_math_n),
        },
        "vs_page": {
            "delta_correct": int(report["overall"]["delta_correct"]),
            "fixed": int(report["overall"]["fixed"]),
            "regressed": int(report["overall"]["regressed"]),
            "mcnemar_exact_two_sided_p": float(
                report["overall"]["mcnemar_exact_two_sided_p"]
            ),
        },
        "combined_solver_tokens": int(report["execution"]["combined_tokens_total"]),
        "logical_model_calls": None,
    }


def _subject_counts(
    standard: dict[str, dict[str, Any]], element: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_subjects = standard["page_rag"]["by_subject"]
    assert set(page_subjects) == set(SUBJECT_ORDER)
    for subject in SUBJECT_ORDER:
        n = int(page_subjects[subject]["n"])
        counts = {
            key: int(report["by_subject"][subject]["new_correct"])
            for key, report in standard.items()
        }
        counts["element_proxy"] = int(element["by_subject"][subject]["treatment_correct"])
        rows.append({"subject": subject, "n": n, "correct": counts})
    return rows


def _artifact_hashes() -> dict[str, str]:
    relative_paths = [
        "artifacts/baselines/basic_page_rag_v1/validation_274.jsonl",
        "artifacts/baselines/basic_page_rag_v1/agent_rag_274.jsonl",
        "artifacts/baselines/basic_page_rag_v1/agent_rag_judge.jsonl",
        "artifacts/baselines/no_tools_v1/b0_no_tools_raw.jsonl",
        "scripts/score_maxim_full274.py",
        "reports/maxim_ideas_full274_20260731/element_chunking_proxy_v1/result.json",
        "reports/maxim_ideas_full274_20260731/direct_reasoning_first_v2/solver.jsonl",
        "reports/maxim_ideas_full274_20260731/decompose_reasoning_first_v2/solver.jsonl",
        "reports/maxim_ideas_full274_20260731/parallel8_reasoning_first_v2/solver.jsonl",
        "reports/maxim_ideas_full274_20260731/graph253_v1/solver.jsonl",
    ]
    return {path: _sha256(REPO / path) for path in relative_paths}


def build_summary() -> dict[str, Any]:
    standard = {key: _load(path) for key, path in STANDARD_SCORES.items()}
    element = _load(ELEMENT_RESULT)
    rows = {key: _standard_row(key, report) for key, report in standard.items()}
    rows["element_proxy"] = _element_row(element)

    for row in rows.values():
        assert row["overall"]["n"] == 274
        assert row["math"]["n"] == 139
        assert row["non_math"]["n"] == 135
        assert row["math"]["correct"] + row["non_math"]["correct"] == row["overall"]["correct"]

    # These are the two causal comparisons with a matched one-call RF-v2 control.
    matched = {
        "decompose_vs_direct": {
            "treatment_correct": 150,
            "control_correct": 137,
            "fixed": 33,
            "regressed": 20,
            "mcnemar_exact_two_sided_p": _mcnemar_exact(33, 20),
            "math": {
                "treatment_correct": 75,
                "control_correct": 65,
                "fixed": 18,
                "regressed": 8,
                "mcnemar_exact_two_sided_p": _mcnemar_exact(18, 8),
            },
            "non_math": {
                "treatment_correct": 75,
                "control_correct": 72,
                "fixed": 15,
                "regressed": 12,
                "mcnemar_exact_two_sided_p": _mcnemar_exact(15, 12),
            },
        },
        "parallel8_vs_direct": {
            "treatment_correct": 151,
            "control_correct": 137,
            "fixed": 32,
            "regressed": 18,
            "mcnemar_exact_two_sided_p": _mcnemar_exact(32, 18),
            "math": {
                "treatment_correct": 79,
                "control_correct": 65,
                "fixed": 19,
                "regressed": 5,
                "mcnemar_exact_two_sided_p": _mcnemar_exact(19, 5),
            },
            "non_math": {
                "treatment_correct": 72,
                "control_correct": 72,
                "fixed": 13,
                "regressed": 13,
                "mcnemar_exact_two_sided_p": 1.0,
            },
        },
        "parallel8_vs_decompose": {
            "treatment_correct": 151,
            "control_correct": 150,
            "fixed": 26,
            "regressed": 25,
            "mcnemar_exact_two_sided_p": 1.0,
            "token_ratio": round(2_843_423 / 647_833, 3),
        },
    }

    return {
        "schema_version": "maxim-ideas-full274-summary-v1",
        "owner": "Максим",
        "benchmark": {
            "rows": 274,
            "math_rows": 139,
            "non_math_rows": 135,
            "deterministic_rows": 177,
            "image_judge_rows": 97,
            "sha256": _sha256(BENCHMARK),
        },
        "headline": {
            "general_upgrade_found": False,
            "best_practical_idea": "decompose_rf_v2",
            "best_point_estimate_idea": "parallel8_rf_v2",
            "reason": (
                "Decomposition and parallel8 improve the 274-task point estimate only "
                "because of Math; every Maksim treatment remains below page-RAG on "
                "the 135 non-Math tasks. Parallel8 adds one answer over decomposition "
                "while using 4.39x its solver tokens."
            ),
        },
        "method_status": {
            "element_proxy": "proxy_only_not_colpali",
            "decompose_rf_v2": "full274_tested",
            "parallel8_rf_v2": "full274_tested",
            "graph253": "safe_partial_solutions_disabled",
        },
        "rows": [rows[key] for key in ORDER],
        "matched_comparisons": matched,
        "subjects": _subject_counts(standard, element),
        "limitations": [
            "All p-values are exploratory, unadjusted two-sided exact McNemar tests from one benchmark run.",
            "Math is 139/274 tasks, so overall gains can hide regressions on the 135 non-Math tasks.",
            "Element-level result is OCR/text MiniLM+HNSW; ColPali/ColQwen was unavailable and remains untested.",
            "Graph253 has solution_of edges but include_solutions=false; the unsafe full solution variant was not tested.",
            "Frozen no-tools is a strong historical long-reasoning reference, not a matched control: 16,384-token v2_cot versus 1,536-token, thinking-off RF-v2 and an older image-judge lineage.",
            "The frozen page baseline is output-locked by SHA; the exact historical retrieval index/code environment is not fully reconstructible from Git.",
            "Gold/reference fields were unavailable during generation and were read only by the post-generation scorer/judge.",
        ],
        "artifact_sha256": _artifact_hashes(),
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%".replace(".", ",")


def _p(value: float) -> str:
    if value < 0.0001:
        return f"{value:.2e}".replace(".", ",")
    return f"{value:.4f}".replace(".", ",")


def _score(cell: dict[str, Any]) -> str:
    return f"{cell['correct']}/{cell['n']} ({_pct(cell['accuracy'])})"


def render_markdown(summary: dict[str, Any]) -> str:
    by_id = {row["id"]: row for row in summary["rows"]}
    lines = [
        "# Идеи Максима на общем benchmark-274",
        "",
        "## Короткий вывод",
        "",
        "**Общего апгрейда пока нет.** Декомпозиция и 8 параллельных рассуждений "
        "поднимают aggregate относительно frozen page-RAG, но весь чистый выигрыш "
        "приходит из Math. На 135 non-Math задачах все четыре идеи Максима ниже page-RAG.",
        "",
        "Практический кандидат — **decomposition**: 150/274 против 151/274 у 8×+judge, "
        "но parallel8 расходует в 4,39 раза больше solver-токенов. Parallel8 даёт "
        "исследовательский сигнал только как Math-route; его Math 79/139 против "
        "page-RAG 62/139 (`p=0,0115`, raw exact McNemar).",
        "",
        "## Единая таблица",
        "",
        "| Условие | Статус | Overall | Δ к page | Math | Non-Math | Solver tokens | raw p vs page |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    statuses = {
        "page_rag": "frozen control",
        "no_tools_frozen": "historical quality ref*",
        "direct_rf_v2": "matched agent control",
        "element_proxy": "proxy; не ColPali†",
        "decompose_rf_v2": "full274",
        "parallel8_rf_v2": "full274",
        "graph253": "safe partial‡",
    }
    for key in ORDER:
        row = by_id[key]
        delta = row["vs_page"]["delta_correct"]
        delta_text = "—" if key == "page_rag" else f"{delta:+d}"
        p_text = "—" if key == "page_rag" else _p(row["vs_page"]["mcnemar_exact_two_sided_p"])
        tokens = row["combined_solver_tokens"]
        token_text = f"{tokens:,}".replace(",", " ") if tokens is not None else "н/д"
        lines.append(
            f"| {row['display']} | {statuses[key]} | {_score(row['overall'])} | "
            f"{delta_text} | {_score(row['math'])} | {_score(row['non_math'])} | "
            f"{token_text} | {p_text} |"
        )
    lines.extend(
        [
            "",
            "`*` No-tools 191/274 — historical long-reasoning reference, а не matched control: "
            "v2_cot с лимитом 16 384 токена против 1 536 у direct RF-v2; image judge также другой версии. "
            "Преимущество остаётся на judge-независимой deterministic-ветке: 132/177.",
            "",
            "`†` Проверена основа гипотезы — чанки `theory/exercise/worked_example/solution` "
            "и text dense retrieval MiniLM/HNSW. Настоящий ColPali не был доступен и не измерен.",
            "",
            "`‡` Граф настоящий (212 317 узлов, 157 390 рёбер, 253 книги), но решения "
            "не выдавались агенту (`include_solutions=false`), поэтому полная версия идеи с решениями не проверена.",
            "",
            "## Что означает результат",
            "",
            "1. **Element-level proxy — минус.** 123/274, `−18` к page-RAG; 22 исправления и 40 регрессий, "
            "raw `p=0,0300`. Положительного предметного среза нет.",
            "2. **Decomposition — лучший практический кандидат.** 150/274; против matched direct "
            "137/274 это `+13`, 33/20, `p=0,0984`. На non-Math 75/135 против page 79/135.",
            "3. **8×+judge — дорогой Math-кандидат.** 151/274; против matched direct `+14`, "
            "`p=0,0649`; на Math `+14`, `p=0,00661`. Против decomposition всего `+1`, "
            "26/25, `p=1,0`, при 4,39× токенов.",
            "4. **Graph253 always-on — минус.** 129/274, `−12` к page. Retrieval принимал любой "
            "непустой результат без relevance threshold; 21,45% возвращённых слотов повторялись, "
            "а solution/example coverage была только 6,83%/17,36%.",
            "",
            "## По предметам: число правильных",
            "",
            "Малые предметные срезы только диагностические; поправка на множественные сравнения не применялась.",
            "",
            "| Предмет | n | Page | Elements | Decompose | Parallel8 | Graph253 | No-tools* |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["subjects"]:
        c = item["correct"]
        lines.append(
            f"| {item['subject']} | {item['n']} | {c['page_rag']} | {c['element_proxy']} | "
            f"{c['decompose_rf_v2']} | {c['parallel8_rf_v2']} | {c['graph253']} | "
            f"{c['no_tools_frozen']} |"
        )
    lines.extend(
        [
            "",
            "## Почему retrieval пока не помогает",
            "",
            "- Сильный historical no-tools показывает, что модель часто решает задачу лучше без внешнего контекста; "
            "нерелевантный учебный фрагмент закрепляет раннюю ошибку чтения изображения.",
            "- В Graph253 не было evidence gate: непустая выдача принималась даже при низком score; grade отсутствовал "
            "в 57,92% вызовов, а top-score слабо разделял правильные и неправильные ответы.",
            "- Граф преимущественно добавляет теорию. Связанных разобранных примеров и безопасных решений мало.",
            "- 170/274 Graph253-ответов пришлось восстанавливать fallback-парсером; они имели 38,82% accuracy против "
            "60,58% у штатного structured output. Эффект retrieval смешан с проблемой формата.",
            "- В parallel8 восемь кандидатов создают полезное разнообразие, но selector остаётся узким местом: "
            "добавление кандидатов почти не обгоняет двухшаговую decomposition.",
            "",
            "## Что делать следующим",
            "",
            "1. Не менять frozen page-RAG; он остаётся контрольной точкой с SHA.",
            "2. Сначала сделать **matched long-reasoning no-tools** под тем же backend/judge и бюджетом: это проверит, "
            "нужен ли retrieval вообще, без смешения prompt и compute.",
            "3. Продолжать с decomposition как дешёвым агентным кандидатом. Parallel8 оставить для Math-only held-out "
            "репликации или сократить до 2–3 действительно разных кандидатов плюс более сильный verifier.",
            "4. Следующий RAG должен быть gated: no-tools candidate сохраняется по умолчанию, retrieval включается только "
            "при независимом evidence/relevance signal; обязательны grade, cross-call dedup и полные graph paths в trace.",
            "5. ColPali оценивать отдельным pilot после установки и построения визуального индекса; текущие 123/274 "
            "нельзя приписывать ColPali.",
            "",
            "## Протокол и ограничения",
            "",
            f"Benchmark: 274 фиксированных task_id, SHA-256 `{summary['benchmark']['sha256']}`; "
            "177 deterministic + 97 image-judge. Gold/reference был доступен только post-generation scorer/judge.",
            "",
            "Все p-value — exploratory raw exact McNemar из одного прогона, без multiplicity correction. "
            "Frozen page-RAG закреплён на уровне выходов и SHA; точная историческая retrieval-среда не полностью "
            "восстанавливается из Git. Answer-first v1 сохранён как диагностическая абляция, а основная агентная "
            "таблица использует единый reasoning-first v2 контракт.",
            "",
            "Основные SHA и машинно-читаемые значения находятся в [SUMMARY.json](SUMMARY.json). "
            "Детали каждого прогона — в соседних каталогах `element_chunking_proxy_v1`, "
            "`decompose_reasoning_first_v2`, `parallel8_reasoning_first_v2` и `graph253_v1`.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write SUMMARY.json and SUMMARY.md")
    parser.add_argument("--check", action="store_true", help="verify existing summaries byte-for-byte")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = build_summary()
    json_text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown = render_markdown(summary)
    json_path = REPORT_DIR / "SUMMARY.json"
    md_path = REPORT_DIR / "SUMMARY.md"
    if args.check:
        assert json_path.read_text(encoding="utf-8") == json_text
        assert md_path.read_text(encoding="utf-8") == markdown
        print("summary_check=ok")
        return 0
    if args.write:
        json_path.write_text(json_text, encoding="utf-8", newline="\n")
        md_path.write_text(markdown, encoding="utf-8", newline="\n")
        print(json_path)
        print(md_path)
        return 0
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
