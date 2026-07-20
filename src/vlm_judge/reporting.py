from __future__ import annotations

import html
from pathlib import Path
from typing import Any


SETUP_LABELS = {
    "no_tools": "Без инструментов",
    "web_search": "Веб-поиск",
    "textbook_retrieval": "Поиск по учебникам",
}
SETUP_ORDER = ("no_tools", "web_search", "textbook_retrieval")


def _ordered_setups(summary: dict[str, Any]) -> list[str]:
    available = summary.get("by_setup", {})
    return [value for value in SETUP_ORDER if value in available] + sorted(
        value for value in available if value not in SETUP_ORDER
    )


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _number(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _interval(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "—"
    return f"[{_number(value[0])}, {_number(value[1])}]"


def _setup_cards(summary: dict[str, Any]) -> str:
    cards = []
    for setup in _ordered_setups(summary):
        metrics = summary["by_setup"][setup]
        accuracy = metrics.get("strict_accuracy")
        width = max(0.0, min(100.0, float(accuracy or 0) * 100))
        cards.append(
            f"""
            <article class="metric-card">
              <div class="metric-label">{html.escape(SETUP_LABELS.get(setup, setup))}</div>
              <div class="metric-value">{_percent(accuracy)}</div>
              <div class="bar" aria-label="Accuracy {_percent(accuracy)}"><span style="width:{width:.2f}%"></span></div>
              <div class="metric-context">score {_number(metrics.get('mean_score_0_4'))} / 4 · {metrics.get('scored', 0)} оценено</div>
            </article>
            """
        )
    return "".join(cards)


def _pairwise_rows(summary: dict[str, Any]) -> str:
    rows = []
    comparisons = summary.get("paired_comparisons", {})
    preferred = (
        "web_search_vs_no_tools",
        "textbook_retrieval_vs_no_tools",
        "web_search_vs_textbook_retrieval",
        "textbook_retrieval_vs_web_search",
    )
    names = [name for name in preferred if name in comparisons] + sorted(
        name for name in comparisons if name not in preferred
    )
    for name in names:
        metrics = comparisons[name]
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{metrics.get('paired_tasks', 0)}</td>"
            f"<td>{_number(metrics.get('mean_score_delta'))}</td>"
            f"<td>{_interval(metrics.get('mean_score_delta_ci95'))}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="4">Нет парных сравнений</td></tr>'


def _answer_type_rows(summary: dict[str, Any]) -> str:
    setups = _ordered_setups(summary)
    rows = []
    for answer_type, values in summary.get("by_answer_type", {}).items():
        cells = "".join(
            f"<td>{_percent((values.get(setup) or {}).get('strict_accuracy'))}</td>"
            for setup in setups
        )
        rows.append(f"<tr><td>{html.escape(answer_type)}</td>{cells}</tr>")
    return "".join(rows), setups


def render_experiment_report(summary: dict[str, Any], output_path: Path) -> None:
    answer_rows, setups = _answer_type_rows(summary)
    synthetic = bool(summary.get("synthetic_smoke_test"))
    disclaimer = (
        '<div class="notice"><strong>Синтетический dry run.</strong> '
        "Числа проверяют механику пайплайна и не являются результатом исследования.</div>"
        if synthetic
        else ""
    )
    setup_headers = "".join(
        f"<th>{html.escape(SETUP_LABELS.get(setup, setup))}</th>" for setup in setups
    )
    document = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VLM experiment report</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f4f5f7; --surface:#fff; --text:#17191e; --muted:#69717d; --line:#dce1e8; --accent:#315efb; --notice:#fff4d6; --notice-text:#604700; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,system-ui,sans-serif; background:var(--bg); color:var(--text); }}
    main {{ width:min(1180px,100%); margin:auto; padding:32px 20px 72px; }}
    h1 {{ margin:0 0 6px; font-size:clamp(26px,4vw,42px); }}
    h2 {{ margin:34px 0 12px; font-size:20px; }}
    .subtitle,.metric-context {{ color:var(--muted); }}
    .notice {{ margin:20px 0; padding:14px 16px; border:1px solid #d9b650; border-radius:12px; background:var(--notice); color:var(--notice-text); }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin-top:22px; }}
    .metric-card {{ padding:18px; border:1px solid var(--line); border-radius:14px; background:var(--surface); }}
    .metric-label {{ color:var(--muted); font-size:13px; }}
    .metric-value {{ font-size:34px; font-weight:650; margin:7px 0 12px; }}
    .metric-context {{ font-size:12px; margin-top:9px; }}
    .bar {{ height:7px; background:var(--line); border-radius:999px; overflow:hidden; }}
    .bar span {{ display:block; height:100%; background:var(--accent); }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:14px; background:var(--surface); }}
    table {{ width:100%; border-collapse:collapse; min-width:640px; }}
    th,td {{ padding:12px 14px; text-align:left; border-bottom:1px solid var(--line); }}
    th {{ color:var(--muted); font-size:12px; font-weight:600; }}
    tr:last-child td {{ border-bottom:0; }}
    @media (prefers-color-scheme:dark) {{ :root {{ --bg:#111318; --surface:#191c22; --text:#f1f3f6; --muted:#a4acb8; --line:#303640; --accent:#7d9cff; --notice:#392f18; --notice-text:#ffe5a0; }} }}
  </style>
</head>
<body><main>
  <h1>VLM experiment report</h1>
  <div class="subtitle">Hybrid accuracy: exact metrics where possible, blinded judge otherwise</div>
  {disclaimer}
  <section class="metrics" aria-label="Результаты по сетапам">{_setup_cards(summary)}</section>
  <h2>Парные различия</h2>
  <div class="table-wrap"><table><thead><tr><th>Сравнение</th><th>Задач</th><th>Δ score</th><th>95% CI</th></tr></thead><tbody>{_pairwise_rows(summary)}</tbody></table></div>
  <h2>Accuracy по типу ответа</h2>
  <div class="table-wrap"><table><thead><tr><th>Тип</th>{setup_headers}</tr></thead><tbody>{answer_rows}</tbody></table></div>
</main></body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
