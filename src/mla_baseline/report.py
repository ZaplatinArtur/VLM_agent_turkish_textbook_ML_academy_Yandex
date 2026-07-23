"""HTML-отчёт по прогону: KPI, графики по предметам, промахи.

Самодостаточный файл (инлайн-SVG, без внешних зависимостей и CDN),
светлая/тёмная тема. Палитра — валидированный референс: статусные цвета
для долей (верно/судья/неверно/обрыв), один синий для величин.

Запуск:
  python -m mla_baseline.report --results results/b0_full_v2_cot.jsonl \\
      --tasks data/validation.jsonl [--meta data/validation.meta.jsonl] [--out ...]
или флагом --report у runner-а.
"""

import argparse
import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .eval import match

# --- палитра (референс dataviz; light / dark) --------------------------------
CSS_VARS = """
:root { color-scheme: light dark; }
.rep {
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --ring: rgba(11,11,11,.10);
  --accent: #2a78d6;
  --series-a: #2a78d6; --series-b: #eb6834;  /* категориальные слоты 1-2 */
  --good: #0ca30c; --bad: #d03b3b; --judge: #898781; --err: #ec835a;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .rep {
    --surface: #1a1a19; --page: #0d0d0d;
    --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --ring: rgba(255,255,255,.10);
    --accent: #3987e5;
    --series-a: #3987e5; --series-b: #d95926;
  }
}
:root[data-theme="dark"] .rep {
  --surface: #1a1a19; --page: #0d0d0d;
  --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --baseline: #383835; --ring: rgba(255,255,255,.10);
  --accent: #3987e5;
  --series-a: #3987e5; --series-b: #d95926;
}
"""

STYLE = CSS_VARS + """
.rep { background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 24px 16px; }
.rep .wrap { max-width: 980px; margin: 0 auto; }
.rep h1 { font-size: 20px; margin: 0 0 4px; }
.rep h2 { font-size: 15px; margin: 28px 0 10px; }
.rep .sub { color: var(--ink2); margin: 0 0 20px; }
.rep .card { background: var(--surface); border: 1px solid var(--ring);
  border-radius: 10px; padding: 16px; }
.rep .kpis { display: flex; flex-wrap: wrap; gap: 12px; }
.rep .kpi { flex: 1 1 130px; }
.rep .kpi .v { font-size: 32px; font-weight: 650; }
.rep .kpi .l { color: var(--ink2); font-size: 12px; margin-top: 2px; }
.rep .kpi .d { color: var(--muted); font-size: 12px; }
.rep svg text { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; }
.rep svg .num { font-variant-numeric: tabular-nums; }
.rep .row:hover rect.bar { opacity: .82; }
.rep .legend { display: flex; gap: 16px; flex-wrap: wrap;
  color: var(--ink2); font-size: 12px; margin: 8px 2px 0; }
.rep .legend i { display: inline-block; width: 10px; height: 10px;
  border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.rep table { border-collapse: collapse; width: 100%; font-size: 13px;
  font-variant-numeric: tabular-nums; }
.rep th { text-align: left; color: var(--ink2); font-weight: 600;
  border-bottom: 1px solid var(--baseline); padding: 6px 10px 6px 0; }
.rep td { border-bottom: 1px solid var(--grid); padding: 6px 10px 6px 0; }
.rep .overflow { overflow-x: auto; }
.rep .foot { color: var(--muted); font-size: 12px; margin-top: 24px; }
"""


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- расчёт ------------------------------------------------------------------

def compute(results, tasks, meta):
    subj = defaultdict(lambda: {"total": 0, "ok": 0, "wrong": 0, "judge": 0,
                                "err": 0, "forced": 0, "out_tok": []})
    misses, tok_all, lat_all = [], [], []
    for r in results:
        t = tasks.get(r["task_id"])
        if t is None:
            continue
        s = subj[t["subject"]]
        s["total"] += 1
        if r["usage"].get("output_tokens"):
            s["out_tok"].append(r["usage"]["output_tokens"])
            tok_all.append(r["usage"]["output_tokens"])
        if r["usage"].get("latency_s"):
            lat_all.append(r["usage"]["latency_s"])
        s["forced"] += bool(r.get("forced_answer"))
        if r.get("error"):
            s["err"] += 1
            continue
        if meta.get(r["task_id"], {}).get("answer_is_url") or t["answer_type"] == "free_form":
            s["judge"] += 1
            continue
        verdict = match(r.get("final_answer") or "", t["reference_answer"], t["answer_type"])
        if verdict:
            s["ok"] += 1
        else:
            s["wrong"] += 1
            misses.append({
                "task_id": r["task_id"], "subject": t["subject"],
                "answer_type": t["answer_type"], "expected": t["reference_answer"],
                "got": r.get("final_answer"), "forced": bool(r.get("forced_answer")),
            })
    return subj, misses, tok_all, lat_all


# --- SVG-примитивы -----------------------------------------------------------

def bar_chart_accuracy(subj) -> str:
    """Горизонтальные бары: точность по предметам (один синий, величина)."""
    rows = [(name, s) for name, s in subj.items() if s["ok"] + s["wrong"] > 0]
    rows.sort(key=lambda kv: -(kv[1]["ok"] + kv[1]["wrong"]))
    label_w, bar_w, row_h, pad_t = 220, 620, 26, 18
    height = pad_t + row_h * len(rows) + 24
    parts = [f'<svg viewBox="0 0 {label_w + bar_w + 70} {height}" role="img" '
             f'aria-label="Точность по предметам">']
    for pct in (0, 25, 50, 75, 100):
        x = label_w + bar_w * pct / 100
        parts.append(f'<line x1="{x:.0f}" y1="{pad_t - 6}" x2="{x:.0f}" '
                     f'y2="{height - 22}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{x:.0f}" y="{height - 8}" text-anchor="middle" '
                     f'class="num" fill="var(--muted)">{pct}%</text>')
    for i, (name, s) in enumerate(rows):
        auto = s["ok"] + s["wrong"]
        acc = s["ok"] / auto
        y = pad_t + i * row_h
        w = max(bar_w * acc, 2)
        parts.append('<g class="row">')
        parts.append(f'<title>{_esc(name)}: {s["ok"]}/{auto} верно '
                     f'({acc * 100:.0f}%)</title>')
        parts.append(f'<text x="{label_w - 10}" y="{y + 15}" text-anchor="end" '
                     f'fill="var(--ink)">{_esc(name)} '
                     f'<tspan fill="var(--muted)">({auto})</tspan></text>')
        parts.append(f'<rect class="bar" x="{label_w}" y="{y + 3}" width="{w:.1f}" '
                     f'height="16" rx="4" fill="var(--accent)"/>')
        parts.append(f'<rect x="{label_w}" y="{y + 3}" width="2" height="16" '
                     f'fill="var(--accent)"/>')  # плоский старт у baseline
        parts.append(f'<text x="{label_w + w + 8:.1f}" y="{y + 15}" class="num" '
                     f'fill="var(--ink2)">{acc * 100:.0f}%</text>')
        parts.append('</g>')
    parts.append(f'<line x1="{label_w}" y1="{pad_t - 6}" x2="{label_w}" '
                 f'y2="{height - 22}" stroke="var(--baseline)" stroke-width="1"/>')
    parts.append('</svg>')
    return "".join(parts)


def stacked_composition(subj) -> str:
    """Состав по предметам: верно | судья | неверно | обрывы (доли, статусы)."""
    rows = sorted(subj.items(), key=lambda kv: -kv[1]["total"])
    label_w, bar_w, row_h, pad_t = 220, 620, 26, 8
    height = pad_t + row_h * len(rows) + 8
    total_max = max(s["total"] for _, s in rows)
    segs = [("ok", "var(--good)", "верно"), ("judge", "var(--judge)", "ждёт судью"),
            ("wrong", "var(--bad)", "неверно"), ("err", "var(--err)", "обрыв")]
    parts = [f'<svg viewBox="0 0 {label_w + bar_w + 70} {height}" role="img" '
             f'aria-label="Состав результатов по предметам">']
    for i, (name, s) in enumerate(rows):
        y = pad_t + i * row_h
        x = float(label_w)
        parts.append('<g class="row">')
        tip = (f'{name}: всего {s["total"]} — верно {s["ok"]}, ждёт судью '
               f'{s["judge"]}, неверно {s["wrong"]}, обрывов {s["err"]}')
        parts.append(f'<title>{_esc(tip)}</title>')
        parts.append(f'<text x="{label_w - 10}" y="{y + 15}" text-anchor="end" '
                     f'fill="var(--ink)">{_esc(name)}</text>')
        for key, color, _label in segs:
            n = s[key]
            if not n:
                continue
            w = bar_w * n / total_max
            parts.append(f'<rect x="{x:.1f}" y="{y + 3}" width="{max(w - 2, 1):.1f}" '
                         f'height="16" rx="3" fill="{color}"/>')
            if w >= 22:
                parts.append(f'<text x="{x + w / 2:.1f}" y="{y + 15}" class="num" '
                             f'text-anchor="middle" fill="var(--surface)" '
                             f'font-weight="600">{n}</text>')
            x += w
        parts.append('</g>')
    parts.append('</svg>')
    legend = "".join(f'<span><i style="background:{c}"></i>{_esc(l)}</span>'
                     for _, c, l in segs)
    return "".join(parts) + f'<div class="legend">{legend}</div>'


def hist_tokens(tok_all) -> str:
    """Гистограмма длины ответа (output-токены): один синий, величина."""
    if not tok_all:
        return ""
    bin_w = 1024
    bins = defaultdict(int)
    for t in tok_all:
        bins[min(t // bin_w, 12)] += 1
    max_bin = max(bins)
    max_n = max(bins.values())
    col_w, gap, plot_h, pad_l = 52, 2, 150, 10
    width = pad_l + (max_bin + 1) * col_w + 10
    height = plot_h + 46
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Распределение длины ответа в токенах">']
    parts.append(f'<line x1="{pad_l}" y1="{plot_h + 10}" x2="{width - 6}" '
                 f'y2="{plot_h + 10}" stroke="var(--baseline)" stroke-width="1"/>')
    for b in range(max_bin + 1):
        n = bins.get(b, 0)
        h = (plot_h - 14) * n / max_n if n else 0
        x = pad_l + b * col_w
        y = plot_h + 10 - h
        label = f"{b}k–{b + 1}k" if b < 12 else "12k+"
        parts.append('<g class="row">')
        parts.append(f'<title>{label} токенов: {n} задач</title>')
        if n:
            parts.append(f'<rect class="bar" x="{x + gap}" y="{y:.1f}" '
                         f'width="{col_w - 2 * gap}" height="{h:.1f}" rx="4" '
                         f'fill="var(--accent)"/>')
            parts.append(f'<rect x="{x + gap}" y="{plot_h + 8:.1f}" '
                         f'width="{col_w - 2 * gap}" height="2" fill="var(--accent)"/>')
            parts.append(f'<text x="{x + col_w / 2}" y="{y - 5:.1f}" class="num" '
                         f'text-anchor="middle" fill="var(--ink2)">{n}</text>')
        parts.append(f'<text x="{x + col_w / 2}" y="{plot_h + 26}" class="num" '
                     f'text-anchor="middle" fill="var(--muted)" '
                     f'font-size="10">{label}</text>')
        parts.append('</g>')
    parts.append('</svg>')
    return "".join(parts)


# --- сравнение двух прогонов -------------------------------------------------

def grouped_accuracy(subj_a, subj_b, label_a, label_b) -> str:
    """Две серии баров на предмет: точность условия A и B."""
    names = [n for n in subj_a
             if subj_a[n]["ok"] + subj_a[n]["wrong"] > 0
             or (n in subj_b and subj_b[n]["ok"] + subj_b[n]["wrong"] > 0)]
    names.sort(key=lambda n: -(subj_a.get(n, {}).get("ok", 0)
                               + subj_a.get(n, {}).get("wrong", 0)))
    label_w, bar_w, row_h, pad_t = 220, 620, 44, 18
    height = pad_t + row_h * len(names) + 24
    parts = [f'<svg viewBox="0 0 {label_w + bar_w + 70} {height}" role="img" '
             f'aria-label="Точность по предметам: {_esc(label_a)} и {_esc(label_b)}">']
    for pct in (0, 25, 50, 75, 100):
        x = label_w + bar_w * pct / 100
        parts.append(f'<line x1="{x:.0f}" y1="{pad_t - 6}" x2="{x:.0f}" '
                     f'y2="{height - 22}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{x:.0f}" y="{height - 8}" text-anchor="middle" '
                     f'class="num" fill="var(--muted)">{pct}%</text>')
    for i, name in enumerate(names):
        y = pad_t + i * row_h
        parts.append(f'<text x="{label_w - 10}" y="{y + 20}" text-anchor="end" '
                     f'fill="var(--ink)">{_esc(name)}</text>')
        for j, (subj, color, label) in enumerate(
                ((subj_a, "var(--series-a)", label_a),
                 (subj_b, "var(--series-b)", label_b))):
            s = subj.get(name)
            auto = (s["ok"] + s["wrong"]) if s else 0
            if not auto:
                continue
            acc = s["ok"] / auto
            w = max(bar_w * acc, 2)
            by = y + 2 + j * 18
            parts.append('<g class="row">')
            parts.append(f'<title>{_esc(label)} — {_esc(name)}: {s["ok"]}/{auto} '
                         f'({acc * 100:.0f}%)</title>')
            parts.append(f'<rect class="bar" x="{label_w}" y="{by}" width="{w:.1f}" '
                         f'height="14" rx="4" fill="{color}"/>')
            parts.append(f'<rect x="{label_w}" y="{by}" width="2" height="14" fill="{color}"/>')
            parts.append(f'<text x="{label_w + w + 8:.1f}" y="{by + 11}" class="num" '
                         f'fill="var(--ink2)">{acc * 100:.0f}%</text>')
            parts.append('</g>')
    parts.append(f'<line x1="{label_w}" y1="{pad_t - 6}" x2="{label_w}" '
                 f'y2="{height - 22}" stroke="var(--baseline)" stroke-width="1"/>')
    parts.append('</svg>')
    legend = (f'<div class="legend"><span><i style="background:var(--series-a)"></i>'
              f'{_esc(label_a)}</span><span><i style="background:var(--series-b)"></i>'
              f'{_esc(label_b)}</span></div>')
    return "".join(parts) + legend


def compare_table(subj_a, subj_b, label_a, label_b) -> str:
    names = sorted(set(subj_a) | set(subj_b),
                   key=lambda n: -(subj_a.get(n, {}).get("total", 0)))
    rows = []
    for n in names:
        a, b = subj_a.get(n), subj_b.get(n)
        auto_a = (a["ok"] + a["wrong"]) if a else 0
        auto_b = (b["ok"] + b["wrong"]) if b else 0
        acc_a = a["ok"] / auto_a if auto_a else None
        acc_b = b["ok"] / auto_b if auto_b else None
        delta = ""
        if acc_a is not None and acc_b is not None:
            d = (acc_b - acc_a) * 100
            color = "var(--good)" if d > 0 else ("var(--bad)" if d < 0 else "var(--muted)")
            delta = f'<span style="color:{color}">{d:+.0f} пп</span>'
        fmt = lambda acc, ok, auto: (f'{acc * 100:.0f}% ({ok}/{auto})' if acc is not None else "—")
        rows.append(f'<tr><td>{_esc(n)}</td>'
                    f'<td>{fmt(acc_a, a["ok"] if a else 0, auto_a)}</td>'
                    f'<td>{fmt(acc_b, b["ok"] if b else 0, auto_b)}</td>'
                    f'<td>{delta}</td></tr>')
    return (f'<table><tr><th>Предмет</th><th>{_esc(label_a)}</th>'
            f'<th>{_esc(label_b)}</th><th>Δ</th></tr>' + "".join(rows) + '</table>')


def generate_compare(path_a: Path, path_b: Path, tasks_path: Path,
                     meta_path: Path | None, out_path: Path | None = None) -> Path:
    results_a, results_b = _load(path_a), _load(path_b)
    tasks = {t["task_id"]: t for t in _load(tasks_path)}
    meta = {m["task_id"]: m for m in _load(meta_path)} if meta_path else {}
    label_a = results_a[0].get("condition", path_a.stem) if results_a else path_a.stem
    label_b = results_b[0].get("condition", path_b.stem) if results_b else path_b.stem

    subj_a, _, tok_a, lat_a = compute(results_a, tasks, meta)
    subj_b, _, tok_b, lat_b = compute(results_b, tasks, meta)
    tot_a = {k: sum(s[k] for s in subj_a.values()) for k in ("total", "ok", "wrong", "err", "forced")}
    tot_b = {k: sum(s[k] for s in subj_b.values()) for k in ("total", "ok", "wrong", "err", "forced")}
    auto_a, auto_b = tot_a["ok"] + tot_a["wrong"], tot_b["ok"] + tot_b["wrong"]
    acc_a = 100 * tot_a["ok"] / auto_a if auto_a else 0
    acc_b = 100 * tot_b["ok"] / auto_b if auto_b else 0
    n_tools = sum(1 for r in results_b if r.get("tool_calls"))
    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    body = f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Сравнение: {_esc(label_a)} vs {_esc(label_b)}</title>
<style>{STYLE}</style>
<div class="rep"><div class="wrap">
<h1>Сравнение условий: {_esc(label_a)} vs {_esc(label_b)}</h1>
<p class="sub">Модель {_esc(results_a[0].get("model", "?"))} · промпт
{_esc(results_a[0].get("prompt_version", "?"))} · {now}</p>

<div class="kpis">
{kpi(f"{acc_a:.1f}%", f"точность {label_a}", f'{tot_a["ok"]}/{auto_a} авто-оцененных')}
{kpi(f"{acc_b:.1f}%", f"точность {label_b}", f'{tot_b["ok"]}/{auto_b} авто-оцененных')}
{kpi(f"{acc_b - acc_a:+.1f} пп", "разница (поиск)", "плюс — в пользу " + label_b)}
{kpi(n_tools, f"задач с поиском в {label_b}", f"из {tot_b['total']}")}
{kpi(f"{avg(lat_b) / max(avg(lat_a), 0.001):.1f}×", "латентность с поиском",
     f"{avg(lat_a):.0f}с → {avg(lat_b):.0f}с в среднем")}
</div>

<h2>Точность по предметам</h2>
<div class="card">{grouped_accuracy(subj_a, subj_b, label_a, label_b)}</div>

<h2>Таблица (авто-оцененные задачи)</h2>
<div class="card overflow">{compare_table(subj_a, subj_b, label_a, label_b)}</div>

<p class="foot">Сгенерировано mla_baseline.report · {_esc(path_a.name)} vs
{_esc(path_b.name)} · задач в наборе: {len(tasks)}</p>
</div></div>
"""
    out = out_path or path_b.with_name(f"compare_{label_a}_vs_{label_b}.html")
    out.write_text(body, encoding="utf-8")
    return out


# --- сборка ------------------------------------------------------------------

def kpi(v, label, detail="") -> str:
    d = f'<div class="d">{_esc(detail)}</div>' if detail else ""
    return (f'<div class="kpi card"><div class="v num">{_esc(v)}</div>'
            f'<div class="l">{_esc(label)}</div>{d}</div>')


def generate(results_path: Path, tasks_path: Path, meta_path: Path | None,
             out_path: Path | None = None) -> Path:
    results = _load(results_path)
    tasks = {t["task_id"]: t for t in _load(tasks_path)}
    meta = {m["task_id"]: m for m in _load(meta_path)} if meta_path else {}

    subj, misses, tok_all, lat_all = compute(results, tasks, meta)
    tot = {k: sum(s[k] for s in subj.values())
           for k in ("total", "ok", "wrong", "judge", "err", "forced")}
    auto = tot["ok"] + tot["wrong"]
    acc = f'{100 * tot["ok"] / auto:.1f}%' if auto else "—"
    first = results[0] if results else {}

    misses_rows = "".join(
        f'<tr><td>{_esc(m["task_id"])}</td><td>{_esc(m["subject"])}</td>'
        f'<td>{_esc(m["answer_type"])}</td><td>{_esc(m["expected"])}</td>'
        f'<td>{_esc(m["got"] or "—")}</td>'
        f'<td>{"да" if m["forced"] else ""}</td></tr>'
        for m in misses[:100])
    misses_note = (f"<p class='sub'>Показаны первые 100 из {len(misses)}.</p>"
                   if len(misses) > 100 else "")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    avg_lat = f"{sum(lat_all) / len(lat_all):.1f} c" if lat_all else "—"
    avg_tok = f"{sum(tok_all) // len(tok_all)}" if tok_all else "—"

    body = f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Отчёт: {_esc(first.get("condition", "?"))} / {_esc(first.get("prompt_version", "?"))}</title>
<style>{STYLE}</style>
<div class="rep"><div class="wrap">
<h1>Отчёт по прогону: {_esc(first.get("condition", "?"))}</h1>
<p class="sub">Модель {_esc(first.get("model", "?"))} · промпт
{_esc(first.get("prompt_version", "?"))} · {_esc(results_path.name)} · {now}</p>

<div class="kpis">
{kpi(tot["total"], "задач прогнано")}
{kpi(acc, "точность (exact match)", f'{tot["ok"]} из {auto} авто-оцененных')}
{kpi(tot["judge"], "ждут LLM-судью", "эталон-картинка или free form")}
{kpi(tot["forced"], "принудительный финал", "forced_answer=true")}
{kpi(tot["err"], "обрывы/ошибки")}
{kpi(avg_lat, "средняя латентность", f"средний ответ {avg_tok} ток.")}
</div>

<h2>Точность по предметам (exact match, авто-оцененные)</h2>
<div class="card">{bar_chart_accuracy(subj)}</div>

<h2>Состав результатов по предметам</h2>
<div class="card">{stacked_composition(subj)}</div>

<h2>Длина ответа (output-токены)</h2>
<div class="card">{hist_tokens(tok_all)}</div>

<h2>Промахи (для разбора)</h2>
{misses_note}
<div class="card overflow"><table>
<tr><th>task_id</th><th>предмет</th><th>тип</th><th>эталон</th><th>ответ</th><th>форс.</th></tr>
{misses_rows}
</table></div>

<p class="foot">Сгенерировано mla_baseline.report · строк результата:
{len(results)} · задач в наборе: {len(tasks)}</p>
</div></div>
"""
    out = out_path or results_path.with_suffix(".html")
    out.write_text(body, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--compare", type=Path, default=None,
                    help="второй results JSONL: сравнительный отчёт "
                         "(--results = базовое условие, --compare = новое)")
    ap.add_argument("--out", type=Path, default=None,
                    help="куда писать HTML (по умолчанию рядом с results)")
    args = ap.parse_args()
    if args.compare:
        out = generate_compare(args.results, args.compare, args.tasks, args.meta, args.out)
    else:
        out = generate(args.results, args.tasks, args.meta, args.out)
    print(f"Отчёт: {out}")


if __name__ == "__main__":
    main()
