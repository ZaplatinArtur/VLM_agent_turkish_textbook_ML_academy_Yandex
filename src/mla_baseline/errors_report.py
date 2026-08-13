"""Единый отчёт по ошибкам модели: все задачи, где хоть одно условие ошиблось.

Ошибка определяется по основной метрике задачи: exact-match для
авто-оцениваемых, вердикт LLM-судьи для остальных (free_form и
эталоны-картинки с транскрипцией). Задачи без валидного эталона
(нерасшифрованные картинки) в отчёт не попадают.

Запуск:
    python -m mla_baseline.errors_report \
        --tasks data/eval/validation.jsonl --meta data/eval/validation.meta.jsonl \
        --transcripts reports/answer_transcripts.json \
        --run B0:reports/b0_full_32k.jsonl:reports/judge_out_b0.jsonl:reports/judge_out_b0_delta.jsonl \
        --run B1dr:reports/b1_deep_routed_32k.jsonl:reports/judge_out_b1dr.jsonl:reports/judge_out_b1dr_delta.jsonl \
        --out reports/errors_report.html
"""

import argparse
import base64
import html
import io
import json
from collections import defaultdict
from pathlib import Path

from .eval import match

THUMB_WIDTH = 440


def _load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["task_id"]] = row
    return out


def _load_judge(paths: list[Path]) -> dict[str, dict]:
    """base + delta; delta (транскрибированные эталоны) перекрывает base."""
    out: dict[str, dict] = {}
    for path in paths:
        if not path or not path.exists():
            continue
        for row in _load_jsonl(path).values():
            v = row.get("verdict")
            if v and v.get("score") is not None:
                out[row["task_id"]] = v
    return out


def _thumb(path: Path) -> str | None:
    try:
        from PIL import Image

        img = Image.open(path)
        img = img.convert("RGB")
        if img.width > THUMB_WIDTH:
            img = img.resize(
                (THUMB_WIDTH, round(img.height * THUMB_WIDTH / img.width)))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=60)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _category(task: dict, m: dict) -> str:
    qt = (m.get("question_type") or "").strip()
    if qt == "single-choice question" or task["answer_type"] == "choice":
        return "закрытый"
    if qt == "open question (precise answer)" or task["answer_type"] in ("numeric", "short_text"):
        return "открытый точный"
    return "открытый произвольный"


def collect(tasks_path: Path, meta_path: Path, transcripts_path: Path | None,
            runs: list[tuple[str, Path, list[Path]]], data_root: Path) -> dict:
    tasks = _load_jsonl(tasks_path)
    meta = _load_jsonl(meta_path)
    transcripts = set()
    if transcripts_path and transcripts_path.exists():
        transcripts = set(json.load(transcripts_path.open(encoding="utf-8")))

    loaded = [(name, _load_jsonl(res), _load_judge(judges)) for name, res, judges in runs]

    cards = []
    for tid, task in tasks.items():
        m = meta.get(tid, {})
        no_reference = m.get("answer_is_url") and tid not in transcripts
        if no_reference:
            continue  # эталон-картинка не расшифрована — судить не по чему
        auto = not m.get("answer_is_url") and task["answer_type"] != "free_form"

        per_run = []
        any_wrong = False
        for name, results, judge in loaded:
            r = results.get(tid) or {}
            answer = r.get("final_answer") or ""
            exact_ok = (bool(match(answer, task["reference_answer"], task["answer_type"]))
                        if auto and not r.get("error") else None)
            verdict = judge.get(tid)
            judge_ok = bool(verdict["score"]) if verdict else None
            primary_ok = exact_ok if auto else judge_ok
            if primary_ok is False:
                any_wrong = True
            per_run.append({
                "name": name, "answer": answer,
                "exact_ok": exact_ok, "judge_ok": judge_ok, "primary_ok": primary_ok,
                "rationale": (verdict or {}).get("rationale") or "",
                "steps": (r.get("solution_steps") or r.get("reasoning") or "")[:1200],
                "forced": bool(r.get("forced_answer")),
                "searched": bool(r.get("tool_calls")),
                "error": r.get("error"),
            })
        if not any_wrong:
            continue

        images = [data_root / ref["data"] for ref in task.get("question_images", [])
                  if ref.get("format") == "file_path"]
        cards.append({
            "task_id": tid, "subject": task["subject"], "grade": task.get("grade"),
            "category": _category(task, m), "auto": auto,
            "reference": task["reference_answer"],
            "images": images, "runs": per_run,
        })

    return {"cards": cards, "n_tasks": len(tasks),
            "n_scored": sum(1 for tid in tasks
                            if not (meta.get(tid, {}).get("answer_is_url")
                                    and tid not in transcripts)),
            "run_names": [name for name, *_ in runs]}


_CSS = """
:root { color-scheme: light dark; }
.rep { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --ring:rgba(11,11,11,.10);
  --accent:#2a78d6; --good:#0ca30c; --bad:#d03b3b; --badbg:rgba(208,59,59,.08); }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .rep {
  --surface:#1a1a19; --page:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --ring:rgba(255,255,255,.10); --accent:#3987e5;
  --good:#27b027; --bad:#e05252; --badbg:rgba(224,82,82,.10); } }
:root[data-theme="dark"] .rep { --surface:#1a1a19; --page:#0d0d0d; --ink:#fff;
  --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --ring:rgba(255,255,255,.10);
  --accent:#3987e5; --good:#27b027; --bad:#e05252; --badbg:rgba(224,82,82,.10); }
.rep { background:var(--page); color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; margin:0; padding:24px 16px; }
.rep .wrap { max-width:1060px; margin:0 auto; }
.rep h1 { font-size:20px; margin:0 0 4px; }
.rep .sub { color:var(--ink2); margin:0 0 16px; }
.rep .card { background:var(--surface); border:1px solid var(--ring);
  border-radius:10px; padding:14px 16px; margin-bottom:14px; }
.rep .filters { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
.rep select { background:var(--surface); color:var(--ink); border:1px solid var(--grid);
  border-radius:6px; padding:5px 8px; font:inherit; }
.rep .head { display:flex; flex-wrap:wrap; gap:8px 14px; align-items:baseline; }
.rep .tid { font-weight:650; }
.rep .tag { color:var(--ink2); font-size:12px; }
.rep .ref { margin:6px 0; }
.rep .ref b { color:var(--good); }
.rep img { max-width:100%; border-radius:6px; border:1px solid var(--grid); margin:8px 0 4px; }
.rep .run { border-top:1px solid var(--grid); margin-top:8px; padding-top:8px; }
.rep .run.wrong { background:var(--badbg); margin-left:-16px; margin-right:-16px;
  padding-left:16px; padding-right:16px; }
.rep .vv { font-variant-numeric:tabular-nums; }
.rep .ok { color:var(--good); } .rep .bad { color:var(--bad); font-weight:650; }
.rep .rat { color:var(--ink2); font-size:13px; margin:2px 0 0; }
.rep details { margin-top:4px; } .rep summary { cursor:pointer; color:var(--muted); font-size:12px; }
.rep pre { white-space:pre-wrap; font-size:12px; color:var(--ink2);
  background:var(--page); border-radius:6px; padding:8px; overflow-x:auto; }
.rep table { border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }
.rep th { text-align:left; color:var(--ink2); padding:4px 14px 4px 0;
  border-bottom:1px solid var(--grid); }
.rep td { padding:4px 14px 4px 0; border-bottom:1px solid var(--grid); }
"""

_JS = """
function applyFilters() {
  const s = document.getElementById('f-subj').value,
        c = document.getElementById('f-cat').value,
        w = document.getElementById('f-who').value;
  document.querySelectorAll('.err-card').forEach(el => {
    const okS = s === 'all' || el.dataset.subject === s,
          okC = c === 'all' || el.dataset.category === c,
          okW = w === 'all' || el.dataset.who === w || (w === 'both' && el.dataset.who === 'both');
    el.style.display = (okS && okC && okW) ? '' : 'none';
  });
  let n = 0;
  document.querySelectorAll('.err-card').forEach(el => { if (el.style.display !== 'none') n++; });
  document.getElementById('f-count').textContent = n;
}
"""


def generate(data: dict, out_path: Path) -> Path:
    cards = data["cards"]
    names = data["run_names"]

    by_subject = defaultdict(lambda: defaultdict(int))
    who_counts = defaultdict(int)
    for c in cards:
        wrongs = [r["name"] for r in c["runs"] if r["primary_ok"] is False]
        who = "both" if len(wrongs) == len(names) else wrongs[0]
        c["who"] = who
        who_counts[who] += 1
        for r in c["runs"]:
            if r["primary_ok"] is False:
                by_subject[c["subject"]][r["name"]] += 1
        by_subject[c["subject"]]["n"] += 1

    subjects = sorted({c["subject"] for c in cards})
    cats = sorted({c["category"] for c in cards})

    p: list[str] = []
    p.append(f"<meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>")
    p.append(f"<title>Ошибки модели: {' vs '.join(names)}</title>")
    p.append(f"<style>{_CSS}</style><script>{_JS}</script>")
    p.append("<div class='rep'><div class='wrap'>")
    p.append(f"<h1>Ошибки модели: {' и '.join(names)}</h1>")
    both = who_counts.get("both", 0)
    p.append(f"<p class='sub'>задач с валидным эталоном: {data['n_scored']} из {data['n_tasks']} · "
             f"задач с ошибкой хотя бы одного условия: {len(cards)} · обе ошиблись: {both} "
             f"(устойчивые пробелы модели) · метрика: exact-match для авто-задач, LLM-судья для остальных</p>")

    p.append("<div class='card'><table><tr><th>Предмет</th><th>ошибок всего</th>"
             + "".join(f"<th>{html.escape(n)}</th>" for n in names) + "</tr>")
    for s, row in sorted(by_subject.items(), key=lambda kv: -kv[1]["n"]):
        p.append(f"<tr><td>{html.escape(s)}</td><td>{row['n']}</td>"
                 + "".join(f"<td>{row.get(n, 0)}</td>" for n in names) + "</tr>")
    p.append("</table></div>")

    opt = lambda vals: "".join(f"<option>{html.escape(v)}</option>" for v in vals)
    p.append("<div class='filters'>"
             f"<select id='f-subj' onchange='applyFilters()'><option value='all'>все предметы</option>{opt(subjects)}</select>"
             f"<select id='f-cat' onchange='applyFilters()'><option value='all'>все категории</option>{opt(cats)}</select>"
             "<select id='f-who' onchange='applyFilters()'><option value='all'>любая ошибка</option>"
             "<option value='both'>обе ошиблись</option>"
             + "".join(f"<option>{html.escape(n)}</option>" for n in names)
             + f"</select><span class='tag' style='align-self:center'>показано: <span id='f-count'>{len(cards)}</span></span></div>")

    order = {"both": 0}
    for c in sorted(cards, key=lambda c: (order.get(c["who"], 1), c["subject"], c["task_id"])):
        p.append(f"<div class='card err-card' data-subject='{html.escape(c['subject'])}' "
                 f"data-category='{html.escape(c['category'])}' data-who='{html.escape(c['who'])}'>")
        badge = "обе ошиблись" if c["who"] == "both" else f"ошиблась {c['who']}"
        p.append(f"<div class='head'><span class='tid'>{c['task_id']}</span>"
                 f"<span class='tag'>{html.escape(c['subject'])} · {c['grade'] or '?'} кл · "
                 f"{c['category']} · <b class='bad'>{badge}</b></span></div>")
        for img in c["images"]:
            data_uri = _thumb(img)
            if data_uri:
                p.append(f"<img src='{data_uri}' alt='{html.escape(img.name)}' loading='lazy'>")
        p.append(f"<div class='ref'>Эталон: <b>{html.escape(str(c['reference'])[:200])}</b></div>")
        for r in c["runs"]:
            cls = "run wrong" if r["primary_ok"] is False else "run"
            marks = []
            if r["exact_ok"] is not None:
                marks.append(f"exact <span class='{'ok' if r['exact_ok'] else 'bad'}'>{'✓' if r['exact_ok'] else '✗'}</span>")
            if r["judge_ok"] is not None:
                marks.append(f"судья <span class='{'ok' if r['judge_ok'] else 'bad'}'>{'✓' if r['judge_ok'] else '✗'}</span>")
            extra = ("".join([" · forced" if r["forced"] else "", " · искала" if r["searched"] else ""]))
            p.append(f"<div class='{cls}'><b>{html.escape(r['name'])}</b>: "
                     f"<span class='vv'>{html.escape((r['answer'] or '—')[:200])}</span> "
                     f"<span class='tag'>{' · '.join(marks)}{extra}</span>")
            if r["rationale"] and r["primary_ok"] is False:
                p.append(f"<p class='rat'>Судья: {html.escape(r['rationale'][:300])}</p>")
            if r["steps"]:
                p.append(f"<details><summary>решение</summary><pre>{html.escape(r['steps'])}</pre></details>")
            p.append("</div>")
        p.append("</div>")

    p.append("</div></div>")
    out_path.write_text("\n".join(p), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Единый отчёт по ошибкам модели")
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path, default=None)
    ap.add_argument("--run", action="append", required=True,
                    help="ИМЯ:results.jsonl[:judge.jsonl[:judge_delta.jsonl]]")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("reports/errors_report.html"))
    ap.add_argument("--digest", type=Path, default=None,
                    help="доп. JSON-дайджест ошибок для анализа")
    args = ap.parse_args(argv)

    runs = []
    for spec in args.run:
        parts = spec.split(":")
        name, res = parts[0], Path(parts[1])
        judges = [Path(x) for x in parts[2:]]
        runs.append((name, res, judges))

    data = collect(args.tasks, args.meta, args.transcripts, runs, args.data_root)
    out = generate(data, args.out)
    print(f"Ошибок: {len(data['cards'])} из {data['n_scored']} задач -> {out}")

    if args.digest:
        digest = [{k: c[k] for k in ("task_id", "subject", "grade", "category", "auto",
                                     "reference", "who")}
                  | {"runs": [{k: r[k] for k in ("name", "answer", "exact_ok", "judge_ok",
                                                 "primary_ok", "rationale", "forced",
                                                 "searched")} for r in c["runs"]]}
                  for c in data["cards"]]
        args.digest.write_text(json.dumps(digest, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        print(f"Дайджест: {args.digest}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
