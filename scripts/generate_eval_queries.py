"""Синтетический набор запросов для замера: поровну по предметам, три уровня сложности.

Зачем: реальный эталон на 93% математический, а корпус на 79% — нет. На таком
наборе нельзя отличить «профиль лучше вообще» от «профиль лучше на математике».
Синтетика делается одинаково для всех предметов, поэтому предметные срезы
сравнимы между собой честно.

Со страницы запрашивается сразу три запроса разной сложности:
  kolay — прямой поиск термина или определения;
  orta  — вопрос в духе задачи, своими словами;
  zor   — ситуация, в которой нужное правило не названо вовсе.
Уровень сохраняется рядом с запросом, чтобы в отчёте была разбивка по сложности.

Это только запросы. Эталон к ним строится как обычно — пулингом и разметкой
(scripts/build_qrels_llm.py), потому что одну тему объясняют медиана 9 книг и
страница-источник далеко не единственный верный ответ.

    RETRIEVE_GATE_URL=http://127.0.0.1:8000/v1 \
    python scripts/generate_eval_queries.py --per-subject 30 \
        --output data/eval/synthetic_queries.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from paths import DATA_DIR

MAX_PAGE_CHARS = 1400

# Предмет книги определяется по её slug — других меток у корпуса нет.
SUBJECTS = {
    "matematik": "математика",
    "fen-bilimleri": "естествознание",
    "turkce": "турецкий",
    "ingilizce": "английский",
    "sosyal-bilgiler": "обществознание",
    "hayat-bilgisi": "окружающий мир",
    "din-kulturu": "религия",
    "inkilap": "история",
}

LEVELS = ("kolay", "orta", "zor")

SYSTEM = (
    "You write realistic search queries for a Turkish school textbook search engine. "
    "Answer with JSON only."
)

USER = """This is one page from a Turkish school textbook ({subject}).

{page}

Write exactly three Turkish search queries that would lead a student to THIS page,
one per difficulty level:

- "kolay": a direct lookup of the term or definition the page teaches.
  May reuse the page's own wording. 3-6 words.
- "orta": phrased as a task the student is stuck on, in their own words.
  Must NOT reuse distinctive phrases from the page. 4-8 words.
- "zor": describes a concrete situation WITHOUT naming the rule, formula or
  concept at all — the reader must infer which topic is needed. Must not contain
  the key term itself. 6-12 words.

Rules for all three: Turkish only, lowercase, no numbers copied from the page,
no question marks.
If the page teaches nothing (cover, contents, blank, unreadable OCR), return
{{"queries": []}}.

Reply with JSON: {{"queries": [{{"level": "kolay", "query": "..."}},
{{"level": "orta", "query": "..."}}, {{"level": "zor", "query": "..."}}]}}"""


def ask(base_url, model, system, user, timeout, max_tokens=400):
    import urllib.request

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": max_tokens, "stream": False,
        # Без этого Qwen3.5 уводит бюджет токенов в reasoning и content пустой.
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def extract(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("нет JSON в ответе")
    return json.loads(text[start:end + 1])


def subject_of(chunk_id: str) -> str | None:
    book = chunk_id.split(":")[0]
    for marker, name in SUBJECTS.items():
        if marker in book:
            return name
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Синтетические запросы для замера")
    parser.add_argument("--per-subject", type=int, default=30, help="запросов на предмет")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="добрать только эти предметы (остальные не трогать)")
    parser.add_argument("--min-chars", type=int, default=500, help="пропускать короткие страницы")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--deadline-minutes", type=float, default=None)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "eval" / "synthetic_queries.jsonl")
    parser.add_argument("--base-url", default=os.environ.get("RETRIEVE_GATE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("RETRIEVE_GATE_MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    from retrieve.evaluation.evaluate import load_corpus

    corpus = [c for c in load_corpus(None) if len(c.text.strip()) >= args.min_chars]
    by_subject = defaultdict(list)
    for chunk in corpus:
        name = subject_of(chunk.chunk_id)
        if name:
            by_subject[name].append(chunk)
    if args.subjects:
        unknown = set(args.subjects) - set(by_subject)
        if unknown:
            raise SystemExit(f"неизвестные предметы: {sorted(unknown)}; есть {sorted(by_subject)}")
        by_subject = {k: v for k, v in by_subject.items() if k in set(args.subjects)}
    rng = random.Random(args.seed)
    for pages in by_subject.values():
        rng.shuffle(pages)
    print("страниц по предметам:", {k: len(v) for k, v in sorted(by_subject.items())}, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_pages, produced = set(), Counter()
    if args.output.exists():
        for line in args.output.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                done_pages.add(row["source_chunk_id"])
                produced[row["subject"]] += 1
        print(f"уже есть: {sum(produced.values())} запросов", flush=True)

    started = time.time()
    written = skipped = failed = 0
    with args.output.open("a", encoding="utf-8") as sink:
        # По кругу между предметами: при обрыве набор остаётся сбалансированным.
        cursors = {name: 0 for name in by_subject}
        while any(produced[name] < args.per_subject for name in by_subject):
            if args.deadline_minutes and (time.time() - started) / 60 > args.deadline_minutes:
                print("достигнут предел времени, останавливаюсь", flush=True)
                break
            progressed = False
            for name, pages in sorted(by_subject.items()):
                if produced[name] >= args.per_subject:
                    continue
                while cursors[name] < len(pages) and pages[cursors[name]].chunk_id in done_pages:
                    cursors[name] += 1
                if cursors[name] >= len(pages):
                    continue
                page = pages[cursors[name]]
                cursors[name] += 1
                progressed = True
                try:
                    payload = extract(ask(args.base_url, args.model, SYSTEM,
                                          USER.format(subject=name,
                                                      page=page.text.strip()[:MAX_PAGE_CHARS]),
                                          args.timeout))
                    items = payload.get("queries") or []
                except Exception as exc:
                    failed += 1
                    print(f"  {page.chunk_id}: {type(exc).__name__}: {str(exc)[:60]}", flush=True)
                    continue
                if not items:
                    skipped += 1
                    continue
                for item in items:
                    level = str(item.get("level", "")).strip().lower()
                    query = re.sub(r"\s+", " ", str(item.get("query", ""))).strip(" ?.")
                    if level not in LEVELS or len(query) < 8:
                        continue
                    sink.write(json.dumps({
                        "task_id": f"syn_{name}_{page.chunk_id}_{level}".replace(" ", ""),
                        "query": query,
                        "subject": name,
                        "difficulty": level,
                        "source_chunk_id": page.chunk_id,
                    }, ensure_ascii=False) + "\n")
                    written += 1
                    produced[name] += 1
                sink.flush()
                done_pages.add(page.chunk_id)
                if written and written % 30 == 0:
                    print(f"  запросов {written} | {(time.time()-started)/60:.0f} мин | "
                          f"{dict(sorted(produced.items()))}", flush=True)
            if not progressed:
                print("страницы кончились раньше квоты", flush=True)
                break

    print(f"\nготово: запросов {written}, страниц без темы {skipped}, сбоев {failed}")
    print("по предметам:", dict(sorted(produced.items())))
    print(f"файл: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
