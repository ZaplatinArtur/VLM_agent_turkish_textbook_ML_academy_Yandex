# -*- coding: utf-8 -*-
"""Проба живого SearXNG: работает ли он вообще и на наших ли запросах.

Прогон b1 нельзя запускать, не проверив бэкенд: в прошлый раз половина
условия измеряла неисправность, а не поиск. Скрипт берёт РЕАЛЬНЫЕ запросы
модели из логов (по умолчанию те, что вернули пустоту) и прогоняет их через
инстанс, показывая, что именно отвечает, а что молчит.

    python scripts/probe_searxng.py --url http://localhost:8080 [-n 40] [--all]

Код возврата 1, если доля пустых ответов выше порога (--max-empty, по
умолчанию 0.2) — годится как гейт перед запуском прогона.
"""

import argparse
import json
import random
import statistics
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_LOGS = ["reports/b1_full_32k_v2.jsonl", "reports/b1_deep_32k.jsonl",
            "reports/b1_deep_routed_32k.jsonl"]
EMPTY_PREFIXES = ("Sonuç bulunamadı", "Sayfalar açılamadı", "Arama hatası")
HDRS = {"User-Agent": "Mozilla/5.0 (compatible; mla-baseline/0.2)"}


def load_queries(only_failed: bool) -> dict[str, str]:
    """Запросы модели из логов: {запрос: прежний исход}.

    Если один запрос встречался с разным исходом, прежним считаем удачный —
    так сравнение «до/после» получается консервативным.
    """
    out: dict[str, str] = {}
    for rel in RUN_LOGS:
        path = ROOT / rel
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            for call in json.loads(line).get("tool_calls") or []:
                if call.get("tool") != "web_search":
                    continue
                query = str((call.get("args") or {}).get("query") or "").strip()
                preview = call.get("result_preview") or ""
                if not query or preview.startswith(("Bu sorguyu", "Arama limitine")):
                    continue  # до бэкенда такой вызов не доходил
                was = "empty" if preview.startswith(EMPTY_PREFIXES) else "hit"
                if out.get(query) != "hit":
                    out[query] = was
    if only_failed:
        return {q: w for q, w in out.items() if w == "empty"}
    return out


def probe_config(url: str) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/config", headers=HDRS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def probe_query(url: str, query: str, timeout: float) -> dict:
    endpoint = url.rstrip("/") + "/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "language": "tr"})
    started = time.perf_counter()
    try:
        req = urllib.request.Request(endpoint, headers=HDRS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return {"query": query, "error": type(exc).__name__,
                "ms": round((time.perf_counter() - started) * 1000)}
    results = [r for r in (data.get("results") or []) if r.get("url")]
    return {
        "query": query,
        "results": len(results),
        "engines": Counter(r.get("engine") or "?" for r in results),
        "dead": [e[0] if isinstance(e, (list, tuple)) and e else str(e)
                 for e in (data.get("unresponsive_engines") or [])],
        "ms": round((time.perf_counter() - started) * 1000),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("-n", type=int, default=30,
                        help="сколько запросов взять (0 — все)")
    parser.add_argument("--out-jsonl", default="",
                        help="куда писать результат по каждому запросу")
    parser.add_argument("--all", action="store_true",
                        help="брать все запросы, а не только провалившиеся")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--pause", type=float, default=0.5,
                        help="пауза между запросами (бурст провоцирует бан)")
    parser.add_argument("--max-empty", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"инстанс: {args.url}")
    try:
        config = probe_config(args.url)
    except Exception as exc:
        print(f"  /config недоступен: {type(exc).__name__}: {exc}")
        return 1
    print(f"  версия: {config.get('version', '?')}, "
          f"движков включено: {len(config.get('engines') or [])}")
    # /config не показывает search.formats, поэтому проверяем JSON канарейкой:
    # при выключенном json API отдаёт 403, и прогон получил бы сплошную пустоту
    canary = probe_query(args.url, "test", args.timeout)
    if canary.get("error"):
        print(f"  ОШИБКА: JSON-запрос не проходит ({canary['error']}). "
              "Проверь, что json есть в search.formats.")
        return 1
    print(f"  JSON API отвечает: {canary['results']} результатов на 'test'")

    previous = load_queries(only_failed=not args.all)
    if not previous:
        print("нет запросов в логах — укажи --all или проверь reports/")
        return 1
    queries = list(previous)
    random.Random(args.seed).shuffle(queries)
    if args.n > 0:
        queries = queries[: args.n]
    kind = "все" if args.all else "ранее провалившиеся"
    print(f"\nпробуем {len(queries)} запросов ({kind})\n")

    sink = (ROOT / args.out_jsonl).open("w", encoding="utf-8") if args.out_jsonl else None
    rows = []
    for i, query in enumerate(queries, 1):
        row = probe_query(args.url, query, args.timeout)
        row["was"] = previous[query]
        rows.append(row)
        if sink:
            sink.write(json.dumps({k: (dict(v) if isinstance(v, Counter) else v)
                                   for k, v in row.items()}, ensure_ascii=False) + "\n")
            sink.flush()
        mark = ("ERR " + row["error"]) if row.get("error") else f"{row['results']:>3} рез."
        dead = f"  мертвы: {','.join(row['dead'])}" if row.get("dead") else ""
        print(f"{i:>4}. было:{row['was']:<5} стало:{mark}  {row['ms']:>5} мс  "
              f"{query[:55]}{dead}", flush=True)
        if args.pause:
            time.sleep(args.pause)
    if sink:
        sink.close()

    empty = [r for r in rows if r.get("error") or not r.get("results")]
    engines = Counter()
    dead = Counter()
    for r in rows:
        engines.update(r.get("engines") or {})
        dead.update(r.get("dead") or [])
    latencies = [r["ms"] for r in rows]

    share = len(empty) / len(rows)
    print(f"\nпусто/ошибка: {len(empty)}/{len(rows)} ({share:.0%})")

    # сравнение с прежним исходом того же запроса
    cross = Counter((r["was"], "empty" if (r.get("error") or not r.get("results"))
                     else "hit") for r in rows)
    for was in ("empty", "hit"):
        total = sum(v for (w, _), v in cross.items() if w == was)
        if not total:
            continue
        fixed = cross[(was, "hit")]
        print(f"  было {was:<5}: {total:>4} запросов → сейчас находится "
              f"{fixed} ({fixed / total:.0%})")
    print(f"медиана задержки: {statistics.median(latencies):.0f} мс")
    print("результаты по движкам: "
          + (", ".join(f"{k} {v}" for k, v in engines.most_common(10)) or "—"))
    print("отвалившиеся движки: "
          + (", ".join(f"{k} ×{v}" for k, v in dead.most_common(10)) or "нет"))

    if share > args.max_empty:
        print(f"\nГЕЙТ НЕ ПРОЙДЕН: пустых {share:.0%} > {args.max_empty:.0%}. "
              "Прогон запускать нельзя — измерите бэкенд, а не модель.")
        return 1
    print("\nгейт пройден: бэкенд отвечает, прогон запускать можно")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
