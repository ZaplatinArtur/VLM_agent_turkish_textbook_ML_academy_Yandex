"""Калибровка порога уверенности для визуального ретрива.

Порог визуального профиля нельзя взять из реестра: шкала MaxSim — сумма
максимумов косинуса по токенам запроса, она зависит от индекса, и дефолтные
0.57 пропустили бы вообще любую выдачу. Поэтому VisualSearchClient без порога
не создаётся, а порог снимается здесь.

Метод тот же, что у текстовых профилей (retrieve.compare.calibrate_gate):
гоняем свои и чужие запросы, берём top-1 score каждого и ищем порог, который
их разделяет.

    python -m retrieve.calibrate_visual
    python -m retrieve.calibrate_visual --transport http --url http://localhost:8780
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from paths import EVAL_DIR

from .compare import calibrate_gate

DEFAULT_QUERIES_FILE = EVAL_DIR / "eval_queries_tr.txt"
DEFAULT_OOD_FILE = EVAL_DIR / "eval_queries_ood_tr.txt"
DEFAULT_OUTPUT = Path("configs/visual_threshold.json")


def load_queries(path: Path, limit: int | None = None) -> list[str]:
    queries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not queries:
        raise ValueError(f"нет запросов в {path}")
    return queries[:limit] if limit else queries


def top_scores(client: Any, queries: list[str], *, top_k: int = 5) -> list[float]:
    """Top-1 score каждого запроса; пустые выдачи пропускаем."""
    scores: list[float] = []
    for number, query in enumerate(queries, 1):
        try:
            result = client.search(query, top_k=top_k)
        except Exception as exc:
            print(f"  [{number}/{len(queries)}] пропущен: {exc}", file=sys.stderr)
            continue
        hits = result.get("hits") or []
        if hits and hits[0].get("score") is not None:
            scores.append(float(hits[0]["score"]))
    return scores


def build_client(transport: str, url: str | None, index_dir: str | None) -> Any:
    from mla_baseline.tools.visual_search import VisualSearchClient

    # Порог -inf: на калибровке гейт должен пропускать всё, мы как раз
    # измеряем сырые счета, чтобы его и получить.
    return VisualSearchClient(
        transport=transport,  # type: ignore[arg-type]
        url=url,
        index_dir=index_dir,
        min_score=float("-inf"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", default="inprocess",
                       choices=("inprocess", "http"))
    parser.add_argument("--url", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_FILE)
    parser.add_argument("--ood-queries", type=Path, default=DEFAULT_OOD_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    in_queries = load_queries(args.queries, args.limit)
    ood_queries = load_queries(args.ood_queries, args.limit)
    client = build_client(args.transport, args.url, args.index_dir)

    print(f"свои запросы: {len(in_queries)}")
    in_domain = top_scores(client, in_queries, top_k=args.top_k)
    print(f"чужие запросы: {len(ood_queries)}")
    out_domain = top_scores(client, ood_queries, top_k=args.top_k)

    if not in_domain or not out_domain:
        print("нечего калибровать: обе выборки должны быть непустыми",
              file=sys.stderr)
        return 1

    report = calibrate_gate(in_domain, out_domain)
    report["profile"] = "visual_colqwen25_cascade"
    report["in_domain_queries"] = len(in_domain)
    report["out_domain_queries"] = len(out_domain)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    for key in ("threshold", "accuracy", "in_domain_kept", "out_domain_leaked",
                "in_domain_min", "in_domain_median",
                "out_domain_median", "out_domain_max", "separable"):
        print(f"  {key:20} {report[key]}")
    print(f"\nотчёт: {args.output}")
    print(f"\nMLA_VISUAL_MIN_SCORE={report['threshold']}")
    if not report["separable"]:
        # Пересечение выборок значит, что порогом их не разделить: гейт будет
        # ошибаться в обе стороны, и это надо знать до прогона, а не после.
        print("\nвнимание: свои и чужие запросы по счёту пересекаются, "
              "порог разделяет их лишь частично", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
