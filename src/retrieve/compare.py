"""A/B профилей ретрива без размеченного qrels.

Гоняет один набор запросов через несколько профилей и печатает скорость,
согласие с эталоном и распределение score; --calibrate подбирает по нему порог
для confidence.assess_relevance.

Эталон по умолчанию — пулинг всех систем плюс кросс-энкодер. Это silver, а не
разметка: годится сравнивать системы между собой, но не как абсолютная
accuracy. Требует GPU, иначе --reference none.

    python -m retrieve.compare --systems e5-small rrf_e5-small_bm25 \
        --index-root data/cache/index --output reports/ab.json
    python -m retrieve.compare --systems <профиль> --reference none --calibrate
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterator

from paths import EVAL_DIR

from schemas.retrieve import RetrievedChunk

from .evaluate import _mean, _percentile, load_corpus
from .index import Index
from .pipelines import PROFILES, build_profile

DEFAULT_QUERIES_FILE = EVAL_DIR / "eval_queries_tr.txt"
DEFAULT_OOD_FILE = EVAL_DIR / "eval_queries_ood_tr.txt"

FALLBACK_QUERIES = (
    "dikdörtgenin alanı nasıl hesaplanır",
    "kesirlerde toplama işlemi nasıl yapılır",
    "üçgenin iç açıları toplamı kaç derecedir",
    "maddenin hâl değişimi buharlaşma ve yoğuşma",
    "hücre ve organellerinin görevleri",
    "Osmanlı Devleti'nin kuruluşu",
    "Kurtuluş Savaşı cepheleri",
    "noktalama işaretlerinin kullanımı",
)

METRIC_LABELS = (
    ("agree_at_k", "agree@k"),
    ("recall_ref_at_k", "recall_ref@k"),
    ("ndcg_ref_at_k", "ndcg_ref@k"),
    ("mrr_ref_at_k", "mrr_ref@k"),
    ("top1_match", "top1_match"),
)


# --------------------------------------------------------------------------- метрики


def silver_metrics(ranked: list[str], reference: list[str], k: int) -> dict[str, float]:
    """Сравнивает выдачу системы с silver-эталоном (упорядоченный список id).

    Знаменатель min(k, |R|), веса для nDCG линейные по позиции в эталоне.
    """
    top = ranked[:k]
    if not reference or not top:
        return {name: 0.0 for name, _ in METRIC_LABELS}

    relevance = {chunk_id: len(reference) - position for position, chunk_id in enumerate(reference)}
    denominator = min(k, len(reference))
    head = set(reference[:k])

    gains = [relevance.get(chunk_id, 0) for chunk_id in top]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1) if gain)
    ideal_gains = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    first_hit = next((rank for rank, cid in enumerate(top, start=1) if cid in relevance), None)

    return {
        "agree_at_k": len(head.intersection(top)) / denominator,
        "recall_ref_at_k": len([c for c in top if c in relevance]) / denominator,
        "ndcg_ref_at_k": dcg / idcg if idcg else 0.0,
        "mrr_ref_at_k": 1.0 / first_hit if first_hit else 0.0,
        "top1_match": float(top[0] == reference[0]),
    }


def score_stats(scores: list[float], k: int) -> dict[str, float]:
    if not scores:
        return {"score_top1": 0.0, "score_at_k": 0.0, "score_margin": 0.0}
    return {
        "score_top1": scores[0],
        "score_at_k": scores[min(k, len(scores)) - 1],
        "score_margin": scores[0] - scores[1] if len(scores) > 1 else 0.0,
    }


# --------------------------------------------------------------------------- прогон


def iter_rankers(node: Any) -> Iterator[Any]:
    """Обходит ранкеры пайплайна вглубь: fusion прячет свои ранкеры внутри."""
    for ranker in getattr(node, "rankers", []) or []:
        yield ranker
        yield from iter_rankers(ranker)


def build_everything(pipeline) -> None:
    """Строит все индексы заранее, иначе сборка попадёт в замер латентности."""
    for ranker in iter_rankers(pipeline):
        build = getattr(ranker, "build", None)
        if callable(build):
            build()


def dense_components(pipeline) -> list[dict[str, Any]]:
    components = []
    for ranker in iter_rankers(pipeline):
        store = getattr(ranker, "_store", None)
        if store is None:
            continue
        components.append({
            "embedder": getattr(ranker, "_embedder_name", "?"),
            "dim": store.dim,
            "vectors": store.size,
            "index_type": store.index_type,
            "vector_mb": round(store.size * store.dim * 4 / 1e6, 1),
        })
    return components


def run_system(
        profile: str,
        index: Index,
        queries: list[str],
        *,
        depth: int,
        index_root: Path | None,
        fetch_k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    pipeline = build_profile(profile, index, index_root=index_root, fetch_k=fetch_k)
    build_everything(pipeline)
    build_seconds = time.perf_counter() - started

    # Прогрев: первый вызов тянет модель запроса и прогревает FAISS.
    pipeline.run(queries[0], k=1)

    ranked: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        hits = pipeline.run(query, k=depth)
        latencies.append((time.perf_counter() - started) * 1000)
        ranked[query] = [hit.chunk_id for hit in hits]
        scores[query] = [hit.score for hit in hits]
    return {
        "profile": profile,
        "build_seconds": round(build_seconds, 1),
        "dense": dense_components(pipeline),
        "latency_ms": {
            "median": round(statistics.median(latencies), 1),
            "p95": round(_percentile(latencies, 0.95), 1),
        },
        "ranked": ranked,
        "scores": scores,
    }


def pooled_reference(
        queries: list[str],
        runs: dict[str, dict[str, Any]],
        index: Index,
        *,
        pool_depth: int,
        ref_k: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    """Silver-эталон пулингом: (топ-ref_k по запросам, score кросс-энкодера)."""
    from .rankers import CrossEncoderRanker

    # top_n покрывает весь пул, иначе его хвост останется непереранжированным.
    reranker = CrossEncoderRanker(top_n=len(runs) * pool_depth + ref_k)
    reference: dict[str, list[str]] = {}
    ranked_scores: dict[str, dict[str, float]] = {}
    total_pairs = 0
    for number, query in enumerate(queries, start=1):
        pool: dict[str, RetrievedChunk] = {}
        for run in runs.values():
            for chunk_id in run["ranked"][query][:pool_depth]:
                if chunk_id not in pool:
                    chunk = index.get_by_id(chunk_id)
                    if chunk is not None:
                        pool[chunk_id] = chunk
        total_pairs += len(pool)
        hits = reranker.rank(query, list(pool.values()))
        reference[query] = [hit.chunk_id for hit in hits[:ref_k]]
        ranked_scores[query] = {hit.chunk_id: hit.score for hit in hits[:ref_k]}
        print(f"  пул {number}/{len(queries)}: {len(pool)} кандидатов", flush=True)
    print(f"  всего пар (запрос, чанк) через кросс-энкодер: {total_pairs}")
    return reference, ranked_scores


def profile_reference(
        run: dict[str, Any],
        queries: list[str],
        ref_k: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    """Silver-эталон из выдачи одного профиля (например с кросс-энкодером)."""
    reference = {query: run["ranked"][query][:ref_k] for query in queries}
    scores = {
        query: dict(zip(run["ranked"][query][:ref_k], run["scores"][query][:ref_k]))
        for query in queries
    }
    return reference, scores


def summarize(
        run: dict[str, Any],
        queries: list[str],
        reference: dict[str, list[str]] | None,
        k: int,
) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    if reference:
        rows = [silver_metrics(run["ranked"][q], reference.get(q, []), k) for q in queries]
        metrics = {name: _mean(row[name] for row in rows) for name, _ in METRIC_LABELS}
    stats_rows = [score_stats(run["scores"][q], k) for q in queries]
    return {
        "profile": run["profile"],
        "build_seconds": run["build_seconds"],
        "latency_ms": run["latency_ms"],
        "dense": run["dense"],
        "metrics": metrics,
        "scores": {name: round(_mean(row[name] for row in stats_rows), 4)
                   for name in ("score_top1", "score_at_k", "score_margin")},
    }


def calibrate_gate(in_domain: list[float], out_domain: list[float]) -> dict[str, Any]:
    """Подбирает порог top-1 score, разделяющий свои и чужие запросы.

    Кандидаты — середины между соседними значениями; берём максимум точности,
    при равенстве — меньше просочившихся чужих.
    """
    if not in_domain or not out_domain:
        return {}
    values = sorted(set(in_domain + out_domain))
    candidates = [(a + b) / 2 for a, b in zip(values, values[1:])]
    candidates += [values[0] - 0.01, values[-1] + 0.01]
    total = len(in_domain) + len(out_domain)

    best: tuple[float, float, float] | None = None  # (accuracy, -просочилось, порог)
    for threshold in candidates:
        kept = sum(1 for value in in_domain if value >= threshold)
        leaked = sum(1 for value in out_domain if value >= threshold)
        accuracy = (kept + len(out_domain) - leaked) / total
        key = (accuracy, -leaked, threshold)
        if best is None or key > best:
            best = key
    accuracy, negative_leaked, threshold = best
    kept = sum(1 for value in in_domain if value >= threshold)
    return {
        "threshold": round(threshold, 4),
        "accuracy": accuracy,
        "in_domain_kept": kept / len(in_domain),
        "out_domain_leaked": -negative_leaked / len(out_domain),
        "in_domain_min": round(min(in_domain), 4),
        "in_domain_median": round(statistics.median(in_domain), 4),
        "out_domain_median": round(statistics.median(out_domain), 4),
        "out_domain_max": round(max(out_domain), 4),
        "separable": min(in_domain) > max(out_domain),
    }


def run_calibration(
        systems: list[str],
        runs: dict[str, dict[str, Any]],
        index: Index,
        ood_queries: list[str],
        *,
        depth: int,
        index_root: Path | None,
        fetch_k: int,
) -> dict[str, dict[str, Any]]:
    calibration: dict[str, dict[str, Any]] = {}
    for system in systems:
        ood_run = run_system(system, index, ood_queries, depth=depth,
                             index_root=index_root, fetch_k=fetch_k)
        in_domain = [scores[0] for scores in runs[system]["scores"].values() if scores]
        out_domain = [scores[0] for scores in ood_run["scores"].values() if scores]
        calibration[system] = calibrate_gate(in_domain, out_domain)
    return calibration


def _calibration_table(systems: list[str], calibration: dict[str, dict]) -> str:
    width = 26
    header = "порог гейта".ljust(width) + "".join(f"{name:>22}" for name in systems)
    lines = [header, "-" * len(header)]
    rows = (
        ("in-domain медиана", "in_domain_median", "{:.4f}"),
        ("in-domain минимум", "in_domain_min", "{:.4f}"),
        ("out-of-domain медиана", "out_domain_median", "{:.4f}"),
        ("out-of-domain максимум", "out_domain_max", "{:.4f}"),
        ("рекомендуемый min_score", "threshold", "{:.4f}"),
        ("точность разделения", "accuracy", "{:.1%}"),
        ("своих пропущено", "in_domain_kept", "{:.1%}"),
        ("чужих просочилось", "out_domain_leaked", "{:.1%}"),
    )
    for label, key, fmt in rows:
        cells = [fmt.format(calibration[s][key]) if calibration.get(s) else "—" for s in systems]
        lines.append(label.ljust(width) + "".join(f"{cell:>22}" for cell in cells))
    return "\n".join(lines)


# --------------------------------------------------------------------------- вывод


def _table(systems: list[str], summaries: dict[str, dict], k: int, with_metrics: bool) -> str:
    baseline = systems[0]
    width = max(20, *(len(label) for _, label in METRIC_LABELS))
    header = "показатель".ljust(width) + "".join(f"{name:>22}" for name in systems)
    lines = [header, "-" * len(header)]

    def row(label: str, values: list[str]) -> str:
        return label.ljust(width) + "".join(f"{value:>22}" for value in values)

    if with_metrics:
        lines.append(f"[качество vs silver-эталон, k={k}; дельты к {baseline}]")
        for name, label in METRIC_LABELS:
            cells = []
            for system in systems:
                value = summaries[system]["metrics"].get(name, 0.0) * 100
                if system == baseline:
                    cells.append(f"{value:.1f}%")
                else:
                    delta = value - summaries[baseline]["metrics"].get(name, 0.0) * 100
                    cells.append(f"{value:.1f}% ({delta:+.1f})")
            lines.append(row(label, cells))
        lines.append("-" * len(header))

    lines.append("[скорость]")
    lines.append(row("build, s", [f"{summaries[s]['build_seconds']:.1f}" for s in systems]))
    lines.append(row("latency median, ms",
                     [f"{summaries[s]['latency_ms']['median']:.1f}" for s in systems]))
    lines.append(row("latency p95, ms",
                     [f"{summaries[s]['latency_ms']['p95']:.1f}" for s in systems]))
    lines.append(row("dim", [
        "/".join(str(component["dim"]) for component in summaries[s]["dense"]) or "—"
        for s in systems
    ]))
    lines.append(row("векторы, МБ", [
        "/".join(str(component["vector_mb"]) for component in summaries[s]["dense"]) or "—"
        for s in systems
    ]))
    lines.append("-" * len(header))
    lines.append("[шкала score — сравнима только внутри профиля]")
    for name, label in (("score_top1", "score top1"),
                        ("score_at_k", f"score @{k}"),
                        ("score_margin", "margin top1-top2")):
        lines.append(row(label, [f"{summaries[s]['scores'][name]:.4f}" for s in systems]))
    return "\n".join(lines)


def _markdown(systems: list[str], summaries: dict[str, dict], k: int, with_metrics: bool) -> str:
    head = "| показатель | " + " | ".join(systems) + " |"
    sep = "|---" * (len(systems) + 1) + "|"
    lines = [head, sep]
    if with_metrics:
        for name, label in METRIC_LABELS:
            cells = [f"{summaries[s]['metrics'].get(name, 0.0) * 100:.1f}%" for s in systems]
            lines.append(f"| {label} (k={k}) | " + " | ".join(cells) + " |")
    lines.append("| build, s | " + " | ".join(f"{summaries[s]['build_seconds']:.1f}" for s in systems) + " |")
    lines.append("| latency median, ms | " + " | ".join(
        f"{summaries[s]['latency_ms']['median']:.1f}" for s in systems) + " |")
    lines.append("| latency p95, ms | " + " | ".join(
        f"{summaries[s]['latency_ms']['p95']:.1f}" for s in systems) + " |")
    lines.append("| score top1 | " + " | ".join(
        f"{summaries[s]['scores']['score_top1']:.4f}" for s in systems) + " |")
    return "\n".join(lines)


def _print_examples(
        queries: list[str],
        runs: dict[str, dict],
        reference: dict[str, list[str]] | None,
        show: int,
        index: Index,
) -> None:
    for query in queries:
        print(f"\nq: {query}")
        silver = set((reference or {}).get(query, [])[:show])
        for profile, run in runs.items():
            print(f"  [{profile}]")
            for position, chunk_id in enumerate(run["ranked"][query][:show], start=1):
                chunk = index.get_by_id(chunk_id)
                snippet = " ".join(chunk.text.split())[:70] if chunk else ""
                mark = "*" if chunk_id in silver else " "
                print(f"   {mark}{position}. {chunk_id}  {run['scores'][query][position - 1]:.4f}  {snippet}")


# --------------------------------------------------------------------------- CLI


def load_books(books: list[str] | None) -> list[RetrievedChunk]:
    """Корпус целиком либо срез из нескольких книг (A/B без GPU за минуты)."""
    if not books:
        return load_corpus(None)
    chunks: list[RetrievedChunk] = []
    for book in books:
        chunks.extend(load_corpus(book))
    return chunks


def load_queries(path: Path | None, limit: int | None) -> list[str]:
    """Читает запросы из файла (строка = запрос, '#' — комментарий).

    Явно указанный путь обязан существовать; встроенный набор — только замена
    отсутствующему файлу по умолчанию.
    """
    if path is not None:
        if not path.exists():
            raise SystemExit(f"Нет файла с запросами: {path}")
        source = path
    elif DEFAULT_QUERIES_FILE.exists():
        source = DEFAULT_QUERIES_FILE
    else:
        print(f"{DEFAULT_QUERIES_FILE} не найден — беру встроенный набор")
        source = None
    if source is None:
        queries = list(FALLBACK_QUERIES)
    else:
        lines = source.read_text(encoding="utf-8").splitlines()
        queries = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if not queries:
        raise SystemExit("Пустой список запросов")
    return queries[:limit] if limit else queries


def write_qrels(
        path: Path,
        queries: list[str],
        reference: dict[str, list[str]],
        ref_scores: dict[str, dict[str, float]],
        *,
        min_score: float,
        origin: str,
) -> tuple[int, int]:
    """Выгружает silver-эталон в формате retrieve.evaluate — как черновик."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with path.open("w", encoding="utf-8") as handle:
        for number, query in enumerate(queries, start=1):
            scores = ref_scores.get(query, {})
            relevant = [cid for cid in reference.get(query, []) if scores.get(cid, 0.0) >= min_score]
            if not relevant:
                skipped += 1
                continue
            handle.write(json.dumps({
                "query_id": f"q{number:03d}",
                "query": query,
                "subject": None,
                "relevant_chunk_ids": relevant,
                "relevant_page_ids": [],
                "silver": {
                    "reference": origin,
                    "min_score": min_score,
                    "scores": {cid: round(scores.get(cid, 0.0), 4) for cid in relevant},
                },
            }, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A/B профилей ретрива без размеченного qrels")
    parser.add_argument("--systems", nargs="+", default=["e5-small", "rrf_e5-small_bm25"],
                        choices=PROFILES, help="первый — бейзлайн для дельт")
    parser.add_argument("--queries", default=None, type=Path,
                        help=f"файл с запросами (по умолчанию {DEFAULT_QUERIES_FILE})")
    parser.add_argument("--limit", default=None, type=int, help="взять первые N запросов")
    parser.add_argument("--books", nargs="+", default=None,
                        help="прогон на срезе из этих книг (иначе весь корпус)")
    parser.add_argument("--k", type=int, default=5, help="глубина, на которой сравниваем")
    parser.add_argument("--ref-k", type=int, default=10, help="сколько чанков берём в эталон")
    parser.add_argument("--pool-depth", type=int, default=20,
                        help="сколько кандидатов берём от каждой системы в пул")
    parser.add_argument("--reference", default="pool",
                        help="pool (кросс-энкодер поверх объединения) | имя профиля | none")
    parser.add_argument("--index-root", default=None, type=Path,
                        help="каталог снапшотов dense-индексов (иначе строятся в памяти)")
    parser.add_argument("--fetch-k", type=int, default=200)
    parser.add_argument("--calibrate", action="store_true",
                        help="подобрать порог гейта по out-of-domain запросам")
    parser.add_argument("--ood-queries", default=None, type=Path,
                        help=f"файл нерелевантных запросов (по умолчанию {DEFAULT_OOD_FILE})")
    parser.add_argument("--show", type=int, default=0, help="печатать top-N по каждому запросу")
    parser.add_argument("--output", default=None, type=Path, help="JSON-отчёт (рядом ляжет .md)")
    parser.add_argument("--emit-qrels", default=None, type=Path,
                        help="выгрузить silver-эталон как qrels для retrieve.evaluate")
    parser.add_argument("--qrels-min-score", type=float, default=0.5,
                        help="порог score эталона для попадания в qrels")
    args = parser.parse_args(argv)

    queries = load_queries(args.queries, args.limit)
    corpus = load_books(args.books)
    index = Index(corpus)
    depth = max(args.k, args.ref_k, args.pool_depth)
    print(f"Корпус: {len(corpus)} чанков | запросов: {len(queries)} | системы: {args.systems}")
    print(f"k={args.k} ref_k={args.ref_k} pool_depth={args.pool_depth} эталон={args.reference}\n")

    runs: dict[str, dict[str, Any]] = {}
    for system in args.systems:
        run = run_system(system, index, queries, depth=depth,
                         index_root=args.index_root, fetch_k=args.fetch_k)
        runs[system] = run
        print(f"[{system}] build {run['build_seconds']}s | "
              f"latency median {run['latency_ms']['median']} ms", flush=True)

    reference: dict[str, list[str]] | None = None
    ref_scores: dict[str, dict[str, float]] = {}
    origin = args.reference
    if args.reference == "pool":
        print("\nЭталон: пулинг всех систем + кросс-энкодер")
        reference, ref_scores = pooled_reference(
            queries, runs, index, pool_depth=args.pool_depth, ref_k=args.ref_k
        )
    elif args.reference != "none":
        if args.reference not in PROFILES:
            raise SystemExit(f"Неизвестный эталон: {args.reference}")
        print(f"\nЭталон: профиль {args.reference}")
        print("ВНИМАНИЕ: системы, входящие в этот профиль, получают фору — "
              "для честного сравнения используйте --reference pool")
        ref_run = runs.get(args.reference) or run_system(
            args.reference, index, queries, depth=depth,
            index_root=args.index_root, fetch_k=args.fetch_k
        )
        reference, ref_scores = profile_reference(ref_run, queries, args.ref_k)

    summaries = {s: summarize(runs[s], queries, reference, args.k) for s in args.systems}
    print("\n" + _table(args.systems, summaries, args.k, with_metrics=reference is not None))
    if reference is not None:
        print("\nSilver-эталон — не разметка: он показывает, кто ближе к выдаче "
              "кросс-энкодера, а не абсолютную правильность.")

    calibration: dict[str, dict[str, Any]] = {}
    if args.calibrate:
        # --limit на OOD не переносим: берём весь файл чужих запросов.
        ood_queries = load_queries(args.ood_queries or DEFAULT_OOD_FILE, None)
        print(f"\nКалибровка гейта: {len(ood_queries)} out-of-domain запросов")
        calibration = run_calibration(
            args.systems, runs, index, ood_queries,
            depth=depth, index_root=args.index_root, fetch_k=args.fetch_k,
        )
        print("\n" + _calibration_table(args.systems, calibration))
        print("\nПорог подставляется в confidence.assess_relevance(min_score=...) "
              "и валиден только для своего профиля.")

    if args.show:
        _print_examples(queries, runs, reference, args.show, index)

    if args.emit_qrels and reference is not None:
        written, skipped = write_qrels(
            args.emit_qrels, queries, reference, ref_scores,
            min_score=args.qrels_min_score, origin=origin,
        )
        print(f"\nSilver-qrels: {args.emit_qrels} ({written} запросов, {skipped} без кандидатов "
              f"выше {args.qrels_min_score})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corpus_chunks": len(corpus),
            "queries": queries,
            "k": args.k,
            "ref_k": args.ref_k,
            "reference": origin,
            "systems": summaries,
            "calibration": calibration,
            "ranked": {s: runs[s]["ranked"] for s in args.systems},
            "silver": reference,
        }
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown = args.output.with_suffix(".md")
        markdown.write_text(
            _markdown(args.systems, summaries, args.k, reference is not None) + "\n",
            encoding="utf-8",
        )
        print(f"\nОтчёт: {args.output} и {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
