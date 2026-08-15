"""Offline ColModern retrieval eval + error taxonomy + fix recommendations.

Runs against train_splits/val.jsonl (query → positive_page_id). Optionally
samples failures for an LLM-as-judge relevance check when --judge-base-url
is set.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from visual_retrive.maxsim_index import DEFAULT_MAXSIM_INDEX_DIR
from visual_retrive.paths import CATALOG_DIR, VISUAL_RETRIVE_DIR
from visual_retrive.search import get_page_index


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _hit_at_k(ranked_ids: list[str], gold: str, k: int) -> bool:
    return gold in ranked_ids[:k]


def _ndcg_at_k(ranked_ids: list[str], gold: str, k: int) -> float:
    for i, pid in enumerate(ranked_ids[:k]):
        if pid == gold:
            return 1.0 / math.log2(i + 2)
    return 0.0


def _classify_error(
    *,
    gold: str,
    ranked: list[dict[str, Any]],
    gold_meta: dict[str, Any] | None,
) -> str:
    if not ranked:
        return "empty_results"
    top_ids = [str(h["page_id"]) for h in ranked]
    if gold in top_ids[:1]:
        return "ok_hit1"
    if gold in top_ids[:5]:
        return "ok_hit5_miss1"
    if gold in top_ids[:10]:
        return "ok_hit10_miss5"
    # subject/grade confusion on top-1
    top = ranked[0]
    if gold_meta:
        if str(top.get("subject")) != str(gold_meta.get("subject")):
            return "subject_mismatch_top1"
        if str(top.get("grade")) != str(gold_meta.get("grade")):
            return "grade_mismatch_top1"
        if str(top.get("book_slug")) == str(gold_meta.get("book_slug")):
            return "same_book_near_miss"
    return "irrelevant_top10"


def _fix_recommendations(counts: Counter[str], metrics: dict[str, Any]) -> list[str]:
    tips: list[str] = []
    n = max(int(metrics.get("n") or 1), 1)
    hit1 = float(metrics.get("hit_rate@1") or 0)
    if hit1 < 0.55:
        tips.append(
            "Hit@1 düşük: pooled mean-vector proxy MaxSim'i zayıflatıyor olabilir. "
            "Agent'ta top_k=20 FAISS + MaxSim rerank (multi-vector) ekle; "
            "veya query-side augmentation token sayısını eğitimle hizala."
        )
    if counts.get("subject_mismatch_top1", 0) / n > 0.08:
        tips.append(
            "Subject mismatch sık: agent tool'unda subject/grade filtresini "
            "zorunlu tut (task metadata'dan) ve catalog subject etiketlerini normalize et."
        )
    if counts.get("same_book_near_miss", 0) / n > 0.1:
        tips.append(
            "Same-book near-miss: hard-negative mining'i aynı kitaptan komşu sayfalarla "
            "güçlendir; training'de page±1/±2 negatif oranını artır."
        )
    if counts.get("grade_mismatch_top1", 0) / n > 0.05:
        tips.append(
            "Grade mismatch: retrieval sonrası grade re-rank / hard filter; "
            "index'e grade embedding bias veya metadata filter default=on."
        )
    if counts.get("empty_results", 0) > 0:
        tips.append("Empty results: index eksik veya encoder yüklenemedi — index meta'yı kontrol et.")
    if counts.get("irrelevant_top10", 0) / n > 0.25:
        tips.append(
            "Irrelevant@10 yüksek: solution-only index dışında theory pages'i de indexle "
            "(--all-pages); query generation'ı OCR/question-text'e yaklaştır; "
            "LoRA'yı maxsim loss ile daha uzun train (eval probe büyüt)."
        )
    if not tips:
        tips.append(
            "Retrieval metrikleri makul. Asıl kazanç agent tarafında: "
            "tool çıktısındaki answer_text'i kör kopyalamayı engelleyen policy + "
            "LLM-judge error buckets (wrong_number / wrong_choice / ignored_evidence)."
        )
    tips.append(
        "RAG+agent hataları için: paired_eval (b0 vs agent_visual_rag) + vlm_judge; "
        "flip analizi (retrieval_ok & answer_wrong → grounded generation bug)."
    )
    return tips


def _maybe_judge_failures(
    failures: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    max_cases: int,
) -> list[dict[str, Any]]:
    """Lightweight relevance judge: does top-1 answer_text help solve the query?"""
    try:
        from openai import OpenAI
    except ImportError:
        return [{"error": "openai package missing"}]

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    out: list[dict[str, Any]] = []
    for case in failures[:max_cases]:
        prompt = (
            "Sen bir retrieval hakemisin. Sorgu ve bulunan sayfa cevabına bak.\n"
            "JSON döndür: {\"relevant\": true|false, \"reason\": \"...\"}\n\n"
            f"SORGU: {case['query']}\n\n"
            f"GOLD_PAGE: {case['gold_page_id']}\n"
            f"TOP1_PAGE: {case['top1_page_id']}\n"
            f"TOP1_ANSWER:\n{str(case.get('top1_answer_text') or '')[:1500]}\n"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content or ""
            start = text.find("{")
            end = text.rfind("}")
            verdict = json.loads(text[start : end + 1]) if start >= 0 and end > start else {
                "raw": text
            }
        except Exception as exc:
            verdict = {"error": f"{type(exc).__name__}: {exc}"}
        out.append({**case, "judge": verdict})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dir", type=Path, default=DEFAULT_MAXSIM_INDEX_DIR)
    p.add_argument(
        "--val",
        type=Path,
        default=CATALOG_DIR / "train_splits" / "val.jsonl",
    )
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", type=Path, default=VISUAL_RETRIVE_DIR / "eval" / "visual_retrieval")
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--judge-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--judge-max", type=int, default=20)
    args = p.parse_args()

    rows = _read_jsonl(args.val)
    useful = [r for r in rows if r.get("useful") and r.get("query") and r.get("positive_page_id")]
    rng = random.Random(args.seed)
    if args.limit and len(useful) > args.limit:
        useful = rng.sample(useful, args.limit)

    print(f"[eval] loading index {args.index_dir}", flush=True)
    index = get_page_index(str(args.index_dir), load_model=True)
    if hasattr(index, "prepare_gpu_corpus"):
        index.prepare_gpu_corpus()
    print(
        f"[eval] pages={len(index.pages)} dim={index.dim} "
        f"scoring={index.meta.get('scoring')} queries={len(useful)}",
        flush=True,
    )

    metrics_sum = {"hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}
    error_counts: Counter[str] = Counter()
    by_subject: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []

    t0 = time.time()
    for i, row in enumerate(useful, start=1):
        q = str(row["query"])
        gold = str(row["positive_page_id"])
        hits = index.search(q, top_k=args.top_k)
        ids = [str(h["page_id"]) for h in hits]
        gold_meta = {
            "subject": row.get("subject"),
            "grade": row.get("grade"),
            "book_slug": row.get("book_slug"),
        }
        bucket = _classify_error(gold=gold, ranked=hits, gold_meta=gold_meta)
        error_counts[bucket] += 1
        by_subject[str(row.get("subject") or "?")][bucket] += 1

        m = {
            "hit@1": float(_hit_at_k(ids, gold, 1)),
            "hit@5": float(_hit_at_k(ids, gold, 5)),
            "hit@10": float(_hit_at_k(ids, gold, 10)),
            "ndcg@5": _ndcg_at_k(ids, gold, 5),
            "ndcg@10": _ndcg_at_k(ids, gold, 10),
        }
        for k, v in m.items():
            metrics_sum[k] += v

        rec = {
            "query": q,
            "gold_page_id": gold,
            "ranked_page_ids": ids,
            "top1_page_id": ids[0] if ids else None,
            "top1_score": hits[0]["score"] if hits else None,
            "top1_answer_text": hits[0].get("answer_text") if hits else None,
            "error_bucket": bucket,
            "subject": row.get("subject"),
            "grade": row.get("grade"),
            **m,
        }
        per_query.append(rec)
        if bucket.startswith("ok_hit1"):
            pass
        else:
            failures.append(rec)
        if i % 50 == 0 or i == len(useful):
            print(f"[eval] {i}/{len(useful)}  hit@1_so_far={metrics_sum['hit@1']/i:.3f}", flush=True)

    n = len(useful) or 1
    metrics = {f"hit_rate@{k}": metrics_sum[f"hit@{k}"] / n for k in (1, 5, 10)}
    metrics.update({f"ndcg@{k}": metrics_sum[f"ndcg@{k}"] / n for k in (5, 10)})
    metrics["n"] = len(useful)
    metrics["seconds"] = round(time.time() - t0, 2)
    metrics["index"] = str(args.index_dir)
    metrics["adapter"] = index.meta.get("adapter")

    judge_rows: list[dict[str, Any]] = []
    if args.judge_base_url:
        hard = [f for f in failures if f["error_bucket"] in {
            "irrelevant_top10", "subject_mismatch_top1", "same_book_near_miss"
        }]
        judge_rows = _maybe_judge_failures(
            hard,
            base_url=args.judge_base_url,
            model=args.judge_model,
            max_cases=args.judge_max,
        )

    tips = _fix_recommendations(error_counts, metrics)
    report = {
        "metrics": metrics,
        "error_buckets": dict(error_counts),
        "error_buckets_by_subject": {
            s: dict(c) for s, c in sorted(by_subject.items(), key=lambda kv: -sum(kv[1].values()))[:20]
        },
        "fix_recommendations": tips,
        "judge_sample_n": len(judge_rows),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if judge_rows:
        with (args.out_dir / "judge_failures.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for row in judge_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = [
        f"# Visual retrieval eval (ColModern {index.meta.get('scoring', 'pooled')})",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Error buckets",
        "",
        "```json",
        json.dumps(dict(error_counts), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Fix recommendations",
        "",
    ]
    for tip in tips:
        md.append(f"- {tip}")
    md.append("")
    (args.out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[eval] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
