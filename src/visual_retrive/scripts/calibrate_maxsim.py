"""Calibrate MaxSim retrieval: hit/recall@K (± grade/subject) + comparison chart."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from visual_retrive.maxsim_index import DEFAULT_MAXSIM_INDEX_DIR, MaxSimPageIndex
from visual_retrive.paths import CATALOG_DIR, VISUAL_RETRIVE_DIR

DEFAULT_KS = (1, 5, 10, 20, 50, 100, 200, 500, 1000)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def _plot(
    *,
    out_png: Path,
    ks: list[int],
    recall_nofilt: dict[int, float],
    recall_meta: dict[int, float],
    hit_nofilt: dict[int, float],
    pooled_path: Path | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(
        ks,
        [recall_nofilt[k] for k in ks],
        marker="o",
        label="MaxSim recall@K (no filter)",
    )
    ax.plot(
        ks,
        [recall_meta[k] for k in ks],
        marker="s",
        label="MaxSim recall@K (grade+subject)",
    )
    ax.plot(
        ks,
        [hit_nofilt.get(k, float("nan")) for k in ks],
        marker="^",
        linestyle="--",
        label="MaxSim hit@K (no filter)",
    )

    if pooled_path and pooled_path.is_file():
        pooled = json.loads(pooled_path.read_text(encoding="utf-8"))
        p_rec = (
            pooled.get("recall_by_k_meta")
            or (pooled.get("with_grade_subject_filter") or {}).get("recall_by_k")
            or pooled.get("recall_by_k_grade_subject")
        )
        if isinstance(p_rec, dict) and p_rec:
            p_ks = sorted(int(k) for k in p_rec)
            ax.plot(
                p_ks,
                [float(p_rec[str(k)]) for k in p_ks],
                marker="x",
                alpha=0.7,
                label="Pooled recall@K (grade+subject)",
            )

    ax.axhline(0.9, color="gray", linestyle=":", linewidth=1, label="target recall 0.90")
    ax.set_xscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("ColModern MaxSim calibration (val)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dir", type=Path, default=DEFAULT_MAXSIM_INDEX_DIR)
    p.add_argument("--val", type=Path, default=CATALOG_DIR / "train_splits" / "val.jsonl")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--ks",
        default=",".join(str(k) for k in DEFAULT_KS),
        help="comma-separated K values",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/visual_rag_eval/maxsim_calibration"),
    )
    p.add_argument(
        "--pooled-summary",
        type=Path,
        default=Path("reports/visual_rag_eval/pooled_first_stage_calibration/summary.json"),
    )
    args = p.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    max_k = max(ks)

    rows = [
        r
        for r in _read_jsonl(args.val)
        if r.get("useful") and r.get("query") and r.get("positive_page_id")
    ]
    rng = random.Random(args.seed)
    if args.limit and len(rows) > args.limit:
        rows = rng.sample(rows, args.limit)

    print(f"[maxsim] loading {args.index_dir}", flush=True)
    index = MaxSimPageIndex.load(args.index_dir, load_model=True)
    page_to_i = {p.page_id: i for i, p in enumerate(index.pages)}
    print(
        f"[maxsim] pages={len(index.pages)} packing GPU corpus…",
        flush=True,
    )
    t_pack = time.time()
    index.prepare_gpu_corpus()
    print(f"[maxsim] corpus packed in {time.time() - t_pack:.1f}s", flush=True)

    ranks_nf: list[int] = []
    ranks_meta: list[int] = []
    missing = 0
    per_query: list[dict[str, Any]] = []
    t0 = time.time()

    for qi, row in enumerate(rows, start=1):
        gold = str(row["positive_page_id"])
        gi = page_to_i.get(gold)
        if gi is None:
            missing += 1
            continue
        q_emb = index.encode_query(str(row["query"]))
        scores = index.score_all(q_emb)
        order = np.argsort(-scores)
        rank_nf = int(np.where(order == gi)[0][0]) + 1
        ranks_nf.append(rank_nf)

        subject, grade = row.get("subject"), row.get("grade")
        cand = index._candidate_ids(subject=subject, grade=grade, candidate_ids=None)
        if gi in cand:
            scores_m = index.score_all(q_emb, candidate_ids=cand)
            # map local argmax order to global ids
            local_order = np.argsort(-scores_m)
            local_pos = {cid: j for j, cid in enumerate(cand)}
            rank_m = int(np.where(local_order == local_pos[gi])[0][0]) + 1
        else:
            rank_m = 10**9
        ranks_meta.append(rank_m)

        per_query.append(
            {
                "query": row["query"],
                "gold_page_id": gold,
                "rank_nofilt": rank_nf,
                "rank_meta": rank_m if rank_m < 10**9 else None,
                "score_gold": float(scores[gi]),
                "subject": subject,
                "grade": grade,
            }
        )
        if qi % 25 == 0 or qi == len(rows):
            hit1 = sum(1 for r in ranks_nf if r <= 1) / max(len(ranks_nf), 1)
            print(
                f"[maxsim] {qi}/{len(rows)} scored={len(ranks_nf)} "
                f"missing={missing} hit@1={hit1:.3f}",
                flush=True,
            )

    scored = len(ranks_nf) or 1
    recall_nf = {k: sum(1 for r in ranks_nf if r <= k) / scored for k in ks}
    recall_meta = {k: sum(1 for r in ranks_meta if r <= k) / scored for k in ks}
    hit_nf = {k: recall_nf[k] for k in ks if k <= max_k}

    summary = {
        "index": str(args.index_dir),
        "scoring": "maxsim",
        "n": len(rows),
        "seed": args.seed,
        "scored": len(ranks_nf),
        "missing_gold_in_index": missing,
        "ks": ks,
        "hit_rate": {str(k): hit_nf[k] for k in (1, 5, 10) if k in hit_nf},
        "recall_by_k_nofilt": {str(k): recall_nf[k] for k in ks},
        "recall_by_k_meta": {str(k): recall_meta[k] for k in ks},
        "gold_rank_nofilt": {
            "median": _percentile([float(x) for x in ranks_nf], 50),
            "p90": _percentile([float(x) for x in ranks_nf], 90),
            "mean": float(np.mean(ranks_nf)) if ranks_nf else None,
            "max": max(ranks_nf) if ranks_nf else None,
        },
        "gold_rank_meta": {
            "median": _percentile([float(x) for x in ranks_meta if x < 10**9], 50),
            "p90": _percentile([float(x) for x in ranks_meta if x < 10**9], 90),
            "mean": float(np.mean([x for x in ranks_meta if x < 10**9]))
            if any(x < 10**9 for x in ranks_meta)
            else None,
            "max": max((x for x in ranks_meta if x < 10**9), default=None),
        },
        "seconds": round(time.time() - t0, 2),
        "recommendation": {
            "prefer": "maxsim",
            "require_grade_subject_filter": recall_meta.get(5, 0) - recall_nf.get(5, 0)
            > 0.05,
            "agent_top_k": 5 if recall_meta.get(5, 0) >= 0.55 else 10,
            "note": (
                "ColModern was trained with MaxSim; pooled cosine is only a first-stage "
                "candidate generator (see pooled_first_stage_calibration)."
            ),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    png = args.out_dir / "recall_at_k.png"
    # Normalize pooled summary keys if present
    pooled_summary = args.pooled_summary
    if pooled_summary.is_file():
        raw = json.loads(pooled_summary.read_text(encoding="utf-8"))
        if "recall_by_k_meta" not in raw and "recall_grade_subject" in raw:
            raw["recall_by_k_meta"] = {
                str(r["k"]): r["recall"] for r in raw["recall_grade_subject"]
            }
            raw["ks"] = [int(r["k"]) for r in raw["recall_grade_subject"]]
            tmp = args.out_dir / "_pooled_for_plot.json"
            tmp.write_text(json.dumps(raw), encoding="utf-8")
            pooled_summary = tmp
        elif "recall_by_k_meta" not in raw:
            # try nested recommendation tables from our calibrate script
            meta_table = raw.get("recall_with_meta") or raw.get("recall_meta")
            if isinstance(meta_table, dict):
                raw["recall_by_k_meta"] = {str(k): v for k, v in meta_table.items()}
                raw["ks"] = sorted(int(k) for k in meta_table)
                tmp = args.out_dir / "_pooled_for_plot.json"
                tmp.write_text(json.dumps(raw), encoding="utf-8")
                pooled_summary = tmp

    _plot(
        out_png=png,
        ks=ks,
        recall_nofilt=recall_nf,
        recall_meta=recall_meta,
        hit_nofilt=hit_nf,
        pooled_path=pooled_summary if pooled_summary.is_file() else None,
    )

    md = [
        "# MaxSim calibration",
        "",
        f"- index: `{args.index_dir}`",
        f"- n={len(rows)}, seed={args.seed}, scored={len(ranks_nf)}, missing_gold={missing}",
        f"- seconds: {summary['seconds']}",
        "",
        "## Hit rates (no filter)",
        "",
        "```json",
        json.dumps(summary["hit_rate"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Recall@K (no filter)",
        "",
        "| K | recall |",
        "|---|--------|",
    ]
    for k in ks:
        md.append(f"| {k} | {recall_nf[k]:.4f} |")
    md += [
        "",
        "## Recall@K (grade+subject filter)",
        "",
        "| K | recall |",
        "|---|--------|",
    ]
    for k in ks:
        md.append(f"| {k} | {recall_meta[k]:.4f} |")
    md += [
        "",
        f"Chart: `{png}`",
        "",
        "## Recommendation",
        "",
        "```json",
        json.dumps(summary["recommendation"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (args.out_dir / "chart_legend_ru.md").write_text(
        "\n".join(
            [
                "# Легенда графика MaxSim",
                "",
                "- **MaxSim recall@K (no filter)** — доля запросов, где gold-страница в top-K по late-interaction MaxSim без метаданных.",
                "- **MaxSim recall@K (grade+subject)** — то же, но корпус сужен фильтром grade+subject (как в агенте).",
                "- **MaxSim hit@K** — то же, что recall при одном релевантном документе.",
                "- **Pooled recall@K** — для сравнения: mean-cosine first-stage (слабый proxy для ColModern).",
                "- Пунктир **0.90** — целевой recall для first-stage / agent top-K выбора.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[maxsim] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
