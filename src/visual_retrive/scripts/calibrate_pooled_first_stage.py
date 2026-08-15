"""Calibrate pooled cosine first-stage recall@K and optionally plot it."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from visual_retrive.paths import CATALOG_DIR, VISUAL_RETRIVE_DIR

DEFAULT_INDEX = VISUAL_RETRIVE_DIR / "indexes" / "colmodern_v2_best_pooled"
DEFAULT_KS = (1, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000)


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


def _load_encoder(adapter: Path | None):
    from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
    from peft import PeftModel

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_name = "ModernVBERT/colmodernvbert-merged"
    processor = ColModernVBertProcessor.from_pretrained(model_name)
    model = ColModernVBert.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    if adapter is not None and Path(adapter).is_dir():
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, processor


@torch.inference_mode()
def _encode_queries(model, processor, queries: list[str]) -> np.ndarray:
    texts = [
        processor.query_prefix + q + processor.query_augmentation_token * 10
        for q in queries
    ]
    batch = processor.process_texts(texts)
    device = next(model.parameters()).device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    emb = model(**batch)
    if emb.ndim == 3:
        vec = emb.float().mean(dim=1)
    else:
        vec = emb.float()
    vec = F.normalize(vec, p=2, dim=-1)
    return vec.detach().cpu().numpy().astype(np.float32)


def _choose_k(recall_by_k: dict[int, float], *, target: float, ks: list[int]) -> dict:
    chosen = None
    for k in ks:
        if recall_by_k.get(k, 0.0) >= target:
            chosen = k
            break
    best_k = max(ks, key=lambda k: (recall_by_k.get(k, 0.0), -k))
    return {
        "target_recall": target,
        "chosen_k": chosen,
        "chosen_recall": recall_by_k.get(chosen) if chosen is not None else None,
        "best_k": best_k,
        "best_recall": recall_by_k.get(best_k),
        "ceiling": chosen is None,
        "recommended_k": chosen if chosen is not None else best_k,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--val", type=Path, default=CATALOG_DIR / "train_splits" / "val.jsonl")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--target-recall", type=float, default=0.90)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/visual_rag_eval/pooled_first_stage_calibration"),
    )
    p.add_argument("--plot", action="store_true", help="Write recall@K PNG")
    args = p.parse_args()
    ks = sorted(set(int(k) for k in args.ks if int(k) > 0))

    meta = json.loads((args.index_dir / "meta.json").read_text(encoding="utf-8"))
    emb = np.load(args.index_dir / "embeddings.f32.npy")
    pages = []
    with (args.index_dir / "pages.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pages.append(json.loads(line))
    page_id_to_idx = {str(r["page_id"]): i for i, r in enumerate(pages)}
    page_subjects = np.asarray([str(r.get("subject") or "") for r in pages])
    page_grades = np.asarray([str(r.get("grade") if r.get("grade") is not None else "") for r in pages])

    rows = [
        r
        for r in _read_jsonl(args.val)
        if r.get("useful") and r.get("query") and r.get("positive_page_id")
    ]
    rng = random.Random(args.seed)
    if args.limit and len(rows) > args.limit:
        rows = rng.sample(rows, args.limit)

    adapter = Path(meta["adapter"]) if meta.get("adapter") else None
    print(f"[calib] loading encoder adapter={adapter}", flush=True)
    model, processor = _load_encoder(adapter)
    print(f"[calib] pages={len(pages)} queries={len(rows)} ks={ks}", flush=True)

    def eval_mode(use_meta: bool) -> dict[str, Any]:
        hits = {k: 0 for k in ks}
        gold_ranks: list[int] = []
        missing = 0
        n_scored = 0
        t0 = time.time()
        for i, row in enumerate(rows, start=1):
            gold = str(row["positive_page_id"])
            gi = page_id_to_idx.get(gold)
            if gi is None:
                missing += 1
                continue
            qv = _encode_queries(model, processor, [str(row["query"])])[0]
            scores = emb @ qv
            if use_meta:
                mask = (page_subjects == str(row.get("subject") or "")) & (
                    page_grades == str(row.get("grade") if row.get("grade") is not None else "")
                )
                if not mask.any():
                    continue
                scores = np.where(mask, scores, -np.inf)
            order = np.argsort(-scores)
            valid = order[np.isfinite(scores[order])]
            if valid.size == 0:
                continue
            n_scored += 1
            gold_pos = np.where(valid == gi)[0]
            rank = int(gold_pos[0]) + 1 if gold_pos.size else int(valid.size) + 1
            gold_ranks.append(rank)
            ranked = [str(pages[int(j)]["page_id"]) for j in valid[: max(ks)].tolist()]
            for k in ks:
                if gold in ranked[:k]:
                    hits[k] += 1
            if i % 50 == 0 or i == len(rows):
                kref = 50 if 50 in hits else ks[0]
                print(
                    f"[calib{'-meta' if use_meta else ''}] {i}/{len(rows)} "
                    f"recall@{kref}={hits[kref]/max(n_scored,1):.3f}",
                    flush=True,
                )
        denom = max(n_scored, 1)
        recall = {k: hits[k] / denom for k in ks}
        return {
            "n_scored": n_scored,
            "missing_gold": missing,
            "seconds": round(time.time() - t0, 2),
            "recall_by_k": {str(k): recall[k] for k in ks},
            "_recall": recall,
            "gold_rank": {
                "n": len(gold_ranks),
                "median": float(np.median(gold_ranks)) if gold_ranks else None,
                "p90": _percentile([float(x) for x in gold_ranks], 90),
                "max": int(max(gold_ranks)) if gold_ranks else None,
            },
        }

    no_filter = eval_mode(False)
    with_meta = eval_mode(True)
    choice_nf = _choose_k(no_filter["_recall"], target=args.target_recall, ks=ks)
    choice_meta = _choose_k(with_meta["_recall"], target=args.target_recall, ks=ks)
    use_meta = bool(choice_nf["ceiling"]) and not bool(choice_meta["ceiling"])
    rec_k = int(choice_meta["recommended_k"] if use_meta else choice_nf["recommended_k"])

    report = {
        "index": str(args.index_dir),
        "adapter": str(adapter) if adapter else None,
        "n_sample": len(rows),
        "seed": args.seed,
        "ks": ks,
        "target_recall": args.target_recall,
        "no_filter": {
            **{k: v for k, v in no_filter.items() if not k.startswith("_")},
            "k_selection": choice_nf,
        },
        "with_grade_subject_filter": {
            **{k: v for k, v in with_meta.items() if not k.startswith("_")},
            "k_selection": choice_meta,
        },
        "recommendation": {
            "require_grade_subject_filter": use_meta,
            "candidate_k": rec_k,
            "min_score": None,
            "env": {
                "MLA_VISUAL_POOL_CANDIDATE_K": rec_k,
                "MLA_VISUAL_POOL_REQUIRE_META_FILTER": use_meta,
                "MLA_VISUAL_POOL_MIN_SCORE": None,
            },
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Markdown with plain-language explanation
    md = [
        "# Что такое порог K (простыми словами)",
        "",
        "Pooled-поиск быстро ставит все страницы учебника «по похожести» к вопросу.",
        "**K** — сколько лучших страниц мы оставляем для второго шага (точного MaxSim).",
        "",
        "- Маленький K (1–50) = быстрее, но чаще выкидываем правильную страницу.",
        "- Большой K (500–1000) = почти всегда оставляем правильную, MaxSim дольше.",
        "- **recall@K** = доля вопросов, где правильная страница попала в эти K штук.",
        "- Цель: recall ≥ 90%.",
        "",
        f"**Выбор:** K=`{rec_k}`, фильтр grade+subject=`{use_meta}`, min_score=нет.",
        "",
        "## recall@K без фильтра",
        "",
        "| K | recall |",
        "|---|--------|",
    ]
    for k in ks:
        md.append(f"| {k} | {no_filter['recall_by_k'][str(k)]:.4f} |")
    md += ["", "## recall@K с фильтром класс+предмет", "", "| K | recall |", "|---|--------|"]
    for k in ks:
        md.append(f"| {k} | {with_meta['recall_by_k'][str(k)]:.4f} |")
    md.append("")
    (args.out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(
            ks,
            [no_filter["_recall"][k] * 100 for k in ks],
            "o-",
            label="Без фильтра (весь корпус)",
            color="#555555",
        )
        ax.plot(
            ks,
            [with_meta["_recall"][k] * 100 for k in ks],
            "s-",
            label="С фильтром класс + предмет",
            color="#1f4e79",
        )
        ax.axhline(90, color="#b85c38", linestyle="--", linewidth=1.5, label="Цель 90%")
        ax.axvline(rec_k, color="#2a7f62", linestyle=":", linewidth=1.5, label=f"Выбранный K={rec_k}")
        for k in ks:
            y = with_meta["_recall"][k] * 100
            ax.annotate(
                f"K={k}\n{y:.0f}%",
                (k, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color="#1f4e79",
            )
        ax.set_xlabel("K — сколько кандидатов оставляем после быстрого поиска")
        ax.set_ylabel("recall@K, % — как часто правильная страница в топ-K")
        ax.set_title(
            "Pooled first-stage: какой K брать\n"
            "(N=500 val, seed=7; scored subset без пропавших page_id)"
        )
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        png = args.out_dir / "recall_at_k.png"
        fig.savefig(png, dpi=160)
        plt.close(fig)
        print(f"[calib] plot → {png}", flush=True)

    (args.out_dir / "first_stage.json").write_text(
        json.dumps(report["recommendation"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["recommendation"], ensure_ascii=False, indent=2))
    print(f"[calib] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
