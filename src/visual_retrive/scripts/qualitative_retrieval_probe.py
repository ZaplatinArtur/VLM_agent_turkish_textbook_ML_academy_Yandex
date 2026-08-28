"""Qualitative probe: invented Turkish queries × image-only MaxSim corpus."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from PIL import Image

from visual_retrive.train.dataset import read_jsonl, resolve_image_path
from visual_retrive.train.train_colqwen_lora import (
    DEFAULT_MODEL,
    _load_model_and_processor,
    _torch_dtype,
)

QUERIES = [
    "kesirlerde payda eşitleme nasıl yapılır",
    "üçgende açıortay nedir örnekle anlat",
    "Fotosentez olayını kısaca açıkla",
    "Osmanlı Devleti'nin kuruluşu hangi yüzyılda",
    "Present Continuous tense örnek cümleler",
    "hücre zarının görevi nedir",
    "namazın farzları nelerdir",
    "kesirlerle toplama işlemi örnekleri",
]


def _tokenize_keywords(q: str) -> set[str]:
    toks = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+", q.lower())
    return {t for t in toks if len(t) > 2}


def _relevance(query: str, answer: str) -> tuple[str, int]:
    kws = _tokenize_keywords(query)
    text = (answer or "").lower()
    hits = sum(1 for k in kws if k in text)
    if hits >= 3:
        return "likely", hits
    if hits >= 1:
        return "weak", hits
    return "unlikely", hits


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/visual_retrive"))
    p.add_argument("--pairs", type=Path, default=None)
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir (optional)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--corpus-n", type=int, default=800)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/visual_retrive/catalog/logs/qualitative_retrieval_probe_canonical.md"),
    )
    args = p.parse_args(argv)

    pairs_path = args.pairs or (args.data_root / "catalog" / "train_splits" / "val.jsonl")
    rows = [r for r in read_jsonl(pairs_path) if r.get("useful")]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    corpus: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        pid = str(row.get("positive_page_id") or "")
        if not pid or pid in seen:
            continue
        path = resolve_image_path(row, args.data_root)
        if path is None:
            continue
        seen.add(pid)
        corpus.append(
            {
                "page_id": pid,
                "image": path,
                "answer": str(row.get("positive_answer_text") or "")[:400],
            }
        )
        if len(corpus) >= args.corpus_n:
            break
    if len(corpus) < 50:
        raise SystemExit(f"Corpus too small: {len(corpus)}")

    dtype = _torch_dtype()
    model, processor, _family = _load_model_and_processor(args.model, dtype)
    adapter = args.adapter
    if adapter is not None and str(adapter).strip() and Path(adapter).is_dir():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        print(f"[probe] loaded adapter {adapter}", flush=True)
        args.adapter = adapter
    else:
        args.adapter = None
    model.eval()
    device = next(model.parameters()).device

    # Encode corpus (image-only).
    doc_embs: list[torch.Tensor] = []
    for start in range(0, len(corpus), args.batch_size):
        chunk = corpus[start : start + args.batch_size]
        imgs = [Image.open(c["image"]).convert("RGB") for c in chunk]
        batch = processor.process_images(imgs)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        emb = model(**batch)
        for i in range(emb.size(0)):
            doc_embs.append(emb[i].detach().float().cpu())

    lines: list[str] = []
    adapter_tag = str(args.adapter) if args.adapter else "base"
    lines.append("# Qualitative retrieval probe (canonical MaxSim, image-only)")
    lines.append("")
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- adapter: `{adapter_tag}`")
    lines.append(f"- corpus: {len(corpus)} random pages (image-only, no answer fuse)")
    lines.append("- scoring: late-interaction MaxSim")
    lines.append("")

    hub_top1: list[str] = []
    for qi, query in enumerate(QUERIES, start=1):
        texts = [
            processor.query_prefix + query + processor.query_augmentation_token * 10
        ]
        q_batch = processor.process_texts(texts)
        q_batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in q_batch.items()
        }
        q_emb = model(**q_batch)[0].detach().float().cpu()
        scored: list[tuple[float, int]] = []
        for di, d_emb in enumerate(doc_embs):
            sim = torch.einsum("qd,sd->qs", q_emb, d_emb).amax(dim=1).sum().item()
            scored.append((sim, di))
        scored.sort(reverse=True)
        hub_top1.append(corpus[scored[0][1]]["page_id"])
        lines.append(f"## Q{qi}. {query}")
        lines.append("")
        for rank, (score, di) in enumerate(scored[:5], start=1):
            row = corpus[di]
            rel, overlap = _relevance(query, row["answer"])
            snippet = row["answer"].replace("\n", " ")[:180]
            lines.append(
                f"{rank}. **{row['page_id']}**  score=`{score:.3f}`  "
                f"relevance≈`{rel}` (keyword overlap={overlap})"
            )
            if snippet:
                lines.append(f"   - {snippet}…")
            lines.append("")

    unique_top1 = len(set(hub_top1))
    lines.append("## Collapse check")
    lines.append("")
    lines.append(f"- unique top-1 pages across {len(QUERIES)} queries: **{unique_top1}**")
    lines.append(f"- top-1 list: `{hub_top1}`")
    if unique_top1 <= 2:
        lines.append("- verdict: **hub-collapse likely** (≤2 unique top-1)")
    else:
        lines.append("- verdict: **diversified top-1** (no single-hub collapse)")
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "unique_top1": unique_top1,
        "top1": hub_top1,
        "corpus_n": len(corpus),
        "adapter": adapter_tag,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[probe] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
