"""LoRA fine-tune ColQwen2 for Turkish textbook page retrieval.

Requires optional deps on the GPU host:
  pip install \"colpali-engine>=0.3.4\" peft bitsandbytes accelerate datasets

Training recipe: contrastive in-batch negatives over (query text, page image).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import read_jsonl, resolve_image_path, split_by_page, write_jsonl

DEFAULT_MODEL = "vidore/colqwen2-v1.0"


def _require_colpali():
    try:
        import colpali_engine  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing GPU training deps. On the server run:\n"
            '  pip install "colpali-engine>=0.3.4" peft bitsandbytes '
            "accelerate datasets pillow\n"
            f"Original error: {exc}"
        ) from exc


def _load_rows_with_images(rows: list[dict], data_root: Path) -> list[dict]:
    from PIL import Image

    usable: list[dict] = []
    for row in rows:
        image_path = resolve_image_path(row, data_root)
        query = str(row.get("query") or "").strip()
        if image_path is None or len(query) < 3:
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            continue
        usable.append(
            {
                "query": query,
                "image": image,
                "page_id": row["positive_page_id"],
            }
        )
    if not usable:
        raise ValueError(
            "no usable ColQwen training rows (need query + existing page image)"
        )
    return usable


def train_colqwen_lora(
    train_rows: list[dict],
    val_rows: list[dict],
    *,
    data_root: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 5e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_steps: int | None = None,
) -> Path:
    _require_colpali()

    import torch
    from colpali_engine.loss import ColbertPairwiseCELoss
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset

    class _PairDataset(Dataset):
        def __init__(self, items: list[dict]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict:
            return self.items[index]

    processor = ColQwen2Processor.from_pretrained(model_name)
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_items = _load_rows_with_images(train_rows, data_root)
    _ = val_rows  # reserved for future retrieval eval hooks
    loss_fn = ColbertPairwiseCELoss()

    def collate(batch: list[dict]):
        images = [item["image"] for item in batch]
        queries = [item["query"] for item in batch]
        batch_images = processor.process_images(images)
        batch_queries = processor.process_queries(queries)
        device = next(model.parameters()).device
        batch_images = {key: value.to(device) for key, value in batch_images.items()}
        batch_queries = {key: value.to(device) for key, value in batch_queries.items()}
        return batch_queries, batch_images

    loader = DataLoader(
        _PairDataset(train_items),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=lr,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        for query_inputs, image_inputs in loader:
            query_emb = model(**query_inputs)
            image_emb = model(**image_inputs)
            loss = loss_fn(query_emb, image_emb) / grad_accum
            loss.backward()
            step += 1
            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 20 == 0:
                print(
                    f"epoch={epoch + 1} step={step} loss={float(loss) * grad_accum:.4f}",
                    flush=True,
                )
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
        # flush leftover grads at epoch end
        if step % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    meta = {
        "base_model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "train_rows": len(train_items),
        "steps": step,
        "data_root": str(data_root),
    }
    (output_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune ColQwen2 on Turkish page retrieval pairs.",
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing relative positive_image paths (usually data/visual_retrive).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--write-splits-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = read_jsonl(args.pairs)
    splits = split_by_page(
        rows,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if args.write_splits_dir is not None:
        for name, split_rows in splits.items():
            write_jsonl(args.write_splits_dir / f"{name}.jsonl", split_rows)

    train_colqwen_lora(
        splits["train"],
        splits["val"],
        data_root=args.data_root,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_steps=args.max_steps,
    )
    print(f"Saved ColQwen LoRA -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
