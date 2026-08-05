"""Fine-tune a multilingual text embedder on query ↔ answer_text pairs.

Uses sentence-transformers MultipleNegativesRankingLoss. This is the cheap
first fine-tune before / alongside ColQwen LoRA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import read_jsonl, split_by_page, write_jsonl

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _as_input_examples(rows: list[dict]):
    from sentence_transformers import InputExample

    examples = []
    for row in rows:
        query = str(row.get("query") or "").strip()
        positive = str(row.get("positive_answer_text") or "").strip()
        if len(query) < 3 or len(positive) < 20:
            continue
        # Keep positives short enough for MiniLM context.
        examples.append(InputExample(texts=[query, positive[:2_000]]))
    return examples


def train_text_embedder(
    train_rows: list[dict],
    val_rows: list[dict],
    *,
    model_name: str = DEFAULT_MODEL,
    output_dir: Path,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
) -> Path:
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader

    model = SentenceTransformer(model_name)
    train_examples = _as_input_examples(train_rows)
    if not train_examples:
        raise ValueError("no usable train pairs (need query + positive_answer_text)")

    loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = None
    if val_rows:
        queries = {
            f"q{i}": str(row["query"])
            for i, row in enumerate(val_rows)
            if row.get("query") and row.get("positive_answer_text")
        }
        corpus = {
            f"d{i}": str(row["positive_answer_text"])[:2_000]
            for i, row in enumerate(val_rows)
            if row.get("query") and row.get("positive_answer_text")
        }
        relevant = {
            f"q{i}": {f"d{i}"}
            for i, row in enumerate(val_rows)
            if row.get("query") and row.get("positive_answer_text")
        }
        if queries and corpus:
            evaluator = InformationRetrievalEvaluator(
                queries,
                corpus,
                relevant,
                name="val",
                show_progress_bar=False,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    warmup = max(10, int(0.05 * len(loader) * epochs))
    model.fit(
        train_objectives=[(loader, loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": lr},
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    meta = {
        "base_model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_pairs": len(train_examples),
        "val_rows": len(val_rows),
    }
    (output_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune multilingual text embedder on generated page queries.",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="JSONL from generate_train_queries.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--write-splits-dir",
        type=Path,
        default=None,
        help="Optional directory to dump train/val/test JSONL splits.",
    )
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

    train_text_embedder(
        splits["train"],
        splits["val"],
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"Saved text embedder -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
