from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL = "google/siglip2-base-patch16-512"


def load_encoder(model_or_dir: str | Path, *, dtype=torch.float16, device="cuda"):
    source = str(model_or_dir)
    processor = AutoProcessor.from_pretrained(source)
    model = AutoModel.from_pretrained(source, dtype=dtype).to(device)
    return model, processor


def configure_trainable(model: torch.nn.Module, last_blocks: int = 1) -> int:
    for p in model.parameters():
        p.requires_grad = False
    needles = ["text_model.head", "vision_model.head", "logit_scale", "logit_bias"]
    for tower in ("text_model", "vision_model"):
        layers = getattr(getattr(model, tower).encoder, "layers")
        start = max(0, len(layers) - last_blocks)
        needles.extend(f"{tower}.encoder.layers.{i}." for i in range(start, len(layers)))
    for name, p in model.named_parameters():
        if any(needle in name for needle in needles):
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def encode_text(model, processor, texts: list[str], device: torch.device):
    batch = processor(text=texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    return F.normalize(model.get_text_features(**batch), dim=-1)


def encode_images(model, processor, images, device: torch.device):
    batch = processor(images=images, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    return F.normalize(model.get_image_features(**batch), dim=-1)


def save_checkpoint(model, processor, output: Path, meta: dict):
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    (output / "visrag_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
