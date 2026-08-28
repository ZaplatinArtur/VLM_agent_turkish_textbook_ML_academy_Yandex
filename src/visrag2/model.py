from __future__ import annotations

import torch
import torch.nn.functional as F
import os
from transformers import AutoModel, AutoProcessor


def load_model(source, device=None):
    processor = AutoProcessor.from_pretrained(source)
    dtype = torch.float32 if os.environ.get("MIXED_PRECISION") == "fp16" else torch.bfloat16
    model = AutoModel.from_pretrained(source, torch_dtype=dtype)
    if device is not None: model.to(device)
    model.config.use_cache = False
    return model, processor


def configure_last_blocks(model, count=3):
    for parameter in model.parameters(): parameter.requires_grad = False
    train_names = ["text_model.head", "vision_model.head", "logit_scale", "logit_bias"]
    for tower_name in ("text_model", "vision_model"):
        tower = getattr(model, tower_name); layers = tower.encoder.layers
        train_names += [f"{tower_name}.encoder.layers.{i}." for i in range(max(0,len(layers)-count),len(layers))]
    for name, parameter in model.named_parameters():
        if any(token in name for token in train_names): parameter.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def encode_texts(model, processor, texts, device):
    batch = processor(text=texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    batch = {k:v.to(device) for k,v in batch.items()}
    base = getattr(model, "module", model)
    return F.normalize(base.get_text_features(**batch), dim=-1)


def encode_images(model, processor, images, device):
    batch = processor(images=images, return_tensors="pt")
    batch = {k:v.to(device) for k,v in batch.items()}
    base = getattr(model, "module", model)
    return F.normalize(base.get_image_features(**batch), dim=-1)
