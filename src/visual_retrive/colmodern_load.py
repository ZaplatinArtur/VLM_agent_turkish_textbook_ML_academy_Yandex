"""Robust ColModern load for current colpali_engine + transformers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

_DEFAULT_FREEZE_CONFIG = {
    "freeze_text_layers": False,
    "freeze_vision_layers": False,
    "freeze_projector": False,
    "freeze_embedding_layers": False,
    "freeze_head": False,
}


def configure_torch_runtime() -> None:
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        import torch._dynamo as dynamo

        dynamo.config.disable = True
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True


def torch_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _minor = torch.cuda.get_device_capability()
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_colmodern_weight_ids(model_name: str) -> tuple[str, str | None]:
    """Return (full_weights_repo, optional_adapter_repo)."""
    name = model_name.strip().rstrip("/")
    try:
        from huggingface_hub import hf_hub_download
        import json

        adapter_path = hf_hub_download(name, "adapter_config.json")
        base = json.loads(Path(adapter_path).read_text(encoding="utf-8")).get(
            "base_model_name_or_path"
        )
        if base:
            return str(base), name
    except Exception:
        pass
    if name.endswith("colmodernvbert"):
        return "ModernVBERT/colmodernvbert-merged", None
    return name, None


def remap_colmodern_state_dict_key(key: str) -> str:
    """Align Hub safetensors keys with current colpali@vbert module layout."""
    if key.startswith("model.vision_model.vision_model."):
        key = "model.vision_model." + key[len("model.vision_model.vision_model.") :]
    if key == "model.connector.modality_projection.weight":
        key = "model.connector.modality_projection.proj.weight"
    return key


def prepare_modernvbert_config(config: Any) -> Any:
    """Hub configs omit freeze_config; embedding vocab must match checkpoint (50408)."""
    from colpali_engine.models.modernvbert.modeling_modernvbert import (
        DecoupledEmbedding,
        ModernVBertModel,
    )
    from transformers import AutoConfig, AutoModel

    if getattr(config, "freeze_config", None) is None:
        config.freeze_config = dict(_DEFAULT_FREEZE_CONFIG)
    text_vocab = getattr(getattr(config, "text_config", None), "vocab_size", None)
    if text_vocab:
        config.vocab_size = int(text_vocab)

    def _init_language_model(cfg: Any):
        text_model_config = AutoConfig.from_pretrained(
            cfg.text_config.text_model_name,
            _attn_implementation=cfg._attn_implementation,
            trust_remote_code=True,
        )
        text_model_config.vocab_size = int(cfg.text_config.vocab_size)
        if hasattr(text_model_config, "reference_compile"):
            text_model_config.reference_compile = False
        text_model = AutoModel.from_config(text_model_config, trust_remote_code=True)
        if hasattr(text_model, "config") and hasattr(text_model.config, "reference_compile"):
            text_model.config.reference_compile = False
        embed_layer = DecoupledEmbedding(
            num_embeddings=text_model_config.vocab_size,
            num_additional_embeddings=cfg.additional_vocab_size,
            embedding_dim=cfg.hidden_size,
            partially_freeze=cfg.freeze_config["freeze_text_layers"],
            padding_idx=cfg.pad_token_id,
        )
        text_model.set_input_embeddings(embed_layer)
        return text_model

    ModernVBertModel.init_language_model = staticmethod(_init_language_model)
    return config


def load_colmodern_weights(model: Any, weights_id: str) -> None:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(weights_id, "model.safetensors")
    state: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as handle:
        for key in handle.keys():
            state[remap_colmodern_state_dict_key(key)] = handle.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[colmodern] load_state_dict missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"[colmodern] missing[:8]={missing[:8]}", flush=True)
    if unexpected:
        print(f"[colmodern] unexpected[:8]={unexpected[:8]}", flush=True)


def load_model_and_processor(
    model_name: str = "ModernVBERT/colmodernvbert-merged",
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    adapter: str | Path | None = None,
) -> tuple[Any, Any, torch.device]:
    """Load ColModern (+ optional LoRA adapter) ready for query/page encode."""
    configure_torch_runtime()
    dtype = dtype or torch_dtype()
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
    from colpali_engine.models.modernvbert.configuration_modernvbert import (
        ModernVBertConfig,
    )

    weights_id, hub_adapter = resolve_colmodern_weight_ids(model_name)
    config = ModernVBertConfig.from_pretrained(weights_id, trust_remote_code=True)
    config = prepare_modernvbert_config(config)
    processor = ColModernVBertProcessor.from_pretrained(weights_id, trust_remote_code=True)

    # Prefer explicit safetensors remap: HF from_pretrained often leaves vision
    # randomly initialized due to nested key layout drift.
    model = ColModernVBert(config)
    load_colmodern_weights(model, weights_id)
    model.to(dtype=dtype)

    # Ensure ModernBERT does not call broken torch.compile MLP.
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "reference_compile"):
            cfg.reference_compile = False

    adapter_path = adapter or hub_adapter
    if adapter_path and Path(adapter_path).is_dir():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.to(device)
    model.eval()
    return model, processor, device
