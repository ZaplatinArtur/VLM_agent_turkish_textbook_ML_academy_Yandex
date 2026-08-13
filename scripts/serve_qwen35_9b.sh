#!/usr/bin/env bash
set -euo pipefail

# Some bare GPU hosts provide the CUDA runtime but not nvcc. Disable only the
# FlashInfer sampling JIT; attention, vision, and tool calling remain enabled.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# OpenAI-compatible Qwen server for mla_baseline and the retrieval gate
# (retrieve.gate points RETRIEVE_GATE_URL at this endpoint). The shorter default
# context keeps the KV cache practical on a single card; homework prompts do not
# need the model's full native 262k context window.
model_name="${MLA_MODEL_NAME:-Qwen/Qwen3.5-9B}"
host="${MLA_VLLM_HOST:-127.0.0.1}"
port="${MLA_VLLM_PORT:-8000}"
max_model_len="${MLA_VLLM_MAX_MODEL_LEN:-32768}"
gpu_memory_utilization="${MLA_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

# bfloat16 needs compute capability 8.0+; the V100s here are Volta (7.0) and vLLM
# refuses to start. fp16 has tensor-core support on Volta, so speed is unaffected —
# its dynamic range is narrower, so switch to float32 if the model emits NaNs.
dtype="${MLA_VLLM_DTYPE:-float16}"

exec vllm serve "${model_name}" \
    --host "${host}" \
    --port "${port}" \
    --served-model-name "${model_name}" \
    --tensor-parallel-size 1 \
    --dtype "${dtype}" \
    --max-model-len "${max_model_len}" \
    --gpu-memory-utilization "${gpu_memory_utilization}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
