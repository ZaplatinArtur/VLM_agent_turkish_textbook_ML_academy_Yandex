#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/src"
export WANDB_PROJECT="${WANDB_PROJECT:-turkish-visrag-v3-v100}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"$PWD/.venv/bin/accelerate" launch --num_processes 4 --mixed_precision fp16 -m visrag2.train \
  --pairs catalog/train_queries.cleaned.jsonl \
  --groups catalog/visrag_query_relevance_e5_v3_reviewed.json \
  --data-root . \
  --model /mnt/storage-1/d.bykov/VLM_agent_turkish_textbook_ML_academy_Yandex/data/visual_retrive/models/visrag_siglip_e5_v3 \
  --output-dir /mnt/storage-1/d.bykov/visrag_training/checkpoints/visrag2_siglip_e5_v3_v100 \
  --pages-per-subject 120 \
  --batch-size 32 --epochs 100 --lr 5e-6 --temperature 0.05 \
  --unfreeze-blocks 3 --eval-every 200 --save-every 200 \
  --deadline 12:00 --stop-buffer-minutes 10 \
  --wandb-project "$WANDB_PROJECT" \
  "${@}"
