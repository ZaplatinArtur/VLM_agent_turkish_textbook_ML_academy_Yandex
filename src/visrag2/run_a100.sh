#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/src"
export WANDB_PROJECT="${WANDB_PROJECT:-turkish-visrag2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
accelerate launch --num_processes 2 --mixed_precision bf16 -m visrag2.train \
  --pairs catalog/train_queries_grades_1_12_blocks.cleaned.jsonl \
  --groups catalog/visrag_relevance_groups_v3_reviewed.json \
  --data-root . \
  --output-dir models/visrag2_colqwen35_turkish \
  --model athrael-soju/colqwen3.5-4.5B-v3 \
  --pages-per-subject 120 \
  --batch-size 4 --grad-accum 16 --epochs 1 \
  --eval-every 500 --save-every 250 \
  --deadline 11:00 --stop-buffer-minutes 8
