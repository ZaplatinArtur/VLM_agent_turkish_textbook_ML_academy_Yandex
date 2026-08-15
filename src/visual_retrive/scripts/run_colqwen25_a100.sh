#!/usr/bin/env bash
set -euo pipefail
cd /home/d.teslov/VLM_agent_turkish_textbook_ML_academy_Yandex
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="${WANDB_PROJECT:-turkish-colqwen25}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/src:$PWD/.tools/colpali"
RESUME=()
if [[ $# -gt 0 ]]; then RESUME=(--resume-from-checkpoint "$1"); fi
.venv-colqwen/bin/torchrun --standalone --nproc_per_node=2 src/visual_retrive/scripts/train_colqwen25.py \
  --pairs catalog/train_queries_grades_1_12_blocks.cleaned.jsonl \
  --data-root . --output-dir models/colqwen25_turkish \
  --pages-per-subject 120 --batch-size 8 --grad-accum 2 \
  --wandb-project "$WANDB_PROJECT" "${RESUME[@]}"
