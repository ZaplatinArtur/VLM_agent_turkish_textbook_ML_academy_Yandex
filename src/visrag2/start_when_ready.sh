#!/usr/bin/env bash
set -euo pipefail
project=/home/d.bykov/VLM_agent_turkish_textbook_ML_academy_Yandex
log=/mnt/storage-1/d.bykov/visrag_training/visrag2_autostart.log
mkdir -p "$(dirname "$log")"
expected_pairs_bytes=544912135
while test "$(stat -c %s catalog/train_queries.cleaned.jsonl 2>/dev/null || echo 0)" != "$expected_pairs_bytes"; do
  printf '%s waiting for complete query manifest\n' "$(date -Is)" >>"$log"
  sleep 60
done
while ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 < 1024 {n++} END {exit !(n == 4)}'; do
  printf '%s waiting for all four GPUs to become free\n' "$(date -Is)" >>"$log"
  sleep 60
done
cd "$project"
test -x .venv/bin/accelerate
test -s catalog/train_queries.cleaned.jsonl
test -d data/visual_retrive/models/visrag_siglip_e5_v3
.venv/bin/python -c "import torch, transformers, accelerate, wandb; print('imports OK', torch.__version__)" >>"$log" 2>&1
PYTHONPATH=src .venv/bin/python -m visrag2.audit_split \
  --pairs catalog/train_queries.cleaned.jsonl \
  --groups catalog/visrag_query_relevance_e5_v3_reviewed.json \
  --data-root . --pages-per-subject 120 >>"$log" 2>&1
exec bash src/visrag2/run_a100.sh >>/mnt/storage-1/d.bykov/visrag_training/visrag2_train.log 2>&1
