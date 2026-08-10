#!/usr/bin/env bash
# Разведка стенда перед прогоном. НИЧЕГО НЕ МЕНЯЕТ — только читает.
#
#   ssh <user>@<host> 'bash -s' < scripts/stand_preflight.sh
#
# Отвечает на вопросы, от которых зависит план: свободна ли наша GPU, есть ли
# докер под SearXNG, потянет ли диск веса, откуда брать питон.

set -uo pipefail
echo "=== стенд ==="
hostname; uname -sr; echo

echo "=== GPU (наши — 3 и 4) ==="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader
  echo "--- чьи процессы ---"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
    | head -20
else
  echo "nvidia-smi нет"
fi
echo

echo "=== докер (нужен для SearXNG) ==="
if command -v docker >/dev/null; then
  docker version --format '{{.Server.Version}}' 2>&1 | head -2
  docker ps --format '{{.Names}}\t{{.Image}}' 2>&1 | head -10
else
  echo "docker нет — SearXNG поднимем обратным туннелем с локальной машины"
fi
echo

echo "=== питон и окружение ==="
for p in python3.12 python3.11 python3 python; do
  command -v $p >/dev/null && { echo -n "$p: "; $p -V 2>&1; }
done
command -v uv >/dev/null && echo "uv есть"
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())' 2>/dev/null \
  || echo "torch в системном питоне нет (норм, поставим в venv)"
echo

echo "=== место ==="
df -h "$HOME" / /tmp 2>/dev/null | awk 'NR==1 || /\// {print}' | head -5
echo "кэш HF: $(du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null | cut -f1 || echo нет)"
echo

echo "=== сеть наружу (нужна для весов и поиска) ==="
for url in https://huggingface.co https://duckduckgo.com; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 12 "$url" 2>/dev/null || echo timeout)
  echo "  $url -> $code"
done
echo
echo "=== занятые порты (ищем свободные под vLLM/SearXNG) ==="
(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | awk 'NR>1 {print $4}' \
  | grep -oE '[0-9]+$' | sort -un | tr '\n' ' ' | head -c 400
echo
