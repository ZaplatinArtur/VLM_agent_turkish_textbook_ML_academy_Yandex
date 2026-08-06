#!/usr/bin/env bash
# Прогон условий с веб-поиском на исправленном бэкенде.
#
# Правила стенда: работаем ТОЛЬКО в своей папке и ТОЛЬКО на своей GPU.
# Номер GPU передаётся явно — скрипт сам ничего не выбирает и не занимает.
#
#   GPU=3 bash scripts/stand_run_b1.sh
#
# Что делает: поднимает SearXNG (докер) и vLLM на указанной GPU, прогоняет
# гейт поиска, затем условия b1_search и b1_deep_routed на канонической
# конфигурации 32k. Гейт обязателен: без него прогон снова измерит бэкенд.

set -euo pipefail

GPU="${GPU:?укажи GPU=3 или GPU=4 — только наши}"
case "$GPU" in 3|4) ;; *) echo "GPU $GPU не наша, только 3 и 4"; exit 1;; esac

WORK="${WORK:-$HOME/shabrov_mla}"
VLLM_PORT="${VLLM_PORT:-8003}"
SEARX_PORT="${SEARX_PORT:-8090}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
CONDITIONS="${CONDITIONS:-b1_search b1_deep_routed}"

mkdir -p "$WORK"; cd "$WORK"
echo "рабочая папка: $WORK, GPU $GPU, vLLM :$VLLM_PORT, SearXNG :$SEARX_PORT"

# --- SearXNG -----------------------------------------------------------------
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  if ! docker ps --format '{{.Names}}' | grep -qx mla-searxng-$USER; then
    docker run -d --name "mla-searxng-$USER" --restart unless-stopped \
      -p "127.0.0.1:$SEARX_PORT:8080" \
      -v "$WORK/searxng/settings.yml:/etc/searxng/settings.yml:ro" \
      -e "SEARXNG_SECRET=$(head -c32 /dev/urandom | od -An -tx1 | tr -d ' \n')" \
      -e SEARXNG_BASE_URL="http://localhost:$SEARX_PORT/" \
      searxng/searxng:latest
    sleep 15
  fi
  SEARX_URL="http://127.0.0.1:$SEARX_PORT"
else
  # без докера ждём обратный туннель с локальной машины:
  #   ssh -R 8090:localhost:8080 <user>@<host>
  SEARX_URL="http://127.0.0.1:$SEARX_PORT"
  echo "докера нет — жду SearXNG в обратном туннеле на :$SEARX_PORT"
fi

# --- код и окружение ---------------------------------------------------------
[ -d MLA_Baseline ] || git clone -b mla_baseline \
  https://github.com/ZaplatinArtur/VLM_agent_turkish_textbook_ML_academy_Yandex.git MLA_Baseline
cd MLA_Baseline && git pull --ff-only && cd ..
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/pip install -q -U pip
./.venv/bin/pip install -q -e MLA_Baseline
./.venv/bin/pip install -q vllm trafilatura

# --- vLLM на нашей GPU -------------------------------------------------------
if ! curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
  # V100 — это Volta: bfloat16 она не умеет, только float16. На A100 vLLM
  # выбирал bf16 сам, здесь это надо задать явно, иначе падение на старте.
  DTYPE="${DTYPE:-float16}"
  CUDA_VISIBLE_DEVICES="$GPU" nohup ./.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$VLLM_PORT" --max-model-len 32768 --dtype "$DTYPE" \
    --gpu-memory-utilization 0.90 > "$WORK/vllm.log" 2>&1 &
  echo "vLLM стартует, лог $WORK/vllm.log"
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null && break
    sleep 10
  done
fi
curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null || { echo "vLLM не поднялся"; exit 1; }

# --- конфиг прогона ----------------------------------------------------------
cd MLA_Baseline
cat > .env <<ENV
MLA_VLLM_BASE_URL=http://127.0.0.1:$VLLM_PORT/v1
MLA_MODEL_NAME=$MODEL
MLA_MAX_TOKENS=16384
MLA_PROMPT_VERSION=v2_cot
MLA_SEARXNG_URL=$SEARX_URL
MLA_SEARX_MIN_INTERVAL_S=3.0
MLA_CONCURRENCY=4
ENV

# --- ГЕЙТ: бэкенд обязан отвечать до прогона ---------------------------------
../.venv/bin/python scripts/probe_searxng.py --url "$SEARX_URL" -n 40 --pause 3 \
  || { echo "ГЕЙТ НЕ ПРОЙДЕН — прогон отменён, чинить поиск"; exit 1; }

# --- прогоны -----------------------------------------------------------------
TASKS="${TASKS:-data/validation.jsonl}"
[ -f "$TASKS" ] || { echo "нет набора задач: $TASKS (скопируй с локальной машины)"; exit 1; }
for cond in $CONDITIONS; do
  echo "=== $cond ==="
  ../.venv/bin/python -m mla_baseline.runner --tasks "$TASKS" --condition "$cond" \
    --out "results/${cond}_fixed_web.jsonl" 2>&1 | tail -5
done
echo "готово; результаты в $WORK/MLA_Baseline/results/"
