#!/bin/bash
# Монитор захвата карт: как только GPU 2/3 освобождается - поднимаем на ней
# наш vLLM и воркер EXAMS-V. Живёт на стенде, ssh не нужен.
cd ~/shabrov_mla

# вернуть отложенные работы EXAMS-V в очередь (однократно)
if ! grep -q "e300" joblist.txt; then
  cat joblist_e300.txt >> joblist.txt
  echo "=== e300 возвращён в joblist ($(wc -l < joblist.txt) строк) $(date) ==="
fi

# снять локи незавершённых работ, иначе воркеры их вечно пропускают
for d in locks/*/; do
  J=$(basename "$d")
  OUT="results/$J.jsonl"
  TASKS=$(awk -v j="results/$J.jsonl" '$1==j {print $3}' joblist.txt)
  [ -z "$TASKS" ] && continue
  TOTAL=$(wc -l < "$TASKS" 2>/dev/null || echo 0)
  DONE=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  if [ "$DONE" -lt "$TOTAL" ]; then
    rmdir "$d" 2>/dev/null && echo "=== лок снят: $J ($DONE/$TOTAL) ==="
  fi
done

start_stack () {  # $1 = индекс GPU, $2 = порт, $3 = лог сервера
  CUDA_VISIBLE_DEVICES=$1 setsid nohup .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B --port $2 --max-model-len 32768 --dtype float16 \
    --gpu-memory-utilization 0.85 --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml > $3 2>&1 < /dev/null &
  sleep 10
  PORT=$2 setsid nohup ./worker.sh > worker_$2.log 2>&1 < /dev/null &
  echo "=== GPU $1 захвачена: сервер :$2 + воркер $(date) ==="
}

G2=0; G3=0
while [ "$G2" -eq 0 ] || [ "$G3" -eq 0 ]; do
  if [ "$G2" -eq 0 ]; then
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
    [ "$M" -lt 500 ] && { start_stack 2 8010 vllm.log; G2=1; }
  fi
  if [ "$G3" -eq 0 ]; then
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    [ "$M" -lt 500 ] && { start_stack 3 8011 vllm2.log; G3=1; }
  fi
  [ "$G2" -eq 1 ] && [ "$G3" -eq 1 ] && break
  sleep 300
done
echo "=== WATCHER_DONE: обе карты в работе $(date) ==="
