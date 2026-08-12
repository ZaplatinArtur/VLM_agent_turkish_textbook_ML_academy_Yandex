#!/bin/bash
# Воркер общей очереди. Запуск: PORT=8010 ./worker.sh
# Работы разбираются mkdir-локами - два воркера не столкнутся.
cd ~/shabrov_mla
export MLA_CONCURRENCY=8
export MLA_VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
mkdir -p locks
until curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do sleep 60; done
while true; do
  GOT=0
  while read -r OUT COND TASKS; do
    [ -z "$OUT" ] && continue
    J=$(basename "$OUT" .jsonl)
    TOTAL=$(wc -l < "$TASKS"); [ -f "$OUT" ] && DONE=$(wc -l < "$OUT") || DONE=0
    [ "$DONE" -ge "$TOTAL" ] && continue
    curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null || { echo "=== SERVER $PORT DEAD, стоп $(date) ==="; exit 1; }
    mkdir "locks/$J" 2>/dev/null || continue
    GOT=1
    echo "=== $J старт (:$PORT) $(date) ==="
    .venv/bin/python -m mla_baseline.runner --tasks "$TASKS" --condition "$COND" --out "$OUT"
    echo "=== $J готов $(date) ==="
  done < joblist.txt
  [ "$GOT" -eq 0 ] && break
done
echo "=== WORKER_${PORT}_DONE $(date) ==="
