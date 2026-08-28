#!/usr/bin/env bash
set -euo pipefail

# Сервис визуального поиска ColQwen для agent_rag с MLA_RETRIEVAL_BACKEND=visual
# и MLA_VISUAL_TRANSPORT=http. Нужен, когда воркеров больше, чем карт: при
# раскладке «воркер на карту» дешевле транспорт inprocess, сервис не нужен.
#
#     CUDA_VISIBLE_DEVICES=2 ./scripts/serve_visual_search.sh 8780
#
# Карта выбирается снаружи через CUDA_VISIBLE_DEVICES. Без неё модель уедет на
# CPU: MaxSim по сотням кандидатов там медленный, но цепочку отладить можно.

port="${1:-${MLA_VISUAL_PORT:-8780}}"
host="${MLA_VISUAL_HOST:-127.0.0.1}"
index_dir="${MLA_VISUAL_INDEX_DIR:-}"

# Веса ищутся локально: без этого HF_HOME модель уходит в интернет, а от
# упавшей попытки остаётся замок, который роняет следующие запуски.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES не задан — сервис пойдёт на CPU" >&2
fi

args=(--host "${host}" --port "${port}")
[[ -n "${index_dir}" ]] && args+=(--index-dir "${index_dir}")
[[ "${MLA_VISUAL_NO_IMAGES:-0}" == "1" ]] && args+=(--no-images)

# Индекс грузится до первого запроса, поэтому падение конфигурации видно сразу,
# а не через час прогона. Готовность — 200 на /health.
exec python -m visual_retrive.search_service "${args[@]}"
