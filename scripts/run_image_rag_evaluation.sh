#!/usr/bin/env bash
set -euo pipefail

# Full photo-only B0 vs textbook-RAG evaluation through OpenRouter. Both agents receive
# only the original question screenshot. Text/reference images are attached
# later to the judge and are never exposed to the solving agents.
python_bin="${PYTHON_BIN:-python}"
data_root="${DATA_ROOT:-outputs/validation_merged_20260723}"
manifest="${MANIFEST:-${data_root}/validation_manifest.jsonl}"
tasks="${TASKS:-${data_root}/validation_image_tasks.jsonl}"
chunks_dir="${CHUNKS_DIR:-data/corpus/chunks/jsonl}"
output_dir="${OUTPUT_DIR:-results/validation_images_full}"
report_dir="${REPORT_DIR:-reports/validation_images_full}"
base_url="${BASE_URL:-https://openrouter.ai/api/v1}"
model="${MODEL:-qwen/qwen3.5-9b}"
expected_tasks="${EXPECTED_TASKS:-198}"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running the evaluation}"

export MLA_LLM_PROVIDER=openrouter
export MLA_OPENROUTER_BASE_URL="${MLA_OPENROUTER_BASE_URL:-${base_url}}"
export MLA_OPENROUTER_MODEL_NAME="${MLA_OPENROUTER_MODEL_NAME:-${model}}"
export MLA_DATA_ROOT="${MLA_DATA_ROOT:-${data_root}}"
export MLA_TEXT_ONLY=false
export MLA_INCLUDE_QUESTION_TEXT_WITH_IMAGES=false
export MLA_PROMPT_VERSION="${MLA_PROMPT_VERSION:-v2_cot}"
export MLA_CONCURRENCY="${MLA_CONCURRENCY:-4}"
export MLA_RETRIEVAL_TOP_K="${MLA_RETRIEVAL_TOP_K:-5}"
export MLA_RETRIEVAL_MAX_CONTEXT_CHARS="${MLA_RETRIEVAL_MAX_CONTEXT_CHARS:-6000}"
export MLA_RETRIEVAL_MAX_CALLS="${MLA_RETRIEVAL_MAX_CALLS:-2}"

mkdir -p "${output_dir}" "${report_dir}"

test -f "${manifest}"
test -d "${data_root}/images"
test -d "${chunks_dir}"

"${python_bin}" -m vlm_judge.cli build-image-validation-tasks \
  --manifest "${manifest}" \
  --data-root "${data_root}" \
  --output "${tasks}"

actual_tasks="$("${python_bin}" -c \
  'from pathlib import Path; import sys; print(sum(bool(line.strip()) for line in Path(sys.argv[1]).open(encoding="utf-8")))' \
  "${tasks}")"
if [[ "${actual_tasks}" != "${expected_tasks}" ]]; then
  echo "Expected ${expected_tasks} tasks, found ${actual_tasks}: ${tasks}" >&2
  exit 2
fi

"${python_bin}" -m mla_baseline.preflight \
  --tasks "${tasks}" \
  --data-root "${data_root}"

"${python_bin}" -m retrieve.build_index \
  --sample-query "dikdörtgen alan formülü" \
  --k 3

b0_results="${output_dir}/b0_no_tools.jsonl"
rag_results="${output_dir}/agent_rag.jsonl"
b0_input="${output_dir}/b0_image_judge_input.jsonl"
rag_input="${output_dir}/agent_rag_image_judge_input.jsonl"
b0_judge="${output_dir}/b0_image_judge.jsonl"
rag_judge="${output_dir}/agent_rag_image_judge.jsonl"
judge_cache="${output_dir}/judge_cache"

"${python_bin}" -m mla_baseline.runner \
  --tasks "${tasks}" \
  --condition b0_no_tools \
  --out "${b0_results}" \
  --retry-errors

"${python_bin}" -m mla_baseline.runner \
  --tasks "${tasks}" \
  --condition agent_rag \
  --out "${rag_results}" \
  --retry-errors

"${python_bin}" -m vlm_judge.cli prepare-image-judge-input \
  --manifest "${manifest}" \
  --results "${b0_results}" \
  --data-root "${data_root}" \
  --output "${b0_input}" \
  --require-all

"${python_bin}" -m vlm_judge.cli prepare-image-judge-input \
  --manifest "${manifest}" \
  --results "${rag_results}" \
  --data-root "${data_root}" \
  --output "${rag_input}" \
  --require-all

"${python_bin}" -m vlm_judge.cli run-judge \
  --input "${b0_input}" \
  --output "${b0_judge}" \
  --base-url "${base_url}" \
  --model "${model}" \
  --api-key-env OPENROUTER_API_KEY \
  --provider openrouter \
  --image-mode data_url \
  --disable-thinking \
  --cache-dir "${judge_cache}" \
  --workers "${JUDGE_WORKERS:-2}"

"${python_bin}" -m vlm_judge.cli run-judge \
  --input "${rag_input}" \
  --output "${rag_judge}" \
  --base-url "${base_url}" \
  --model "${model}" \
  --api-key-env OPENROUTER_API_KEY \
  --provider openrouter \
  --image-mode data_url \
  --disable-thinking \
  --cache-dir "${judge_cache}" \
  --workers "${JUDGE_WORKERS:-2}"

"${python_bin}" -m mla_baseline.paired_eval \
  --tasks "${tasks}" \
  --baseline-results "${b0_results}" \
  --rag-results "${rag_results}" \
  --baseline-judge "${b0_judge}" \
  --rag-judge "${rag_judge}" \
  --chunks-dir "${chunks_dir}" \
  --out-json "${report_dir}/summary.json" \
  --out-md "${report_dir}/summary.md"

echo "Done. Report: ${report_dir}/summary.md"
