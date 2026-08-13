#!/usr/bin/env bash
set -euo pipefail

# Reproducible paired experiment on one fixed task file through OpenRouter.
python_bin="${PYTHON_BIN:-python}"
tasks="${TASKS:-data/eval/validation.jsonl}"
chunks_dir="data/corpus/chunks/jsonl"
output_dir="${OUTPUT_DIR:-results/rag_eval}"
report_dir="${REPORT_DIR:-reports/rag_eval}"
base_url="${BASE_URL:-https://openrouter.ai/api/v1}"
model="${MODEL:-qwen/qwen3.5-9b}"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running the evaluation}"

export MLA_PROMPT_VERSION="${MLA_PROMPT_VERSION:-v2_cot}"
export MLA_LLM_PROVIDER=openrouter
export MLA_OPENROUTER_MODEL_NAME="${MLA_OPENROUTER_MODEL_NAME:-${model}}"
export MLA_OPENROUTER_BASE_URL="${MLA_OPENROUTER_BASE_URL:-${base_url}}"
export MLA_CONCURRENCY="${MLA_CONCURRENCY:-4}"
export MLA_TEXT_ONLY="${MLA_TEXT_ONLY:-true}"

mkdir -p "${output_dir}" "${report_dir}"

b0_results="${output_dir}/b0_no_tools.jsonl"
rag_results="${output_dir}/agent_rag.jsonl"
b0_input="${output_dir}/b0_judge_input.jsonl"
rag_input="${output_dir}/agent_rag_judge_input.jsonl"
b0_judge="${output_dir}/b0_judge.jsonl"
rag_judge="${output_dir}/agent_rag_judge.jsonl"

"${python_bin}" -m mla_baseline.preflight \
  --tasks "${tasks}" \
  --text-only

"${python_bin}" -m retrieve.build_index \
  --sample-query "dikdörtgen alan formülü" --k 3

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

"${python_bin}" -m vlm_judge.cli prepare-mla-judge-input \
  --tasks "${tasks}" \
  --results "${b0_results}" \
  --output "${b0_input}" \
  --require-all

"${python_bin}" -m vlm_judge.cli prepare-mla-judge-input \
  --tasks "${tasks}" \
  --results "${rag_results}" \
  --output "${rag_input}" \
  --require-all

"${python_bin}" -m vlm_judge.cli run-text-judge \
  --input "${b0_input}" \
  --output "${b0_judge}" \
  --base-url "${base_url}" \
  --model "${model}" \
  --api-key-env OPENROUTER_API_KEY \
  --provider openrouter \
  --retry-failures

"${python_bin}" -m vlm_judge.cli run-text-judge \
  --input "${rag_input}" \
  --output "${rag_judge}" \
  --base-url "${base_url}" \
  --model "${model}" \
  --api-key-env OPENROUTER_API_KEY \
  --provider openrouter \
  --retry-failures

"${python_bin}" -m mla_baseline.paired_eval \
  --tasks "${tasks}" \
  --baseline-results "${b0_results}" \
  --rag-results "${rag_results}" \
  --baseline-judge "${b0_judge}" \
  --rag-judge "${rag_judge}" \
  --chunks-dir "${chunks_dir}" \
  --out-json "${report_dir}/summary.json" \
  --out-md "${report_dir}/summary.md"
