#!/bin/bash
cd ~/mla_baseline
export PYTHONUNBUFFERED=1
.venv/bin/python -m mla_baseline.runner --tasks data/validation.jsonl --condition agent_rag_routed --out results/agent_rag_routed_32k.jsonl 2>&1 | tee logs/agent_rag_routed.log
J=judge_repo/.venv/bin/vlm-judge
$J prepare-mla-judge-input --tasks data/validation.jsonl --results results/agent_rag_routed_32k.jsonl --output results/judge_in_aragr.jsonl
$J prepare-mla-judge-input --tasks data/validation.bridge.jsonl --results results/agent_rag_routed_32k.jsonl --output results/judge_in_aragr_bridge.jsonl
.venv/bin/python - <<'PYEOF'
import json
tr = set(json.load(open('data/answer_transcripts.json')))
rows = [json.loads(l) for l in open('results/judge_in_aragr_bridge.jsonl')]
with open('results/judge_in_aragr_delta.jsonl', 'w') as f:
    for r in rows:
        if r['task_id'] in tr:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
PYEOF
$J run-text-judge --input results/judge_in_aragr.jsonl --output results/judge_out_aragr.jsonl --base-url http://localhost:8001/v1 --model Qwen/Qwen3.5-9B --retry-failures
$J run-text-judge --input results/judge_in_aragr_delta.jsonl --output results/judge_out_aragr_delta.jsonl --base-url http://localhost:8001/v1 --model Qwen/Qwen3.5-9B --retry-failures
echo DONE_ARAG_ALL
