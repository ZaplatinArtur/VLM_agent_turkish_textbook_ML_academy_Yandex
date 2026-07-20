# VLM Judge

Reproducible evaluation for three homework-agent setups:

1. `no_tools`;
2. `web_search`;
3. `textbook_retrieval`.

The package combines deterministic exact metrics, a blinded multimodal LLM judge, human calibration, LMArena-style pairwise validation, and paired statistical reporting. It is usable without a GPU; the model backend is an adapter boundary that can be connected when the Qwen endpoint is known.

## Current source inventory

- Main workbook: 823 usable task records with 20 raw subject labels.
- Planning sheet and mentor target: 17 subject categories. The 20-to-17 correspondence is unconfirmed, so every raw label is preserved.
- Math workbook: 200 question/reference-image pairs, grades 1–12; all 400 image links have been verified.
- ÖdevJet corpus: 45,920 records from 215 books and 8 source subjects.
- Corpus risks: 344 duplicate rows, 263 conflicting duplicate IDs, and 16,520 low-information pages.
- Prepared retrieval layer: 45,576 canonical pages and 280,822 stable chunks (103,070 text and 177,752 image chunks); conflicting variants remain quarantined.
- CPU lexical baseline: SQLite FTS5/BM25 over all 103,070 text chunks, with a local agent-tool API and provenance-rich hits.

The math gold answers are often annotated images. Judge requests therefore support both a question image and a reference-answer image. Setup labels are removed from model prompts and hidden by default in the human interface.

## Install and test

```powershell
python -m pip install -e ".[sources,dev]"
python -m unittest discover -s tests -v
```

## Main commands

```powershell
# Normalize source workbooks and audit the corpus.
vlm-judge prepare-sources --main-workbook sheet1.xlsx --math-workbook sheet2.xlsx --corpus odevjet.jsonl --output-dir artifacts

# Canonical pages plus stable text/image chunks; conflicts are quarantined.
vlm-judge prepare-corpus --input odevjet.jsonl --output-dir artifacts/corpus --max-chars 1600 --overlap-chars 200

# Attach an agent run to the benchmark. Empty and failed responses are preserved.
vlm-judge import-candidates --benchmark artifacts/math_benchmark.jsonl --responses run.csv --setup no_tools --output artifacts/runs/no_tools.jsonl

# Exact metrics and blinded judge requests.
vlm-judge score-deterministic --input artifacts/runs/no_tools.jsonl --output artifacts/runs/no_tools_exact.jsonl
vlm-judge prepare-requests --input artifacts/runs/no_tools.jsonl --output artifacts/runs/no_tools_requests.jsonl

# Build and serve the CPU BM25 textbook baseline.
vlm-judge build-bm25 --chunks artifacts/corpus/chunks.jsonl --index artifacts/retrieval/bm25.sqlite
vlm-judge serve-retrieval --index artifacts/retrieval/bm25.sqlite --port 8770
vlm-judge prepare-retrieval-qrels --benchmark artifacts/math_benchmark.jsonl --output artifacts/retrieval/math_qrels_template.jsonl
vlm-judge evaluate-retrieval --index artifacts/retrieval/bm25.sqlite --qrels artifacts/retrieval/math_qrels.jsonl --k 1 --k 5 --k 10 --output artifacts/reports/bm25_eval.json

# Refuse to evaluate an incomplete or contract-drifting three-setup grid.
vlm-judge validate-runs --benchmark artifacts/math_benchmark.jsonl `
  --run no_tools=artifacts/runs/no_tools.jsonl `
  --run web_search=artifacts/runs/web_search.jsonl `
  --run textbook_retrieval=artifacts/runs/textbook_retrieval.jsonl `
  --strict-metadata `
  --output artifacts/reports/run_validation.json

# Exercise the entire pipeline without a model. Outputs are synthetic mechanics tests only.
vlm-judge synthetic-dry-run --benchmark artifacts/math_benchmark.jsonl --output-dir artifacts/dryrun

# Randomized, optionally mirrored A/B records.
vlm-judge prepare-pairs --input artifacts/runs/no_tools.jsonl --input artifacts/runs/textbook_retrieval.jsonl --setup-a no_tools --setup-b textbook_retrieval --mirrored --output artifacts/calibration/arena.jsonl

# Human UI: adjacent gold/candidate, gold transcription, and optional adjudication queue.
vlm-judge sample-calibration-responses --input artifacts/runs/no_tools.jsonl --input artifacts/runs/web_search.jsonl --input artifacts/runs/textbook_retrieval.jsonl --size 120 --output artifacts/calibration/response_sample.jsonl
vlm-judge-ui --dataset artifacts/calibration/response_sample.jsonl `
  --annotations artifacts/annotations/human.jsonl `
  --gold artifacts/annotations/gold.jsonl `
  --judge-results artifacts/runs/judge_results.jsonl `
  --adjudications artifacts/annotations/adjudications.jsonl `
  --open-browser

# Apply only verified task-scoped transcriptions before building judge requests.
vlm-judge apply-gold --dataset artifacts/runs/no_tools.jsonl --gold artifacts/annotations/gold.jsonl --output artifacts/runs/no_tools_with_gold.jsonl

# Calibration and position-bias reports.
vlm-judge analyze-calibration --human artifacts/annotations/human.jsonl --judge artifacts/runs/judge_results.jsonl --output artifacts/reports/calibration.json
vlm-judge audit-judge-run --input artifacts/runs/judge_results.jsonl --output artifacts/reports/judge_operational_audit.json
vlm-judge analyze-arena --annotations artifacts/annotations/arena.jsonl --output artifacts/reports/arena_bias.json

# Prioritize human/LLM disagreements, judge errors, low-confidence cases, reference issues,
# plus a stable 10% control sample of agreements.
vlm-judge prepare-adjudication --dataset artifacts/runs/no_tools.jsonl `
  --judge artifacts/runs/judge_results.jsonl `
  --human artifacts/annotations/human.jsonl `
  --output artifacts/reports/adjudication_queue.jsonl `
  --summary artifacts/reports/adjudication_summary.json

# Final hybrid/exact/judge views, paired deltas, bootstrap intervals, and coverage audit.
vlm-judge aggregate --input artifacts/runs/scored_records.jsonl --output artifacts/reports/summary.json
```

## Prepared calibration assets

`artifacts/calibration` contains 120 stratified real tasks, a UTF-8 human-labeling template, and 150 synthetic multiple-choice smoke cases. Synthetic records validate judge mechanics only; they are not evidence of model quality.

## Integration contract

Each setup must emit one record per `(task_id, setup, run_id)`. Candidate text is preserved verbatim, including failures and timeouts. The judge can already call an OpenAI-compatible Qwen/vLLM endpoint through `vlm-judge run-judge`; use `--limit 10` for the first endpoint/image/JSON smoke test, then remove the limit for the cached full run. Only deployment values remain external. The other missing inputs are real three-setup outputs, the mentor gold set, 771 unresolved main-workbook image assets, and the confirmed 17-subject mapping.

See [evaluation protocol](docs/evaluation_protocol.md), [judge acceptance criteria](docs/judge_acceptance_criteria.md), [data contract](docs/data_contract.md), [interface design](docs/interface_design.md), [retrieval tool contract](docs/retrieval_tool_contract.md), and [data/retrieval strategy](docs/data_and_retrieval_strategy.md).
