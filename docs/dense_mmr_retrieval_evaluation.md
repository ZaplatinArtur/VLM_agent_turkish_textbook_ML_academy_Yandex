# Dense and MMR retrieval evaluation

This benchmark evaluates retrieval separately from the homework-solving agent.
It compares the current Dense FAISS ranking with MMR applied to the same Dense
candidate pool.

## Qrels

Each JSONL row fixes one query and its independently annotated relevant chunks:

```json
{"query_id":"task-1","query":"Mısır Uygarlığı özellikleri","subject":"history","grade":5,"relevant_chunk_ids":["5-sinif-sosyal-bilgiler-ders-kitabi-cevaplari-e-kare-yayinlari:0039"]}
```

Do not derive relevance labels only from the evaluated top-50 results. Search
the corpus independently and record every page/chunk that contains evidence
usable for the task. Rows without labels are kept in the candidate report but
are not included in metric denominators. Rows whose labels are all absent from
the current corpus are reported as `uncovered` and are also excluded.

## LLM-assisted pooled qrels

The reproducible proxy protocol pools all-corpus Dense@200 and BM25@100 before
annotation. This is intentionally wider than the evaluated top-50 and includes
an independent lexical retriever, reducing direct top-50 pooling bias.

```powershell
python -m retrieve.annotate build-pool `
  --qrels artifacts\retrieval\textbook_qrels.jsonl `
  --tasks outputs\validation_merged_20260723\validation_image_tasks.jsonl `
  --run results\openrouter_routed_experiment\full_math_router_v1\agent_rag_raw.jsonl `
  --output artifacts\retrieval\textbook_annotation_pool.jsonl `
  --dense-k 200 --bm25-k 100 --text-chars 400
```

Set `OPENROUTER_API_KEY` only in the local environment or `.env`; never put it
in a tracked script. First annotate five tasks as a schema/cost smoke test:

```powershell
$env:OPENROUTER_API_KEY = "<your key>"
python -m retrieve.annotate run `
  --pool artifacts\retrieval\textbook_annotation_pool.jsonl `
  --output artifacts\retrieval\textbook_qrels_llm.jsonl `
  --limit 5
```

Inspect the five rows, then rerun without `--limit`. The command resumes completed
tasks and atomically replaces errors on retry. `relevant_chunk_ids` contains only
directly useful evidence; borderline chunks are retained separately as
`uncertain_chunk_ids`. The resulting metrics are LLM-assisted pooled estimates,
not human gold. Audit a stratified sample before reporting them as final.

For image-only tasks, freeze the agent's first textbook-search query before
annotating qrels:

```powershell
python -m retrieve.evaluate prepare-qrels `
  --tasks outputs\validation_merged_20260723\validation_image_tasks.jsonl `
  --run results\openrouter_routed_experiment\full_math_router_v1\agent_rag_raw.jsonl `
  --output artifacts\retrieval\textbook_qrels.jsonl
```

Fill `relevant_chunk_ids` manually, then run the benchmark:

```powershell
python -m retrieve.evaluate run `
  --qrels artifacts\retrieval\textbook_qrels.jsonl `
  --k 1 --k 5 --k 10 --k 50 `
  --fetch-k 50 `
  --mmr-lambda 0.5 `
  --output results\retrieval_eval\dense_mmr.json
```

The JSON report contains Recall, Hit Rate, MAP, and MRR at every requested
cutoff for both variants, Dense-to-MMR deltas, per-subject aggregates, latency,
coverage counts, and per-query ranked candidates with text snippets.

The qrels template keeps the task `subject` for reporting and separately freezes
the first tool call's `retrieval_subject` for filtering. A null
`retrieval_subject` reproduces a tool call that searched the whole corpus. Use
`--no-subject-filter` only for an explicit all-corpus ablation.

At `fetch_k=50`, Dense and MMR must contain the same candidate set at cutoff 50:
MMR changes their order, not candidate generation. Consequently Recall@50 must
match, while MAP@50 may change.
