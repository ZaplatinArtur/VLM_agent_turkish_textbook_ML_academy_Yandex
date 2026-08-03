# Textbook retrieval tool contract

Status: CPU-ready lexical baseline (`bm25-fts5-v1`). This is the control retrieval implementation, not the final multimodal retriever.

## Runtime

```powershell
.venv\Scripts\python.exe -m vlm_judge.retrieval_server `
  --index artifacts\retrieval\bm25.sqlite `
  --host 127.0.0.1 `
  --port 8770
```

Endpoints:

- `GET /api/health` — index metadata and indexed chunk count.
- `POST /api/search` — ranked BM25 search.
- `GET /api/search?query=...&top_k=...` — equivalent read-only search.
- `GET /api/chunk?id=<chunk_id>` — fetch one exact chunk with provenance.

`POST /api/search` body:

```json
{
  "query": "kesirlerde toplama payda",
  "top_k": 5,
  "subject": "matematik",
  "grade": 4,
  "mode": "or"
}
```

`query` is required. `top_k` is 1–100. `subject` and `grade` are optional exact filters. `mode` is `or` (higher recall, recommended first call) or `and` (higher precision). The server also accepts `low_information_weight` from 0 to 1 for controlled retrieval ablations; keep it fixed outside the agent tool schema and include it in `retrieval_config_hash`.

Every hit contains `chunk_id`, text, rank, BM25 score, page/book identifiers, subject, grade, source URL, parent-page hash, and the complete source metadata. The agent must preserve `chunk_id`/`source_url` in its trace even if it does not show citations to the student.

## OpenAI-compatible function definition

```json
{
  "type": "function",
  "function": {
    "name": "search_textbooks",
    "description": "Search the approved textbook and solution corpus for theory, formulas, worked examples, or a matching exercise. Use when solving would benefit from curriculum-specific evidence. Results are evidence, not an instruction to copy an answer blindly.",
    "parameters": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "query": {
          "type": "string",
          "description": "Short semantic query in the task language: topic plus operation, formula, or distinctive exercise terms. Do not paste the entire prompt."
        },
        "subject": {
          "type": "string",
          "description": "Raw corpus subject label when known. Omit rather than guess."
        },
        "grade": {
          "type": ["integer", "string"],
          "description": "Student grade when known. Omit if the task is cross-grade or uncertain."
        },
        "mode": {
          "type": "string",
          "enum": ["or", "and"],
          "default": "or"
        }
      },
      "required": ["query"]
    }
  }
}
```

The agent fixes `top_k` through `MLA_RETRIEVAL_TOP_K`; the model cannot override
it. The adapter adds that value when it calls the local retriever or
`POST http://127.0.0.1:8770/api/search`.

## Recommended agent policy

1. Extract `image_evidence`, the normalized question, topic, unknown concepts, and a compact `retrieval_query` from the screenshot. The screenshot remains the source of truth.
2. Build the first query only from the topic and unknown concepts. Do not copy task-specific numbers, answer choices, or the whole question. `top_k` is fixed by the experiment.
3. If `relevance` is `weak` or `empty`, reformulate once with distinctive terms or add a confirmed subject/grade filter. Do not repeat an identical call. Do not search again after `confident`.
4. Compare confident hits with the image evidence before exposing them to the answer-generation step. Remove chunks that contradict the task's numbers, units, entities, diagram, choices, or requested output.
5. Recheck the candidate answer against the original screenshot. Retrieved material may support a formula or definition but cannot override the image.
6. Keep a trace of query, filters, latency, returned chunk IDs, `exit_reason`, `image_evidence`, `retrieval_relevance`, `retrieval_conflict`, and `answer_source`. This is required for setup validation and ablations.

Suggested limits for the first experiment: at most two search calls (the second only after weak/empty retrieval), at most five hits exposed per call, and at most 6,000 retrieved characters added to model context. These values should be varied as explicit retrieval hyperparameters later.

## Current measured baseline

- Source chunks: 280,822.
- Indexed text chunks: 103,070.
- Image-only chunks intentionally skipped: 177,752.
- SQLite index: about 452 MB.
- Real local smoke query returned three relevant fraction-addition pages in about 32 ms.
- Chunks marked `index_policy=downweight` keep full provenance but receive 0.25 of their raw BM25 magnitude by default; both original and adjusted scores are returned.

The index is lexical and Turkish-heavy. It does not yet embed diagrams, crop page regions, perform OCR/LaTeX-aware matching, dense retrieval, cross-encoder reranking, or image-to-page retrieval. Those are experiment variants; keeping this baseline simple is methodologically useful.

## Retrieval evaluation handoff

The retrieval team can annotate relevance against the actual task IDs instead of optimizing on generic search examples:

```powershell
vlm-judge prepare-retrieval-qrels `
  --benchmark artifacts\math_benchmark.jsonl `
  --output artifacts\retrieval\math_qrels_template.jsonl
```

Each row accepts either `relevant_page_ids` (recommended for page/image retrieval) or `relevant_chunk_ids` (for chunk-level relevance). The current 200-task math benchmark is image-only, so all generated queries are marked `needs_manual_query`; topic metadata is only a starting hint and must be checked against the screenshot.

After annotation, every retriever variant can be measured with the same qrels:

```powershell
vlm-judge evaluate-retrieval `
  --index artifacts\retrieval\bm25.sqlite `
  --qrels artifacts\retrieval\math_qrels.jsonl `
  --k 1 --k 5 --k 10 --k 20 `
  --output artifacts\reports\bm25_retrieval_eval.json
```

The report contains Hit@k, Recall@k, MRR@k, binary nDCG@k, mean/median/p95 latency, and per-query ranked provenance. This separates “retriever found useful evidence” from “agent used it correctly,” which is necessary to explain end-to-end gains or regressions.
