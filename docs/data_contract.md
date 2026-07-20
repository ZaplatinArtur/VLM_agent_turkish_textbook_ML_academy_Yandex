# Evaluation data contract

The experiment must emit one JSONL record per `(task_id, setup, run_id)`.

```json
{
  "task_id": "m0101",
  "subject": "Mathematics",
  "grade": 1,
  "answer_type": "multi_answer",
  "setup": "textbook_retrieval",
  "question_image_url": "https://...",
  "reference_image_url": "https://...",
  "reference_answer": null,
  "acceptable_answers": [],
  "candidate_answer": "...",
  "metadata": {
    "run_id": "run-001",
    "pairing_id": "seed-42-replicate-1",
    "agent_model": "...",
    "agent_prompt_version": "...",
    "seed": 42,
    "retrieval_calls": 1,
    "retrieval_config_hash": "...",
    "retrieved_chunk_ids": ["book:page:chunk"],
    "tool_trace": [],
    "latency_ms": 1200,
    "input_tokens": 1000,
    "output_tokens": 250
  }
}
```

Required team guarantees:

- the same `task_id` is used for all three setups;
- setup names are exactly `no_tools`, `web_search`, or `textbook_retrieval`;
- candidate output is preserved verbatim;
- failed, empty, and timed-out runs are recorded rather than silently removed;
- model, prompt, retrieval configuration, seed/run ID, latency, and token usage are logged;
- setup-specific `run_id` values do not define pairing; if there are repeated stochastic runs, provide the same `pairing_id` or `replicate_id` for corresponding records across all three setups;
- every retrieved chunk is referenced by a stable ID and rank, with score and index version;
- benchmark references are never included in agent context.

`annotation_id` in the human UI is setup-aware (`task_id::setup`). This prevents labels for the three candidate answers of one task from overwriting each other. Gold transcriptions are intentionally keyed only by `task_id`; adjudications use `adj::task_id::setup`. Pairwise records use their stable `pair_id` and retain the hidden A/B-to-setup mapping outside the visible comparison.

Human pointwise labels use the same score mapping as the LLM judge: `0=incorrect`, `1/2=partially_correct`, `3=mostly_correct`, `4=fully_correct`; `strict_correct` is true only at score 4. The server rejects type coercion, out-of-range confidence/scores, inconsistent label/score pairs, and completed records without a primary pointwise score or pairwise winner. Verified readable gold requires both an explicit quality (`clear` or `ambiguous`) and a transcription or required subanswers; `unknown`, `incorrect`, and `unreadable` gold is never substituted into judge input.

The judge prompt does not contain the setup label, task identifier, or arbitrary metadata. Task IDs are removed because source/split names and synthetic suffixes can accidentally encode expected outcomes. Run IDs, model names, retrieval traces, expected synthetic labels, latency, and tool configuration are also stripped at the prompt boundary. Only the explicit safe fields `required_subanswers`, `reference_notes`, and `gold_quality` can cross from metadata into the judge prompt. Cache IDs combine a hash of the actual blinded prompt/image list with a hash of model, endpoint, decoding, JSON mode, and image mode. Prompt or backend-configuration changes therefore cannot silently reuse stale verdicts.
