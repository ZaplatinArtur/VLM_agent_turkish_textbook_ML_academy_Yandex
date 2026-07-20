# Current status and handoff

## Ready without compute

- both shared workbooks normalized into benchmark JSONL;
- 45,920-row textbook corpus audited and reduced to 45,576 canonical pages;
- 280,822 provenance-preserving retrieval units prepared: 103,070 text chunks and 177,752 image chunks;
- 452 MB SQLite FTS5/BM25 index built over all 103,070 text chunks; local search API verified on real corpus data at about 32 ms for a smoke query;
- all 400 math question/reference assets verified;
- deterministic metrics for choice, numeric, and short answers;
- blind multimodal judge prompt, strict output schema, retry/cache runner, and working OpenAI-compatible Qwen/vLLM backend (including image URL or cached data-URL input);
- 120-task stratified human calibration selection and 150 synthetic mechanics tests;
- imports from JSON/JSONL/CSV/TSV with explicit preservation of empty failures;
- LMArena-style randomized/mirrored pair generation;
- local labeling UI with adjacent expected/candidate panes, Arena mode, task-scoped gold transcription, and human-vs-LLM adjudication;
- human-vs-judge agreement report with Wilson intervals, 5-score macro-F1, confusion matrix, weighted kappa, per-setup/subject/type slices, selective confidence curves, and mirrored-position audit;
- reproducible adjudication queue: all disagreements/errors/reference issues/low-confidence cases plus a stable agreement control sample;
- three-run contract validator covering completeness, setup drift, duplicate/unknown IDs, failures, retrieval traces, and file hashes;
- synthetic 106-task/318-output end-to-end dry run with dashboard; explicitly excluded from quality claims;
- hybrid aggregation with bootstrap intervals, setup coverage checks, and separate deterministic/judge views.

## External dependencies

1. Exact Qwen checkpoint, OpenAI-compatible endpoint URL/model identifier, and GPU limits.
2. Candidate outputs for all three setups under the shared contract.
3. Mentor's gold set and confirmation of which records form calibration versus held-out evaluation.
4. Mapping for 771 unresolved main-workbook question asset IDs.
5. Confirmed mapping from 20 raw workbook labels to the planned 17 categories. Until then, raw labels remain untouched.

## Estimated effort after access arrives

- Qwen deployment connection and 150-case smoke test: roughly 2–4 hours; the adapter and cache/retry path are already implemented.
- Human labeling of 120 visual tasks: roughly 3–6 hours per annotator, then 1–2 hours for adjudication.
- Judge calibration, prompt freeze, and regression checks: 1–2 days.
- Full three-setup run plus retries: 0.5–1 day excluding model queue time.
- Statistical report and failure analysis: 0.5–1 day.

The largest schedule risk is not GPU speed; it is incomplete task/setup grids, unresolved images, or late changes to gold/schema. Candidate and retrieval teams should adopt the data contract before generating runs.
