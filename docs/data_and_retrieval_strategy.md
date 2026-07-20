# Data and retrieval strategy

## Parsing architecture

Use four immutable layers:

1. **Raw source** — original workbook, JSONL, PDF, HTML, and images with acquisition timestamp and source URL.
2. **Normalized page** — common metadata, repaired encoding, page/book identity, text blocks, formulas, tables, diagrams, and image references.
3. **Retrieval units** — content-addressed chunks with provenance and parent-page links.
4. **Experiment evidence** — exact index version, query, ranks/scores, returned chunks, agent tool trace, and final answer.

Every object should have a stable content hash. This gives reproducible indexes, makes duplicate removal auditable, and lets the evaluation layer reconstruct exactly what the agent saw.

For the current ÖdevJet export:

- keep one canonical variant per exact duplicate ID, but preserve all 263 conflicting-ID variants in a quarantine file;
- do not blindly delete all 16,520 low-information pages: exclude boilerplate text from the text index while retaining useful page images and provenance;
- chunk by layout/page semantics before applying a character/token window;
- create a page-bundle retrieval unit linking text, formulas, diagrams, and all image regions instead of treating modalities as unrelated documents;
- store raw subject labels until the 17-category mapping is confirmed.

The current CPU pass already emits 45,576 canonical pages and 280,822 stable retrieval units. Text chunks use 1,600-character windows with 200-character overlap; every page/image keeps parent hashes and source metadata. The 263 conflicting IDs are written separately instead of being silently discarded. A reproducible FTS5/BM25 baseline now indexes all 103,070 text chunks, exposes a local agent-tool API, and has a task-aligned qrels template plus Hit/Recall/MRR/latency evaluator.

Adapters worth supporting next are PDF folders, image folders plus CSV manifests, exported Google Sheets/XLSX, generic JSONL/CSV, and trusted curriculum web pages. All adapters should emit the same normalized page schema and validation report.

## The high-leverage experimental idea: causal retrieval ladder

Keep the mentor's three setups as the primary experiment. On a smaller diagnostic subset, add shadow retrieval conditions:

- real retrieved chunks;
- the same number and length of random corpus chunks;
- theory-only chunks with exact answers masked;
- an oracle chunk known to contain the solution;
- real chunks with order reversed or low-ranked evidence only.

This separates four effects that the basic three-arm comparison mixes together: access to a tool, extra context/tokens, successful evidence retrieval, and correct use of evidence. It answers not only “did textbook RAG help?” but “where in the retrieval-to-answer chain did the gain or failure occur?”

For each task, record a retrieval utility decomposition:

1. **Found** — did top-k contain sufficient evidence or the exact task?
2. **Used** — is the final reasoning supported by that evidence?
3. **Helped** — did the answer improve relative to the paired no-tools run?

These labels should be diagnostic and collected after blind answer scoring.

## Two additional high-impact ideas

### Gold compiler

Use a one-time VLM pass plus human verification to turn annotated answer images into structured gold objects: final answer, ordered subanswers, units, acceptable equivalents, and bounding boxes. Deterministic scoring can then handle many cases cheaply, while the LLM judge is reserved for genuine semantic/reasoning ambiguity. The task-level transcription/verification UI and the verified-gold compiler are now implemented; automatic VLM prefill and bounding boxes remain compute-time additions.

### Evidence graph and visual fingerprinting

Represent tasks, source pages, chunks, retrieval calls, answers, and judge verdicts as a content-addressed graph. Add exact hashes, normalized-text hashes, and perceptual image fingerprints. This detects near-duplicate benchmark/corpus pages, distinguishes exact-solution retrieval from theory retrieval, and prevents accidental cross-split leakage. It also produces mentor-friendly failure traces: `task → query → chunk → answer → verdict`.

### Screenshot-to-page retrieval instead of OCR-only RAG

The strongest multimodal variant should retrieve whole textbook pages directly from the homework screenshot before extracting text. Use a page-image encoder or late-interaction document model to get visual candidates, then fuse four signals at page level:

1. BM25 over OCR/layout text;
2. dense multilingual text similarity;
3. screenshot-to-page visual similarity;
4. perceptual/local-feature matches for exact or near-exact exercise regions.

Rerank the fused top pages with a VLM that sees the query screenshot and candidate page together, then return a compact page bundle: matching crop, surrounding theory, OCR/LaTeX, book/page identity, and provenance. This avoids the common failure where OCR destroys geometry, formulas, or diagram labels before retrieval starts. It also provides an explicit exact-task leakage detector: a strong visual match can be tagged separately from theory-only support and analyzed as its own experimental stratum.

## Retrieval metrics to request from teammates

- Recall@k / MRR / nDCG on evidence labels;
- exact-task, sufficient-theory, irrelevant, and empty retrieval rates;
- answer accuracy conditioned on those retrieval categories;
- tool-call rate, repeated-query rate, latency, and token cost;
- oracle-vs-actual gap and random-context control gap;
- performance by raw subject, grade, visual density, and answer type.
