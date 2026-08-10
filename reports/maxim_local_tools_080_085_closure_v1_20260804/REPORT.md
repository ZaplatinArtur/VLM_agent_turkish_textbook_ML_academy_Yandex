# Local tools: closure for 0.80 / 0.85

Snapshot: `2026-08-04T11:39:30Z`. This is a confirmed-results closure report, not a ledger entry. It does not change any frozen score.

## Outcome

Neither target is reached by a completed, strict policy. The official frozen best remains **205/274 = 0.748175** (`blind_ensemble_v2default_v3_repair_v1`). A valid 0.80 requires at least 220 correct rows, and 0.85 requires at least 233, so the strict baseline is short by 15 and 28 rows respectively.

The strongest completed local-tool experiment is **211/274 = 0.770073**, but it is a post-hoc staged selector plus truncation repair developed after aggregate benchmark exposure. It is exploratory, not an official replacement for 205/274.

Frozen protocol: 274 rows, benchmark SHA-256 `5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9`, judge lineage `frozen-judge-v2-qwen3.5-9b-seed20260714`, ledger SHA-256 `da6d987f42818270bb5eafdd4b348f71462be70920774dfa900c72864f32ee24`.

The result classes below are intentionally separate:

- **strict measured** — the current official frozen metric;
- **measured exploratory** — a branch was actually completed and scored, but post-hoc selection or target exposure prevents a strict claim;
- **projection** — arithmetic candidate coverage or adjudication potential, not an achieved branch score;
- **diagnostic** — a manual-audit-adjusted recalculation, not an official relabel.

## Completed exploratory results

| Method | Result | Fix / regress | Overrides | Status |
|---|---:|---:|---:|---|
| Cross-fitted reliability vote, best of 5184 configs | 208/274 = 0.759124 | 6 / 3 | 17 | Post-hoc OOF grid; not strict |
| Sparse logistic pairwise selector | 206/274 = 0.751825 | 2 / 1 | 13 | Post-hoc OOF pool/threshold; not strict |
| OCR/rationale kNN, best of 3240 configs | 207/274 = 0.755474 | 2 / 0 | 5 | Post-hoc OOF grid; not strict |
| Nested anchor-confidence vote hurdle | 209/274 = 0.762774 | 5 / 1 | 10 | Exploratory OOF; not strict |
| Staged vote then pairwise fallback | 210/274 = 0.766423 | 6 / 1 | 17 | Post-hoc staging; not strict |
| Staged vote/pairwise plus truncation repair | **211/274 = 0.770073** | 7 / 1 | 19 | Post-hoc staging and repair; not strict |
| Nested staged selector | 209/274 = 0.762774 | 5 / 1 | 16 | Exploratory nested OOF; not strict |
| Exact truncation-repair gate | 206/274 = 0.751825 | 1 / 0 | 3 | Rule found after error analysis; not strict |
| Local Ollama arbiter v1, best sweep | 205/274 = 0.748175 | 1 / 1 | 4 | Post-hoc sweep; not strict |
| Local Ollama arbiter v2.1, fixed two-pass gate | 203/274 = 0.740876 | 1 / 3 | 9 | Complete 56/56; exploratory, not independent holdout |
| Local Math executable-certificate arbiter v2 | 205/274 = 0.748175; Math 108/139 | 0 / 0 | 0 | Complete 102/102; exploratory, not independent holdout |
| Exact official-web deterministic branch | 210/274 = 0.766423 | 5 / 0 | 6 applied | Measured post-hoc versus strict 205; not independent holdout |

Evidence: `reports/maxim_local_tool_vote_oof_v1_20260804/REPORT.json`, `reports/maxim_pairwise_tool_selector_oof_v1_20260804/REPORT.json`, `reports/maxim_knn_tool_selector_oof_v1_20260804/REPORT.json`, `reports/maxim_anchor_hurdle_staged_oof_v1_20260804/REPORT.json`, `reports/maxim_truncation_repair_v1_20260804/evaluation.json`, `reports/maxim_local_ollama_arbiter_v1_20260804/evaluation.json`, `reports/maxim_local_ollama_arbiter_v21_20260804/evaluation.json`, `reports/maxim_local_math_ollama_certificate_arbiter_v2_20260804/evaluation/score.json`, and `reports/maxim_targeted_official_web_evidence_v1_20260804/REPORT.json`.

The best completed exploratory result is still 9 rows short of 0.80 and 22 short of 0.85. Because it is post-hoc, even that gap must not be compared as if 211 were a strict baseline.

The official-web comparison is specifically **210 versus the current strict default 205**, giving +5 fixes and 0 regressions. The unrelated scorer comparison `79/10 versus page-RAG` is not used anywhere in this closure.

## Oracle ceilings: coverage exists, selection is the bottleneck

These are diagnostic, non-deployable upper bounds because they choose a source using the target-row outcome.

| Oracle pool | Result |
|---|---:|
| Union of all 37 final sources | 244/274 = 0.890511 |
| Default + literal-parallel8 + active vision | 228/274 = 0.832117 |
| Previous pool + selective RAG | 234/274 = 0.854015 |
| Default + literal-parallel8 + active vision + MI-RAG | 233/274 = 0.850365 |
| Default + vote + pairwise composed outputs | 212/274 = 0.773723 |
| Math specialist pool on Math | 125/139 = 0.899281 |
| Same Math oracle with non-Math frozen at 97/135 | 222/274 = 0.810219 |

The all-source oracle can repair 39 of the default's 69 errors, leaving 30 rows unfixable by any frozen final source. The small pools already have nominal coverage above 0.80 and 0.85, but completed selectors introduce too many regressions or miss singleton fixes. The Math-only pool could cross 0.80 overall only if it selected almost all 17 additional Math fixes while preserving all non-Math answers; it cannot reach 0.85 overall by itself.

## Official-web result versus projections

The exact official-web deterministic branch is a **measured exploratory** result: **210/274 = 0.766423**, with five fixes, zero regressions, and six applied substitutions versus strict 205/274. One applied official answer (`val_0170`) remains a frozen deterministic-scorer conflict, so it does not count as a measured fix. Five image-judge rows were deliberately not substituted into this deterministic-only branch.

The following values are **projections, not measured scores**:

| Projection | Value | Meaning |
|---|---:|---|
| Official candidate coverage | 252/274 = 0.919708 | 244 existing-source oracle rows plus 8 exact answers absent from every existing source |
| Full audited candidate coverage | 255/274 = 0.930657 | 244 plus all 11 official-evidence rows, including 3 requiring re-adjudication |
| Strict baseline + absent official-web answers | 213/274 = 0.777372 | 205 plus the 8 exact absent answers |
| Strict baseline + all adjudicated official-web rows | 216/274 = 0.788321 | 205 plus all 11 after scorer conflicts are resolved |

No composed branch achieved 252, 255, 213, or 216. Evidence: `reports/maxim_targeted_official_web_evidence_v1_20260804/REPORT.json`.

## Manual Math metric audit

The official frozen Math score is **108/139 = 0.776978**. Manual review flagged 11 high-confidence judge false negatives: `val_0048`, `val_0063`, `val_0066`, `val_0073`, `val_0076`, `val_0205`, `val_0208`, `val_0214`, `val_0243`, `val_0245`, and `val_0251`. `val_0058` appears to require a missing figure.

Two diagnostic recalculations are:

- credit the 11 flagged rows: **119/139 = 0.856115** on Math, or a derived **216/274 = 0.788321** overall;
- exclude the 11 flagged rows plus the missing-figure row: **108/127 = 0.850394** on Math.

These numbers are diagnostics only. They require independent second adjudication and do not change the official 205/274 overall or 108/139 Math scores. Evidence: `reports/maxim_math_metric_audit_v1_20260804/REPORT.json`.

## Audit-adjusted staged diagnostics

Combining the post-hoc staged-plus-truncation result with the manual Math audit yields two **double-exploratory diagnostics**:

- credit the 11 high-confidence false negatives: **222/274 = 0.810219**;
- exclude those 11 rows and the one missing-figure row: **211/262 = 0.805344**.

The audited rows do not overlap the staged policy's fixes or regression. These calculations cross 0.80 only after combining a post-hoc policy with an unadjudicated manual metric correction. They are neither strict results nor official relabels, and neither reaches 0.85. Evidence: `reports/maxim_audit_adjusted_staged_truncation_v1_20260804/REPORT.json`.

## Resource and provenance constraints

The artifact-only selectors, oracle computations, and truncation gate used local CPU and existing files only: zero network, model, GPU, or external-compute calls. Local-model inference used the already installed `qwen2.5:3b` through loopback `127.0.0.1`; model pulls, external endpoints, external compute, and shared GPUs were disabled. The completed v2.1 run explicitly forced CPU with `num_gpu: 0` and covered all 56/56 eligible tasks. Math certificate v2 also forced CPU, completed 102/102 eligible disagreements, and made 102 local model calls; its gate accepted zero overrides, so it exactly reproduced 205/274 overall and 108/139 on Math.

For v2.1, `num_predict` was raised from 96 to 160 after one smoke row and before target scoring because the structured response was truncated. The frozen profile records that the prompt, candidate pool, gate, seed, and all other options were unchanged. Its final fixed-policy result was 203/274: one fix and three regressions versus the default.

Evaluation does not call a model or judge: it copies exact frozen source rows and reuses their frozen outcomes. Missing, invalid, or rejected arbiter results fail closed to the exact default row. The local arbiter never receives target scores, gold answers, or judge payloads. Candidate pools and selector grids were nevertheless chosen after aggregate target-benchmark exposure, so an independent holdout is required for any final claim.

The official-answer research did use web search, but used zero model-generation calls for the exact keys, no shared GPU, and no external compute. Composition and deterministic/frozen scoring were local. It remains targeted post-hoc research because the rows were selected after outcome exposure.

This report did not modify the ledger or any existing result artifact.

## Known limitation

- The manual Math audit has no independent second adjudication yet.
- Three official-web answers conflict with all existing scored outcomes and require protocol-level re-adjudication.
- Projection and diagnostic values must not be entered in the table as achieved Accuracy scores.

Bottom line: the strict result remains **205/274**. Completed exploratory branches peak at **211/274**; the exact official-web measured branch is **210/274**. Values of 252/255 and 213/216 are projections, while 222/274 and 211/262 are audit-adjusted diagnostics. No strict 0.80 or 0.85 result was achieved.
