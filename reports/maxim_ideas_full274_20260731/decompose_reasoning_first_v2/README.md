# Decompose reasoning-first v2: paired audit

## Verdict

The decomposition treatment reaches **150/274 (54.74%)**. It is better than the matched direct v2 control at **137/274 (50.00%)**, but the paired evidence is still only suggestive: **+13 answers, +4.75 pp, 33 fixed / 20 regressed, exact McNemar p=0.0984**. On Math the signal is stronger but remains just outside 0.05: **75/139 vs 65/139, +7.19 pp, 18/8, p=0.0755**.

This is not yet a promotion candidate. The treatment doubles model calls, uses **2.17x** as many combined tokens, and the observed quality gain is not statistically confirmed on this single run.

## Main paired comparisons

`fixed/regressed` always means decomposition correct/direct comparator wrong, then decomposition wrong/comparator correct. All p-values are two-sided exact McNemar tests.

| Slice | Decompose | Comparator | Delta | Fixed / regressed | Exact p |
|---|---:|---:|---:|---:|---:|
| Overall vs matched direct v2 | 150/274 | 137/274 | +13 (+4.75 pp) | 33 / 20 | 0.0984 |
| Math vs matched direct v2 | 75/139 | 65/139 | +10 (+7.19 pp) | 18 / 8 | 0.0755 |
| Overall vs frozen page-RAG | 150/274 | 141/274 | +9 (+3.29 pp) | 40 / 31 | 0.3425 |
| Math vs frozen page-RAG | 75/139 | 62/139 | +13 (+9.35 pp) | 27 / 14 | 0.0596 |
| Overall vs frozen no-tools | 150/274 | 191/274 | -41 (-14.96 pp) | 28 / 69 | 0.0000380 |
| Math vs frozen no-tools | 75/139 | 105/139 | -30 (-21.58 pp) | 12 / 42 | 0.0000521 |

The no-tools comparison needs a protocol caveat: its 97 judged rows were produced with `presentation-hybrid-v1`, while both v2 runs use `judge-v2`. Therefore its aggregate and image-judge p-values are historical-reference comparisons, not fully judge-matched controls. The deficit is nevertheless already decisive on the **177 deterministic rows**: 100 vs 132, 12/44, p=0.0000209.

## Score-source slices

| Comparator | Source | Decompose | Comparator | Delta pp | Fixed / regressed | Exact p |
|---|---|---:|---:|---:|---:|---:|
| matched direct v2 | deterministic | 100/177 | 91/177 | +5.09 | 22 / 13 | 0.1755 |
| matched direct v2 | image judge | 50/97 | 46/97 | +4.12 | 11 / 7 | 0.4807 |
| frozen page-RAG | deterministic | 100/177 | 96/177 | +2.26 | 22 / 18 | 0.6358 |
| frozen page-RAG | image judge | 50/97 | 45/97 | +5.16 | 18 / 13 | 0.4731 |
| frozen no-tools | deterministic | 100/177 | 132/177 | -18.08 | 12 / 44 | 0.0000209 |
| frozen no-tools | image judge | 50/97 | 59/97 | -9.28 | 16 / 25 | 0.2110* |

`*` Judge protocol is not matched for the frozen no-tools image slice.

## Subject slices

Each comparison cell is `delta correct; fixed/regressed; exact p`. These unadjusted small-slice tests are exploratory.

| Subject | n | Decompose / direct / page / no-tools | vs direct | vs page | vs no-tools |
|---|---:|---:|---:|---:|---:|
| ATATÜRKÇÜLÜK | 1 | 0 / 1 / 1 / 0 | -1; 0/1; 1.000 | -1; 0/1; 1.000 | 0; 0/0; 1.000 |
| Biology | 19 | 8 / 7 / 8 / 11 | +1; 3/2; 1.000 | 0; 1/1; 1.000 | -3; 0/3; 0.250 |
| Chemistry | 32 | 19 / 19 / 19 / 20 | 0; 2/2; 1.000 | 0; 4/4; 1.000 | -1; 4/5; 1.000 |
| English | 9 | 7 / 6 / 7 / 5 | +1; 1/0; 1.000 | 0; 0/0; 1.000 | +2; 3/1; 0.625 |
| Geography | 14 | 8 / 6 / 11 / 12 | +2; 4/2; 0.688 | -3; 1/4; 0.375 | -4; 1/5; 0.219 |
| History | 10 | 4 / 5 / 6 / 5 | -1; 0/1; 1.000 | -2; 0/2; 0.500 | -1; 1/2; 1.000 |
| Math | 139 | 75 / 65 / 62 / 105 | +10; 18/8; 0.0755 | +13; 27/14; 0.0596 | -30; 12/42; 0.0000521 |
| Philosophy | 2 | 2 / 2 / 2 / 1 | 0; 0/0; 1.000 | 0; 0/0; 1.000 | +1; 1/0; 1.000 |
| Physics | 19 | 13 / 12 / 12 / 14 | +1; 2/1; 1.000 | +1; 4/3; 1.000 | -1; 3/4; 1.000 |
| Science | 5 | 1 / 1 / 2 / 3 | 0; 0/0; 1.000 | -1; 0/1; 1.000 | -2; 0/2; 0.500 |
| Sociology | 3 | 2 / 2 / 2 / 2 | 0; 0/0; 1.000 | 0; 0/0; 1.000 | 0; 0/0; 1.000 |
| Turkish language and literature | 21 | 11 / 11 / 9 / 13 | 0; 3/3; 1.000 | +2; 3/1; 0.625 | -2; 3/5; 0.727 |

No subject-level claim should be made from the tiny slices, and no multiplicity correction was applied. The main useful signal is concentrated in Math; Geography and History regress relative to frozen page-RAG.

## Compute and latency

| Metric | Decompose v2 | Matched direct v2 | Ratio |
|---|---:|---:|---:|
| Model calls | 548 | 274 | 2.00x |
| Input tokens | 471,210 | 200,597 | 2.35x |
| Output tokens | 176,623 | 97,364 | 1.81x |
| Combined tokens | 647,833 | 297,961 | 2.17x |
| Mean latency per task | 12.437 s | 7.552 s | 1.65x |
| Median latency per task | 11.124 s | 5.765 s | 1.93x |
| p95 latency | 23.402 s | 16.950 s | 1.38x |

The matched direct run controls model, benchmark, vision input, zero temperature, answer schema and the reasoning-first contract. It is not compute-matched, so the +13 result estimates the complete two-call treatment, not a compute-neutral decomposition effect.

## Fail-closed retry audit

The initial 274-row run recorded three structured-output truncations after two endpoint attempts each. The operational IDs are `val_0059`, `val_0072`, and `val_0212`; no answers or rationales are reproduced here.

- Two numeric rows (`val_0059`, `val_0212`) were regenerated with a constrained compact schema and atomically replaced by task ID. Both archived retry rows match their final solver rows exactly.
- `val_0072` has a healthy standard-v2 row in the final solver. Its initial failure is present in `generation.log`, but a standalone retry JSONL or retry command log is not archived locally. This is a narrow provenance gap, not a scoring error.
- Final artifact audit: **274 rows, 274 unique IDs, 0 errors, 0 missing answers, 0 forced answers, 0 forbidden gold fields, 0 non-false `gold_access`, 0 tool calls**.

## Judge and scoring tags

- Fixed partition: 177 deterministic rows plus 97 image-judge rows.
- v2 image tag: `judge-v2`; setup `maxim_decompose_reasoning_first_v2`.
- Judge: `Qwen/Qwen3.5-9B`, temperature 0, max tokens 900, seed 20260714, response format on, thinking off, `data_url` images.
- Direct and decomposition use the same semantic judge configuration and have zero judge/projection errors.
- Gold/reference data are only read by the post-generation scorer and are absent from solver JSONL.

Machine-readable paired counts, exact p-values, hashes and guardrails are in [diagnostics.json](diagnostics.json). Raw model answers and rationales are intentionally not copied into this audit.
