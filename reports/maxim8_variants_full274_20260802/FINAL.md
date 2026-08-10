# Maksim: 8 ideas on the frozen common benchmark

Benchmark: **274** tasks; frozen baseline: **141/274 (51.460%)**. Matched judge lineage: `frozen-judge-v2-qwen3.5-9b-seed20260714`.

`exact` means a score from the configured matched judge lineage; `replay` is historical/non-matched evidence and is not an exact matched result; `partial` reports measured incomplete progress only, never accuracy; `pending` has no full score yet. Pre-judge bounds are shown only for `partial`/`pending` and are not scores.

| Variant | Status | Progress | Overall | Math | Non-Math | Δ correct | Fixed / regressed | Pre-judge bounds | Model calls | Tokens | Judge lineage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Basic page-RAG | exact | — | 141/274 (51.460%) | 62/139 (44.604%) | 79/135 (58.518%) | +0 | 0 / 0 | — | — | 4 662 595 | frozen-judge-v2-qwen3.5-9b-seed20260714 (matched) |
| Solver - Critic - Repair | exact | — | 154/274 (56.204%) | 79/139 (56.834%) | 75/135 (55.556%) | +13 | 52 / 39 | — | 797 | 1 307 934 | frozen-judge-v2-qwen3.5-9b-seed20260714 (matched) |
| Selective RAG (retrieval reliability gate) | exact | — | 165/274 (60.219%) | 76/139 (54.676%) | 89/135 (65.926%) | +24 | 31 / 7 | — | 548 | 6 158 705 | frozen-judge-v2-qwen3.5-9b-seed20260714 (matched) |
| Router by task type | exact | — | 182/274 (66.423%) | 103/139 (74.101%) | 79/135 (58.518%) | +41 | 48 / 7 | — | — | 3 009 395 | frozen-judge-v2-qwen3.5-9b-seed20260714 (matched) |
| 8-way answer clustering / consensus | exact | — | 139/274 (50.730%) | 74/139 (53.237%) | 65/135 (48.148%) | -2 | 32 / 34 | — | 2 466 | 2 843 423 | frozen-judge-v2-qwen3.5-9b-seed20260714 (matched) |
| Calculator / SymPy verification | exact | — | 143/274 (52.190%) | 64/139 (46.043%) | 79/135 (58.518%) | +2 | 2 / 0 | — | 125 | 4 846 139 | frozen-judge-v2-qwen3.5-9b-seed20260714 (matched) |
| Two-pass transcription and solve | exact | — | 146/274 (53.285%) | 75/139 (53.957%) | 71/135 (52.593%) | +5 | 44 / 39 | — | 546 | 689 448 | frozen-judge-v2-qwen3.5-9b-seed20260714 (matched) |
| Answer canonicalization | exact | — | 141/274 (51.460%) | 62/139 (44.604%) | 79/135 (58.518%) | +0 | 0 / 0 | — | — | 4 662 595 | frozen-judge-v2-qwen3.5-9b-seed20260714 (matched) |
| Frozen memory of common error patterns | exact | — | 155/274 (56.569%) | 81/139 (58.273%) | 74/135 (54.815%) | +14 | 47 / 33 | — | 271 | 396 248 | frozen-judge-v2-qwen3.5-9b-seed20260714 (matched) |

## Provenance

- Benchmark SHA256: `5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9`
- Scorer SHA256: `bca10e6546b68f8a66eb4d68aa13316429e4689789be1bef14cd592955e4eacf`
- Config SHA256: `19542da2eed4287596244ca71bc5d5cfa24144dbdf9e9f418b3a0b09e932469d`
