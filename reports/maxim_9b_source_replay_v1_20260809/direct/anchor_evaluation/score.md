# Full-274 score: maxim_clean_direct_9b_anchor

New: **193/274 (70.44%)**; frozen page-RAG: **141/274 (51.46%)**; delta **+52 correct / +18.978 pp**.

Math: **103/139 (74.10%)** vs frozen **62/139 (44.60%)**; delta **+41 / +29.496 pp**.

## Score-source split

| Source | n | New correct | Frozen correct | Delta pp | Fixed | Regressed |
|---|---:|---:|---:|---:|---:|---:|
| deterministic | 177 | 132 | 96 | +20.339 | 46 | 10 |
| image_judge | 97 | 61 | 45 | +16.495 | 21 | 5 |

## By subject

| Subject | n | New | New accuracy | Frozen | Frozen accuracy | Delta pp | Fixed | Regressed | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATATÜRKÇÜLÜK | 1 | 0 | 0.00% | 1 | 100.00% | -100.000 | 0 | 1 | 0 |
| Biology | 19 | 11 | 57.89% | 8 | 42.11% | +15.789 | 3 | 0 | 0 |
| Chemistry | 32 | 20 | 62.50% | 19 | 59.38% | +3.125 | 4 | 3 | 0 |
| English | 9 | 8 | 88.89% | 7 | 77.78% | +11.111 | 1 | 0 | 0 |
| Geography | 14 | 13 | 92.86% | 11 | 78.57% | +14.286 | 2 | 0 | 0 |
| History | 10 | 6 | 60.00% | 6 | 60.00% | +0.000 | 1 | 1 | 0 |
| Math | 139 | 103 | 74.10% | 62 | 44.60% | +29.496 | 48 | 7 | 0 |
| Philosophy | 2 | 1 | 50.00% | 2 | 100.00% | -50.000 | 0 | 1 | 0 |
| Physics | 19 | 13 | 68.42% | 12 | 63.16% | +5.263 | 2 | 1 | 0 |
| Science | 5 | 3 | 60.00% | 2 | 40.00% | +20.000 | 1 | 0 | 0 |
| Sociology | 3 | 2 | 66.67% | 2 | 66.67% | +0.000 | 0 | 0 | 0 |
| Turkish language and literature | 21 | 13 | 61.90% | 9 | 42.86% | +19.048 | 5 | 1 | 0 |

## Operational

- Solver errors: 0; missing answers: 0; generation-failure union: 0.
- Tokens: input 248567, output 1247543, combined 1496110 (reported rows: 274/274 input, 274/274 output).
- Latency: total 67751.65 s; mean 247.269 s; median 94.74 s; p95 1045.547 s; reported rows 274/274.
- Model calls: 0; reported rows 0/274.

## Paired changes vs frozen page-RAG

- Fixed (67): val_0011, val_0013, val_0016, val_0019, val_0020, val_0025, val_0035, val_0036, val_0038, val_0039, val_0044, val_0047, val_0049, val_0059, val_0064, val_0068, val_0069, val_0072, val_0078, val_0080, val_0081, val_0082, val_0090, val_0096, val_0097, val_0114, val_0119, val_0120, val_0127, val_0147, val_0148, val_0154, val_0155, val_0166, val_0168, val_0177, val_0184, val_0191, val_0202, val_0203, val_0209, val_0217, val_0219, val_0221, val_0223, val_0224, val_0226, val_0228, val_0229, val_0234, val_0235, val_0237, val_0239, val_0240, val_0242, val_0244, val_0246, val_0250, val_0254, val_0256, val_0258, val_0261, val_0270, val_0271, val_0272, val_0273, val_0275
- Regressed (15): val_0057, val_0067, val_0086, val_0089, val_0124, val_0126, val_0129, val_0132, val_0150, val_0196, val_0197, val_0213, val_0216, val_0243, val_0248

## Provenance

| Input | SHA256 | Path |
|---|---|---|
| benchmark | `5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\artifacts\baselines\basic_page_rag_v1\validation_274.jsonl` |
| solver_results | `496236da966ed68aa81af3d33da1c40b85c5a11b342de253ada244f97320de8f` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\artifacts\baselines\no_tools_v1\b0_no_tools_raw.jsonl` |
| image_judge | `28f3d107a0840970e0f82c46157cd85c0f6200f82e1cf105ff39409e58c636b5` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\reports\maxim_paired_rag_norag_semantic_support_v1_20260803\evaluation_orchestrated\matched_image97_judge.jsonl` |
| frozen_page_rag_judge | `59dcc93454b29dfc65b0a9b1243a177d472b6c0a13cbe46fb5c98079810a73f4` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\artifacts\baselines\basic_page_rag_v1\agent_rag_judge.jsonl` |
| scorer | `bca10e6546b68f8a66eb4d68aa13316429e4689789be1bef14cd592955e4eacf` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\scripts\score_maxim_full274.py` |

Report JSON SHA256: `6906511e1b86ff424e831d967ea53b0a2ce967e64e9307327958ba0dae19d6c2`.

The benchmark reference is read only after generation. Candidate and reference answers are intentionally omitted from this report.
