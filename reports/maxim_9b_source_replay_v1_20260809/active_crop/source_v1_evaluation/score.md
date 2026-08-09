# Full-274 score: maxim_9b_active_crop_source_v1_materialized

New: **218/274 (79.56%)**; frozen page-RAG: **141/274 (51.46%)**; delta **+77 correct / +28.102 pp**.

Math: **106/139 (76.26%)** vs frozen **62/139 (44.60%)**; delta **+44 / +31.655 pp**.

## Score-source split

| Source | n | New correct | Frozen correct | Delta pp | Fixed | Regressed |
|---|---:|---:|---:|---:|---:|---:|
| deterministic | 177 | 148 | 96 | +29.379 | 60 | 8 |
| image_judge | 97 | 70 | 45 | +25.773 | 28 | 3 |

## By subject

| Subject | n | New | New accuracy | Frozen | Frozen accuracy | Delta pp | Fixed | Regressed | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATATÜRKÇÜLÜK | 1 | 0 | 0.00% | 1 | 100.00% | -100.000 | 0 | 1 | 0 |
| Biology | 19 | 12 | 63.16% | 8 | 42.11% | +21.053 | 4 | 0 | 0 |
| Chemistry | 32 | 30 | 93.75% | 19 | 59.38% | +34.375 | 11 | 0 | 0 |
| English | 9 | 8 | 88.89% | 7 | 77.78% | +11.111 | 1 | 0 | 0 |
| Geography | 14 | 14 | 100.00% | 11 | 78.57% | +21.429 | 3 | 0 | 0 |
| History | 10 | 6 | 60.00% | 6 | 60.00% | +0.000 | 1 | 1 | 0 |
| Math | 139 | 106 | 76.26% | 62 | 44.60% | +31.655 | 51 | 7 | 0 |
| Philosophy | 2 | 1 | 50.00% | 2 | 100.00% | -50.000 | 0 | 1 | 0 |
| Physics | 19 | 19 | 100.00% | 12 | 63.16% | +36.842 | 7 | 0 | 0 |
| Science | 5 | 5 | 100.00% | 2 | 40.00% | +60.000 | 3 | 0 | 0 |
| Sociology | 3 | 2 | 66.67% | 2 | 66.67% | +0.000 | 0 | 0 | 0 |
| Turkish language and literature | 21 | 15 | 71.43% | 9 | 42.86% | +28.571 | 7 | 1 | 0 |

## Operational

- Solver errors: 0; missing answers: 0; generation-failure union: 0.
- Tokens: input 288993, output 1156807, combined 1445800 (reported rows: 274/274 input, 274/274 output).
- Latency: total 55771.489 s; mean 203.546 s; median 90.624 s; p95 811.978 s; reported rows 274/274.
- Model calls: 0; reported rows 0/274.

## Paired changes vs frozen page-RAG

- Fixed (88): val_0003, val_0008, val_0011, val_0013, val_0016, val_0018, val_0019, val_0020, val_0022, val_0025, val_0027, val_0028, val_0035, val_0036, val_0037, val_0038, val_0039, val_0044, val_0047, val_0049, val_0059, val_0064, val_0068, val_0069, val_0072, val_0078, val_0080, val_0081, val_0082, val_0084, val_0090, val_0096, val_0097, val_0102, val_0109, val_0110, val_0114, val_0115, val_0116, val_0119, val_0120, val_0123, val_0127, val_0128, val_0130, val_0131, val_0137, val_0139, val_0141, val_0147, val_0148, val_0154, val_0155, val_0166, val_0168, val_0173, val_0177, val_0184, val_0191, val_0202, val_0203, val_0209, val_0217, val_0219, val_0221, val_0223, val_0224, val_0226, val_0228, val_0229, val_0234, val_0235, val_0237, val_0239, val_0240, val_0242, val_0244, val_0246, val_0250, val_0254, val_0256, val_0258, val_0261, val_0270, val_0271, val_0272, val_0273, val_0275
- Regressed (11): val_0057, val_0067, val_0086, val_0089, val_0150, val_0196, val_0197, val_0213, val_0216, val_0243, val_0248

## Provenance

| Input | SHA256 | Path |
|---|---|---|
| benchmark | `5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\artifacts\baselines\basic_page_rag_v1\validation_274.jsonl` |
| solver_results | `de864af37cfafc9e283127bb798ceeef9f014f871ea1cb0f3dbd6d5bc77de1f4` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\reports\maxim_9b_source_replay_v1_20260809\active_crop\source_v1_composed\solver.jsonl` |
| image_judge | `2bd8200d623f373286111cf634befe9dc73bef1d668aa261869866ff484b0d2b` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\reports\maxim_9b_source_replay_v1_20260809\active_crop\source_v1_evaluation\matched_image97_judge.jsonl` |
| frozen_page_rag_judge | `59dcc93454b29dfc65b0a9b1243a177d472b6c0a13cbe46fb5c98079810a73f4` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\artifacts\baselines\basic_page_rag_v1\agent_rag_judge.jsonl` |
| scorer | `bca10e6546b68f8a66eb4d68aa13316429e4689789be1bef14cd592955e4eacf` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\scripts\score_maxim_full274.py` |

Report JSON SHA256: `33eb8dff8a3142666329c34c4156443bb90b4bcb0b429320e5637a5f036fb053`.

The benchmark reference is read only after generation. Candidate and reference answers are intentionally omitted from this report.
