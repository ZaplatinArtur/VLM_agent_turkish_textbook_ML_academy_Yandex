# Full-274 score: maxim_9b_active_crop_source_v6

New: **235/274 (85.77%)**; frozen page-RAG: **141/274 (51.46%)**; delta **+94 correct / +34.307 pp**.

Math: **108/139 (77.70%)** vs frozen **62/139 (44.60%)**; delta **+46 / +33.094 pp**.

## Score-source split

| Source | n | New correct | Frozen correct | Delta pp | Fixed | Regressed |
|---|---:|---:|---:|---:|---:|---:|
| deterministic | 177 | 156 | 96 | +33.898 | 67 | 7 |
| image_judge | 97 | 79 | 45 | +35.052 | 37 | 3 |

## By subject

| Subject | n | New | New accuracy | Frozen | Frozen accuracy | Delta pp | Fixed | Regressed | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATATÜRKÇÜLÜK | 1 | 0 | 0.00% | 1 | 100.00% | -100.000 | 0 | 1 | 0 |
| Biology | 19 | 16 | 84.21% | 8 | 42.11% | +42.105 | 8 | 0 | 0 |
| Chemistry | 32 | 31 | 96.88% | 19 | 59.38% | +37.500 | 12 | 0 | 0 |
| English | 9 | 8 | 88.89% | 7 | 77.78% | +11.111 | 1 | 0 | 0 |
| Geography | 14 | 14 | 100.00% | 11 | 78.57% | +21.429 | 3 | 0 | 0 |
| History | 10 | 9 | 90.00% | 6 | 60.00% | +30.000 | 4 | 1 | 0 |
| Math | 139 | 108 | 77.70% | 62 | 44.60% | +33.094 | 53 | 7 | 0 |
| Philosophy | 2 | 1 | 50.00% | 2 | 100.00% | -50.000 | 0 | 1 | 0 |
| Physics | 19 | 19 | 100.00% | 12 | 63.16% | +36.842 | 7 | 0 | 0 |
| Science | 5 | 5 | 100.00% | 2 | 40.00% | +60.000 | 3 | 0 | 0 |
| Sociology | 3 | 3 | 100.00% | 2 | 66.67% | +33.333 | 1 | 0 | 0 |
| Turkish language and literature | 21 | 21 | 100.00% | 9 | 42.86% | +57.143 | 12 | 0 | 0 |

## Operational

- Solver errors: 0; missing answers: 0; generation-failure union: 0.
- Tokens: input 288993, output 1156807, combined 1445800 (reported rows: 274/274 input, 274/274 output).
- Latency: total 55771.489 s; mean 203.546 s; median 90.624 s; p95 811.978 s; reported rows 274/274.
- Model calls: 0; reported rows 0/274.

## Paired changes vs frozen page-RAG

- Fixed (104): val_0003, val_0008, val_0011, val_0013, val_0016, val_0018, val_0019, val_0020, val_0022, val_0025, val_0027, val_0028, val_0035, val_0036, val_0037, val_0038, val_0039, val_0044, val_0047, val_0049, val_0059, val_0062, val_0064, val_0066, val_0068, val_0069, val_0072, val_0078, val_0080, val_0081, val_0082, val_0084, val_0087, val_0088, val_0090, val_0092, val_0094, val_0096, val_0097, val_0101, val_0102, val_0109, val_0110, val_0114, val_0115, val_0116, val_0119, val_0120, val_0123, val_0127, val_0128, val_0130, val_0131, val_0137, val_0139, val_0141, val_0147, val_0148, val_0154, val_0155, val_0162, val_0163, val_0164, val_0166, val_0168, val_0173, val_0177, val_0179, val_0184, val_0186, val_0189, val_0191, val_0194, val_0199, val_0200, val_0202, val_0203, val_0209, val_0217, val_0219, val_0221, val_0223, val_0224, val_0226, val_0228, val_0229, val_0234, val_0235, val_0237, val_0239, val_0240, val_0242, val_0244, val_0246, val_0250, val_0254, val_0256, val_0258, val_0261, val_0270, val_0271, val_0272, val_0273, val_0275
- Regressed (10): val_0057, val_0067, val_0086, val_0089, val_0150, val_0196, val_0213, val_0216, val_0243, val_0248

## Provenance

| Input | SHA256 | Path |
|---|---|---|
| benchmark | `5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\artifacts\baselines\basic_page_rag_v1\validation_274.jsonl` |
| solver_results | `2d18fc738b3757590cb3c92e7c0a76ad9410620d6bf13a928198f3a8df42ca19` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\reports\maxim_9b_source_replay_v1_20260809\active_crop\v6_composed\solver.jsonl` |
| image_judge | `e06e3fb36a7ef591da579c3ebe49636d535c56afa7dd180dd10c61407b3e155e` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\reports\maxim_9b_source_replay_v1_20260809\active_crop\v6_evaluation\matched_image97_judge.jsonl` |
| frozen_page_rag_judge | `59dcc93454b29dfc65b0a9b1243a177d472b6c0a13cbe46fb5c98079810a73f4` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\artifacts\baselines\basic_page_rag_v1\agent_rag_judge.jsonl` |
| scorer | `bca10e6546b68f8a66eb4d68aa13316429e4689789be1bef14cd592955e4eacf` | `C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag\scripts\score_maxim_full274.py` |

Report JSON SHA256: `0ec79eb701f9a2c69b7f4145df4ff44feca79ab6f056ab91be26cbd8bd15cd1b`.

The benchmark reference is read only after generation. Candidate and reference answers are intentionally omitted from this report.
