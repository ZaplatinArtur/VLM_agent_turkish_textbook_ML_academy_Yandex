# Maxim: detailed frozen full274 report

Benchmark: `5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9` (274 tasks; Math 139 / non-Math 135).
Judge lineage: `frozen-judge-v2-qwen3.5-9b-seed20260714`.
Strict ledger: `reports/maxim_full274_results_ledger_v1_20260803/RESULTS.json` (`da6d987f42818270bb5eafdd4b348f71462be70920774dfa900c72864f32ee24`).

Only ledger-accepted `status=final` score artifacts are opened and ranked. Rejected candidates and non-final/pending scores are not read.

## Reference anchors

- Page baseline (`answer_canonicalization`): 141/274 (51.460%); Math 62/139 (44.604%); non-Math 79/135 (58.519%).
- Frozen Router (`subject_router`): 182/274 (66.423%); Math 103/139 (74.101%); non-Math 79/135 (58.519%).

## Final accepted results

Final branches: **37**. Best: **Gold-blind V2.1 default with conservative V3.1 repair**, 205/274 (74.818%).

| Rank | Branch | Overall | Math139 | nonMath135 | vs page | vs Router | Calls | Tokens | Latency | Errors | Fallbacks |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Gold-blind V2.1 default with conservative V3.1 repair | 205/274 (74.818%) | 108/139 (77.698%) | 97/135 (71.852%) | +64 (fix 74 / reg 10) | +23 (fix 30 / reg 7) | — | 1,308,556 | 39,478.0s | 0 | — |
| 2 | Gold-blind V2.1 default with conservative Active-Crop repair | 204/274 (74.453%) | 107/139 (76.978%) | 97/135 (71.852%) | +63 (fix 73 / reg 10) | +22 (fix 29 / reg 7) | — | 1,307,423 | 39,391.4s | 0 | — |
| 3 | Gold-blind V2.1 default with strict triple-agreement repair | 204/274 (74.453%) | 107/139 (76.978%) | 97/135 (71.852%) | +63 (fix 73 / reg 10) | +22 (fix 29 / reg 7) | — | 1,322,287 | 40,772.0s | 0 | — |
| 4 | Outcome-blind consensus repair: V3.1, Active-Crop, and no-tools triple | 204/274 (74.453%) | 109/139 (78.417%) | 95/135 (70.370%) | +63 (fix 72 / reg 9) | +22 (fix 27 / reg 5) | — | 1,322,326 | 40,836.3s | 0 | — |
| 5 | Outcome-blind consensus repair: V3.1 and Active-Crop pair | 204/274 (74.453%) | 109/139 (78.417%) | 95/135 (70.370%) | +63 (fix 72 / reg 9) | +22 (fix 27 / reg 5) | — | 1,322,326 | 40,836.3s | 0 | — |
| 6 | Outcome-blind consensus repair: V3.1 and no-tools pair | 204/274 (74.453%) | 109/139 (78.417%) | 95/135 (70.370%) | +63 (fix 72 / reg 9) | +22 (fix 27 / reg 5) | — | 1,322,326 | 40,836.3s | 0 | — |
| 7 | Meta-verifier v2.1 numeric choice-token compatibility | 204/274 (74.453%) | 107/139 (76.978%) | 97/135 (71.852%) | +63 (fix 73 / reg 10) | +22 (fix 29 / reg 7) | — | 1,322,287 | 40,772.0s | 0 | — |
| 8 | Frozen 10-way final meta-verifier v2 | 203/274 (74.088%) | 107/139 (76.978%) | 96/135 (71.111%) | +62 (fix 72 / reg 10) | +21 (fix 28 / reg 7) | — | 1,343,481 | 40,680.5s | 0 | — |
| 9 | Meta-verifier v3.1 numeric choice-token compatibility | 199/274 (72.628%) | 106/139 (76.259%) | 93/135 (68.889%) | +58 (fix 69 / reg 11) | +17 (fix 29 / reg 12) | — | 1,323,996 | 39,708.9s | 0 | — |
| 10 | Frozen 12-way final meta-verifier v3 | 198/274 (72.263%) | 106/139 (76.259%) | 92/135 (68.148%) | +57 (fix 68 / reg 11) | +16 (fix 28 / reg 12) | — | 1,344,887 | 39,604.5s | 0 | — |
| 11 | Outcome-blind consensus repair: Active-Crop and no-tools pair | 195/274 (71.168%) | 104/139 (74.820%) | 91/135 (67.407%) | +54 (fix 70 / reg 16) | +13 (fix 22 / reg 9) | — | 1,394,215 | 45,660.9s | 0 | — |
| 12 | Outcome-blind consensus repair: unique two-of-three majority | 194/274 (70.803%) | 104/139 (74.820%) | 90/135 (66.667%) | +53 (fix 69 / reg 16) | +12 (fix 21 / reg 9) | — | 1,379,423 | 45,385.8s | 0 | — |
| 13 | Query-conditioned active crop verifier v2 | 194/274 (70.803%) | 104/139 (74.820%) | 90/135 (66.667%) | +53 (fix 68 / reg 15) | +12 (fix 20 / reg 8) | — | 1,445,800 | 55,771.5s | 0 | — |
| 14 | Conservative Visual Sketchpad v2 | 194/274 (70.803%) | 104/139 (74.820%) | 90/135 (66.667%) | +53 (fix 68 / reg 15) | +12 (fix 20 / reg 8) | — | 1,448,242 | 55,719.3s | 0 | — |
| 15 | Evidence-gap Seeker to Inspector | 193/274 (70.438%) | 103/139 (74.101%) | 90/135 (66.667%) | +52 (fix 67 / reg 15) | +11 (fix 19 / reg 8) | 170 | 256,125 | 7,276.2s | 0 | — |
| 16 | Frozen no-tools (semantic gate selected RAG 0/274) | 193/274 (70.438%) | 103/139 (74.101%) | 90/135 (66.667%) | +52 (fix 67 / reg 15) | +11 (fix 19 / reg 8) | — | 1,496,110 | 67,751.6s | 0 | — |
| 17 | Evidence-gap fact-intent compatibility v1.1 | 190/274 (69.343%) | 100/139 (71.942%) | 90/135 (66.667%) | +49 (fix 64 / reg 15) | +8 (fix 19 / reg 11) | 295 | 773,701 | 9,683.9s | 0 | — |
| 18 | Qwen3.5-27B direct full274 | 190/274 (69.343%) | 95/139 (68.345%) | 95/135 (70.370%) | +49 (fix 68 / reg 19) | +8 (fix 31 / reg 23) | 266 | 565,780 | 32,719.6s | 0 | — |
| 19 | Blind disagreement verifier (preregistered raw selector) | 189/274 (68.978%) | 105/139 (75.540%) | 84/135 (62.222%) | +48 (fix 62 / reg 14) | +7 (fix 20 / reg 13) | 176 | — | — | 2 | — |
| 20 | Frozen subject router | 182/274 (66.423%) | 103/139 (74.101%) | 79/135 (58.519%) | +41 (fix 48 / reg 7) | +0 (fix 0 / reg 0) | — | 3,009,395 | 44,422.9s | 0 | — |
| 21 | Qwen3.5-27B hard86 composite | 180/274 (65.693%) | 101/139 (72.662%) | 79/135 (58.519%) | +39 (fix 45 / reg 6) | -2 (fix 8 / reg 10) | 82 | 2,645,780 | 30,062.2s | 0 | 7 |
| 22 | Native thinking math router v4 | 173/274 (63.139%) | 94/139 (67.626%) | 79/135 (58.519%) | +32 (fix 38 / reg 6) | -9 (fix 7 / reg 16) | 117 | 2,967,843 | 32,192.1s | 0 | — |
| 23 | Selective RAG (retrieval reliability gate) | 165/274 (60.219%) | 76/139 (54.676%) | 89/135 (65.926%) | +24 (fix 31 / reg 7) | -17 (fix 18 / reg 35) | 548 | 6,158,705 | 73,843.2s | 0 | — |
| 24 | Active vision: localize, crop, sketch, solve | 163/274 (59.489%) | 80/139 (57.554%) | 83/135 (61.481%) | +22 (fix 50 / reg 28) | -19 (fix 30 / reg 49) | 538 | 1,453,752 | 15,701.1s | 0 | 5 |
| 25 | Budgeted thinking math router v5 | 157/274 (57.299%) | 78/139 (56.115%) | 79/135 (58.519%) | +16 (fix 25 / reg 9) | -25 (fix 3 / reg 28) | 172 | 2,825,527 | 19,865.0s | 0 | — |
| 26 | Frozen memory of common error patterns | 155/274 (56.569%) | 81/139 (58.273%) | 74/135 (54.815%) | +14 (fix 47 / reg 33) | -27 (fix 20 / reg 47) | 271 | 396,248 | 11,225.8s | 3 | — |
| 27 | Solver - Critic - Repair | 154/274 (56.204%) | 79/139 (56.835%) | 75/135 (55.556%) | +13 (fix 52 / reg 39) | -28 (fix 22 / reg 50) | 797 | 1,307,934 | 27,907.7s | 3 | — |
| 28 | Multi-iteration RAG | 151/274 (55.109%) | 73/139 (52.518%) | 78/135 (57.778%) | +10 (fix 35 / reg 25) | -31 (fix 7 / reg 38) | 336 | 632,316 | 34,127.4s | 0 | — |
| 29 | Literal parallel-8 consensus with low-confidence arbiter | 147/274 (53.650%) | 75/139 (53.957%) | 72/135 (53.333%) | +6 (fix 41 / reg 35) | -35 (fix 23 / reg 58) | — | 38,349 | 2,598.4s | 0 | — |
| 30 | Two-pass transcription and solve | 146/274 (53.285%) | 75/139 (53.957%) | 71/135 (52.593%) | +5 (fix 44 / reg 39) | -36 (fix 22 / reg 58) | 546 | 689,448 | 15,611.3s | 2 | — |
| 31 | Calculator / SymPy verification | 143/274 (52.190%) | 64/139 (46.043%) | 79/135 (58.519%) | +2 (fix 2 / reg 0) | -39 (fix 7 / reg 46) | 125 | 4,846,139 | 18,672.0s | 0 | — |
| 32 | Answer canonicalization | 141/274 (51.460%) | 62/139 (44.604%) | 79/135 (58.519%) | +0 (fix 0 / reg 0) | -41 (fix 7 / reg 48) | — | 4,662,595 | 6,091.6s | 0 | — |
| 33 | 8-way answer clustering / consensus | 139/274 (50.730%) | 74/139 (53.237%) | 65/135 (48.148%) | -2 (fix 32 / reg 34) | -43 (fix 16 / reg 59) | 2,466 | 2,843,423 | 3,635.2s | 0 | — |
| 34 | Overlapping tiled vision | 125/274 (45.620%) | 53/139 (38.129%) | 72/135 (53.333%) | -16 (fix 28 / reg 44) | -57 (fix 19 / reg 76) | 272 | 1,632,215 | 13,731.8s | 0 | 2 |
| 35 | Structural element RAG | 116/274 (42.336%) | 49/139 (35.252%) | 67/135 (49.630%) | -25 (fix 24 / reg 49) | -66 (fix 5 / reg 71) | 169 | 1,172,661 | 17,758.8s | 0 | — |
| 36 | Parser-augmented solver with generic saturation and repetition gate | 98/274 (35.766%) | 37/139 (26.619%) | 61/135 (45.185%) | -43 (fix 26 / reg 69) | -84 (fix 17 / reg 101) | — | 639,420 | 17,716.6s | 0 | — |
| 37 | PaddleOCR-VL structured parser + Qwen3.5-9B solver | 98/274 (35.766%) | 37/139 (26.619%) | 61/135 (45.185%) | -43 (fix 26 / reg 69) | -84 (fix 17 / reg 101) | — | 639,442 | 17,592.3s | 0 | — |

### Per-branch details

#### Gold-blind V2.1 default with conservative V3.1 repair (`blind_ensemble_v2default_v3_repair_v1`)

Accepted: `reports/maxim_blind_ensemble_v2default_v3_repair_v1_20260804/evaluation_cached_reuse_v1/score.json` (`0550b1b59b6f9680ad8e7462c1b30fc6c97960bd16fa82576a3f917192fbb24a`).

- Overall: 205/274 (74.818%); vs page +64 (fix 74 / reg 10); vs Router +23 (fix 30 / reg 7).
- Math139: 108/139 (77.698%); vs page +46 (fix 52 / reg 6); vs Router +5 (fix 8 / reg 3).
- nonMath135: 97/135 (71.852%); vs page +18 (fix 22 / reg 4); vs Router +18 (fix 22 / reg 4).
- Bound/explicit manifests summarized: 1.

#### Gold-blind V2.1 default with conservative Active-Crop repair (`blind_ensemble_v2default_active_repair_v1`)

Accepted: `reports/maxim_blind_ensemble_v2default_active_repair_v1_20260804/evaluation_cached_reuse_v1/score.json` (`80e2f522c03e93b28b67be04a35d458f93d6f633a7b0d1f7cc48bd2aac237358`).

- Overall: 204/274 (74.453%); vs page +63 (fix 73 / reg 10); vs Router +22 (fix 29 / reg 7).
- Math139: 107/139 (76.978%); vs page +45 (fix 51 / reg 6); vs Router +4 (fix 7 / reg 3).
- nonMath135: 97/135 (71.852%); vs page +18 (fix 22 / reg 4); vs Router +18 (fix 22 / reg 4).
- Bound/explicit manifests summarized: 1.

#### Gold-blind V2.1 default with strict triple-agreement repair (`blind_ensemble_v2default_triple_agreement_v1`)

Accepted: `reports/maxim_blind_ensemble_v2default_triple_agreement_v1_20260804/evaluation_cached_reuse_v1/score.json` (`81ec64edad7d8aa21c11a3a4faaf2e5e9fc4b741be7043a75827981e9fe4face`).

- Overall: 204/274 (74.453%); vs page +63 (fix 73 / reg 10); vs Router +22 (fix 29 / reg 7).
- Math139: 107/139 (76.978%); vs page +45 (fix 51 / reg 6); vs Router +4 (fix 7 / reg 3).
- nonMath135: 97/135 (71.852%); vs page +18 (fix 22 / reg 4); vs Router +18 (fix 22 / reg 4).
- Bound/explicit manifests summarized: 1.

#### Outcome-blind consensus repair: V3.1, Active-Crop, and no-tools triple (`consensus_repair_v31_active_no_tools_triple_v1`)

Accepted: `reports/maxim_consensus_repair_sweep_v1_20260804/v31_active_no_tools_triple/evaluation_cached_reuse_v1/score.json` (`2e5d6e101692adbe033806a308db28e3616c114ed71bd1a3830e076a3ad3dc79`).

- Overall: 204/274 (74.453%); vs page +63 (fix 72 / reg 9); vs Router +22 (fix 27 / reg 5).
- Math139: 109/139 (78.417%); vs page +47 (fix 52 / reg 5); vs Router +6 (fix 7 / reg 1).
- nonMath135: 95/135 (70.370%); vs page +16 (fix 20 / reg 4); vs Router +16 (fix 20 / reg 4).
- Bound/explicit manifests summarized: 1.

#### Outcome-blind consensus repair: V3.1 and Active-Crop pair (`consensus_repair_v31_active_pair_v1`)

Accepted: `reports/maxim_consensus_repair_sweep_v1_20260804/v31_active_pair/evaluation_cached_reuse_v1/score.json` (`71f0eaf40a16574bd6cb79072b82e43d6d5acb01a1afc4f5e39669f98fcc1b89`).

- Overall: 204/274 (74.453%); vs page +63 (fix 72 / reg 9); vs Router +22 (fix 27 / reg 5).
- Math139: 109/139 (78.417%); vs page +47 (fix 52 / reg 5); vs Router +6 (fix 7 / reg 1).
- nonMath135: 95/135 (70.370%); vs page +16 (fix 20 / reg 4); vs Router +16 (fix 20 / reg 4).
- Bound/explicit manifests summarized: 1.

#### Outcome-blind consensus repair: V3.1 and no-tools pair (`consensus_repair_v31_no_tools_pair_v1`)

Accepted: `reports/maxim_consensus_repair_sweep_v1_20260804/v31_no_tools_pair/evaluation_cached_reuse_v1/score.json` (`24913917abc4e61e1d5215c7061dc1843f01303b3184f74b6a4c324ca5737861`).

- Overall: 204/274 (74.453%); vs page +63 (fix 72 / reg 9); vs Router +22 (fix 27 / reg 5).
- Math139: 109/139 (78.417%); vs page +47 (fix 52 / reg 5); vs Router +6 (fix 7 / reg 1).
- nonMath135: 95/135 (70.370%); vs page +16 (fix 20 / reg 4); vs Router +16 (fix 20 / reg 4).
- Bound/explicit manifests summarized: 1.

#### Meta-verifier v2.1 numeric choice-token compatibility (`final_meta_verifier_v2_choice_token_compat_v21`)

Accepted: `reports/maxim_final_meta_verifier_v2_choice_token_compat_v21_20260803/evaluation/score.json` (`64f42117dde176136cf028152916752eaa81eca3f0d9316c396f93fb81e41182`).

- Overall: 204/274 (74.453%); vs page +63 (fix 73 / reg 10); vs Router +22 (fix 29 / reg 7).
- Math139: 107/139 (76.978%); vs page +45 (fix 51 / reg 6); vs Router +4 (fix 7 / reg 3).
- nonMath135: 97/135 (71.852%); vs page +18 (fix 22 / reg 4); vs Router +18 (fix 22 / reg 4).
- Bound/explicit manifests summarized: 1.

#### Frozen 10-way final meta-verifier v2 (`final_meta_verifier_v2`)

Accepted: `reports/maxim_final_meta_verifier_v2_20260803/evaluation/score.json` (`436ac2eaa80e6bdbcc6c43f7a76b48dfb59f0a3b7e6b9fdb9a8e55f592c0c6cb`).

- Overall: 203/274 (74.088%); vs page +62 (fix 72 / reg 10); vs Router +21 (fix 28 / reg 7).
- Math139: 107/139 (76.978%); vs page +45 (fix 51 / reg 6); vs Router +4 (fix 7 / reg 3).
- nonMath135: 96/135 (71.111%); vs page +17 (fix 21 / reg 4); vs Router +17 (fix 21 / reg 4).
- Bound/explicit manifests summarized: 1.

#### Meta-verifier v3.1 numeric choice-token compatibility (`final_meta_verifier_v3_choice_token_compat_v31`)

Accepted: `reports/maxim_final_meta_verifier_v3_choice_token_compat_v31_20260803/evaluation/score.json` (`7903c1fac1a1a21205693679a3c30e9a9616021d27ff76a181f1fd753dee0a2b`).

- Overall: 199/274 (72.628%); vs page +58 (fix 69 / reg 11); vs Router +17 (fix 29 / reg 12).
- Math139: 106/139 (76.259%); vs page +44 (fix 50 / reg 6); vs Router +3 (fix 10 / reg 7).
- nonMath135: 93/135 (68.889%); vs page +14 (fix 19 / reg 5); vs Router +14 (fix 19 / reg 5).
- Bound/explicit manifests summarized: 1.

#### Frozen 12-way final meta-verifier v3 (`final_meta_verifier_v3`)

Accepted: `reports/maxim_final_meta_verifier_v3_20260803/evaluation/score.json` (`f98687e4202a267d01cf78312e53c1fdf22dac24279d736200addae1ea29b081`).

- Overall: 198/274 (72.263%); vs page +57 (fix 68 / reg 11); vs Router +16 (fix 28 / reg 12).
- Math139: 106/139 (76.259%); vs page +44 (fix 50 / reg 6); vs Router +3 (fix 10 / reg 7).
- nonMath135: 92/135 (68.148%); vs page +13 (fix 18 / reg 5); vs Router +13 (fix 18 / reg 5).
- Bound/explicit manifests summarized: 1.

#### Outcome-blind consensus repair: Active-Crop and no-tools pair (`consensus_repair_active_no_tools_pair_v1`)

Accepted: `reports/maxim_consensus_repair_sweep_v1_20260804/active_no_tools_pair/evaluation_cached_reuse_v1/score.json` (`ba44f07ea2c8bb31c1783e7f297d133917518586a1fc4f309c96be89e695d1d2`).

- Overall: 195/274 (71.168%); vs page +54 (fix 70 / reg 16); vs Router +13 (fix 22 / reg 9).
- Math139: 104/139 (74.820%); vs page +42 (fix 49 / reg 7); vs Router +1 (fix 1 / reg 0).
- nonMath135: 91/135 (67.407%); vs page +12 (fix 21 / reg 9); vs Router +12 (fix 21 / reg 9).
- Bound/explicit manifests summarized: 1.

#### Outcome-blind consensus repair: unique two-of-three majority (`consensus_repair_two_of_three_majority_v1`)

Accepted: `reports/maxim_consensus_repair_sweep_v1_20260804/two_of_three_majority/evaluation_cached_reuse_v1/score.json` (`ae00fc0c3069b5381ad6a3c2e819e1bd22987cfefc9d9a949058c1d04f2161ee`).

- Overall: 194/274 (70.803%); vs page +53 (fix 69 / reg 16); vs Router +12 (fix 21 / reg 9).
- Math139: 104/139 (74.820%); vs page +42 (fix 49 / reg 7); vs Router +1 (fix 1 / reg 0).
- nonMath135: 90/135 (66.667%); vs page +11 (fix 20 / reg 9); vs Router +11 (fix 20 / reg 9).
- Bound/explicit manifests summarized: 1.

#### Query-conditioned active crop verifier v2 (`query_active_crop_v2`)

Accepted: `reports/maxim_query_active_crop_v2_20260803/evaluation_orchestrated/score.json` (`fe7af97028dc51d2ce23ce0b2779d6ff5661af032929223b8e6067a8ded6b374`).

- Overall: 194/274 (70.803%); vs page +53 (fix 68 / reg 15); vs Router +12 (fix 20 / reg 8).
- Math139: 104/139 (74.820%); vs page +42 (fix 49 / reg 7); vs Router +1 (fix 1 / reg 0).
- nonMath135: 90/135 (66.667%); vs page +11 (fix 19 / reg 8); vs Router +11 (fix 19 / reg 8).
- Bound/explicit manifests summarized: 2.

#### Conservative Visual Sketchpad v2 (`visual_sketchpad_v2`)

Accepted: `reports/maxim_visual_sketchpad_v2_20260803/evaluation_cached_union_v1/score.json` (`214d47c86c7f35fb2732a99c781820225fcc53ce2e5668fb54555a3829fccb53`).

- Overall: 194/274 (70.803%); vs page +53 (fix 68 / reg 15); vs Router +12 (fix 20 / reg 8).
- Math139: 104/139 (74.820%); vs page +42 (fix 49 / reg 7); vs Router +1 (fix 1 / reg 0).
- nonMath135: 90/135 (66.667%); vs page +11 (fix 19 / reg 8); vs Router +11 (fix 19 / reg 8).
- Bound/explicit manifests summarized: 1.

#### Evidence-gap Seeker to Inspector (`evidence_gap_seeker_inspector_v1`)

Accepted: `reports/maxim_evidence_gap_seeker_inspector_v1_final_20260803/evaluation/score.json` (`1fe44437dc4dded9a5d38fdc72903cc742443099cdb1ff563a31f5190c90102e`).

- Overall: 193/274 (70.438%); vs page +52 (fix 67 / reg 15); vs Router +11 (fix 19 / reg 8).
- Math139: 103/139 (74.101%); vs page +41 (fix 48 / reg 7); vs Router +0 (fix 0 / reg 0).
- nonMath135: 90/135 (66.667%); vs page +11 (fix 19 / reg 8); vs Router +11 (fix 19 / reg 8).
- Bound/explicit manifests summarized: 1.

#### Frozen no-tools (semantic gate selected RAG 0/274) (`paired_rag_norag_semantic_support`)

Accepted: `reports/maxim_paired_rag_norag_semantic_support_v1_20260803/evaluation_orchestrated/score.json` (`77932f8ef6af9e10df1d79b607902ff54a3144b3c1c17be300ce406a35f2a462`).

- Overall: 193/274 (70.438%); vs page +52 (fix 67 / reg 15); vs Router +11 (fix 19 / reg 8).
- Math139: 103/139 (74.101%); vs page +41 (fix 48 / reg 7); vs Router +0 (fix 0 / reg 0).
- nonMath135: 90/135 (66.667%); vs page +11 (fix 19 / reg 8); vs Router +11 (fix 19 / reg 8).
- Bound/explicit manifests summarized: 2.

#### Evidence-gap fact-intent compatibility v1.1 (`evidence_gap_fact_intent_compat_v11`)

Accepted: `reports/maxim_evidence_gap_seeker_inspector_v1_fact_intent_compat_v11_20260803/evaluation/score.json` (`7d0bd99dd00cb2765f56ecfaadd9ecabbf13b54e5b11481e4327f54c9250c076`).

- Overall: 190/274 (69.343%); vs page +49 (fix 64 / reg 15); vs Router +8 (fix 19 / reg 11).
- Math139: 100/139 (71.942%); vs page +38 (fix 45 / reg 7); vs Router -3 (fix 0 / reg 3).
- nonMath135: 90/135 (66.667%); vs page +11 (fix 19 / reg 8); vs Router +11 (fix 19 / reg 8).
- Bound/explicit manifests summarized: 1.

#### Qwen3.5-27B direct full274 (`stronger_27b_direct`)

Accepted: `reports/maxim_stronger27_direct_full274_20260803/evaluation/score.json` (`985556543c3e865ff9319756ee6c138f6c06775b854db1de8a49547e038342e5`).

- Overall: 190/274 (69.343%); vs page +49 (fix 68 / reg 19); vs Router +8 (fix 31 / reg 23).
- Math139: 95/139 (68.345%); vs page +33 (fix 43 / reg 10); vs Router -8 (fix 6 / reg 14).
- nonMath135: 95/135 (70.370%); vs page +16 (fix 25 / reg 9); vs Router +16 (fix 25 / reg 9).
- Bound/explicit manifests summarized: 1.

#### Blind disagreement verifier (preregistered raw selector) (`blind_verifier_raw_selector`)

Accepted: `reports/maxim_blind_disagreement_verifier_v1_20260803/finalized/raw_selector/score.json` (`3197c5d9044176f560fc3aade8bc23457f141c46a8dcea24dd074c72dd83c1a3`).

- Overall: 189/274 (68.978%); vs page +48 (fix 62 / reg 14); vs Router +7 (fix 20 / reg 13).
- Math139: 105/139 (75.540%); vs page +43 (fix 49 / reg 6); vs Router +2 (fix 7 / reg 5).
- nonMath135: 84/135 (62.222%); vs page +5 (fix 13 / reg 8); vs Router +5 (fix 13 / reg 8).
- Bound/explicit manifests summarized: 2.

#### Frozen subject router (`subject_router`)

Accepted: `reports/maxim8_variants_full274_20260802/offline_v1_evaluation/subject_router/matched_score.json` (`5c50cc2bb882d0c83386d324fafe0470433172845b4ebb5e787d41d288ad306f`).

- Overall: 182/274 (66.423%); vs page +41 (fix 48 / reg 7); vs Router +0 (fix 0 / reg 0).
- Math139: 103/139 (74.101%); vs page +41 (fix 48 / reg 7); vs Router +0 (fix 0 / reg 0).
- nonMath135: 79/135 (58.519%); vs page +0 (fix 0 / reg 0); vs Router +0 (fix 0 / reg 0).
- Bound/explicit manifests summarized: 1.

#### Qwen3.5-27B hard86 composite (`stronger_27b_hard86_composite`)

Accepted: `reports/maxim_stronger27_hard86_20260803/evaluation_orchestrated/score.json` (`e91339c01c54effd654b1f913d03836d11224ea02070456e82e0a79491023746`).

- Overall: 180/274 (65.693%); vs page +39 (fix 45 / reg 6); vs Router -2 (fix 8 / reg 10).
- Math139: 101/139 (72.662%); vs page +39 (fix 45 / reg 6); vs Router -2 (fix 8 / reg 10).
- nonMath135: 79/135 (58.519%); vs page +0 (fix 0 / reg 0); vs Router +0 (fix 0 / reg 0).
- Bound/explicit manifests summarized: 2.

#### Native thinking math router v4 (`native_thinking_v4`)

Accepted: `reports/maxim_native_thinking_math_router_v4_20260803/evaluation_matched/score.json` (`c1b90d27f1d4a0bd25d6a3fbf010aa0e30855b317af709881abb9d0cc83273e2`).

- Overall: 173/274 (63.139%); vs page +32 (fix 38 / reg 6); vs Router -9 (fix 7 / reg 16).
- Math139: 94/139 (67.626%); vs page +32 (fix 38 / reg 6); vs Router -9 (fix 7 / reg 16).
- nonMath135: 79/135 (58.519%); vs page +0 (fix 0 / reg 0); vs Router +0 (fix 0 / reg 0).
- Bound/explicit manifests summarized: 2.

#### Selective RAG (retrieval reliability gate) (`selective_rag_metadata`)

Accepted: `reports/maxim8_variants_full274_20260802/offline_v1_evaluation/selective_rag_metadata/matched_score.json` (`9e90e69a01a17f16ca72d0571a486ce5432382210376e2cd4f66a37edc31f332`).

- Overall: 165/274 (60.219%); vs page +24 (fix 31 / reg 7); vs Router -17 (fix 18 / reg 35).
- Math139: 76/139 (54.676%); vs page +14 (fix 17 / reg 3); vs Router -27 (fix 4 / reg 31).
- nonMath135: 89/135 (65.926%); vs page +10 (fix 14 / reg 4); vs Router +10 (fix 14 / reg 4).
- Bound/explicit manifests summarized: 1.

#### Active vision: localize, crop, sketch, solve (`active_vision`)

Accepted: `reports/maxim_active_vision_v1_20260803/evaluation/score.json` (`bafd2ff3d2f4415f868e46bb64a4b98fcdee5066cf27714fcff518b5d17ad821`).

- Overall: 163/274 (59.489%); vs page +22 (fix 50 / reg 28); vs Router -19 (fix 30 / reg 49).
- Math139: 80/139 (57.554%); vs page +18 (fix 30 / reg 12); vs Router -23 (fix 10 / reg 33).
- nonMath135: 83/135 (61.481%); vs page +4 (fix 20 / reg 16); vs Router +4 (fix 20 / reg 16).
- Bound/explicit manifests summarized: 2.

#### Budgeted thinking math router v5 (`budgeted_thinking_v5`)

Accepted: `reports/maxim_native_thinking_math_router_v5_20260803/evaluation_orchestrated/score.json` (`163118466004ad7b06e49dead4844258d6aca1619e8482a9ac6b7afa78f070fd`).

- Overall: 157/274 (57.299%); vs page +16 (fix 25 / reg 9); vs Router -25 (fix 3 / reg 28).
- Math139: 78/139 (56.115%); vs page +16 (fix 25 / reg 9); vs Router -25 (fix 3 / reg 28).
- nonMath135: 79/135 (58.519%); vs page +0 (fix 0 / reg 0); vs Router +0 (fix 0 / reg 0).
- Bound/explicit manifests summarized: 2.

#### Frozen memory of common error patterns (`error_memory`)

Accepted: `reports/maxim8_variants_full274_20260802/online_v1_evaluation/error_memory/score.json` (`9e1780f5e71216508f7b2993a7b5d5d0c98a53953bbacff82031702d0c0db703`).

- Overall: 155/274 (56.569%); vs page +14 (fix 47 / reg 33); vs Router -27 (fix 20 / reg 47).
- Math139: 81/139 (58.273%); vs page +19 (fix 35 / reg 16); vs Router -22 (fix 8 / reg 30).
- nonMath135: 74/135 (54.815%); vs page -5 (fix 12 / reg 17); vs Router -5 (fix 12 / reg 17).
- Bound/explicit manifests summarized: 1.

#### Solver - Critic - Repair (`solver_critic_repair`)

Accepted: `reports/maxim8_variants_full274_20260802/online_v1_evaluation/solver_critic_repair/score.json` (`10ac64bc3145ab5710f5268edb4bf7ecef4961ab291f36179a4f6d23fa7e9c9b`).

- Overall: 154/274 (56.204%); vs page +13 (fix 52 / reg 39); vs Router -28 (fix 22 / reg 50).
- Math139: 79/139 (56.835%); vs page +17 (fix 35 / reg 18); vs Router -24 (fix 5 / reg 29).
- nonMath135: 75/135 (55.556%); vs page -4 (fix 17 / reg 21); vs Router -4 (fix 17 / reg 21).
- Bound/explicit manifests summarized: 1.

#### Multi-iteration RAG (`mi_rag`)

Accepted: `reports/maxim_mi_rag_v1_20260803/evaluation/score.json` (`a319f3885aa2908f3f7b65c5ea07d52e0c1a2017ce899b6d6ddf23a131e5a313`).

- Overall: 151/274 (55.109%); vs page +10 (fix 35 / reg 25); vs Router -31 (fix 7 / reg 38).
- Math139: 73/139 (52.518%); vs page +11 (fix 28 / reg 17); vs Router -30 (fix 0 / reg 30).
- nonMath135: 78/135 (57.778%); vs page -1 (fix 7 / reg 8); vs Router -1 (fix 7 / reg 8).
- Bound/explicit manifests summarized: 1.

#### Literal parallel-8 consensus with low-confidence arbiter (`literal_parallel8_lowconf_arbiter`)

Accepted: `reports/maxim_literal_parallel8_lowconf_v1_20260803/evaluation/score.json` (`84bc5d4d58dc692f9755705d7986e37d7241a6f8dd606a185624797e5d56eb4c`).

- Overall: 147/274 (53.650%); vs page +6 (fix 41 / reg 35); vs Router -35 (fix 23 / reg 58).
- Math139: 75/139 (53.957%); vs page +13 (fix 26 / reg 13); vs Router -28 (fix 8 / reg 36).
- nonMath135: 72/135 (53.333%); vs page -7 (fix 15 / reg 22); vs Router -7 (fix 15 / reg 22).
- Bound/explicit manifests summarized: 2.

#### Two-pass transcription and solve (`two_pass_transcription`)

Accepted: `reports/maxim8_variants_full274_20260802/online_v1_evaluation/two_pass_transcription/score.json` (`22118875d09574ea4a1a1426663133b49348c1c125889682f71214193a41d6b9`).

- Overall: 146/274 (53.285%); vs page +5 (fix 44 / reg 39); vs Router -36 (fix 22 / reg 58).
- Math139: 75/139 (53.957%); vs page +13 (fix 29 / reg 16); vs Router -28 (fix 7 / reg 35).
- nonMath135: 71/135 (52.593%); vs page -8 (fix 15 / reg 23); vs Router -8 (fix 15 / reg 23).
- Bound/explicit manifests summarized: 1.

#### Calculator / SymPy verification (`calculator_sympy`)

Accepted: `reports/maxim8_variants_full274_20260802/online_v1_evaluation/calculator_sympy/score.json` (`04a30f628930b935387330a00ef57f8fde71047ec38b8a53730bc3111cd46bc6`).

- Overall: 143/274 (52.190%); vs page +2 (fix 2 / reg 0); vs Router -39 (fix 7 / reg 46).
- Math139: 64/139 (46.043%); vs page +2 (fix 2 / reg 0); vs Router -39 (fix 7 / reg 46).
- nonMath135: 79/135 (58.519%); vs page +0 (fix 0 / reg 0); vs Router +0 (fix 0 / reg 0).
- Bound/explicit manifests summarized: 1.

#### Answer canonicalization (`answer_canonicalization`)

Accepted: `reports/maxim8_variants_full274_20260802/offline_v1_evaluation/answer_canonicalization/score.json` (`5ae6af3bb9324b89aa365e06f7335cbce9a43b289332114cd81943e464951589`).

- Overall: 141/274 (51.460%); vs page +0 (fix 0 / reg 0); vs Router -41 (fix 7 / reg 48).
- Math139: 62/139 (44.604%); vs page +0 (fix 0 / reg 0); vs Router -41 (fix 7 / reg 48).
- nonMath135: 79/135 (58.519%); vs page +0 (fix 0 / reg 0); vs Router +0 (fix 0 / reg 0).
- Bound/explicit manifests summarized: 1.

#### 8-way answer clustering / consensus (`parallel_consensus`)

Accepted: `reports/maxim8_variants_full274_20260802/offline_v1_evaluation/parallel_consensus/matched_score.json` (`8955844dede3c03bb5938755f8eafde6555bd9e88b2de92e173d8b12640cd7f9`).

- Overall: 139/274 (50.730%); vs page -2 (fix 32 / reg 34); vs Router -43 (fix 16 / reg 59).
- Math139: 74/139 (53.237%); vs page +12 (fix 23 / reg 11); vs Router -29 (fix 7 / reg 36).
- nonMath135: 65/135 (48.148%); vs page -14 (fix 9 / reg 23); vs Router -14 (fix 9 / reg 23).
- Bound/explicit manifests summarized: 1.

#### Overlapping tiled vision (`tiled_vision`)

Accepted: `reports/maxim_tiled_vision_v1_20260803/evaluation/score.json` (`7217a2104e0231b93689e5b573bcd6d6129af7f9b501cd41151e32066f483a83`).

- Overall: 125/274 (45.620%); vs page -16 (fix 28 / reg 44); vs Router -57 (fix 19 / reg 76).
- Math139: 53/139 (38.129%); vs page -9 (fix 14 / reg 23); vs Router -50 (fix 5 / reg 55).
- nonMath135: 72/135 (53.333%); vs page -7 (fix 14 / reg 21); vs Router -7 (fix 14 / reg 21).
- Bound/explicit manifests summarized: 2.

#### Structural element RAG (`structural_rag`)

Accepted: `reports/maxim_structural_evidence_rag_v1_20260803/evaluation/score.json` (`29763fffcb3bf361b8560c2c68eb6249716150ea61efc8c424035b78805badec`).

- Overall: 116/274 (42.336%); vs page -25 (fix 24 / reg 49); vs Router -66 (fix 5 / reg 71).
- Math139: 49/139 (35.252%); vs page -13 (fix 20 / reg 33); vs Router -54 (fix 1 / reg 55).
- nonMath135: 67/135 (49.630%); vs page -12 (fix 4 / reg 16); vs Router -12 (fix 4 / reg 16).
- Bound/explicit manifests summarized: 1.

#### Parser-augmented solver with generic saturation and repetition gate (`parser_augmented_conservative_v2`)

Accepted: `reports/maxim_document_parser_v1_20260803/parser_augmented_conservative_v2/evaluation/score.json` (`05f734098946a6715418f6f39cf37ff6372a2a5a28bad189bc6d63cb1f709bfb`).

- Overall: 98/274 (35.766%); vs page -43 (fix 26 / reg 69); vs Router -84 (fix 17 / reg 101).
- Math139: 37/139 (26.619%); vs page -25 (fix 12 / reg 37); vs Router -66 (fix 3 / reg 69).
- nonMath135: 61/135 (45.185%); vs page -18 (fix 14 / reg 32); vs Router -18 (fix 14 / reg 32).
- Bound/explicit manifests summarized: 1.

#### PaddleOCR-VL structured parser + Qwen3.5-9B solver (`parser_augmented_solver_v1`)

Accepted: `reports/maxim_document_parser_v1_20260803/parser_augmented_solver_v1/evaluation/score.json` (`ee722d46be78e33aac005a0078963d1a9cde400e87452f0013e22f3bebc04ef0`).

- Overall: 98/274 (35.766%); vs page -43 (fix 26 / reg 69); vs Router -84 (fix 17 / reg 101).
- Math139: 37/139 (26.619%); vs page -25 (fix 12 / reg 37); vs Router -66 (fix 3 / reg 69).
- nonMath135: 61/135 (45.185%); vs page -18 (fix 14 / reg 32); vs Router -18 (fix 14 / reg 32).
- Bound/explicit manifests summarized: 1.

## Non-final (excluded from ranking)

| Branch | Reason | Ledger artifact (not opened) |
|---|---|---|
| Cross-fit subject router (OOF estimate; non-final) | OOF same-benchmark estimate; not an external holdout final | `reports/maxim_crossfit_router_v1_20260803/score.json` |
| Fail-closed V2.1 vs Active-Crop selector trained on explicit semantic adapters v1.1 | superseded before any source/model calls; excluded from ranking and never eligible for a final score | `reports/maxim_pairwise_selector_v21_active_crop_semantic_adapter_v11_20260804/superseded_before_calls_attestation.json` |
| Superseded-before-calls runtime-pinned semantic adapter v1.2 | superseded before any source/model calls; excluded from ranking and never eligible for a final score | `reports/maxim_pairwise_selector_v21_active_crop_semantic_adapter_runtime_pinned_v12_20260804/superseded_before_calls_attestation.json` |
| Superseded-before-calls runtime-exact semantic adapter v1.3 | superseded before any source/model calls; excluded from ranking and never eligible for a final score | `reports/maxim_pairwise_selector_v21_active_crop_semantic_adapter_runtime_exact_v13_20260804/superseded_before_calls_attestation.json` |

## Pending (no accepted final)

| Branch | Reason |
|---|---|
| Element-aware base dense RAG v2 control | no finalized matched-judge report found |
| BGE-M3 whole-book hard-negative RAG | no finalized matched-judge report found |
| Qwen3.5-9B DocVQA LoRA | no finalized matched-judge report found |
| Element-aware sparse BM25 plus graph RAG v3 | no finalized matched-judge report found |
| Element-aware dense sparse graph RAG v3 | no finalized matched-judge report found |
| Dual-granularity page and element RRF RAG | no finalized matched-judge report found |
| Synthetic whole-book pairwise selector | no finalized matched-judge report found |
| Fail-closed synthetic pairwise selector: V2.1 default vs Active-Crop override | no finalized matched-judge report found |
| Active runtime-closure semantic adapter v1.4 with exact seven-module staged import closure | no finalized matched-judge report found |
| Frozen CPU TF-IDF/logistic pairwise selector: V2.1 default vs Active-Crop override | no finalized matched-judge report found |

> Operational totals describe solver artifacts as reported/recomputed from accepted task outcomes and exact-output-bound manifests; unavailable fields remain `—` and are never inferred from unrelated files.
