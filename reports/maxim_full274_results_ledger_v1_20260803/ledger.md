# Frozen full274 results ledger

Benchmark: `5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9` (274 tasks).
Matched judge lineage: `frozen-judge-v2-qwen3.5-9b-seed20260714`.

Final: **28**; pending: **15**; non-final estimates: **1**; conflicts: **0**.

| Branch | Status | Correct | Accuracy | Accepted artifact |
|---|---:|---:|---:|---|
| Solver - Critic - Repair | final | 154/274 | 56.204% | `reports/maxim8_variants_full274_20260802/online_v1_evaluation/solver_critic_repair/score.json` |
| Selective RAG (retrieval reliability gate) | final | 165/274 | 60.219% | `reports/maxim8_variants_full274_20260802/offline_v1_evaluation/selective_rag_metadata/matched_score.json` |
| Frozen subject router | final | 182/274 | 66.423% | `reports/maxim8_variants_full274_20260802/offline_v1_evaluation/subject_router/matched_score.json` |
| 8-way answer clustering / consensus | final | 139/274 | 50.730% | `reports/maxim8_variants_full274_20260802/offline_v1_evaluation/parallel_consensus/matched_score.json` |
| Calculator / SymPy verification | final | 143/274 | 52.190% | `reports/maxim8_variants_full274_20260802/online_v1_evaluation/calculator_sympy/score.json` |
| Two-pass transcription and solve | final | 146/274 | 53.285% | `reports/maxim8_variants_full274_20260802/online_v1_evaluation/two_pass_transcription/score.json` |
| Answer canonicalization | final | 141/274 | 51.460% | `reports/maxim8_variants_full274_20260802/offline_v1_evaluation/answer_canonicalization/score.json` |
| Frozen memory of common error patterns | final | 155/274 | 56.569% | `reports/maxim8_variants_full274_20260802/online_v1_evaluation/error_memory/score.json` |
| Blind disagreement verifier (preregistered raw selector) | final | 189/274 | 68.978% | `reports/maxim_blind_disagreement_verifier_v1_20260803/finalized/raw_selector/score.json` |
| Active vision: localize, crop, sketch, solve | final | 163/274 | 59.489% | `reports/maxim_active_vision_v1_20260803/evaluation/score.json` |
| Query-conditioned active crop verifier v2 | final | 194/274 | 70.803% | `reports/maxim_query_active_crop_v2_20260803/evaluation_orchestrated/score.json` |
| Native thinking math router v4 | final | 173/274 | 63.139% | `reports/maxim_native_thinking_math_router_v4_20260803/evaluation_matched/score.json` |
| Cross-fit subject router (OOF estimate; non-final) | non_final | 184/274 | 67.153% | `reports/maxim_crossfit_router_v1_20260803/score.json` |
| Overlapping tiled vision | final | 125/274 | 45.620% | `reports/maxim_tiled_vision_v1_20260803/evaluation/score.json` |
| Budgeted thinking math router v5 | final | 157/274 | 57.299% | `reports/maxim_native_thinking_math_router_v5_20260803/evaluation_orchestrated/score.json` |
| Structural element RAG | final | 116/274 | 42.336% | `reports/maxim_structural_evidence_rag_v1_20260803/evaluation/score.json` |
| Multi-iteration RAG | final | 151/274 | 55.109% | `reports/maxim_mi_rag_v1_20260803/evaluation/score.json` |
| Frozen no-tools (semantic gate selected RAG 0/274) | final | 193/274 | 70.438% | `reports/maxim_paired_rag_norag_semantic_support_v1_20260803/evaluation_orchestrated/score.json` |
| Literal parallel-8 consensus with low-confidence arbiter | final | 147/274 | 53.650% | `reports/maxim_literal_parallel8_lowconf_v1_20260803/evaluation/score.json` |
| Qwen3.5-27B hard86 composite | final | 180/274 | 65.693% | `reports/maxim_stronger27_hard86_20260803/evaluation_orchestrated/score.json` |
| Qwen3.5-27B direct full274 | final | 190/274 | 69.343% | `reports/maxim_stronger27_direct_full274_20260803/evaluation/score.json` |
| Frozen 10-way final meta-verifier v2 | final | 203/274 | 74.088% | `reports/maxim_final_meta_verifier_v2_20260803/evaluation/score.json` |
| Frozen 12-way final meta-verifier v3 | final | 198/274 | 72.263% | `reports/maxim_final_meta_verifier_v3_20260803/evaluation/score.json` |
| PaddleOCR-VL structured parser + Qwen3.5-9B solver | final | 98/274 | 35.766% | `reports/maxim_document_parser_v1_20260803/parser_augmented_solver_v1/evaluation/score.json` |
| Evidence-gap Seeker to Inspector | final | 193/274 | 70.438% | `reports/maxim_evidence_gap_seeker_inspector_v1_final_20260803/evaluation/score.json` |
| Element-aware base dense RAG v2 control | pending | — | — | — |
| BGE-M3 whole-book hard-negative RAG | pending | — | — | — |
| Qwen3.5-9B DocVQA LoRA | pending | — | — | — |
| Conservative Visual Sketchpad v2 | pending | — | — | — |
| Element-aware sparse BM25 plus graph RAG v3 | pending | — | — | — |
| Element-aware dense sparse graph RAG v3 | pending | — | — | — |
| Parser-augmented solver with generic saturation and repetition gate | final | 98/274 | 35.766% | `reports/maxim_document_parser_v1_20260803/parser_augmented_conservative_v2/evaluation/score.json` |
| Dual-granularity page and element RRF RAG | pending | — | — | — |
| Meta-verifier v2.1 numeric choice-token compatibility | final | 204/274 | 74.453% | `reports/maxim_final_meta_verifier_v2_choice_token_compat_v21_20260803/evaluation/score.json` |
| Meta-verifier v3.1 numeric choice-token compatibility | final | 199/274 | 72.628% | `reports/maxim_final_meta_verifier_v3_choice_token_compat_v31_20260803/evaluation/score.json` |
| Synthetic whole-book pairwise selector | pending | — | — | — |
| Fail-closed synthetic pairwise selector: V2.1 default vs Active-Crop override | pending | — | — | — |
| Fail-closed V2.1 vs Active-Crop selector trained on explicit semantic adapters v1.1 | pending | — | — | — |
| Superseded-before-calls runtime-pinned semantic adapter v1.2 | pending | — | — | — |
| Superseded-before-calls runtime-exact semantic adapter v1.3 | pending | — | — | — |
| Active runtime-closure semantic adapter v1.4 with exact seven-module staged import closure | pending | — | — | — |
| Gold-blind V2.1 default with conservative V3.1 repair | pending | — | — | — |
| Gold-blind V2.1 default with conservative Active-Crop repair | pending | — | — | — |
| Evidence-gap fact-intent compatibility v1.1 | final | 190/274 | 69.343% | `reports/maxim_evidence_gap_seeker_inspector_v1_fact_intent_compat_v11_20260803/evaluation/score.json` |

## Rejected candidates

- **Overlapping tiled vision** — `reports/maxim_tiled_vision_v1_20260803/evaluation/previous-final-20260803T171730Z-b2b41b4b/score.json`: image-judge artifact SHA256 mismatch

> Pending entries are preregistered branches without an admissible final report; bounds, partial/interim, in-sample, and OOF estimates are never shown as final.
