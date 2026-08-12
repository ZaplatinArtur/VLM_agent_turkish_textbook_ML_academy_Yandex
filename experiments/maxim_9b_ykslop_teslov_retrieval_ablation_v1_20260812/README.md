# YKSLOP × Teslov retrieval ablation v1 (unfrozen scaffold)

This new namespace ports only the retrieval shape needed for an offline
ablation. It does not modify or import an audited/frozen execution namespace,
read benchmark gold/per-row outcomes, or call an API.

The source is Daniil Teslov's `origin/feature/embedder_and_search` work:

- hybrid BM25/dense/cross-encoder commit:
  `6aa0683c85c2e69f47c04f8f5da31bc18509503f`;
- final profile/finetuning commit:
  `2116cb1288a8f773e1ccd929dac05382d319327e`;
- reported best shape: `rrf_e5-small_bm25_cross-encoder-tuned` (and its
  gated variant). This scaffold intentionally excludes the LLM gate so the
  first experiment remains retrieval-only and offline.

The local base-model pin available elsewhere in this repository is
`BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
Teslov's tuned LoRA was expected at `data/models/bge-reranker-v2/chosen`.
That directory is absent in this workspace, so no tuned-model result can be
claimed and the real arm must fail closed. No model download was attempted.

`teslov_retrieval.py` provides deterministic BM25, weighted RRF, a strict
grade filter, and an injected cross-encoder adapter. Tests use synthetic
chunks and a deterministic scorer; they neither need model dependencies nor
touch benchmark artifacts. The production retrieval package now also exposes
a narrow current-API port at `src/retrieve/rankers/cross_encoder.py`. It can be
selected with `MLA_RERANK_BACKEND=teslov_cross_encoder`; model loading is lazy,
local-only, and pinned to the base BGE revision above. The default backend is
unchanged.

An offline two-pair smoke using the cached base snapshot ranked a Turkish
triangle-area passage above an unrelated photosynthesis passage (0.998860 vs
0.000016). This is an implementation smoke, not a retrieval or QA metric.

## Outcome-free YKS context census

`run_base_bge_census.py` applies the pinned public base BGE cross-encoder to
all 447 query/chunk pairs permitted by the frozen YKS subject closure. It
reads neither gold answers nor previous predictions and persists aggregate
counts only. The 2026-08-13 offline CPU run produced
`BASE_BGE_CONTEXT_CENSUS.json`, SHA-256
`ecc55bfa57351d28c015c07811050beb7cf2f3c8175e439b705453999f0aca56`:

- exact legacy BM25 nonempty coverage: 126/185 (68.11%);
- base-BGE nonempty coverage: 128/185 (69.19%);
- top-one context changed on 55/126 rows where both arms were nonempty;
- top-two context set changed on 62/185 rows;
- CPU wall time: 1,417.89 seconds for 447 pairs.

`BASE_BGE_CONTEXT_CENSUS_PROVENANCE.json` is a post-run attestation that
binds that immutable result to the runner and transitive local source hashes,
the verified six-file model manifest, offline policy, and runtime package
versions. It does not alter or rerun the census.

These are coverage and disagreement diagnostics, not Hit@K, Recall, MRR,
nDCG, or answer accuracy. Base BGE can rescue only two lexical-zero queries;
57/185 queries have no same-subject theory chunk and therefore cannot be
fixed by reranking this corpus.

Run the same local-only census from the repository root with:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONPATH = 'src'
.\.venv-bge53\Scripts\python.exe `
  experiments\maxim_9b_ykslop_teslov_retrieval_ablation_v1_20260812\run_base_bge_census.py `
  --device cpu
```

The clean evaluation design is not frozen yet. Before any headline retrieval
metric, it still needs an outcome-independent qrels/source-provenance set,
the tuned adapter plus hashes, and preregistered arms (`BM25` vs
`RRF + tuned cross-encoder`) and cutoffs. The public YKS rows expose no grade,
so exact parity deliberately uses no grade filter. Until then this namespace
is code scaffolding and a base-model diagnostic, not a benchmark candidate.

Run the bounded tests with:

```powershell
python -m pytest -q `
  experiments/maxim_9b_ykslop_teslov_retrieval_ablation_v1_20260812/test_teslov_retrieval.py `
  experiments/maxim_9b_ykslop_teslov_retrieval_ablation_v1_20260812/test_base_bge_census.py `
  tests/test_cross_encoder_ranker.py `
  tests/test_bge_m3_service.py
```
