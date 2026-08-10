# TurHistQuad embedder bake-off v1: results

## Frozen scope

This preregistered test uses only the public `mteb/TurHistQuadRetrieval` test
split: revision `b6e74379b7486da28ce81c3d459cd7bbd87d4987`, MIT license,
1,213 corpus rows, 1,024 queries, and 2,048 qrels. It is separate from the
project benchmark; no project queries, references, correctness data, or judge
outputs were accessed, and no project score was computed.

| Condition | Frozen model revision | License | Status |
|---|---|---|---|
| Current MiniLM | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` @ `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` | Apache-2.0 | Completed |
| BGE-M3 dense | `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181` | MIT | Completed |
| GTE multilingual base | `Alibaba-NLP/gte-multilingual-base` @ `9bbca17d9273fd0d03d5725c7a4b0f6b45142062` | Apache-2.0 | Completed with pinned remote code `Alibaba-NLP/new-impl` @ `40ced75c3017eb27626c9d4ea981bde21a2662f4` |
| Qwen3 embedding 0.6B | `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | Apache-2.0 | Optional; skipped |

## Completed comparison

The frozen protocol uses cosine similarity (dot product of L2-normalized dense
vectors), stable corpus-ID tie-breaking, and query-level Recall@1, Recall@5,
and MRR. Confidence intervals are paired, query-level bootstrap intervals over
10,000 samples with seed `20260809`.

| Metric | MiniLM | GTE | BGE-M3 |
|---|---:|---:|---:|
| Recall@1 | 0.10205078125 | 0.20751953125 | 0.24365234375 |
| Recall@5 | 0.2490234375 | 0.39404296875 | 0.47216796875 |
| MRR | 0.3192536587199147 | 0.5515815939849256 | 0.6408793805552973 |
| Hit@1 | 0.2041015625 | 0.4150390625 | 0.4873046875 |
| Hit@5 | 0.4619140625 | 0.7177734375 | 0.841796875 |

| Comparison | Metric | Delta | Paired 95% CI | Bootstrap P(delta > 0) |
|---|---|---:|---:|---:|
| GTE - MiniLM | Recall@1 | +0.10546875 | [0.08935546875, 0.12158203125] | 1.0 |
| GTE - MiniLM | Recall@5 | +0.14501953125 | [0.12744140625, 0.16259765625] | 1.0 |
| GTE - MiniLM | MRR | +0.2323279352650109 | [0.2073215451090453, 0.2575506380607296] | 1.0 |
| BGE-M3 - MiniLM | Recall@1 | +0.1416015625 | [0.12548828125, 0.15771484375] | 1.0 |
| BGE-M3 - MiniLM | Recall@5 | +0.22314453125 | [0.205078125, 0.2412109375] | 1.0 |
| BGE-M3 - MiniLM | MRR | +0.3216257218353825 | [0.2973064354749895, 0.34589655355099097] | 1.0 |
| BGE-M3 - GTE | Recall@1 | +0.0361328125 | [0.02294921875, 0.04931640625] | 1.0 |
| BGE-M3 - GTE | Recall@5 | +0.078125 | [0.0634765625, 0.0927734375] | 1.0 |
| BGE-M3 - GTE | MRR | +0.08929778657037164 | [0.07179959989791902, 0.10682833170783027] | 1.0 |

`Hit@k` is reported separately as binary any-relevant retrieval and is not
labeled as recall.

## CPU latency trade-off

The full first-pass runs used local CPU/float32. Corpus vectors are intended to
be precomputed, so online query encoding is the more relevant production
number.

| Condition | Corpus encode (1,213) | Query encode (1,024) | Mean query encode | Full run wall |
|---|---:|---:|---:|---:|
| MiniLM | 35.983 s | 9.075 s | 8.9 ms/query | 63.404 s |
| GTE | 134.493 s | 43.516 s | 42.5 ms/query | 214.676 s |
| BGE-M3 | 338.022 s | 102.660 s | 100.3 ms/query | 490.016 s |

BGE-M3 buys the strongest retrieval quality at roughly 2.4x GTE query-encoding
latency and 11.3x MiniLM query-encoding latency in this conservative CPU setup.
That supports offline BGE corpus indexing and an explicit latency/quality route,
not recomputing document embeddings per request.

The valid artifact hashes are MiniLM result
`23627f2e99612b2ef3a97a499954350ff74c76ff233165298d2c721f995942ca`,
GTE result
`1536c27cb37193c79913d41ee5325a3c2825f3d3f337a4fb523270649035c403`,
BGE-M3 result
`4af68b76bc70cb16513754fa5c9b5f792ccf0a7570a75829d7d845d6461d74c5`,
all-vs-MiniLM comparison
`9266f04af0fc5e49fb72f122a6f505b236e79011b8a365a3596b3bcc826589b5`,
and BGE-M3-vs-GTE comparison
`2ff0fe3f6af950cca0a6287be6dcc25f42877970212ea5fe8b1fc679629731e0`.

## Metric-definition correction

An earlier draft used binary any-relevant `Hit@1`/`Hit@5` values but labeled
them `Recall@1`/`Recall@5`. TurHistQuad has two relevant passages per query, so
standard recall is the fraction of both relevant passages retrieved. That
draft and its inputs are preserved under `runs/excluded_hit_as_recall_v1` with
an exclusion manifest; they must not be reported as recall. MRR was unaffected.

## GTE runtime incident and validation

The first GTE artifact under `runs/gte_multilingual_base` is excluded. It used
Transformers 5.14.1 with the pinned custom `NewModel` code written for the 4.x
attention-mask path and failed the official model-card cosine smoke test. Its
three cosine values were `[0.7379261255264282, 0.6759868860244751,
0.5983105897903442]`, versus expected `[0.3016996383666992,
0.7503870129585266, 0.3203084468841553]`; it is not evidence about GTE quality
and is absent from the comparison.

The valid replacement in `runs/gte_multilingual_base_t439` is fail-closed to
Transformers 4.39.1. It passed the same official smoke test at absolute
tolerance `5e-5`, producing `[0.30170029401779175, 0.7503871917724609,
0.3203089237213135]`. Only this validated run is reported above.

## BGE-M3 artifact and validation

The pinned 2,271,145,830-byte BGE-M3 weight has SHA-256
`b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`.
Before the full run, the adapter reproduced the official model-card dense
score example: actual `[[0.6259036, 0.3474958], [0.3498677, 0.6782465]]`
versus published rounded `[[0.6260, 0.3474], [0.3499, 0.6782]]`, with an
absolute tolerance of `3e-4`.

## Interpretation and recommendation

This is external Turkish retrieval evidence only. It does not establish an
end-to-end VLM benchmark gain, final-answer accuracy gain, or a safe replacement
for exact lexical retrieval.

Keep the lexical exact arm and use validated BGE-M3 as the preferred semantic
candidate generator. GTE is a smaller fallback but was significantly weaker on
this external Turkish retrieval set. This is a candidate architecture for
separate project validation, not a proven end-to-end improvement. Qwen was
optional and was skipped; no claim about its embedding quality or latency follows.
