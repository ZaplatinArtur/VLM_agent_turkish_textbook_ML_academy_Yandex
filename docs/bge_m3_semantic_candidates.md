# Optional BGE-M3 semantic candidates

The production retrieval service keeps its existing retrieval path unchanged
unless `MLA_BGE_M3_ENABLED=true` is set. BGE-M3 is an optional candidate arm:
it does not replace lexical retrieval, certify an exact match, or directly
decide the final result.

## Immutable model contract

The only accepted model identity is:

- model: `BAAI/bge-m3`
- revision: `5617a9f61b028005a4858fdac845db406aefb181`
- license: MIT (`mit` in the machine-readable manifest)
- task contract: `symmetric_retrieval_text_v1`
- embedding dimension: `1024`
- remote model code: disabled
- the pinned model graph itself contains its `2_Normalize` module;
  `encode_normalize_embeddings=false` means no additional encode-level
  normalization is requested
- vector-store normalization: deterministic L2 normalization before
  inner-product search (`vector_store_normalization=l2`)
- first-load semantic validation: the pinned model-card 2×2 dense-score example,
  with absolute tolerance `3e-4`

Changing the model ID, revision, license, or task contract in the environment
raises `ValueError`; the service never silently falls back to a floating model
revision. Model loading is lazy. By default `MLA_BGE_M3_ALLOW_DOWNLOAD=false`,
so an enabled arm with no local pinned snapshot fails rather than downloading.
Before the model is published to the ranker or any index is built, the first
load runs the official four-text semantic check through the same
SentenceTransformers encode path. A score mismatch fails closed.

## Candidate behavior

`MLA_BGE_M3_CANDIDATE_MODE=union` protects the first
`MLA_BGE_M3_PRIMARY_CANDIDATE_K` lexical candidates, inserts up to
`MLA_BGE_M3_SEMANTIC_CANDIDATE_K` new semantic candidates, then retains the
remaining lexical candidates. Duplicate IDs keep their lexical position.
Semantic candidates are ordered by descending score and then `chunk_id`, which
makes score ties deterministic.

Raw BM25 and cosine scores are not compared directly. Once the semantic arm is
used, both arms receive deterministic within-arm reciprocal-rank scores with
`k=60`: lexical weight `1.0`, semantic weight `0.85`. This preserves lexical
priority while preventing the much larger BM25 numeric scale from suppressing
every semantic candidate in the downstream heuristic reranker.

`MLA_BGE_M3_CANDIDATE_MODE=fallback` does not invoke BGE when lexical retrieval
already returned at least `MLA_BGE_M3_FALLBACK_MIN_CANDIDATES`. Below that
threshold it uses the same lexical-first union. This mode avoids semantic model
work on healthy lexical queries.

The downstream knowledge reranker still receives the candidate list. Exact or
lexical evidence remains the primary arm; BGE can only add recall candidates.
Its window must cover the complete protected union. When `MLA_RERANK_TOP_N` is
unset, the enabled arm derives at least `primary_candidate_k +
semantic_candidate_k` (64 with defaults). An explicitly configured smaller
window is rejected instead of silently making part of the semantic arm inert.

## Cache and index safety

The legacy disabled path keeps its historical `chunk_id` embedding-cache keys
and index compatibility. The enabled BGE arm uses a separate strict contract:

- cache keys are namespaced by a canonical hash of model ID, exact revision,
  license, max length, encode/vector-store normalization contracts,
  remote-code policy, task contract, device, requested/resolved index kind, and the
  configured batch size plus installed `sentence-transformers`, `transformers`,
  `torch`, `tokenizers`, `faiss-cpu`, and `numpy` versions;
- each cache key also includes a projection hash of the chunk retrieval text,
  subject, grade, and textbook, preventing stale reuse after content changes;
- generated, cached, and query vectors must have exactly 1024 finite values;
- the default BGE index directory is a revision-specific subdirectory below
  `MLA_INDEX_DIR`; `MLA_BGE_M3_INDEX_DIR` can override it;
- the FAISS manifest contains the canonical embedder provenance, a content-bound
  corpus projection hash, and SHA-256 plus byte length for `index.faiss` and
  `chunk_ids.json`; it also binds the requested kind to the resolved and
  deserialized FAISS type;
- a model/projection mismatch, missing strict field, malformed manifest, or
  artifact hash mismatch raises an index validation error and leaves every
  existing file untouched. Automatic build is allowed only for a clean or
  nonexistent strict index directory; operators must quarantine/remove an
  invalid snapshot explicitly before rebuilding.

For this contract, “clean” means an existing directory with no entries at all;
even an unrelated file prevents automatic build. The strict index directory,
manifest, FAISS index, and chunk-ID artifact must be regular paths rather than
symlinks or Windows reparse points, so persistence cannot be redirected outside
the configured directory. Parent path components are an operator-supplied trust
boundary and must likewise resolve to the intended storage hierarchy; the
runtime guard does not reject every parent component because legitimate Windows
deployments may live below managed reparse roots such as OneDrive.

Enabling the arm fails closed when installed version metadata is unavailable for
any of those six runtime distributions. Their exact versions are persisted in
the manifest and therefore also participate in cache and index identity. This
does not widen the supported dependency contract: `pyproject.toml` remains the
authority, including `sentence-transformers>=3.0,<5.4`; an installed version
outside that range is rejected before model loading.

The adapter does not currently force a model dtype, so dtype is deliberately not
claimed in provenance. It remains a deployment P2: operators must keep the
intended model dtype stable; the semantic smoke still fails closed on a
score-incompatible runtime change.

When the arm is disabled, BGE-only environment variables are intentionally
ignored so stale optional settings cannot break the legacy retrieval path. An
enabled `MLA_BGE_M3_INDEX_DIR` may not equal the legacy `MLA_INDEX_DIR`, avoiding
cross-arm artifact overwrite.

Strict index construction uses an exclusive `.strict-build.lock` created with
`O_EXCL` before corpus embedding or persistence. Any competing/stale lock fails
closed. The writer removes it only after the completed snapshot reloads and
passes every provenance, content, dimension, and artifact-integrity check; a
failed or crashed build intentionally leaves the lock for operator quarantine.

These hashes detect accidental corruption and ordinary tampering. They are not
a signature or substitute for trusted filesystem permissions.

## Environment variables

| Variable | Default | Contract |
|---|---:|---|
| `MLA_BGE_M3_ENABLED` | `false` | Strict boolean; opt-in only |
| `MLA_BGE_M3_MODEL_ID` | `BAAI/bge-m3` | Immutable |
| `MLA_BGE_M3_REVISION` | pinned SHA above | Immutable |
| `MLA_BGE_M3_LICENSE` | `mit` | Immutable |
| `MLA_BGE_M3_TASK_CONTRACT` | `symmetric_retrieval_text_v1` | Immutable |
| `MLA_BGE_M3_EMBEDDING_DIMENSION` | `1024` | Immutable |
| `MLA_BGE_M3_CANDIDATE_MODE` | `union` | `union` or `fallback` |
| `MLA_BGE_M3_PRIMARY_CANDIDATE_K` | `32` | Positive integer |
| `MLA_BGE_M3_SEMANTIC_CANDIDATE_K` | `32` | Positive integer |
| `MLA_BGE_M3_FALLBACK_MIN_CANDIDATES` | `5` | Positive integer |
| `MLA_BGE_M3_BATCH_SIZE` | `2` | `1..256` |
| `MLA_BGE_M3_MAX_LENGTH` | `1024` | `1..8192`; part of provenance |
| `MLA_BGE_M3_ALLOW_DOWNLOAD` | `false` | Strict boolean |
| `MLA_BGE_M3_CACHE_DIR` | unset | Optional model cache directory |
| `MLA_BGE_M3_INDEX_DIR` | revision-specific | Optional strict index directory |
| `MLA_BGE_M3_DEVICE` | `cpu` | Explicit SentenceTransformers device |
| `MLA_FAISS_INDEX_KIND` | `auto` | `auto`, `flat`, or `hnsw`; strictly enforced and part of provenance |

Before enabling in production, materialize the exact pinned snapshot through an
authorized deployment step and build the strict index in the target environment.
