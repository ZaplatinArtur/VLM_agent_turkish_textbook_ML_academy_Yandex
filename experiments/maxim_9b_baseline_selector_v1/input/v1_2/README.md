# Selector input v1.2

This directory is an isolated extension of frozen input v1.1. It adds one answer-producing role, `structural_strict_9b`, from the exact full-274 Structural solver SHA-256 `4c73b6eb326e5790b19e14c01a79df853be23bb5f55b498ce4c58b78ebc3dff5`.

The package does not contain benchmark answer payloads, references, scores, correctness labels, judge verdicts, or a selector result. It was built for a rule that must be frozen before any new-arm evaluation. No selector or evaluation is run by the builder.

The ordered task-ID projection, evaluator route map, and 156-task source-union membership authority are byte-identical to v1.1. The five inherited normalized role projections are also unchanged. `row_bindings_v1_2.json` adds a sixth projection hash at every authoritative row index and binds the entire upstream file by SHA-256.

Structural provenance is represented honestly. Its normalized `source_access` is `true` for the 169 structural-evidence answers, 79 Router/Page-RAG fallbacks, and 8 fail-closed Page-RAG fallbacks; it is `false` for 18 fail-closed no-tools fallbacks. The build manifest pins the source preparation, determinism, raw-solver, and fail-closed manifests. The recursive answer-producing model closure is exactly `Qwen/Qwen3.5-9B`; inherited 27B output is rejected.

Build once into a new directory:

```powershell
python experiments/maxim_9b_baseline_selector_v1/input/v1_2/build_pool_v1_2.py
```

Verify the frozen package without selection or evaluation:

```powershell
python experiments/maxim_9b_baseline_selector_v1/input/v1_2/build_pool_v1_2.py --verify-existing
```

Runtime entrypoint: `frozen/input_package_v1_2.json`.
