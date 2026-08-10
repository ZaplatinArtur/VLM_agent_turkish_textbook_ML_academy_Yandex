# Frozen normalized input for baseline-selector v1

This directory materializes only the input to the already preregistered selector. It does not run the selector, a model, a judge, or a scorer.

`build_pool.py` verifies exact SHA-256 pins for five full274 Qwen3.5-9B solver artifacts, rejects final errors, non-false gold access, outcome fields, mixed model fields, inherited 27B markers, task-set drift, and malformed parallel traces. The two parallel batches contribute one final answer and exactly eight raw route votes per task.

Evaluator routing is a task-ID-only projection from the frozen 97-row image template. No reference, candidate, score, correctness, or verdict field is copied. The official-source task set is used only as a fail-closed veto. Its 156 IDs are reconstructed from five pinned certificate files and emitted as a separate membership projection; official answers are not copied into candidate rows.

The `frozen/` artifact is self-contained for the selector: it contains byte-identical snapshots of the five upstream solver files because the selector intentionally rejects absolute paths and `..` traversal. `normalization_manifest.json` records the repository authorities, provenance caveats, disagreement matrix, usage/parse diagnostics, and a path-independent content projection.

The hardened v1.1 runtime entrypoint is `frozen/input_package_v1_1.json`. Its pool rows contain only `row_index` plus the five normalized candidate projections: task IDs and evaluator routes come from separately hashed authorities. `row_bindings_v1_1.json` binds every row index and authoritative task ID to the compact canonical SHA-256 of all five role projections. The benchmark itself is not copied into this package; only its exact SHA-256 and an ordered task-ID projection are pinned.

Build once into an absent directory:

```powershell
python experiments/maxim_9b_baseline_selector_v1/input/build_pool.py
```

Verify with focused tests:

```powershell
python -m pytest experiments/maxim_9b_baseline_selector_v1/input/test_build_pool.py -q
```

The selector and evaluation are deliberately outside this step.
