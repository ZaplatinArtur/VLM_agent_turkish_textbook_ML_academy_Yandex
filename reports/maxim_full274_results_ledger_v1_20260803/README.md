# Frozen full274 results ledger v1

This directory is the preregistered skeleton for the final common comparison
table. `registry.json` fixes the 274-task benchmark SHA, the exact matched-judge
lineage, and every branch that must remain visible as `pending` until its final
report exists.

The builder is deliberately read-only with respect to experiment reports. It
accepts `matched_score.json`, hash-manifested finalizer `score.json`, and the
canonical `score.json` emitted through `run_full274_postgeneration_v1.py` in
any output directory. Canonical post-generation scores additionally require a
hash-valid sibling orchestration record with the exact frozen sources, judge,
delegates, and completed stages. The one legacy answer-canonicalization replay
is admitted only through the exact frozen 274-row baseline-judge hash.

Every final recounts all 274 boolean task outcomes and verifies the benchmark
and judge lineage. Bounds, partial/interim, in-sample, and gold-access
generation results are recorded as rejections rather than scores. The
same-benchmark cross-fit router is shown separately as `oof_estimate` /
`non_final`, never promoted or included when selecting the best final result.

A preregistered branch explicitly marked `superseded_before_calls` is also
terminal `non_final` without a metric only when its registry call count is
exactly zero and its reports-relative attestation path, SHA-256, terminal
status, and attested source-call count all validate. Its score globs are not
discovered or opened. An invalid supersession claim fails the ledger build.

Run after one or more branches finish:

```powershell
python scripts/build_maxim_full274_results_ledger_v1.py `
  --repo-root . `
  --registry reports/maxim_full274_results_ledger_v1_20260803/registry.json `
  --out-json reports/maxim_full274_results_ledger_v1_20260803/RESULTS.json `
  --out-md reports/maxim_full274_results_ledger_v1_20260803/RESULTS.md
```

The output is deterministic for a fixed registry and fixed source artifacts.
If two admissible finals match one branch, the branch becomes `conflict`; the
builder never silently picks one.
