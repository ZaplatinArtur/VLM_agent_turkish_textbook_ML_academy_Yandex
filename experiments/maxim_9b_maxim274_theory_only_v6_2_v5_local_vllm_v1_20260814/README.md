# Maxim-274 pure theory-search ablation

This namespace measures the generic pipeline on all 274 frozen Maxim rows with the non-generic pipeline completely absent. The model receives only the frozen public OCR, the public subject, the answer contract, and up to two deterministic BM25 hits from the independently audited 75-chunk strict textbook-theory corpus.

There is no task/example lookup, task database, source database, source router, no-ID router, per-task component map, image input, old answer, base answer, or non-generic fallback. `task_id` exists only in the separate outer alignment used to write the standard solver output. It never enters retrieval, a seed, a route, or model wire data.

The solve lineage is the successful YKS generic V6.2 policy adapted to local vLLM and all Maxim answer types: one medium-reasoning primary request. Only a transport/schema/length failure activates the generic V5 compact derive/falsify/crosscheck path and, when needed, a blind arbiter. If both generic stages fail, the row has an empty answer plus an error and remains wrong in the 274 denominator. No baseline answer is substituted.

## Pre-freeze order

Run hostile tests, print aggregate corpus coverage, and only then freeze:

```powershell
python -m unittest -v test_protocol.py
python prepare_freeze.py --coverage-only
python prepare_freeze.py --freeze
```

The coverage command prints only aggregate and subject-level counts; it does not emit per-task IDs.

## Zero-call validation

Use the SHA printed by the freezer (or `EXECUTION_FREEZE_SHA256.txt`):

```powershell
python dry_run.py --expected-freeze-sha256 <SHA>
```

## Local vLLM execution

The endpoint must advertise the exact served name `Qwen/Qwen3.5-9B`. Repeat `--base-url` to distribute rows deterministically over multiple local servers:

```powershell
python run_candidate.py --execute --expected-freeze-sha256 <SHA> `
  --base-url http://127.0.0.1:18000/v1 `
  --base-url http://127.0.0.1:18001/v1 `
  --workers 6
```

If interrupted, rerun the exact command with `--resume`. Completed journal rows are reused. An intent without a result is treated as an ambiguous interrupted call and becomes a wrong row; it is never silently replayed. Outputs are append-only and never overwritten.

## Post-run scoring

Scoring is intentionally outside the execution freeze. Bind `SCORING_CONTRACT_TEMPLATE.json` to the exact solver SHA from `COMPLETION.json`, then supply either a fresh 97-row judge artifact for that solver or a reusable 97-row artifact whose manifest binds the exact same solver SHA and judge freeze. Run the standard scorer for the fixed 177 deterministic + 97 judge partition. Do not reuse judge rows from another solver and do not remove failures from the denominator.

No inference or scoring is performed by the freezer, tests, coverage audit, or dry run.
