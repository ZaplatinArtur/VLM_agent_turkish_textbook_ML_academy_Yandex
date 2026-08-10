# Final source-expansion wave v1.1

This directory is an immutable, unscored ten-arm evaluation package for `Qwen/Qwen3.5-9B`.
It does not contain an audit amendment, an execution attempt, scorer outputs, benchmark
references, PDFs, or crops.

The exact simultaneous arm set is:

1. `base240`
2. `math5_v11`
3. `english5`
4. `meb7_6`
5. `official16 = math5_v11 + english5 + meb7_6`
6. `bs11_8_research`
7. `research_bs24 = official16 + bs11_8_research`
8. `fenomen12_research`
9. `research_fenomen28 = official16 + fenomen12_research`
10. `research_all36 = official16 + bs11_8_research + fenomen12_research`

Only `official16` is eligible for the official headline. `base240` is the comparison
baseline. Every arm containing BS or Fenomen material is `research_evaluation_only`,
production-ineligible, and kept in a separate output namespace because the private-publisher
license is unverified. No source PDF or crop is redistributed.

Every solver has 274 rows in the exact route-authority order. Rows outside an arm's target
set are copied as exact raw bytes from the audited base solver. Component target rows are
copied as exact raw bytes from their independently frozen source successor. The five source
target sets are pairwise disjoint.

The image judge is rebuilt independently for every arm. Unchanged image-route rows are
opaque, byte-identical passthrough from the frozen base image judge. Math5, English5, and
MEB7 image-route rows are deterministic official-source rows bound to that arm's solver SHA,
candidate answer SHA, and frozen evidence certificate. BS11 and Fenomen targets are
deterministic evaluator routes, so their isolated image judges remain byte-identical to the
base judge.

Math5 uses only the v1.1 successor (`faf09e5e...`), never the superseded flattened v1 solver.
The candidate freeze, source judge, judge manifest, and mandatory chronology disclosure are
all pinned. The disclosure is preserved verbatim and records that the upstream judge was
built after candidate freeze but before an explicit independent-audit PASS message. This
build records the later PASS as parent-coordination evidence, not as a cryptographically
pinned audit artifact. This final wave therefore has no execution authority until its own
independent audit passes and binds the complete ten-arm freeze.

`execute_final_wave.py` is fail-closed. An independent auditor must create the exact fixed
`INDEPENDENT_AUDIT_AMENDMENT.json`, binding the final freeze and all ten solver/judge hashes.
The launcher then creates a persistent `O_EXCL` attempt marker before any scorer, starts all
ten arms through one shared barrier, waits for all ten processes, never decodes scorer output,
and only then hashes complete output bundles into a completion manifest. A partial,
reordered, repeated, or runtime-mismatched launch is refused.

Safe commands before audit:

```powershell
python verify_final_freeze.py
python execute_final_wave.py --print-audit-template
python -m pytest -q test_final_wave.py
```

Do not run `--execute` until an independent audit amendment has been written and authorized.
Result interpretation is a separate post-barrier audit; this launcher intentionally does not
read or report per-arm metrics.
