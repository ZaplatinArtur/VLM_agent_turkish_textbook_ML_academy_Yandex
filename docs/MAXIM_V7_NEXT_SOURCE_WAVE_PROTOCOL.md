# Maxim V7 next-source-wave protocol

## Purpose

V7 extends the retained V6 pipeline only with evidence extracted from pinned official Turkish education PDFs. It must improve source coverage without selecting routes from benchmark correctness outcomes.

V6 was frozen before scoring and achieved 238/274 on the fixed development replay. The V7 candidate set below was produced by a source-only coverage audit and dispatched for implementation before the V6 score was opened. The candidates remain included as a complete wave even where the later V6 report reveals that the current answer is already correct or incorrect.

## Locked candidate set

| Source family | Task IDs | Required certificate |
|---|---|---|
| Samsung LGS1 | `val_0042`, `val_0043`, `val_0044`, `val_0046` | Unique content-page/question binding plus the sectioned official answer list on PDF page 25 |
| MEB-DD TYT Philosophy | `val_0149`, `val_0150` | Unique question binding plus inline official solution; the end-of-book answer table is an independent consistency check |
| MEB-DEF Biology Activity 4 | `val_0178` | Unique global visual page binding to Activity 4 plus the official activity key on PDF page 180 |
| MEB-CK History | `val_0196` | Unique fill-in activity binding plus the explicit official `BOŞLUK DOLDURMA` key on PDF page 18 |

No member of this set may be removed because of its V6 correctness, answer agreement, or effect on aggregate score. A task may abstain only when the locked source-verification rules fail.

## Allowed inputs before freeze

- parser observations and task images without reference answers;
- public source locators;
- official PDFs and their hashes;
- existing source-only indexes, resolver code, and frozen V6 solver as the composition base;
- source-structure tests and exact visual replay tests.

The following are forbidden as router or certificate inputs:

- benchmark reference answers or evaluator verdicts;
- V6 task correctness or transition labels;
- per-task score changes from any earlier run;
- a manual allowlist chosen from observed gains;
- an LLM judge that can replace a fail-closed source certificate.

## Certification and composition rules

1. Every source document is pinned by SHA-256 and reverified at composition time.
2. The task-to-source binding must be unique under document, page/activity, question, layout, and answer-key constraints. Ambiguity means abstention.
3. Visual bindings are generated for the whole eligible page set with a declared deterministic algorithm; a target-only page probe is not sufficient evidence.
4. A certificate may replace V6 only through the same answer-type compatibility and projection gates used by the hardened resolver.
5. Equality with the V6 answer is recorded as certified agreement, not counted as a solver change.
6. All eight candidates are composed in one locked V7 profile. There is no score-guided subset sweep.
7. Before scoring, code, tests, source fragments, indexes, visual evidence, resolver output, projection audit, and composed solver are committed and pushed. Local and remote commit IDs must match.
8. The image-judge artifact and aggregate score are then built once. V7 is retained only if its aggregate score exceeds V6, but a lower result is reported and is not repaired from the same outcomes.

## Interpretation

The resulting number is still a development-replay measurement, not a holdout estimate. The main generalizable claim is narrower: deterministic routing to exact official source elements can safely override a base reasoning answer when a reproducible certificate exists. Production readiness requires an unseen-book or multilingual sealed set.
