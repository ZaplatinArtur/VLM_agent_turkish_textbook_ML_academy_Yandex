# Maxim primary-layout + MEB DD v3 result

## Result

- Full benchmark: **230/274 = 0.839416**.
- Primary-layout v2: **225/274 = 0.821168**.
- Earlier honest best: **223/274 = 0.813869**.
- Net V3 improvement: **+5 correct / +1.825 pp** over v2 and
  **+7 correct / +2.555 pp** over the earlier best.
- Math: **111/139 = 0.798561** (unchanged).
- Non-math: **119/135 = 0.881481**.
- History: **10/10**; Turkish language and literature: **18/21**.

The two-book source wave raised official-source certificates from 74 to 88.
Nine new certificates confirmed the existing anchor, while all five new answer
overrides were correct on the post-freeze scorer.  V3 therefore added no new
regression relative to primary-layout v2.

## Honesty status

This is a previously inspected **development replay**, not a fresh holdout.
The generation profile, strict optional-`nosw` parser, 15 source-native records,
PDF/key review, projection audit, resolver, certificates, and composed solver
were committed and pushed as
`adb14c3a5123849271cf8cc776a4295ee16d40d1` before the V3 image judge or
score was created.

No benchmark candidate, reference, judge result, score, task ordinal, or
outcome-derived routing feature was used during generation.  A separate
source-only reviewer verified all 15 new coordinates and keys and found no
identity-parser blocker after adversarial controls were added.

## Reproducibility

- Pre-score handoff: `reports/maxim_official_exact_source_v2_20260805/PRE_SCORE_HANDOFF_PRIMARY_LAYOUT_MEB_DD_V3.json`
- Score JSON SHA-256: `af9b40d6813b44339f599d56c5c5db311e963a68bdefa59e16d2929b9fe29c28`
- Image-judge SHA-256: `ccdb5f8cb44772fc547ca7575ffea46c2bd74cf2a5710ac4777e5b6a542fc530`
- Solver SHA-256: `1bb9e1850afba9e6a6ec5346bf3dad8dcc6630ea6eed69e838886c72a2f1b8e6`
- Targeted tests: **82 passed**.
