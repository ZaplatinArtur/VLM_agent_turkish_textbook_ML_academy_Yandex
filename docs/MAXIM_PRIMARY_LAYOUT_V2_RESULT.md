# Maxim primary-layout v2 result

## Result

- Full benchmark: **225/274 = 0.821168**.
- Previous honest best: **223/274 = 0.813869**.
- Net improvement over the previous best: **+2 correct / +0.730 pp**.
- Math: **111/139 = 0.798561** (unchanged).
- Non-math: **114/135 = 0.844444**.
- Frozen basic page-RAG: **141/274 = 0.514599**.

The primary-layout source projection raised accepted official-source
certificates from 61 to 74.  The fail-closed composition used 14 answer
overrides; 60 certified answers were already equal to the anchor.

## Honesty status

This is a previously inspected **development replay**, not a fresh holdout.
It is nevertheless a valid pre-score-sealed experiment: profile, code,
projection decisions, source resolver, certificates, and composed solver were
committed and pushed as `b5fb5ded0d9b5eda29e5fb5d935a4b6a37f62035`
before the evaluation image judge and benchmark scorer were created.

No benchmark candidate, reference answer, judge result, score, or
outcome-derived routing feature was used by the frozen generation pipeline.
The benchmark reference was opened only by the post-generation scorer.

## Reproducibility

- Pre-score handoff: `reports/maxim_official_exact_source_v2_20260805/PRE_SCORE_HANDOFF_PRIMARY_LAYOUT_V2.json`
- Score JSON SHA-256: `529deffbb523dbced9cc02b2e03da9e7c78d1a4db7027dd4dbfc0cc67abe5099`
- Image-judge SHA-256: `03f1ab956f9558507c7e1f2d7ca5acf0e47c1fb07e43f85aa49a7431ab9ed09d`
- Solver SHA-256: `8773e59ff43b4e4ecd87eff88479f440a7d3e5908fdb4349c7b3a48fe870f13f`
- Targeted tests: **70 passed**.
