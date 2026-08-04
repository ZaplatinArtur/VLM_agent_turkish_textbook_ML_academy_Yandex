# Maksim Evidence OS v1 — frozen replay result

Status: **development replay; not an untouched-book holdout**.

## Primary result

| Slice | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Overall | 205 | 274 | 0.748175 |
| Math | 108 | 139 | 0.776978 |
| Non-Math | 97 | 135 | 0.718519 |
| Frozen page-RAG baseline | 141 | 274 | 0.514599 |

The Evidence OS output is an exact copy of the strongest frozen anchor, with
solver SHA-256
`aa76740913819b81e23f926e89be68e30501e6f6e14f36867afb3a9f122cc678`.
It made zero overrides because none of the cached specialists supplied a
certificate that was independently verified and bound to both the exact
observable input and the exact proposed answer.

## Cached specialist ablation

| Branch | Frozen standalone score | Evidence OS mode | Decision |
|---|---:|---|---|
| Anchor | 205/274 (0.748175) | default | Keep |
| Active Crop | 194/274 (0.708029) | shadow | No independently pixel-verified visual claim |
| Structural RAG | 116/274 (0.423358) | shadow | Citation is not claim-complete or round-trip verified |
| Calculator/SymPy | 143/274 (0.521898) | shadow | Program inputs/result are not bound to observation and answer |
| Parser | 98/274 (0.357664) | perception only | OCR/layout confidence cannot prove an answer |
| Visual Sketchpad | 194/274 (0.708029) | shadow | No independently verified render certificate |

The previously used legacy gates were also unsafe against this anchor: Active
Crop produced 1 fix and 2 regressions among its differing accepted answers;
Calculator produced 0 fixes and 2 regressions.  They therefore remain disabled
for answer replacement.

## Grouped router diagnostic

The exploratory router used nested source-family cross-validation: five outer
folds and three inner folds, with whole source-document families held out.
Source family was used only for splitting; task ID, source locator/family, row
order, hashes, references, judge outcomes and scores were forbidden features.

| Metric | Router | Anchor | Delta |
|---|---:|---:|---:|
| OOF accuracy | 205/274 (0.748175) | 205/274 (0.748175) | 0 |
| Family-macro accuracy | 0.757933 | 0.757933 | 0 |
| Fixes / regressions | 0 / 0 | — | 0 |
| Answer overrides | 0 | — | 0 |

All five inner validations selected the conservative anchor.  This rejects the
hypothesis that the current observable cached metadata can route specialists
reliably across the 39 represented source families.  It does not estimate
generalization to unseen books because all benchmark rows had been inspected
previously.

## Reproducibility and guardrails

- Frozen profile SHA-256:
  `0ece602811f7e41c51bbeb38c171df556229f196ffe5ba471cd895d82e53c8d5`.
- Public observable-input bundle SHA-256:
  `37dea7d26dfdddb18a59ac300e4ccff26add1ed1ed4063709d4495df5c743e9f`.
- All 274 image-bearing rows were bound to local image bytes.
- `task_id` was used only for alignment and removed before policy execution.
- The policy had no access to references, scores or judge verdicts.
- External certificates require a profile-pinned SHA, verifier/kind allowlist
  and a recomputed inline-trace hash; this profile authorizes none.
- Raw solver rows are checked against the public candidate fields before any
  possible override can be emitted.
- Combined regression and leakage suite: **67 passed**.

The large frozen solver/judge inputs are deliberately not committed.  Their
content hashes are pinned in `configs/maxim_evidence_os_v1.json`; a clean clone
needs the matching local artifacts to reproduce the 274-row replay.

## Table entry

| Author | Pipeline part | Idea | Accuracy |
|---|---|---|---:|
| Максим | Агент/RAG/Тулы | Evidence OS v1: fail-closed evidence router; exact input/answer-bound certificates; unsafe specialists in shadow; nested source-family OOF audit | 0.748 |

Use `0.748` as the current defensible development result.  Do not report the
old post-hoc `0.832` or `0.960` routing experiments as production scores.
