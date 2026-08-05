# Maxim public-workbook primary-layout v2

## Status

This is an outcome-blind **development replay** profile, not a fresh holdout
claim.  The profile and its source-only artifacts must be committed and pushed
before the benchmark scorer is opened for this variant.

## Source-observable change

The previous projection emitted a printed question number only when every
leading numeric marker found by OCR agreed.  A top-level question followed by
numbered subitems therefore became numberless and was forced to abstain.

The v2 projection first looks for exactly one non-image OCR block whose
`block_order` is the integer `1`.  It accepts that block as the top-level marker
only when all of these checks pass:

- `block_label` is exactly `text` or `paragraph_title` after case folding;
- content starts with a strict `N.`, `N)`, or whitespace-delimited `N -` marker;
- bounding-box values are finite non-boolean numbers;
- `0 <= x0 < x1 <= image_width` and `0 <= y0 < y1 <= image_height`;
- `x0 <= 20%` of image width and `y0 <= 15%` of image height.

Duplicate order-one blocks, malformed geometry, non-integral order values, or
any other ambiguity fail closed.  The rule uses layout order rather than JSON
array position and is invariant to block-array permutation.  If it does not
fire, the previous unique-marker projection is retained unchanged.

## Unchanged downstream gates

Recovering a number is not sufficient to change an answer.  The resolver still
requires the unchanged page-coverage and margin thresholds, a unique printed
number on the matched page, and a PDF-bound answer-key context.  Otherwise the
official-source branch abstains and keeps the anchor answer.

No benchmark candidate, reference answer, judge result, score, task-row ordinal,
or outcome-derived feature is accepted by the profile.  `task_id` is used only
to align source-observable records.

## Source-only inventory before scoring

Across all 274 parser rows the frozen `{text, paragraph_title}` allowlist
recovers 17 primary markers over the legacy projection.  Thirteen of those
belong to the currently indexed public-workbook missing-number set and pass far
enough through the unchanged source gates to add certificates:

`val_0101`, `val_0104`, `val_0111`, `val_0117`, `val_0127`, `val_0133`,
`val_0140`, `val_0158`, `val_0167`, `val_0169`, `val_0170`, `val_0176`, and
`val_0179`.

The only `paragraph_title` case is `val_0104`: marker 18 is at normalized
`x0=0.018966`, `y0=0.021341`, while marker 1 occurs in a later `text` block.
This geometry is the source-only reason for including `paragraph_title` in the
explicit allowlist.

## Required pre-score checks

- Parser, workbook, and image-judge tests pass.
- Resolver emits 274 rows and records `gold_access=false`.
- Composition emits 274 rows and records `score_or_outcome_access=false`.
- Profile, code, inputs, resolver outputs, and composed solver are bound by
  SHA-256 in `PRE_SCORE_HANDOFF.json`.
- The handoff is committed and pushed before creating any v2 score artifact.
