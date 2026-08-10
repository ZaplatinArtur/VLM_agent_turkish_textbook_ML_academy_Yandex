# V7 one-shot result

The frozen V7 launch scored **242/274 = 0.8832** on the inspected development
benchmark. The previous frozen V6 score was **238/274 = 0.8686**, so the
reported change is **+4 correct / +1.46 percentage points**. Math remains
**112/139 = 0.8058**.

The run was genuinely one-shot. All eight candidate source bindings, both
judge builders, their fixed output paths, and the exact scorer command were
committed and pushed before the score. The attempt marker was atomically
created before the scorer started. No candidate was removed and no threshold
was changed after outcome access.

## What produced the four-point change

| Task | Subject | V6 | V7 | Mechanism |
|---|---|---:|---:|---|
| `val_0139` | Physics | wrong | correct | unchanged answer; exact official-source certificate corrects the old opaque image-judge verdict |
| `val_0141` | Physics | wrong | correct | unchanged answer; exact official-source certificate corrects the old opaque image-judge verdict |
| `val_0159` | Geography | wrong | correct | unchanged answer; exact official-source certificate corrects the old opaque image-judge verdict |
| `val_0196` | History / Ataturkism | wrong | correct | pre-registered full-page fill-blank source binding changes the solver answer |

This distinction matters: the direct **solver-answer** gain over V6 is one
row. The other three gains are principled evaluator corrections for unchanged
answers that already had strong source certificates. Thus `0.8832` is the
valid score under the new source-adjudicated evaluation protocol, but it is
not an apples-to-apples four-answer model improvement over V6's older opaque
image-judge protocol.

## Reporting line

`Максим | Агент/RAG/Тулы | Fail-closed subject routing + exact official-source retrieval + source-certificate arbitration + full-page fill-blank binding | 0.883`

## Limitations

- This is a previously inspected development replay, not an unseen holdout.
- No web search is used in the online path; it previously added only `0.004`
  while increasing latency and retrieval noise.
- No same-wave retuning or repeat score will be performed.
