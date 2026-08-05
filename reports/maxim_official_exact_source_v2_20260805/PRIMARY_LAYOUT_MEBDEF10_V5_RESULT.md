# Maxim source-native v5 result

The remotely frozen MEB-DEF-10 v5 pipeline scores **234/274 = 0.854015**
on the previously inspected 274-row development replay. This is one correct
answer above the sealed v4 result, **233/274 = 0.850365**.

The source admission profile, implementation, resolver and composition were
committed and pushed as `440b763a15047d78f7396aecdbca70269994b54b`
before the post-freeze image input or aggregate score was created. Three
independent pre-score reviews passed after fail-closed repairs. The remote
commit was verified before the image input was materialized, and the aggregate
scorer then ran once. This is a reproducible development result, not a fresh
holdout claim.

## Measured slices

| Slice | Correct | Accuracy |
|---|---:|---:|
| Overall | 234/274 | 0.854015 |
| Deterministic-reference rows | 157/177 | 0.887006 |
| Image-judge rows | 77/97 | 0.793814 |
| Math | 112/139 | 0.805755 |
| Non-math | 122/135 | 0.903704 |
| Turkish language and literature | 21/21 | 1.000000 |

Compared with the frozen page-RAG baseline, v5 adds 93 net correct answers:
234/274 versus 141/274. Compared with v4, the deterministic and non-math
slices are unchanged. The image slice gains one answer and math rises from
111/139 to 112/139.

## Source-only delta

V5 adds eight reviewed records from the pinned official MEB-DEF-10 mathematics
PDF. Six pass the unchanged source-admission gates. Across the whole resolver,
117 certificates are accepted and 24 source overrides are applied. Only two
final solver answers differ from v4: `val_0066` changes from wrong to correct,
while `val_0065` remains correct. Neither changed row regresses.

The gain did not come from a task-ID rule, a lower similarity threshold, or a
post-score retry. Coordinate-table records require exact PDF identity, physical
page, visible question marker, answer-key geometry, full-string answer grammar,
and the same frozen matching gates as v4. Conflicting or incomplete evidence
abstains. Under the preregistered rule, v5 is promoted because it is strictly
better than v4.
