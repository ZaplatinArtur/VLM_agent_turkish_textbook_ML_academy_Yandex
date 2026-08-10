# Maxim source-native v4 result

The remotely frozen MEB-3A + Samsungis v4 pipeline scores **233/274 =
0.850365** on the previously inspected 274-row development replay. This is
three correct answers above the sealed v3 result, **230/274 = 0.839416**.

The source admission profile, implementation, resolver and composition were
committed and pushed as `36527299ea8c52db18d1f04cfbb1332846eed16f`
before the image judge or scorer was created. Two independent pre-score
reviews passed after a stricter nearest-context repair. This is therefore a
reproducible development result, not a fresh holdout claim.

## Measured slices

| Slice | Correct | Accuracy |
|---|---:|---:|
| Overall | 233/274 | 0.850365 |
| Deterministic-reference rows | 157/177 | 0.887006 |
| Image-judge rows | 76/97 | 0.783505 |
| Math | 111/139 | 0.798561 |
| Non-math | 122/135 | 0.903704 |
| Turkish language and literature | 21/21 | 1.000000 |

Compared with the frozen page-RAG baseline, v4 adds 92 net correct answers:
233/274 versus 141/274. Compared with v3, the image slice and math slice are
unchanged; the entire +3 comes from the three pre-frozen MEB-3A Turkish source
certificates (`val_0189`, `val_0191`, `val_0194`). All three were correct, and
none regressed.

## Interpretation

The gain did not come from lowering a similarity threshold or routing on task
IDs. V4 added task-ID-free source records from two pinned public PDFs. A record
can replace the anchor only after exact document identity, PDF hash, physical
page, printed question number, answer-key cell, same-row ADIM or same-column
subject, and nearest-section checks pass. Samsungis increased independent
certificate coverage but did not create a new answer change; that null delta is
retained rather than tuned away.

The next honest headroom is math: v4 remains 111/139 there. The source-only MEB
10 mathematics wave can extend element-level `Örnek N` binding without changing
the frozen page gates; it must receive its own pre-score commit before any v5
measurement.
