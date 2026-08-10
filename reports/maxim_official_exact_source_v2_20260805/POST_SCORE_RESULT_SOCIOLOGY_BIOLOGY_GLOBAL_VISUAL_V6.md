# V6 post-freeze result

The source-only V6 pipeline scored **238/274 = 0.868613** on the fixed development replay. The previous retained V5 result was **234/274 = 0.854015**, so V6 adds four correct rows, or **+1.460 percentage points**.

| Slice | V6 | Accuracy |
|---|---:|---:|
| Overall | 238/274 | 0.868613 |
| Deterministic | 158/177 | 0.892655 |
| Image-judged | 80/97 | 0.824742 |
| Math | 112/139 | 0.805755 |
| Non-math | 126/135 | 0.933333 |

Exactly four solver rows changed relative to V5. All four changed from incorrect to correct: Biology `val_0162`, `val_0163`, and `val_0164`, plus Sociology `val_0186`. Math was unchanged.

The scored code and source artifacts were committed as `c8c93f3ecdf019e5534a132b20d2320dbe6269bf` and verified against the remote branch before the score was opened. The image-judge artifact was then built once and the aggregate scorer was run once. No post-score rollback, route change, or row selection was applied to V6.

This is an honest development-replay result, not a holdout result. The benchmark had been inspected during earlier experiments, and the official PDFs plus pinned local runtime are not all stored in Git. Generalization must therefore be validated on unseen books, languages, or a newly sealed benchmark before a production claim.

Retained best: **V6**.
