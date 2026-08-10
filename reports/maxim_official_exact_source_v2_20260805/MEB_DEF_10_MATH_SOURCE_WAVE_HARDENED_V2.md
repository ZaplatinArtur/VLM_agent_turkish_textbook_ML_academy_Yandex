# MEB-DEF-10 mathematics source wave - hardened V2

This report supersedes the exploratory V1 alignment summary. It records only
source/runtime admission behavior; benchmark correctness and score artifacts
were not opened.

## Source verification

| Item | Value |
|---|---:|
| pinned PDF | `e71895813859032680b2c66a1045501ed15231e6d029eee1ad09ea20dff80447` |
| physical pages | `356` |
| task-ID-free source records | `8` |
| content projections reproduced | `8/8` |
| connected-grid answer projections reproduced | `8/8` |
| same-page bindings | `7` |
| adjacent-page continuation bindings | `1` |

The continuation is independently derived from the exact prior-page section,
aligned grid edges, and the sequential header transition `37 -> 38`; an
arbitrary context page is rejected.

## Runtime admission with unchanged V4 gates

| Source record | Visible marker | Coverage | Tokens | Margin | Decision |
|---|---|---:|---:|---:|---|
| `p14:q22` | exact primary `Örnek 22` | `0.860540` | `29` | `0.541398` | admit |
| `p14:q24` | exact primary `Örnek 24` | `1.000000` | `7` | `0.000000` | abstain |
| `p39:q1` | no valid primary example title | `1.000000` | `9` | `0.426775` | abstain |
| `p51:q2` | printed `2.` | `1.000000` | `33` | `0.634669` | admit |
| `p53:q9` | exact primary `Örnek 9` | `0.964725` | `34` | `0.631373` | admit |
| `p56:q24` | exact primary `Örnek 24` | `0.928435` | `21` | `0.590491` | admit |
| `p81:q1` | exact primary `Örnek 1` | `0.956451` | `24` | `0.600370` | admit |
| `p89:q43` | exact primary `Örnek 43` | `0.913125` | `53` | `0.682181` | admit |

The gates remain coverage `>= 0.65`, matched tokens `>= 10`, and margin
`>= 0.12`. The earlier `0.600326` value for `p51:q2` came from an exploratory
question-text proxy, not the production `PageMatcher` over the full pinned PDF
page. The unchanged runtime matcher reproducibly yields `1.0` on all 33 query
tokens, so the correct pre-score admission count is `6/8`.

## Leakage and transferability

- The merged runtime index contains no task IDs or benchmark outcome fields.
- Parser layout, public URL identity, PDF bytes, page text, table geometry, and
  source-visible markers are the only features.
- Renaming the opaque benchmark row key cannot change a source decision.
- Missing, ambiguous, non-top-left, non-title, malformed, or conflicting
  example markers fail closed.
- Prefix text, reordered/duplicate multipart labels, generic section
  subphrases, nonadjacent context pages, and Unicode control attacks fail
  closed in tests.
- Trailing multipart delimiters, conflicting numbered/example markers, forged
  source records, forged context pages, and forged source answers also fail
  closed. The resolver, composer, and image-input builder share the same four
  mandatory fail-closed policy gates.
