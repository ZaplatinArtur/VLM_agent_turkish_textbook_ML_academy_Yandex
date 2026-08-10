# MEB 3A TYT Turkish candidate: source-only validation

Date: 2026-08-05

## Outcome

The public PDF contains source-native bindings for all seven locator rows that
name `MEB-3A-SB-TYT-turkce.pdf`. Each observed crop has one dominant physical
page match, and all seven printed question markers, section/ADIM rows, exact
hyphenated key cells, and answer-glyph crops were visually and geometrically
checked.

The final v4 runtime re-verifies all seven source records and admits six under
the unchanged observed-number contract. The p39 row remains an abstention
because its parser crop has no observed printed number. This is not a measured
accuracy gain. No scorer was run.

## Guardrails

- Inputs inspected: the immutable public PDF, frozen source locators, and
  gold-blind parser observations.
- Not inspected or used: benchmark answers, reference answers, score files,
  judge outcomes, rewards, or pass/fail correctness.
- Task IDs occur only in `locator_alignment_audit.jsonl`, where they attest that
  seven frozen locator rows address this exact source. The source fragment and
  merged source index are task-ID-free.
- The source index parses under `public-workbook-source-index-v1`. The v4 PDF
  verifier accepts the book's exact `N-A` syntax only with same-row ADIM and
  exact nearest-section proof.

## Immutable source

| Property | Value |
|---|---|
| File | `MEB-3A-SB-TYT-turkce.pdf` |
| PDF SHA-256 | `7e8d45d46ed391a158121858157c30d37bd16b6256f773cf9b13fbc15c6a8f2f` |
| Physical pages | 283 |
| Content range | 1-276 |
| Printed answer-key range | 277-282 |
| Indexed key page | 277 |
| Source records | 7 |

All seven target content pages and the relevant key page 277 were rendered and
visually inspected. The complete answer-key section, pages 277-282, was also
rendered and text-extracted to confirm its boundaries; none of the seven source
addresses depend on pages 278-282.

## Locator alignment

| Locator row | Source address | Page coverage | Margin | Printed number |
|---|---|---:|---:|---:|
| `val_0189` | p26 q9 | 0.9228 | 0.7531 | 9 |
| `val_0190` | p27 q4 | 0.9521 | 0.8229 | 4 |
| `val_0191` | p30 q5 | 0.9306 | 0.7473 | 5 |
| `val_0192` | p32 q6 | 0.9263 | 0.7610 | 6 |
| `val_0193` | p39 q5 | 0.9415 | 0.7779 | absent in crop projection |
| `val_0194` | p57 q3 | 0.9412 | 0.7920 | 3 |
| `val_0195` | p62 q5 | 0.8888 | 0.7216 | 5 |

The weakest page match still has 0.8888 coverage and a 0.7216 gap over the
runner-up page. The numberless crop on p39 has strong source-side text overlap
(84 of 88 query tokens, weighted coverage 0.9254), but v4 does not use that as
an exception: the frozen policy requires an observed printed number, so this
row abstains.

## PDF/key attestation

Every source record has all of the following:

- an exact PDF hash and page count;
- one printed leading question marker inside a narrow reviewed content crop;
- one exact A-E glyph inside the indexed key crop;
- one unique full key token of the form `N-A` on the same line;
- one matching `1. ADIM`, `2. ADIM`, or `3. ADIM` row label;
- the expected nearest preceding section heading;
- a visual review of both the content question and its answer-key cell.

Page 57 contains a second textual `3.` inside the passage (`3. nüshası`). A
reviewed content bbox isolates the actual leading question marker. This avoids
the unsafe alternative of weakening the global unique-marker rule.

## V4 runtime attestation

The narrow source-format extension is implemented and was rerun against the
pinned 283-page PDF. It accepts an `N-A` cell only when the answer-glyph bbox is
inside that exact token, the indexed ADIM occurs on the same physical row, and
the nearest preceding row without answer cells exactly equals the indexed
section. A more distant matching heading cannot authorize a cell below a newer
section. Adversarial tests cover a nearer wrong section and a wrong same-row
ADIM; both fail closed.

All seven source records pass PDF/key verification. Six produce certificates
under the unchanged page and observed-number gates; p39 abstains. This changes
source parsing, not routing policy, thresholds, or the anchor.

## Frozen artifacts

- Source fragment:
  `../frozen/public_workbook_source_fragment_meb_3a_tyt_turkce_candidate_v1.json`
  - SHA-256: `527255be05de7f4b2235436abacb308b0609896375fb8cec0254a3a61fd6dc8e`
- Merged eight-document candidate index:
  `../frozen/public_workbook_source_index_meb_3a_tyt_turkce_candidate_v1.json`
  - SHA-256: `43df280a10eec1d504fc1c99598eb8b4429fcca230308bb5c2fd0465b864182b`
  - Inventory: 8 documents, 102 task-ID-free source records.
- Merge manifest:
  `../frozen/public_workbook_source_index_meb_3a_tyt_turkce_candidate_v1.manifest.json`
  - SHA-256: `188ef733c8c4faa7beb5da7dad3f93cbe67c1e3335fbe52064d38ac1a47db37f`
- PDF-native source-binding audit: `source_binding_audit.json`
  - SHA-256: `4bf51ddaeed9ca0ba5ed0679b463526f8e814f91255c4a261d7ce926330c108c`
- Locator-only alignment audit: `locator_alignment_audit.jsonl`
  - SHA-256: `180e180c68da23cc846a1cf8a94506f20b5a20d2376c3800ea7012249275d4b2`

## Risks

- This remains a development replay, not an unseen holdout.
- The seven coordinate bindings were manually reviewed; production admission
  should receive a second independent visual review.
- The ADIM/section verifier is implemented, but this remains a development
  replay rather than an unseen holdout.
- The p39 crop has strong source-only text coverage but remains inactive because
  v4 does not relax the observed-number requirement.
- The seven-row ceiling is source coverage only and says nothing by itself about
  benchmark correctness.
