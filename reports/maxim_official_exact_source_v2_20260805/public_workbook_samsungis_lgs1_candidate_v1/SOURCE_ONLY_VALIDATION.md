# Samsungis LGS 1 source-only validation

Date: 2026-08-05

## Outcome

The exact official locator addresses a 25-page PDF with 24 content pages and a
printed answer list on page 25. Twenty-two frozen locator rows point to this
document. Eighteen source-native records were reviewed and indexed without task
IDs; seventeen pass the unchanged page/number gates. No scorer, benchmark
reference, candidate outcome or judge result was opened.

## Immutable source

- URL: `https://samsungis.meb.gov.tr/storage/denemeler/lgs/lgs1.pdf`
- PDF SHA-256: `f88c9f40e3c6f3a2494090ee7635d1f7254b736da561b0f0b98125cb25ad5997`
- physical pages: 25
- answer-key page: 25
- reviewed records: 18/18
- unchanged-gate admissions: 17/18

The answer page contains subject-scoped combined tokens such as `13:A`, `2:B`
and `20:B`. Every indexed cell is tied to the document header, its subject
heading, physical content page, printed question number and exact answer glyph.

## Fail-closed decisions

- Science q3 on physical page 19 is source-attested but its OCR query reaches
  only 0.4632 page coverage, below the frozen 0.65 gate. It remains an
  abstention.
- Four other locator rows were not indexed because their leading question
  markers are not exposed unambiguously by PDF-native text extraction. Their
  visible answers were not used to relax the marker rule.
- Math q3 on page 14 contains another `3.` elsewhere on the page. A reviewed
  content bbox isolates exactly one leading q3 marker; the full-page duplicate
  is never accepted as proof.
- Alternative URLs, query strings, fragments, credentials, ports, percent
  escapes and path normalization are rejected even if they could reach similar
  content.

## Artifacts

- fragment:
  `../frozen/public_workbook_source_fragment_samsungis_lgs1_candidate_v1.json`
- merged v4 index:
  `../frozen/public_workbook_source_index_meb_3a_samsungis_candidate_v4.json`
- merged-index manifest:
  `../frozen/public_workbook_source_index_meb_3a_samsungis_candidate_v4.manifest.json`
- sealed pre-score resolver:
  `../public_workbook_primary_layout_meb3a_samsungis_v4_sealed_resolver/manifest.json`

The fragment and merged index contain no `val_*`, task ordinal, benchmark
answer, correctness flag, metric or score field.
