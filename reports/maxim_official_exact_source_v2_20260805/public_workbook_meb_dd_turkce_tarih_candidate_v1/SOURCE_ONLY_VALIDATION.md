# MEB DD Turkish/History candidate: source-only validation

Date: 2026-08-05

## Guardrails

- Inputs inspected: the two immutable public MEB PDFs, frozen source locators,
  and OCR/parser observations.
- Not inspected or used: benchmark answers, reference answers, score files,
  judge outcomes, pass/fail correctness, rewards, or outcome-derived routing.
- The source index is task-ID-free. Task IDs appear below only to audit which
  frozen locator row is covered by a source-native PDF address.
- No scorer was run.

## Source-native inventory

| PDF | SHA-256 | Source records | PDF/key attestation |
|---|---|---:|---:|
| `MEB-DD-TYT-tarih.pdf` | `7ebb73de46eb8e213771354905090b677386be4aea5e506eb96f0339548c7e20` | 7 | 7/7 |
| `MEB-DD-TYT-turkce.pdf` | `ee4a7e7c202e7b1464f63b61f7896582826d7a5b6f3afa3961942fec695d4d1f` | 8 | 8/8 |

All 15 records pass the existing `verify_workbook_index_pdf` checks: immutable
PDF identity, page count, exactly one printed question number on the content
page, one A-E glyph in the indexed key crop, and source-native inline/table
context binding.

## Resolver coverage under unchanged thresholds

| Locator row | Source address | Binding | Coverage | Margin | Current runtime |
|---|---|---|---:|---:|---|
| `val_0087` | history p213 q1 | inline solution | 1.0000 | 0.6544 | accepted |
| `val_0088` | history p214 q4 | inline solution | 1.0000 | 0.7547 | accepted |
| `val_0089` | history p221 q4 | key table | 0.5841 | 0.4361 | abstain: coverage below 0.65 |
| `val_0090` | history p223 q1 | inline solution | 0.9247 | 0.5792 | accepted |
| `val_0091` | history p223 q2 | inline solution | 0.9064 | 0.7296 | accepted |
| `val_0092` | history p241 q1 | key table | 1.0000 | 0.6308 | accepted |
| `val_0093` | history p244 q12 | key table | 0.9671 | 0.6002 | accepted |
| `val_0094` | Turkish p14 q11 | inline solution | 0.9309 | 0.8005 | accepted |
| `val_0095` | Turkish p14 q10 | inline solution | 0.9399 | 0.8250 | accepted |
| `val_0096` | Turkish p35 q1 | inline solution | 0.8891 | 0.7084 | accepted |
| `val_0097` | Turkish p35 q2 | inline solution | 0.9537 | 0.8110 | accepted |
| `val_0197` | Turkish p21 q1 | key table | 0.8454 | 0.6987 | blocked before match: locator omits `nosw` |
| `val_0198` | Turkish p17 q2 | key table | 0.9363 | 0.7540 | blocked before match: locator omits `nosw` |
| `val_0199` | Turkish p44 q6 | key table | 0.9122 | 0.7859 | blocked before match: locator omits `nosw` |
| `val_0200` | Turkish p48 q5 | inline solution | 0.9638 | 0.8105 | blocked before match: locator omits `nosw` |

The unchanged runtime therefore adds **10 accepted source certificates**. The
full seven-document source-only resolver has 71 accepted certificates versus
61 for the five-document coordinate candidate. This is a coverage count, not
an accuracy measurement.

Four more rows have strong unique page matches and verified keys but are
rejected by URL grammar before matching. Their frozen URLs contain exactly the
same immutable `{url, name}` identity as the accepted Turkish rows and simply
omit the inert viewer parameter `nosw`. A safe identity extension is:

1. accept exactly `{url, name}` or exactly `{url, name, nosw}`;
2. when present, require `nosw` to be numeric;
3. discard `nosw` before identity comparison;
4. reject every other query field, duplicate field, fragment, or host/path.

That source-semantics change has a source-observable ceiling of four additional
certificates. It should be implemented and tested separately; the candidate
profile does not pretend the current runtime already supports it.

The remaining history table crop (`val_0089`) has 48 matched tokens and a very
large 0.4361 page margin, but its OCR contains a large HTML table projection and
coverage is 0.5841. Do not lower the global threshold from this one target.
A defensible follow-up is a preregistered table-aware projection or rescue rule
calibrated on unrelated source-only DD pages.

## Frozen artifacts

- Source fragment: `../frozen/public_workbook_source_fragment_meb_dd_turkce_tarih_candidate_v1.json`
  - SHA-256: `a83211a10b4cd8a64750334cde5d0724ea4cf142f3ebbbca3e4f674a18218811`
- Merged seven-document source index:
  `../frozen/public_workbook_source_index_meb_dd_turkce_tarih_candidate_v1.json`
  - SHA-256: `d9fa48f57961bbef05359e0e03e89146272d306429e807dbd1331387ce66fa52`
- Build manifest:
  `../frozen/public_workbook_source_index_meb_dd_turkce_tarih_candidate_v1.manifest.json`
  - SHA-256: `9158e8c89c89321f49b6bb254b3a4a7c673e063a7a7af1d85922f35bbe8d4fb0`
- Candidate profile:
  `../../../configs/maxim_public_workbook_meb_dd_turkce_tarih_candidate_v1.json`
  - SHA-256: `3b71a10461b53626908bca8c2c1f97f63046cde429014c3b29c87357621b2f2f`
- Source-only resolver manifest: `manifest.json`
  - SHA-256: `061e7371efb7bf506d431600a070b67c00b46614ec3258f8c61de44eda2b2ad7`

## Risks

- This is a development replay, not an unseen holdout.
- The 15-record inventory was manually reviewed against rendered public PDF
  pages; a second reviewer should recheck coordinates before production.
- The four no-`nosw` rows need a small, explicit identity-parser change.
- The low-coverage table row remains fail-closed under the current profile.
- Source coverage cannot establish benchmark accuracy without a separately
  frozen scoring step, which was intentionally not performed here.
