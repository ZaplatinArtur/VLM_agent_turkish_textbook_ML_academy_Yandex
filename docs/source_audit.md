# Source audit — 2026-07-13

## Main workbook

- 823 task records.
- 468 single-choice tasks, 294 open questions, and 61 other structured tasks.
- 453 text-only, 304 text-plus-visual, and 65 text-plus-table records.
- 20 raw subject labels occur in the 823 currently usable task rows.
- 771 question images are represented only by asset IDs such as `q0624`, not resolvable URLs.
- The planning sheet and mentor target contain 17 subject categories, while Sheet 1 contains 20 raw labels because some labels may be aliases.
- No canonical mapping is applied until the team confirms the exact correspondence; raw subject labels are preserved losslessly.

## Math workbook

- 200 unique tasks covering grades 1–12.
- 45 easy, 95 medium, and 60 hard tasks.
- 103 single-answer multiple-choice, 61 multi-answer, 17 numeric, 17 open-ended, and 2 currently unclassified tasks.
- Every task has a question image and annotated reference image.
- All 400 public image links were successfully resolved through the source API after resumable retry.
- 106 tasks have textual answer labels; three of them contain several labels such as `A;A;D` and are evaluated as multi-answer tasks.

## Historical failure sample

- The workbook includes eight previously generated wrong answers with reference choices.
- The deterministic normalizer extracts all eight final choices correctly and reproduces all eight source labels.
- These records are retained as a regression/calibration seed, not as evidence of judge quality by themselves.

## ÖdevJet corpus

- 45,920 rows, 45,576 unique IDs, 215 books, 8 subjects, grades 1–8.
- 344 duplicate rows by ID; 263 duplicated IDs contain conflicting payloads.
- 16,520 records are low-information placeholders or unusually short pages.
- Publisher metadata is absent for 31,299 rows.
- Image count is highly bimodal: most rows contain either one image or nine images.

## Decisions required

1. Confirm the exact mapping from 20 raw labels to the mentor's 17 subject categories.
2. The team must provide an asset-ID-to-image mapping for the 771 unresolved main-workbook questions.
3. The agent team must provide candidate outputs under the shared data contract.
4. The team must select a judge provider/model and supply credentials and cost limits.
5. Retrieval preprocessing should deduplicate IDs and decide whether low-information records are removed or down-weighted.
