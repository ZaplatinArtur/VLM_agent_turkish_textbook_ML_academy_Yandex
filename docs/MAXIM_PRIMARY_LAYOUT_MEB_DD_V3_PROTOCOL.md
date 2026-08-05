# Maxim primary-layout + MEB DD v3 protocol

## Status

This is a pre-score protocol for a previously inspected development replay,
not a fresh holdout claim.  All generation artifacts must be remotely frozen
before this variant is scored.

## Changes from primary-layout v2

The primary-layout question-number projection and all page/key thresholds are
unchanged.  V3 adds two source-native official MEB documents:

- `MEB-DD-TYT-tarih.pdf`: 7 reviewed records;
- `MEB-DD-TYT-turkce.pdf`: 8 reviewed records.

All 15 records were indexed without task IDs and independently re-read from
their pinned PDF pages and answer-key coordinates.  The detailed source-only
audit is in
`reports/maxim_official_exact_source_v2_20260805/public_workbook_meb_dd_turkce_tarih_candidate_v1/SOURCE_ONLY_VALIDATION.md`.

## Optional `nosw` identity projection

Some frozen Yandex Docs links identify the same immutable public resource with
exactly `url` and `name`; other links add a numeric viewer-only `nosw` field.
V3 explicitly freezes `url_name_plus_optional_numeric_nosw_v2`:

- accepted query-key sets are exactly `{url, name}` or
  `{url, name, nosw}`;
- every key must occur exactly once;
- `nosw`, when present, must be non-empty and numeric;
- `nosw` is discarded before source identity comparison;
- extra keys, duplicate keys, fragments, alternate host/path/port/userinfo,
  malformed public locators, and unsafe filenames are rejected.

The optional behavior is passed explicitly from the profile through both
document selection and question resolution.  Legacy/default profiles remain
fail-closed and require numeric `nosw`.

## Unchanged answer-admission gates

An indexed key can replace the anchor only after all existing checks pass:

- strict immutable public-document identity;
- page coverage at least 0.65 and at least 10 matched tokens;
- page margin at least 0.12;
- one source record with the observed printed question number;
- exactly one printed marker on the matched PDF page;
- a reviewed, PDF-bound key context and a valid source answer.

The single low-coverage History table row remains an abstention; no threshold
is reduced for it.  `task_id` remains alignment-only, and benchmark candidates,
references, judge results, scores, and outcome-derived features are forbidden
during generation.

## Required freeze

- Re-run the seven-document source resolver with the final code.
- Freeze projection decisions, source-index inputs, PDFs, certificates,
  composition decisions, implementation files, and tests by SHA-256.
- Commit and push that handoff before creating a V3 image judge or score.
- Report the result as development replay and evaluate future document-grouped
  sources with a one-shot sealed protocol.
