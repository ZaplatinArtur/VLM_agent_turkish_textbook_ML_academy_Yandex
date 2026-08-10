# Maxim primary-layout MEB-3A + Samsungis v4 protocol

## Status

This is a pre-score protocol for a previously inspected development replay,
not a fresh holdout claim. The final profile, source-native records, resolver,
composition, implementation and tests must be committed and pushed before a
v4 image judge or score is created.

## Source-only changes from v3

All v3 page thresholds and the primary-layout question-number projection stay
unchanged. V4 adds two independently inspected public PDFs:

- `MEB-3A-SB-TYT-turkce.pdf`: seven reviewed Turkish records. Six can satisfy
  the unchanged observed-number contract; the numberless p39 record remains an
  abstention.
- `https://samsungis.meb.gov.tr/storage/denemeler/lgs/lgs1.pdf`: eighteen
  reviewed LGS records. Seventeen satisfy the unchanged page gate; the short
  science q3 crop remains an abstention because page coverage is below 0.65.

The merged index has nine documents and 120 task-ID-free source records. No
benchmark answer, candidate outcome, judge result, score or reward was used to
build either source wave.

## Narrow source-format extensions

V4 recognizes only source syntax present in the pinned PDFs:

1. A direct-PDF identity must be an exact, lowercase-host, query-free HTTPS
   `.pdf` URL. Ports, credentials, escapes, dot segments, alternate paths,
   queries and fragments are rejected. The fetched PDF bytes remain pinned by
   SHA-256 and page count.
2. A list key may expose one exact combined token `N:A`, `N-A` or `N.A`. The
   whole token, answer glyph bbox, source-book heading and nearest subject or
   ADIM context must agree. For subject lists, the nearest preceding non-answer
   row in the same page column must exactly equal the indexed subject. For the
   MEB hyphen table, the ADIM must share the key row and the nearest preceding
   non-answer row must exactly equal the indexed section. A farther matching
   heading cannot authorize a cell below a newer section.
3. When a full PDF page repeats the same number inside prose, the global marker
   rule is not weakened. A previously reviewed content bbox must contain exactly
   one printed marker, and the PDF re-verifier records that per-record count in
   the certificate provenance. Missing or forged crop proof is an abstention.

These changes affect source parsing only. They do not change retrieval
thresholds, answer policy, task routing or the anchor.

Adversarial tests freeze both context boundaries: a nearer wrong subject or
section and a wrong same-row ADIM are rejected before any benchmark scoring.

## Frozen admission policy

- page coverage >= 0.65;
- at least 10 matched tokens;
- page margin >= 0.12;
- observed printed question number required;
- one source record and one PDF-verified marker;
- exact immutable public-document identity;
- reviewed PDF-bound key with valid A-E or short-text source answer;
- any ambiguity or malformed input keeps the anchor.

Task ID is alignment-only. The direct source URL, source-index record address,
PDF bytes, parser observation, profile, implementation and composition are all
hash-pinned before scoring.

## Expected source-only delta

The source resolver contains 111 certificates versus 88 in v3. Composition
produces 22 answer overrides versus 19 in v3; the three new overrides all come
from the MEB-3A Turkish source. This is a source-side statement only, not an
accuracy claim. The score remains unknown until the remote pre-score freeze is
verified.
