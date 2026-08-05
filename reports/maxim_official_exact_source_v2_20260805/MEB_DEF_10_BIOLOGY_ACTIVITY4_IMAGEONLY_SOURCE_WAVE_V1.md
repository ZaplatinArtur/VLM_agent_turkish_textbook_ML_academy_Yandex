# MEB-DEF-10 Biology Activity 4 image-only source wave V1

Date: 2026-08-05

## Result

The previously excluded image-only Biology observation now has one isolated,
source-only certificate candidate.  A full 178-page visual sweep selects
physical page 75, and that page contains exactly one indexed, visually reviewed
activity: `ETKİNLİK-4`.  Its official answer key is independently attested on
physical page 180.

This wave does **not** modify the active V6 profile, resolver, composition, or
score.  It does not read a solver answer, image-judge outcome, evaluator output,
reference answer, correctness flag, or metric.  The result is ready for a later
pre-registered merge.

## Why the prior text gate abstained

The frozen parser row contains exactly one `image` block and no textual block.
Consequently, the existing activity-label path correctly refused to invent an
activity number from OCR.  The new path keeps that abstention semantics and
adds a separate contract:

1. parser structure must be one near-full-page image block with no text;
2. the public locator must identify exactly one SHA-pinned official workbook;
3. SIFT/RANSAC must evaluate every indexed content page;
4. the selected page must pass all frozen geometry and rank-margin thresholds;
5. the complete PDF page must contain exactly one canonical activity marker;
6. exactly one PDF-attested, visually reviewed activity may exist on that page;
7. the mapped task polygon must overlap the activity content bbox at IoU >= 0.80.

No task ID, task ordinal, filename number, expected page, or expected activity
number participates in page or record selection.  `task_id` remains only in an
alignment-audit object used to join the parser row to its public locator.

## Pinned inputs

- Parser artifact SHA256:
  `040f9e0884d9be8335d35aa75091315937188eb8e2c76adb17fbad8653400038`
- Public locator artifact SHA256:
  `47e61467c5470c0f50d63199dbfb3ea9a218b0d14dacfcb3f8649b35c62c776a`
- Task image SHA256:
  `e35aadc92dc031135c988b1c24e0840fb8e31f21de74057640cdf055a9c5b1b1`
- Official PDF:
  `tmp/remaining_official_source_audit/pdfs/MEB-DEF-10biyoloji.pdf`
- PDF SHA256:
  `640bb362f2d53d31663326ac303c5065f4670f2a0d506300beb5e41869384e2b`
- PDF pages: `183`; visual content-page sweep: physical pages `1..178`
- Official book:
  `https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/index.html`

Across all 274 pinned parser rows, the structural projector finds exactly one
row satisfying the image-only contract.  Its image block covers
`0.9442198951003304` of the task image.  Changing only its alignment task ID
does not change the parser projection SHA256
`383619ecba70e5857dd8a932e529e6e1ec859820725ccf7832376ab6b5f44d04`.

## Official PDF proof

Source address:

```text
(pdf_sha256, unit_number, printed_page_number, activity_number)
=(640bb362..., 2, 75, 4)
```

- Content page: physical/printed page `75`
- Content marker: `ETKİNLİK-4`
- Marker bbox: `[284.384, 54.266, 341.254, 65.266]`
- Content bbox: `[42.0, 50.0, 582.0, 780.0]`
- Content projection SHA256:
  `c9adf8f44e401f901c0451323b51ed060c5312eff327e5ba62612c1d3d5e329c`
- Key page: physical page `180`
- Key header: `Etkinlik 4 (75. Sayfa)`
- Key header bbox: `[125.762, 508.902, 223.542, 518.902]`
- Key bbox: `[41.0, 506.0, 296.0, 570.0]`
- Key projection SHA256:
  `b9c3d1198dde8d61083470a4f206f2c7843ee7823f0531fb1672f5f981daadda`
- Joint address/projection SHA256:
  `4d01ded97c26ea41d0db157c205d2ce07133d12371cb80e2a727d5b7ca2860bc`
- Complete-page canonical activity-marker inventory: `[4]`
- Marker-inventory projection SHA256:
  `8050f1dbce11c81fef5dddb537fedca3e9e0c4d2f41c0e7d52f7e02be7f97510`

The key exposes all ten ordered components (`a,b,c,ç,d,e,f,g,ğ,h`).  The
existing `activity_answer_key` attestor re-derived every component from the
official PDF and reproduced both PDF projection hashes.

## All-page visual result

The generator rendered all `178` indexed content pages with Poppler 26.05.0 at
144 DPI, then computed one pinned SIFT/RANSAC comparison per page using
OpenCV 5.0.0 and NumPy 2.5.1.

| Rank | Physical page | Rank score | Good matches | Inliers | Inlier ratio | Hull fraction | Median reprojection error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 474.368681 | 1021 | 872 | 0.854065 | 0.686903 | 0.301185 |
| 2 | 125 | 23.568111 | 540 | 262 | 0.485185 | 0.141444 | 1.028502 |
| 3 | 141 | 19.105299 | 494 | 275 | 0.556680 | 0.027454 | 0.184786 |

- Best/runner score margin: `450.80057051913224` (required: `>= 10`)
- Best/runner score ratio: `20.127565001024152` (required: `>= 5`)
- Mapped polygon/content-bbox IoU: `0.9203275201446249`
  (required: `>= 0.80`)
- Selected source record:
  `meb_def_10biyoloji_640bb362f2d5:p75:q4`
- All 17 geometry, identity, uniqueness, projection, and crop checks: passed

## Independent replay

The complete generator was run a second time into a different output path.  It
freshly re-rendered all 178 pages and recomputed all 178 SIFT/RANSAC pairs.
The two outputs were byte-identical:

```text
primary SHA256 = 4a14842d3e2ea83555b300fb1f1509dc39edb6ff2bc176752b948969d65b184e
replay  SHA256 = 4a14842d3e2ea83555b300fb1f1509dc39edb6ff2bc176752b948969d65b184e
byte_equal      = true
```

The source-fragment adapter was also rerun independently.  Both canonical
source-index outputs had SHA256
`2782d62b6b1610af81800c525ffbbecd68947268e925d9e3fedbeb516d88669d`.

## Frozen artifacts

- Raw Activity 4 fragment:
  `frozen/public_workbook_source_fragment_meb_def_10_biology_activity4_imageonly_candidate_v1.json`
  (`dd1e26720523550c38034778b47ce6e87c16bb602d50d4b992182dbe580b3cc8`)
- Canonical one-record source index:
  `frozen/public_workbook_source_index_meb_def_10_biology_activity4_imageonly_candidate_v1.json`
  (`2782d62b6b1610af81800c525ffbbecd68947268e925d9e3fedbeb516d88669d`)
- Adaptation manifest:
  `frozen/public_workbook_source_index_meb_def_10_biology_activity4_imageonly_candidate_v1.manifest.json`
  (`1b6267c14cfbe4794341eb019a74125d440a95609b68f6d33880326b5192da41`)
- Image-only frozen profile:
  `configs/maxim_biology_activity4_imageonly_sourceonly_v1.json`
  (`9213bd972686fbd24e0822530d41c6c1e25212d8ce14a14c58016ba9f4a341ba`)
- Full 178-page visual artifact:
  `frozen/activity_visual_binding_biology_activity4_imageonly_candidate_v1.json`
  (`4a14842d3e2ea83555b300fb1f1509dc39edb6ff2bc176752b948969d65b184e`)

## Integration status

This candidate is deliberately isolated.  A later wave may merge its one
task-ID-free source record and its image-only visual artifact into a newly
frozen profile before any aggregate scoring.  Until that explicit merge, V6
and its measured result remain unchanged.
