# Updated validation archive run — 2026-07-21

## Source and import

- Source ZIP SHA-256: `cae6d1e3cea6e4ec20b996d3a99b4e05056c26091baff3f4eb6a200af1bca21a`
- Workbook rows in `Sheet1`: 823
- Packaged images: 278; all files referenced, readable, and content-unique
- Rows with a resolved local question image: 199
- Trusted text references: 119
- Reference images transcribed by Qwen: 80
- Rows still lacking a packaged local question image: 624
- Historical `Sheet6` failures resolved to local images: 8/8

The importer reads direct paths from columns E/F and the updated mappings from the unnamed
columns L/M. It emits stable row-based task IDs and validates every resolved path inside the
extracted archive root.

## Runtime

- Host: `158.160.42.62`
- GPU: A100 80 GB, device 0
- Model: `Qwen/Qwen3.5-9B`
- Serving: vLLM 0.25.1
- Extraction prompt: `validation-transcription-v1`
- B0 prompt: `v3`, thinking disabled, JSON schema
- Binary judge: `text-binary-v4`

## Pipeline completion

| Stage | Result |
|---|---:|
| Question/reference extraction | 199/199 |
| Initial B0 generation at temperature 0 | 186/199 |
| B0 after error-only retries | 199/199 |
| Binary judge valid output | 199/199 |
| Judge score 1 | 84/199 (42.2%) |
| Judge score 0 | 115/199 (57.8%) |

The 13 initial B0 length failures were retried without recomputing successful rows. Eight
succeeded at temperature 0.3, one at 0.7, three at 1.0, and one at 1.2 with a 1536-token limit.
Therefore 84/199 is an operational full-coverage result, not a clean single-decoding benchmark.

## Preliminary score by subject

| Subject | N | Score 1 | Rate |
|---|---:|---:|---:|
| Math | 64 | 18 | 28.1% |
| Chemistry | 32 | 15 | 46.9% |
| Turkish language and literature | 21 | 10 | 47.6% |
| Physics | 19 | 10 | 52.6% |
| Biology | 19 | 4 | 21.1% |
| Geography | 14 | 9 | 64.3% |
| History | 10 | 5 | 50.0% |
| English | 9 | 7 | 77.8% |
| Science | 5 | 2 | 40.0% |
| Sociology | 3 | 2 | 66.7% |
| Philosophy | 2 | 2 | 100.0% |
| Atatürkçülük | 1 | 0 | 0.0% |

Small-subject rates are descriptive only.

## Judge safeguards and remaining calibration work

The Qwen judge produced six false positives where the explicit multiple-choice final letter did
not match the trusted reference. `text-binary-v4` adds a deterministic mismatch guard: an explicit
wrong final letter is always score 0, while matching letters still go to Qwen so materially false
reasoning can be rejected. The final run has zero wrong-letter score-1 cases and no format errors.

This run does not replace human calibration. Before publishing a benchmark number:

1. manually verify a stratified 20–30-record sample, including image-derived references;
2. inspect exact-letter answers rejected for contradictory reasoning;
3. freeze one retry/temperature policy and rerun the final benchmark with that single policy;
4. keep the 80 image-derived references out of held-out evaluation until their transcription is verified.

A 30-record review sheet with blank `manual_score` and `manual_notes` columns is prepared at
`results/human_review_30.csv`. It includes image-derived references, accepted answers, deterministic
wrong-choice rejections, and Qwen reasoning rejections.

## Remote artifacts

All generated artifacts are retained under `/home/m.kulyamin/vlm-agent-e2e-run/`:

- `data/validation_all_subjects_20260721/validation_manifest.jsonl`
- `data/validation_all_subjects_20260721/tasks_199.jsonl`
- `results/validation_extractions_qwen35_9b.jsonl`
- `results/b0_validation_199_v3.jsonl`
- `results/b0_validation_199_v3_judge_input.jsonl`
- `results/b0_validation_199_v3_judged_v4.jsonl`
- `results/human_review_30.csv`
- `results/b0_seed8_multimodal_v3.jsonl`
- `results/b0_seed8_multimodal_v3_judged_v4.jsonl`
