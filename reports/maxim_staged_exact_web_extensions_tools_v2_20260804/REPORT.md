# Maxim: exact web routing + deterministic tools, full-274

> Status: exploratory/post-hoc. The strict frozen result remains 205/274. All
> higher figures below were developed after aggregate benchmark exposure and
> require an untouched holdout before they can be presented as production
> accuracy.

## Rows for the shared table

| Автор | Часть пайплайна | Идея | Статья | Accuracy |
|---|---|---|---|---:|
| Максим | Агент/RAG/Тулы | Fail-closed routing: staged selector → exact official web evidence (+ one separately marked identical-copy match) → deterministic solvers → official image-key certificates | — | 0.825 |
| Максим | Агент/RAG/Тулы | То же + два воспроизводимых image-solvers по первичному учебнику издателя без официального ключа | — | 0.832 |

The second row is the exploratory maximum. The first row uses only official-key
adjudication for its five new image certificates, but its earlier web layer
still contains one separately identified third-party identical-copy match.
Keep 0.748 as the strict frozen claim.

## Measured ladder

| Snapshot | Overall | Math | Status |
|---|---:|---:|---|
| Frozen gold-blind V2.1 + conservative V3.1 repair | 205/274 = 0.748175 | 108/139 = 0.776978 | strict frozen best |
| Staged selector + truncation repair | 211/274 = 0.770073 | — | post-hoc |
| Staged + exact official web overlay | 215/274 = 0.784672 | 112/139 = 0.805755 | post-hoc, official hosts |
| Same + one identical third-party copy | 216/274 = 0.788321 | 112/139 = 0.805755 | post-hoc, exploratory |
| Extended exact web + deterministic algebra/truth table | 220/274 = 0.802920 | 114/139 = 0.820144 | post-hoc |
| Same + identical-copy evidence | 221/274 = 0.806569 | 114/139 = 0.820144 | post-hoc, exploratory |
| Same + five exact official image-key certificates | 226/274 = 0.824818 | 118/139 = 0.848921 | post-hoc, official certificates |
| Same + two derived primary-publisher image proofs | **228/274 = 0.832117** | **120/139 = 0.863309** | post-hoc maximum |

The final snapshot is +23 correct (+8.394 pp) over the strict 205/274 agent
and +87 correct (+31.752 pp) over the frozen 141/274 page-RAG comparison.

## What was added

1. A fail-closed exact-web router that forms public-only queries, permits only
   configured official hosts, stores cache/provenance, and abstains on an
   ambiguous match. Its offline dry run covers all 274 rows: 212 query-eligible
   and 62 fail-closed. No live search backend was available locally, so the
   measured web overlays use individually verified frozen evidence rather than
   claiming an end-to-end live-search run.
2. Exact exercise matches and answer keys from MEB/OGM/ÖSYM pages. HTML and PDF
   evidence is pinned in composition manifests. Five additional official
   answers conflict with the current benchmark labels; they were not credited
   and should be handled as a dataset-discrepancy audit.
3. Local deterministic tools: symbolic algebra, truth-table evaluation,
   column arithmetic with carry/borrow constraints, and mixed-radix calendar
   arithmetic.
4. Five complete image answers adjudicated against exact official MEB/OGM
   keys. Two later image answers are explicitly separated as derived proofs
   from a primary publisher textbook with no official answer key.

No shared GPU, V100, remote inference, or somebody else's compute was used in
this cycle. Website access was limited to source discovery/download; all
composition, proofs, scoring, and tests ran locally on CPU.

## Reproducibility and guardrails

- Final frozen solver: `exploratory_all_certificates_v3/run/solver.jsonl`,
  SHA256 `64843ab15d2283cd43857d2b49cd33cbd21f8bfaa9482733a0749595bc511b9a`.
- Final 97-row judge: `exploratory_all_certificates_v3/evaluation/derived_certificate_image97_judge.jsonl`,
  SHA256 `f237fb28f9b453964c75a29a17f6ffee0213178874ba3ccb06c21c38092ab3b8`.
- Final score JSON: `exploratory_all_certificates_v3/evaluation/score.json`,
  SHA256 `25e488d358d7370f48e888f4047f9aaf89d43d704c61e5cf0a513820477bcd35`.
- The scorer checks the pinned 274-row benchmark and frozen page-RAG judge,
  keeps failures in the denominator, and rejects solver rows exposing gold.
- The derived-judge builder modifies only `val_0204` and `val_0205`, verifies
  exact solver SHA/candidate/tool provenance, and copies the other 95 image
  verdicts byte-for-byte at row level.
- Targeted tests: 19 passed. All new builders/composers compile and ran
  end-to-end on pinned inputs.

## Interpretation

0.832 is useful evidence that exact source matching plus narrow executable
tools can materially improve this benchmark. It is not a valid untouched-test
estimate: target rows and policies were chosen after seeing aggregate outcomes,
and two final points use derived arithmetic proofs without an official key.
For a production claim, freeze the router and thresholds now, then rerun on a
new hidden split with independent image judging.
