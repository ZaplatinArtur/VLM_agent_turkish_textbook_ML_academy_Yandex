# Evaluation protocol v1

## Primary question

Does textbook retrieval improve answer correctness over the same agent without tools and with web search?

## Metrics

1. Deterministic strict accuracy for parseable multiple-choice, numeric, and short-text answers.
2. Blinded reference-aware LLM strict accuracy for visual, multi-answer, and open-ended tasks.
3. Mean ordinal judge score from 0 to 4 as a secondary metric.
4. Paired win/loss/tie and score delta on identical tasks.
5. Unjudgeable/failure rate, latency, and cost as guardrail metrics.

The primary hybrid score uses deterministic matching whenever it is applicable and the LLM verdict otherwise. Reports also expose deterministic-only and judge-only views; one must never silently overwrite the other. Empty/failed agent responses count as incorrect. Judge execution failures and genuinely unreadable references are reported as different categories.

Accuracy must not be computed by dropping failed agent runs. A failed or empty agent response is incorrect; a genuinely unreadable benchmark reference is unjudgeable and reported separately.

## Bias controls

- Do not reveal the agent setup to the judge.
- Use temperature 0 or the provider's most deterministic mode.
- Pin judge model and prompt versions.
- Randomize A/B order in pairwise validation and repeat a calibration subset with swapped order.
- Do not reward verbosity, citations, confidence, or the presence of retrieved material.
- Keep a human-labeled calibration set stratified by subject, grade, and answer type.

## Calibration

Before the full run, manually label the frozen 120-response calibration sample produced after all three agent runs exist. Balance it 40/40/40 across setups and stratify within setup by answer type, grade, and difficulty. This is a sample of candidate responses, not 120 tasks multiplied by three. Deliberately cover:

- fully correct, partially correct, and incorrect answers;
- correct final result with invalid reasoning;
- equivalent numeric and symbolic forms;
- tasks containing several required sub-answers;
- poor-quality or ambiguous reference images;
- Turkish text and visually dense diagrams.

Report judge agreement with the human labels, a confusion matrix, and the most common disagreement classes. Do not tune the judge prompt on the final test slice.

The precomputed 120-task file is a candidate-neutral source pool and UI fixture. The post-run `sample-calibration-responses` output is the actual human-vs-judge calibration set. Double-label at least 20% of that response sample to estimate the human ceiling.

For double-labeling, give each annotator a separate annotation JSONL file; identical `annotation_id` values are intentional and are joined only during `analyze-calibration`. Sharing one writable file would overwrite the first annotator rather than measure human-human agreement.

## Adjudication

Primary human labels are collected before any LLM verdict or retrieval trace is shown. A separate adjudicator then reviews:

- every human/LLM score disagreement;
- every disagreement on strict correctness;
- judge execution errors and malformed verdicts;
- judge confidence below the frozen threshold;
- any human or judge reference-quality flag;
- a deterministic 10% sample of exact agreements as a control against confirmation bias.

The adjudicator sees expected and candidate answers together, then the two original verdicts. The final record must state whether the human label, LLM label, a custom score, or exclusion was selected, plus the issue source and rationale. Excluded/invalid-reference items are reported separately and are not silently converted to agent failures. Queue rules and the agreement-sample seed are frozen with the judge configuration.

## Statistical reporting

Report paired differences with 95% bootstrap confidence intervals. Show results by raw subject label until the 17-category mapping is confirmed, then by canonical subject, grade, difficulty, question format, and whether retrieval found an exact solution, theory-only support, or no useful evidence.

Before interpreting quality differences, require a complete `(task, setup, run)` grid or explicitly report missing cells. Missing outputs must not disappear from denominators through an inner join.
