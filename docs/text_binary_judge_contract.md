# Text binary judge contract

## Purpose

Evaluate a candidate answer using text only:

```text
judge(question_text, reference_answer, candidate_answer) -> 0 | 1
```

This contract is separate from the existing multimodal `0..4` judge. It does
not accept or inspect question images, reference images, setup names, or tool
traces.

## Decision policy

Return `1` only when the candidate is correct and complete for the supplied
question and reference. Return `0` when the candidate is wrong, incomplete,
contradictory, irrelevant, empty, or cannot be matched to the reference.

Accept equivalent wording, notation, units, language, and mathematically
equivalent forms. Ignore style, verbosity, spelling, and formatting unless
they change the meaning. A materially false explanation makes the answer
incorrect even if an isolated final token happens to match. For multi-part
questions, every required part must be correct.

The reference must be checked before judging. Records with a missing,
ambiguous, or obviously incorrect reference are excluded from the quality
metric and marked separately as reference issues; the judge must not guess.

## Qwen response format

Qwen must return one JSON object and nothing else:

```json
{"score": 1, "rationale": "The candidate matches the reference and answers all parts."}
```

`score` is an integer and must be exactly `0` or `1`. `rationale` is a short
diagnostic string (maximum 120 words) and is not used for scoring.

The pipeline's public score is the integer `score`. The rationale is retained
for manual review. No markdown fences, extra keys, or prose outside the JSON
object are accepted.

## Prompt v1

```text
You are a strict binary evaluator of a candidate answer to a text-only school task.

Inputs:
- QUESTION: the task the candidate had to solve
- REFERENCE: the trusted correct answer
- CANDIDATE: the answer being evaluated

Return score 1 only if the candidate is correct and complete for the question.
Return score 0 if it is wrong, partially correct, incomplete, contradictory,
irrelevant, empty, or unsupported.

Rules:
1. Judge substance, not style, verbosity, confidence, spelling, or formatting.
2. Accept equivalent wording, notation, units, languages, and mathematically
   equivalent forms.
3. For a multi-part question, all required parts must be correct.
4. Do not award 1 for a lucky final token when the explanation contains a
   material contradiction.
5. Do not infer information that is absent from the candidate or reference.
6. If the reference is missing, ambiguous, or obviously invalid, use score 0
   and say "reference_issue" in the rationale.

Return exactly this JSON object:
{"score": 0 or 1, "rationale": "brief reason"}
```

## Input JSONL record

```json
{
  "task_id": "stable-id",
  "question_text": "...",
  "reference_answer": "...",
  "candidate_answer": "..."
}
```

`task_id` is for joining results only and must not be included in the Qwen
prompt. Empty candidate answers are valid records and must receive score `0`.

## Day-1 acceptance checks

- text fields only; no image URLs in the request;
- Qwen output parses as strict JSON;
- score accepts only integer `0` or `1`;
- extra keys and markdown-wrapped output are rejected;
- empty, incomplete, and multi-part cases have explicit expected labels;
- prompt version is recorded as `text-binary-v1`.
