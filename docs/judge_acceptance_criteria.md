# Proposed LLM-judge acceptance criteria

These thresholds are a pre-registration proposal for mentor approval, not claims about the unavailable Qwen model. Freeze them before looking at held-out setup differences.

## Gate 1: execution reliability

- At least 99% schema-valid verdicts after at most one retry.
- Fewer than 1% endpoint/time-out failures after retries.
- 100% of outputs retain request ID, prompt version, exact served model, decoding settings, token usage, finish reason, and cache state.
- Re-running the same prompt/backend configuration from cache produces byte-equivalent verdicts and zero endpoint calls; changing model, endpoint, decoding, JSON mode, or image mode must miss the old cache.
- No setup, run ID, agent model, retrieval trace, tool count, or synthetic expected label reaches the judge prompt.

The parser intentionally rejects extra keys, stringified booleans/numbers, and inconsistent label/score pairs. Silent type coercion would make validity statistics meaningless.

## Gate 2: human agreement on calibration data

Use at least 120 stratified real visual cases after agent responses exist. Deliberately include all setups and difficult patterns rather than drawing only an easy random sample. Double-label at least 20% to estimate the human ceiling.

Proposed minimums:

- coverage over completed human labels: at least 95%;
- exact 0–4 score agreement: at least 70%;
- agreement within one score: at least 90%;
- quadratic weighted kappa: at least 0.75;
- strict-correct F1: at least 0.85;
- strict-correct recall: at least 0.85, because false negatives can erase real setup gains;
- no major subject or answer-type slice below 0.70 strict binary agreement when that slice has at least 15 examples.

Report Wilson 95% intervals; a point estimate alone is not a pass. Compare judge-human agreement with human-human agreement. If humans disagree almost as often as the judge, improve the rubric/gold before changing the model.

## Gate 3: invariance and leakage tests

Run paired metamorphic cases where the substantive answer is unchanged:

- concise versus verbose wording;
- equivalent fraction/decimal/symbolic notation;
- harmless whitespace, language, and unit formatting;
- metadata changed from `no_tools` to retrieval-like values (the blinded prompt hash should remain identical);
- an irrelevant citation or claim that a tool was used;
- identical answer with run order shuffled.

The strict verdict should remain unchanged. Also include adversarial non-invariances: correct final answer with false reasoning, omitted subanswer, wrong unit, and a plausible answer copied from a neighboring textbook problem. The verdict should change in the expected direction.

## Gate 4: confidence policy

Judge confidence is not automatically calibrated probability. Use it only after plotting selective human agreement by minimum-confidence threshold. Freeze a low-confidence adjudication threshold on calibration data; the current default of 0.75 is a starting value.

Every human/judge disagreement, reference-quality flag, judge error, and low-confidence verdict enters adjudication. A stable 10% sample of agreements is also reviewed so adjudicators are not conditioned to assume the judge is wrong.

## Same-family judge risk

If the homework agent and judge are both Qwen-family models, errors and stylistic preferences may be correlated. Blinding removes explicit setup identity but cannot remove model-family bias. Preferred mitigation is a judge from a different model family. If that is unavailable:

1. keep deterministic exact metrics primary where applicable;
2. increase human calibration and adjudication coverage for open-ended items;
3. report results with and without judge-only records;
4. run a small independent second-judge or human audit on cases driving the setup delta;
5. never use synthetic self-generated labels as judge-quality evidence.

## Freeze record

Before the held-out run, record the exact checkpoint/revision, server version, chat template, prompt hash/version, image preprocessing, response-format mode, temperature, seed, token limit, retry policy, worker count, cache policy, gold snapshot hash, calibration report hash, and acceptance decision. Any later change creates a new judge version and requires recalibration.
