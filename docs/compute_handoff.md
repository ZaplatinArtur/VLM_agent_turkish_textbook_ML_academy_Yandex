# Compute/model handoff checklist

The CPU-only evaluation pipeline is ready. Before connecting Qwen, obtain:

- exact checkpoint name and revision;
- confirmation that the model accepts images (a vision-language checkpoint is required for image-only gold references);
- serving mode and endpoint contract;
- chat template and supported structured-output mode;
- maximum context, maximum images per request, and image-size limits;
- GPU type/count/memory and concurrency limits;
- generation defaults, timeout, retry policy, and allowed batch size;
- model-license and data-handling constraints.

Runtime values should be passed through environment/configuration rather than committed credentials. At minimum the adapter will need a model identifier, endpoint or local model path, timeout, concurrency, and optional authentication variable name.

## First compute smoke test

1. Run 10 records with one worker to validate transport, images, JSON mode, and served-model identity.
2. Run 150 synthetic multiple-choice cases and measure JSON-validity plus agreement with expected coarse labels.
3. Run the 120-response calibration selection after real candidate answers are attached and human labels are available.
4. Inspect failures, tune only on the calibration split, and freeze prompt/model parameters.
5. Run the held-out three-setup experiment once, preserving every failure and configuration field.

## OpenAI-compatible Qwen command

The client is implemented and tested against a mock chat-completions server. For a vLLM deployment exposing `/v1/chat/completions`:

```powershell
$env:VLM_JUDGE_API_KEY = "<only if the endpoint requires one>"

vlm-judge run-judge `
  --input artifacts\runs\calibration_items.jsonl `
  --output artifacts\runs\qwen_judge_calibration.jsonl `
  --base-url http://<host>:8000/v1 `
  --model <exact-served-model-name> `
  --api-key-env VLM_JUDGE_API_KEY `
  --temperature 0 `
  --seed 20260714 `
  --max-tokens 900 `
  --max-attempts 2 `
  --retry-delay 1 `
  --workers 1 `
  --limit 10 `
  --cache-dir artifacts\cache\judge
```

Omit `--api-key-env` for an unauthenticated internal endpoint. Start with `--image-mode url`; use `--image-mode data_url` when compute nodes cannot access Yandex public links. The latter downloads through the allowlisted local cache before sending the request.

Before a full run, execute the command above with one worker and confirm:

- the served model is actually the intended VLM checkpoint, not text-only Qwen;
- both question and reference images are accepted;
- `response_format={"type":"json_object"}` is supported (otherwise add `--no-response-format`);
- verdict JSON validity is 100% after at most one retry;
- request/output token counts and finish reasons appear in `judge.response_metadata`;
- cached reruns make zero model calls, while changing model/endpoint/decoding creates a separate cache namespace;
- no setup name is visible in the generated judge prompt.

After that, raise `--workers` gradually (for example 2, 4, then 8) while watching endpoint queueing, p95 latency, and error rate. Output order remains deterministic and each valid verdict is written to a content-addressed cache atomically.
Remove `--limit 10` for the full calibration run; already completed records will be replayed from cache.

Generate the operational audit immediately after each smoke/full run:

```powershell
vlm-judge audit-judge-run `
  --input artifacts\runs\qwen_judge_calibration.jsonl `
  --output artifacts\reports\qwen_judge_audit.json
```

This catches schema failures, retry dependence, duplicate request IDs, unexpected served-model names, finish reasons, cache behavior, and token totals before agreement metrics are interpreted.

Then run `vlm-judge analyze-calibration`, inspect disagreement/adjudication cases, freeze model name/revision, prompt version, decoding parameters, image mode, and serving version, and only then evaluate the held-out set.
