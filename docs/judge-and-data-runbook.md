# Judge и подготовка данных

Этот документ сохраняет рабочие CLI-рецепты evaluation-подсистемы отдельно от
корневого README. Полный список аргументов всегда проверяйте через
`python -m vlm_judge.cli --help`.

## Подготовить источники и корпус

```powershell
vlm-judge prepare-sources `
  --main-workbook sheet1.xlsx `
  --math-workbook sheet2.xlsx `
  --corpus odevjet.jsonl `
  --output-dir artifacts

vlm-judge prepare-corpus `
  --input odevjet.jsonl `
  --output-dir artifacts/corpus `
  --max-chars 1600 `
  --overlap-chars 200
```

Исходные workbook, изображения и полный корпус не коммитятся. В Git допустимы
только небольшие проверяемые manifests и явно замороженные evidence-артефакты.

## Присоединить ответы агента

```powershell
vlm-judge import-candidates `
  --benchmark artifacts/math_benchmark.jsonl `
  --responses run.csv `
  --setup no_tools `
  --output artifacts/runs/no_tools.jsonl

vlm-judge score-deterministic `
  --input artifacts/runs/no_tools.jsonl `
  --output artifacts/runs/no_tools_exact.jsonl

vlm-judge prepare-requests `
  --input artifacts/runs/no_tools.jsonl `
  --output artifacts/runs/no_tools_requests.jsonl
```

Пустые и ошибочные ответы должны сохраняться: они входят в общий знаменатель и
не могут исчезать через inner join.

## Проверить полноту эксперимента

```powershell
vlm-judge validate-runs `
  --benchmark artifacts/math_benchmark.jsonl `
  --run no_tools=artifacts/runs/no_tools.jsonl `
  --run web_search=artifacts/runs/web_search.jsonl `
  --run textbook_retrieval=artifacts/runs/textbook_retrieval.jsonl `
  --strict-metadata `
  --output artifacts/reports/run_validation.json
```

Перед импортом judge-результата в аналитику дополнительно используйте
`verify-judge-output`: expected и judged files должны иметь одинаковые task IDs,
без дублей, ошибок и невалидных verdict.

## Text judge для MLA runner

```powershell
vlm-judge prepare-mla-judge-input `
  --tasks data/validation.jsonl `
  --results results/agent_rag.jsonl `
  --output results/agent_rag_judge_input.jsonl `
  --require-all

vlm-judge run-text-judge `
  --input results/agent_rag_judge_input.jsonl `
  --output results/agent_rag_judge.jsonl `
  --base-url https://openrouter.ai/api/v1 `
  --model qwen/qwen3.5-9b `
  --api-key-env OPENROUTER_API_KEY `
  --provider openrouter `
  --retry-failures
```

Для image-only evaluation используйте `prepare-image-judge-input` и `run-judge`.
Готовый последовательный workflow находится в
`scripts/run_image_rag_evaluation.sh`.

## Human calibration и adjudication

```powershell
vlm-judge sample-calibration-responses `
  --input artifacts/runs/no_tools.jsonl `
  --input artifacts/runs/web_search.jsonl `
  --input artifacts/runs/textbook_retrieval.jsonl `
  --size 120 `
  --output artifacts/calibration/response_sample.jsonl

vlm-judge-ui `
  --dataset artifacts/calibration/response_sample.jsonl `
  --annotations artifacts/annotations/human.jsonl `
  --gold artifacts/annotations/gold.jsonl `
  --judge-results artifacts/runs/judge_results.jsonl `
  --adjudications artifacts/annotations/adjudications.jsonl `
  --open-browser
```

Сначала собирается blind correctness label, и только потом открываются setup и
retrieval trace. Подробная методика: [`evaluation_protocol.md`](evaluation_protocol.md).

## Агрегация

```powershell
vlm-judge aggregate `
  --input reports/judge_out_b0.jsonl `
  --input reports/judge_out_b1dr.jsonl `
  --overlay reports/judge_out_b0_delta.jsonl `
  --overlay reports/judge_out_b1dr_delta.jsonl `
  --output reports/judge_agg_b0_vs_b1dr.json
```

Overlay заменяет строку того же `(task_id, setup)`, а не добавляет второй голос.
Для интерпретации всегда сохраняйте dataset version, model, prompt version,
judge version и retrieval config рядом с отчётом.

