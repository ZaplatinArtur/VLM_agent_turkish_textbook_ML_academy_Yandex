# VLM Judge

Reproducible evaluation for three homework-agent setups:

1. `no_tools`;
2. `web_search`;
3. `textbook_retrieval`.

The package combines deterministic exact metrics, a blinded multimodal LLM judge, human calibration, LMArena-style pairwise validation, and paired statistical reporting. It is usable without a GPU; the model backend is an adapter boundary that can be connected when the Qwen endpoint is known.

## Current source inventory

- Main workbook: 823 usable task records with 20 raw subject labels.
- Planning sheet and mentor target: 17 subject categories. The 20-to-17 correspondence is unconfirmed, so every raw label is preserved.
- Math workbook: 200 question/reference-image pairs, grades 1–12; all 400 image links have been verified.
- ÖdevJet corpus: 45,920 records from 215 books and 8 source subjects.
- Corpus risks: 344 duplicate rows, 263 conflicting duplicate IDs, and 16,520 low-information pages.
- Prepared retrieval layer: 45,576 canonical pages and 280,822 stable chunks (103,070 text and 177,752 image chunks); conflicting variants remain quarantined.
- CPU lexical baseline: SQLite FTS5/BM25 over all 103,070 text chunks, with a local agent-tool API and provenance-rich hits.

The math gold answers are often annotated images. Judge requests therefore support both a question image and a reference-answer image. Setup labels are removed from model prompts and hidden by default in the human interface.

## Desktop analytics application

The source code for the **VLM Analytics** desktop application is maintained in
[`apps/vlm-analytics`](apps/vlm-analytics/README.md). It imports benchmark runs
and judge outputs into a local SQLite database and provides dashboards for
accuracy, subject slices, token usage, latency, audit results, and chunking
experiments.

Only source code, dependency files, build instructions, and tests are tracked.
Generated `.exe` files, local databases, synchronization caches, and imported
benchmark artifacts are intentionally excluded from Git.

## Install and test

```powershell
python -m pip install -e ".[sources,dev]"
python -m unittest discover -s tests -v
```

## Main commands

```powershell
# Normalize source workbooks and audit the corpus.
vlm-judge prepare-sources --main-workbook sheet1.xlsx --math-workbook sheet2.xlsx --corpus odevjet.jsonl --output-dir artifacts

# Canonical pages plus stable text/image chunks; conflicts are quarantined.
vlm-judge prepare-corpus --input odevjet.jsonl --output-dir artifacts/corpus --max-chars 1600 --overlap-chars 200

# Attach an agent run to the benchmark. Empty and failed responses are preserved.
vlm-judge import-candidates --benchmark artifacts/math_benchmark.jsonl --responses run.csv --setup no_tools --output artifacts/runs/no_tools.jsonl

# Exact metrics and blinded judge requests.
vlm-judge score-deterministic --input artifacts/runs/no_tools.jsonl --output artifacts/runs/no_tools_exact.jsonl
vlm-judge prepare-requests --input artifacts/runs/no_tools.jsonl --output artifacts/runs/no_tools_requests.jsonl

# Build and serve the CPU BM25 textbook baseline.
vlm-judge build-bm25 --chunks artifacts/corpus/chunks.jsonl --index artifacts/retrieval/bm25.sqlite
vlm-judge serve-retrieval --index artifacts/retrieval/bm25.sqlite --port 8770
vlm-judge prepare-retrieval-qrels --benchmark artifacts/math_benchmark.jsonl --output artifacts/retrieval/math_qrels_template.jsonl
vlm-judge evaluate-retrieval --index artifacts/retrieval/bm25.sqlite --qrels artifacts/retrieval/math_qrels.jsonl --k 1 --k 5 --k 10 --output artifacts/reports/bm25_eval.json

# Refuse to evaluate an incomplete or contract-drifting three-setup grid.
vlm-judge validate-runs --benchmark artifacts/math_benchmark.jsonl `
  --run no_tools=artifacts/runs/no_tools.jsonl `
  --run web_search=artifacts/runs/web_search.jsonl `
  --run textbook_retrieval=artifacts/runs/textbook_retrieval.jsonl `
  --strict-metadata `
  --output artifacts/reports/run_validation.json

# Exercise the entire pipeline without a model. Outputs are synthetic mechanics tests only.
vlm-judge synthetic-dry-run --benchmark artifacts/math_benchmark.jsonl --output-dir artifacts/dryrun

# Randomized, optionally mirrored A/B records.
vlm-judge prepare-pairs --input artifacts/runs/no_tools.jsonl --input artifacts/runs/textbook_retrieval.jsonl --setup-a no_tools --setup-b textbook_retrieval --mirrored --output artifacts/calibration/arena.jsonl

# Human UI: adjacent gold/candidate, gold transcription, and optional adjudication queue.
vlm-judge sample-calibration-responses --input artifacts/runs/no_tools.jsonl --input artifacts/runs/web_search.jsonl --input artifacts/runs/textbook_retrieval.jsonl --size 120 --output artifacts/calibration/response_sample.jsonl
vlm-judge-ui --dataset artifacts/calibration/response_sample.jsonl `
  --annotations artifacts/annotations/human.jsonl `
  --gold artifacts/annotations/gold.jsonl `
  --judge-results artifacts/runs/judge_results.jsonl `
  --adjudications artifacts/annotations/adjudications.jsonl `
  --open-browser

# Binary 0/1 judge monitor. The UI detects text-binary-v* results and opens this mode automatically.
vlm-judge-ui --dataset artifacts/runs/b0_judge_input.jsonl `
  --judge-results artifacts/runs/b0_judged.jsonl `
  --annotations artifacts/annotations/human.jsonl `
  --gold artifacts/annotations/gold.jsonl `
  --open-browser

# Apply only verified task-scoped transcriptions before building judge requests.
vlm-judge apply-gold --dataset artifacts/runs/no_tools.jsonl --gold artifacts/annotations/gold.jsonl --output artifacts/runs/no_tools_with_gold.jsonl

# Calibration and position-bias reports.
vlm-judge analyze-calibration --human artifacts/annotations/human.jsonl --judge artifacts/runs/judge_results.jsonl --output artifacts/reports/calibration.json
vlm-judge audit-judge-run --input artifacts/runs/judge_results.jsonl --output artifacts/reports/judge_operational_audit.json
vlm-judge analyze-arena --annotations artifacts/annotations/arena.jsonl --output artifacts/reports/arena_bias.json

# Prioritize human/LLM disagreements, judge errors, low-confidence cases, reference issues,
# plus a stable 10% control sample of agreements.
vlm-judge prepare-adjudication --dataset artifacts/runs/no_tools.jsonl `
  --judge artifacts/runs/judge_results.jsonl `
  --human artifacts/annotations/human.jsonl `
  --output artifacts/reports/adjudication_queue.jsonl `
  --summary artifacts/reports/adjudication_summary.json

# Final hybrid/exact/judge views, paired deltas, bootstrap intervals, and coverage audit.
vlm-judge aggregate --input artifacts/runs/scored_records.jsonl --output artifacts/reports/summary.json

# Multiple setups can be aggregated together. Corrected rerun rows override
# matching task_id/setup units without counting them twice.
vlm-judge aggregate `
  --input reports/judge_out_b0.jsonl `
  --input reports/judge_out_b1dr.jsonl `
  --overlay reports/judge_out_b0_delta.jsonl `
  --overlay reports/judge_out_b1dr_delta.jsonl `
  --output reports/judge_agg_b0_vs_b1dr.json
```

## Prepared calibration assets

`artifacts/calibration` contains 120 stratified real tasks, a UTF-8 human-labeling template, and 150 synthetic multiple-choice smoke cases. Synthetic records validate judge mechanics only; they are not evidence of model quality.

## Integration contract

Each setup must emit one record per `(task_id, setup, run_id)`. Candidate text is preserved verbatim, including failures and timeouts. The judge can already call an OpenAI-compatible Qwen/vLLM endpoint through `vlm-judge run-judge`; use `--limit 10` for the first endpoint/image/JSON smoke test, then remove the limit for the cached full run. Only deployment values remain external. The other missing inputs are real three-setup outputs, the mentor gold set, 771 unresolved main-workbook image assets, and the confirmed 17-subject mapping.

See [evaluation protocol](docs/evaluation_protocol.md), [judge acceptance criteria](docs/judge_acceptance_criteria.md), [data contract](docs/data_contract.md), [interface design](docs/interface_design.md), [retrieval tool contract](docs/retrieval_tool_contract.md), and [data/retrieval strategy](docs/data_and_retrieval_strategy.md).

## MLA agent baselines

Бейзлайны для сравнения с агентом решения школьных задач (турецкий рынок,
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) — модель мультимодальна
сама по себе, отдельный VL-вариант не нужен). План проекта — в `BASELINE_PLAN.md`.

Реализованы **B0** (без инструментов), **B1** (веб-поиск), routed/deep варианты
B1 и **AgentRag** с прямым вызовом поиска по учебникам.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env             # и поправить под машину
```

## Проверка без GPU (dry-run)

Собирает сообщения для модели, не вызывая её, — проверка контрактов и картинок:

```bash
python -m mla_baseline.runner --tasks data/eval/tasks.sample.jsonl --dry-run
```

## Локальная проверка на слабом железе (Ollama)

На машине с ~8 ГБ VRAM полноценный vLLM не поднять — для смоук-теста подойдёт
квантованная модель через Ollama (эндпоинт OpenAI-совместимый, код тот же):

```bash
ollama pull qwen3.5:9b-q4_K_M
# .env: MLA_VLLM_BASE_URL=http://localhost:11434/v1
#       MLA_MODEL_NAME=qwen3.5:9b-q4_K_M
python -m mla_baseline.runner --tasks data/eval/tasks.sample.jsonl --condition b0_no_tools
```

Квант Q4 — только для проверки пайплайна; метрики для сравнения условий
снимаем на полном чекпоинте через vLLM.

## Запуск модели (GPU-машина)

Профиль под A100 (полный bf16-чекпоинт, ~18 ГБ весов):

```bash
pip install vllm
vllm serve Qwen/Qwen3.5-9B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 32 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

- A100 40GB: хватает с запасом (весам + KV на ~десятки одновременных запросов);
  при OOM снизить --max-num-seqs до 16.
- A100 80GB: можно --max-num-seqs 64 и/или --max-model-len 32768; остатка VRAM
  хватит и на модель-судью покрупнее на втором процессе.
- В .env на прогонах ставить MLA_CONCURRENCY=16..32 — vLLM батчит их сам.

`--enable-auto-tool-choice --tool-call-parser qwen3_coder` для B0 не нужны, но
понадобятся для B1 (агент с тул-коллами) — поднимаем сразу с ними, чтобы
эндпоинт был один на все условия. Картинки шлём base64 в самом запросе,
поэтому `--allowed-local-media-path` не требуется.

Проверить эндпоинт: `curl http://localhost:8000/v1/models`

## Прогон

```bash
# smoke: 2 задачи из примера
python -m mla_baseline.runner --tasks data/eval/tasks.sample.jsonl --condition b0_no_tools --limit 2

# полный прогон
python -m mla_baseline.runner --tasks data/tasks.jsonl --condition b0_no_tools
```

Результат: `results/b0_no_tools_v1.jsonl` (по строке `SolveResult` на задачу —
формат для LLM-as-Judge). Перезапуск продолжает с места остановки (resume по
`task_id`); чтобы прогнать заново — удалить/переименовать выходной файл.

## Пайплайн валидации (Google Sheets)

Выборка валидации — [таблица](https://docs.google.com/spreadsheets/d/15VJ_gVErnAy2fJLT-JBUO5WvSsBNthhRQyVHti-RVhc/edit?gid=0#gid=0):
823 задачи-скриншота (Visual — URL картинки на s3.mds.yandex.net) с эталонными
ответами. Пайплайн: таблица → Task JSONL → прогон → метрики.

```bash
# 1. Таблица -> data/eval/validation.jsonl + validation.meta.jsonl,
#    картинки -> data/images/ (resume: уже скачанные пропускаются)
python -m mla_baseline.sheet --sheet-id 15VJ_gVErnAy2fJLT-JBUO5WvSsBNthhRQyVHti-RVhc
# либо из локального архива картинок (имена файлов = basename URL из таблицы):
python -m mla_baseline.sheet --csv data/eval/validation_sheet.csv --archive-dir <папка с картинками>

# 2. Прогон (CoT включается через MLA_PROMPT_VERSION=v2_cot в .env)
python -m mla_baseline.runner --tasks data/eval/validation.jsonl --condition b0_no_tools

# 3. Быстрые метрики (exact match до LLM-as-Judge)
python -m mla_baseline.eval --results results/b0_no_tools_v2_cot.jsonl \
    --tasks data/eval/validation.jsonl --meta data/eval/validation.meta.jsonl --by question_type

# 4. HTML-отчёт с графиками (KPI, точность и состав по предметам, длина
#    ответов, таблица промахов). Или сразу флагом: runner ... --report
python -m mla_baseline.report --results results/b0_no_tools_v2_cot.jsonl \
    --tasks data/eval/validation.jsonl --meta data/eval/validation.meta.jsonl
```

Отчёт — самодостаточный HTML (без внешних зависимостей), открывается локально,
поддерживает светлую/тёмную тему. `--meta` можно не указывать, если рядом с
tasks лежит одноимённый `*.meta.jsonl` — подхватится сам.

`eval` считает точный матч с нормализацией (choice — буква шика, numeric —
число, short_text — без регистра/пунктуации); free_form и ответы-URL уходят
в «нужен судья». Срезы: `--by subject|grade|type|class|question_format|question_type`.
Промахи для разбора: `--dump-misses misses.jsonl`.

## Раскладка data/

`data/` разделён по происхождению: `corpus/` — источник (books, chunks, tessdata),
`eval/` — то, чем меряем (validation, qrels, наборы запросов), `cache/` —
производное и пересобираемое. Вне этого деления остался `data/images/` с
картинками задач: на него завязаны `mla_baseline.sheet` и
`vlm_judge.validation_archive`.

Сам `data/` в `.gitignore`, поэтому вместе с кодом раскладка не приезжает. Если
у вас каталог из прежней плоской версии (`data/books`, `data/chunks/jsonl`,
`data/validation.jsonl`), выполните переезд один раз:

```bash
python scripts/migrate_data_layout.py           # показать план
python scripts/migrate_data_layout.py --apply
```

Скрипт заодно правит ссылки на сканы страниц внутри чанков (`books/...` →
`corpus/books/...`). Повторный запуск безопасен.

## Парная оценка B0 против textbook RAG

Корпус должен лежать в `data/corpus/chunks/jsonl/*.jsonl`. Он не коммитится в Git.
Сначала проверьте его состав и один раз постройте постоянный FAISS-индекс:

```bash
python -m retrieve.build_index --dry-run
python -m retrieve.build_index \
  --sample-query "dikdörtgen alan formülü" --k 3
```

Индекс сохраняется в `data/cache/index/` и при неизменном корпусе загружается
повторно. Текущий парсер восстанавливает `grade` и `subject` из slug учебника,
поэтому агентские фильтры на английском и турецком работают одинаково.

Полный воспроизводимый прогон при уже запущенном vLLM:

```bash
bash scripts/run_rag_evaluation.sh
```

Перед дорогим прогоном скрипт проверяет, что локальные изображения существуют,
эталоны заполнены, а вместо текста вопроса нет заглушки `(soru görselde)`.
Последнее обязательно для text-only judge. Если используется старый
`data/eval/validation.jsonl` только с картинками, сначала нужен подготовленный
набор с транскрипциями условий (с теми же `task_id`) либо обновлённый
validation-архив от команды judge.

По умолчанию `run_rag_evaluation.sh` работает в режиме
`MLA_TEXT_ONLY=true`: ссылки `question_images` игнорируются, модель получает
только поле `question`. Поэтому картинки можно не переносить на сервер, но
каждое поле `question` должно содержать настоящее условие, а не заглушку.

Из смешанного файла можно получить чистый text-only JSONL: ссылки на картинки
будут удалены, а задания с заглушками или пропущенными рисунками исключены.

```bash
python -m mla_baseline.prepare_text_only \
  --input data/tasks_with_transcriptions.jsonl \
  --output data/tasks_text_only.jsonl
```

Скрипт на одном и том же `data/eval/validation.jsonl`:

1. запускает `b0_no_tools` и `agent_rag` с одинаковой моделью и промптом;
2. готовит оба результата для строгого binary LLM judge;
3. оценивает оба условия на общем знаменателе;
4. пишет `reports/rag_eval/summary.{json,md}`.

В отчёте отдельно показаны exact match, LLM-as-a-judge, частота вызова tool,
ошибки retrieval, число исправленных/ухудшенных ответов, McNemar p-value,
парный bootstrap CI и срез по покрытию корпуса. Пропущенный ответ или ошибка
остаются в общем знаменателе и не засчитываются как правильные.

Пути и endpoint можно переопределить без изменения скрипта:

```bash
TASKS=data/eval/validation.jsonl \
BASE_URL=http://127.0.0.1:8000/v1 \
MODEL=Qwen/Qwen3.5-9B \
MLA_CONCURRENCY=4 \
bash scripts/run_rag_evaluation.sh
```

### Photo-only validation

Для объединённого validation-архива есть отдельный прогон по всем 198
уникальным изображениям вопросов:

```bash
DATA_ROOT=outputs/validation_merged_20260723 \
bash scripts/run_image_rag_evaluation.sh
```

`validation_image_tasks.jsonl` строится прямо из manifest без OCR. Решающий
агент получает только оригинальный screenshot вопроса. После B0/RAG-прогонов
мультимодальный judge получает тот же screenshot, ответ агента и доверенный
эталон: текст для 118 задач или отдельное изображение ответа для 80 задач.
Эталон никогда не передаётся решающему агенту. Итоговый отчёт находится в
`reports/validation_images_full/summary.{json,md}`.

## Перенос на мощное железо по SSH

Код не содержит локальных путей и ключей — вся конфигурация в `.env`.

```bash
# 1. Залить код (или git clone на той стороне)
rsync -av --exclude .venv --exclude results --exclude .env . user@gpu-host:~/mla_baseline/

# 2. На GPU-машине
ssh user@gpu-host
cd ~/mla_baseline
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # выставить MLA_MODEL_NAME и, если vLLM на другом хосте, MLA_VLLM_BASE_URL

# 3. vLLM в tmux/screen (см. выше), затем прогон
tmux new -s vllm       # внутри: vllm serve ...
python -m mla_baseline.runner --tasks data/tasks.jsonl --condition b0_no_tools

# 4. Забрать результаты
rsync -av user@gpu-host:~/mla_baseline/results/ ./results/
```

Альтернатива для локальной отладки против удалённого vLLM — ssh-туннель:
`ssh -N -L 8000:localhost:8000 user@gpu-host`, тогда локальный `.env` не меняется.

## Настройки (.env)

| Переменная | Что делает |
|---|---|
| `MLA_VLLM_BASE_URL` | OpenAI-совместимый эндпоинт vLLM |
| `MLA_MODEL_NAME` | id модели: HF-id для vLLM (`Qwen/Qwen3.5-9B`) или тег Ollama (`qwen3.5:9b-q4_K_M`) |
| `MLA_STRUCTURED_MODE` | `response_format` (новые vLLM) / `guided_json` (старые) / `none` |
| `MLA_PROMPT_VERSION` | версия промпта; правка промпта = новая версия |
| `MLA_INCLUDE_QUESTION_TEXT_WITH_IMAGES` | `true` — слать текст условия вместе с картинкой (по умолчанию нет: сценарий «ленивый школьник») |
| `MLA_CONCURRENCY` | параллельные запросы к vLLM |

## Структура

```
src/mla_baseline/
  contracts.py    # общекомандные контракты (Task, ImageRef, RetrievedChunk)
  schemas.py      # SolveOutput (JSON от модели), SolveResult (строка для судьи)
  config.py       # настройки из .env (префикс MLA_)
  prompts.py      # версионируемые промпты (турецкий): v1, v2_cot (chain-of-thought)
  images.py       # ImageRef -> OpenAI image block
  parsing.py      # робастный разбор JSON из ответа 9B-модели
  sheet.py        # выборка валидации: Google Sheets CSV -> Task JSONL + картинки
  eval.py         # быстрые метрики (exact match) со срезами по subject/типу
  solvers/
    base.py       # интерфейс Solver (build_messages + solve)
    b0_no_tools.py
  runner.py       # батч-прогон с resume и dry-run
```
