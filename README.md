# MLA Baseline

Бейзлайны для сравнения с агентом решения школьных задач (турецкий рынок,
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) — модель мультимодальна
сама по себе, отдельный VL-вариант не нужен). План проекта — в `BASELINE_PLAN.md`.

Реализовано: **B0** (голая модель, без тулов). Дальше: B1 (веб-поиск), интеграция
RAG-тулов команды ретрива.

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
python -m mla_baseline.runner --tasks data/tasks.sample.jsonl --dry-run
```

## Локальная проверка на слабом железе (Ollama)

На машине с ~8 ГБ VRAM полноценный vLLM не поднять — для смоук-теста подойдёт
квантованная модель через Ollama (эндпоинт OpenAI-совместимый, код тот же):

```bash
ollama pull qwen3.5:9b-q4_K_M
# .env: MLA_VLLM_BASE_URL=http://localhost:11434/v1
#       MLA_MODEL_NAME=qwen3.5:9b-q4_K_M
python -m mla_baseline.runner --tasks data/tasks.sample.jsonl --condition b0_no_tools
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
  --enable-auto-tool-choice --tool-call-parser hermes
```

- A100 40GB: хватает с запасом (весам + KV на ~десятки одновременных запросов);
  при OOM снизить --max-num-seqs до 16.
- A100 80GB: можно --max-num-seqs 64 и/или --max-model-len 32768; остатка VRAM
  хватит и на модель-судью покрупнее на втором процессе.
- В .env на прогонах ставить MLA_CONCURRENCY=16..32 — vLLM батчит их сам.

`--enable-auto-tool-choice --tool-call-parser hermes` для B0 не нужны, но
понадобятся для B1 (агент с тул-коллами) — поднимаем сразу с ними, чтобы
эндпоинт был один на все условия. Картинки шлём base64 в самом запросе,
поэтому `--allowed-local-media-path` не требуется.

Проверить эндпоинт: `curl http://localhost:8000/v1/models`

## Прогон

```bash
# smoke: 2 задачи из примера
python -m mla_baseline.runner --tasks data/tasks.sample.jsonl --condition b0_no_tools --limit 2

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
# 1. Таблица -> data/validation.jsonl + validation.meta.jsonl,
#    картинки -> data/images/ (resume: уже скачанные пропускаются)
python -m mla_baseline.sheet --sheet-id 15VJ_gVErnAy2fJLT-JBUO5WvSsBNthhRQyVHti-RVhc
# либо из локального архива картинок (имена файлов = basename URL из таблицы):
python -m mla_baseline.sheet --csv data/validation_sheet.csv --archive-dir <папка с картинками>

# 2. Прогон (CoT включается через MLA_PROMPT_VERSION=v2_cot в .env)
python -m mla_baseline.runner --tasks data/validation.jsonl --condition b0_no_tools

# 3. Быстрые метрики (exact match до LLM-as-Judge)
python -m mla_baseline.eval --results results/b0_no_tools_v2_cot.jsonl \
    --tasks data/validation.jsonl --meta data/validation.meta.jsonl --by question_type

# 4. HTML-отчёт с графиками (KPI, точность и состав по предметам, длина
#    ответов, таблица промахов). Или сразу флагом: runner ... --report
python -m mla_baseline.report --results results/b0_no_tools_v2_cot.jsonl \
    --tasks data/validation.jsonl --meta data/validation.meta.jsonl
```

Отчёт — самодостаточный HTML (без внешних зависимостей), открывается локально,
поддерживает светлую/тёмную тему. `--meta` можно не указывать, если рядом с
tasks лежит одноимённый `*.meta.jsonl` — подхватится сам.

`eval` считает точный матч с нормализацией (choice — буква шика, numeric —
число, short_text — без регистра/пунктуации); free_form и ответы-URL уходят
в «нужен судья». Срезы: `--by subject|grade|type|class|question_format|question_type`.
Промахи для разбора: `--dump-misses misses.jsonl`.

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
mla_baseline/
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

## Мультиязычный бенч-фреймворк

Готовый конвейер для прогона любых условий на TUMLU (9 тюркских языков),
EXAMS-V (13 языков, скриншоты экзаменов) и MGSM (11 языков, открытая
математика) — всё в Task-контракте, судья не нужен. Как пользоваться:
`docs/mlbench_framework.md`. Канонический срез — `data/mlbench/bench300/`,
базовые цифры — `reports/mlbench300_summary.md`.
