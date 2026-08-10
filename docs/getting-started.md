# Установка и запуск

## 1. Окружение

Проект использует Python 3.11+ и `src`-layout. `pyproject.toml` — единственный
источник зависимостей; `requirements.txt` только устанавливает проект через
`-e .`.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[sources,dev]"
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[sources,dev]"
```

Скопируйте `.env.example` в локальный `.env` только если удобнее хранить
несекретные настройки в файле. Никогда не коммитьте ключ OpenRouter.

## 2. Бесплатная проверка без модели

Dry-run читает `Task`, собирает мультимодальные сообщения и проверяет выбранный
solver, но не отправляет запрос провайдеру:

```powershell
.\.venv\Scripts\python.exe -m mla_baseline.runner `
  --tasks data/tasks.sample.jsonl `
  --condition b0_no_tools `
  --dry-run
```

Базовая проверка кода:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Live-тесты и дорогие OpenRouter-вызовы по умолчанию не должны входить в обычный
локальный test run.

## 3. OpenRouter smoke

```powershell
$env:OPENROUTER_API_KEY = "<your-key>"
$env:MLA_CONCURRENCY = "1"

.\.venv\Scripts\python.exe -m mla_baseline.runner `
  --tasks data/tasks.sample.jsonl `
  --condition b0_no_tools `
  --limit 2 `
  --out results/smoke_b0.jsonl
```

Runner дописывает JSONL и при повторном запуске пропускает уже завершённые
`task_id`. Флаг `--retry-errors` оставляет успешные строки и повторяет только
ошибочные.

## 4. Textbook RAG

Для прямого retrieval нужны чанки в `data/chunks/jsonl/*.jsonl`. Корпус и
построенный индекс не входят в Git.

```powershell
# Проверить найденный корпус без построения индекса.
.\.venv\Scripts\python.exe -m retrieve.build_index --dry-run

# Построить/загрузить индекс и выполнить smoke-query.
.\.venv\Scripts\python.exe -m retrieve.build_index `
  --sample-query "dikdörtgen alan formülü" `
  --k 3

# Прогнать тот же формат задач с RAG tool.
$env:OPENROUTER_API_KEY = "<your-key>"
.\.venv\Scripts\python.exe -m mla_baseline.runner `
  --tasks data/tasks.sample.jsonl `
  --condition agent_rag `
  --limit 2 `
  --out results/smoke_rag.jsonl
```

`agent_rag` по умолчанию вызывает
`retrieve.service.textbook_retrieve_checked()` напрямую. HTTP-сервис для этого
не требуется. Основные настройки retrieval находятся в `.env.example` и имеют
префикс `MLA_RETRIEVAL_`.

## 5. Сравнение B0 и RAG

Linux/WSL, text-only dataset:

```bash
OPENROUTER_API_KEY="<your-key>" \
TASKS=data/validation.jsonl \
bash scripts/run_rag_evaluation.sh
```

Photo-only dataset:

```bash
OPENROUTER_API_KEY="<your-key>" \
DATA_ROOT=outputs/validation_merged_20260723 \
bash scripts/run_image_rag_evaluation.sh
```

Оба скрипта выполняют preflight, B0, RAG, подготовку judge input, одинаковый
judge и paired report. Сырые ответы пишутся в `results/`, итоговые сводки — в
`reports/`.

Для Windows есть воспроизводимый E0/E3/E4 smoke с предметным router:

```powershell
$env:OPENROUTER_API_KEY = "<your-key>"
.\scripts\run_openrouter_routed_experiment.ps1 `
  -RunId "smoke_router_v1" `
  -Limit 10 `
  -Workers 1
```

## 6. Judge и аналитика

CLI judge доступен через `vlm-judge` или `python -m vlm_judge.cli`. Полный список
команд:

```powershell
.\.venv\Scripts\vlm-judge.exe --help
```

Каноническое desktop-приложение:

```powershell
.\.venv\Scripts\python.exe apps/vlm-analytics/main.py
```

Подробности импорта и paired analytics:
[`../apps/vlm-analytics/README.md`](../apps/vlm-analytics/README.md).

## 7. Где появляются файлы

| Каталог | Назначение | Политика Git |
| --- | --- | --- |
| `data/` | локальные датасеты, учебники, чанки, индексы | почти полностью ignored |
| `outputs/` | распакованные архивы и промежуточные датасеты | ignored |
| `results/` | сырые ответы агента и judge | ignored |
| `reports/` | компактные сравнения и проверяемые сводки | коммитятся выборочно |
| `artifacts/` | замороженные inputs/cache/evidence конкретных работ | только при явной необходимости |
| `experiments/` | самодостаточные frozen-пакеты | версия и provenance обязательны |

Не переносите локальные абсолютные пути, `.env`, SQLite-базы аналитики, ключи или
скачанные модели в Git.

