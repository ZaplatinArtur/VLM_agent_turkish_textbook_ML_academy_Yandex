<!-- repo-onboarding
source_commit: 586c95b517af4209b4399425a8665d81c4e8b555
generated_at: 2026-08-10T16:23:55Z
working_tree: dirty
scope: full
-->

# Карта проекта

## Ориентация за 60 секунд

- **Назначение:** сравнить VLM без tools, с веб-поиском и с textbook RAG на
  одинаковых школьных задачах.
- **Основной вход агента:** `src/mla_baseline/runner.py:main`.
- **Прямой RAG-вход:** `src/retrieve/service.py:textbook_retrieve_checked`.
- **Главный поток:** `Task JSONL → Solver → OpenRouter/tool → SolveResult → judge → report`.
- **Быстрая проверка:** `python -m mla_baseline.runner --tasks data/tasks.sample.jsonl --condition b0_no_tools --dry-run`.

## Компоненты

| Компонент | Ответственность | Точка входа | Контракт / зависимость | Статус |
| --- | --- | --- | --- | --- |
| Agent runner | batch, resume, concurrency, JSONL output | `src/mla_baseline/runner.py:main` | `Task → SolveResult` | Confirmed |
| Solvers | B0, web, deep/routed и RAG условия | `src/mla_baseline/solvers/__init__.py:SOLVERS` | общий `Solver` | Confirmed |
| Image-first RAG | bounded tool loop, conflict filter, answer recheck | `src/mla_baseline/solvers/agent_rag.py:AgentRag` | LangChain tool calls | Confirmed |
| Tool adapter | фиксирует `top_k`, сериализует hits и trace | `src/mla_baseline/tools/textbook_search.py:LocalTextbookSearchClient` | direct/optional HTTP backend | Confirmed |
| Retrieval | Dense/BM25 fusion, reranking, relevance gate | `src/retrieve/service.py:textbook_retrieve_checked` | `list[RetrievedChunk] + RelevanceVerdict` | Confirmed |
| Retrieval evaluation | Recall/Hit/MAP/MRR, Dense→MMR | `src/retrieve/evaluation.py` | независимые qrels | Confirmed |
| Judge | deterministic + multimodal/text LLM judge, audit | `src/vlm_judge/cli.py:main` | agent JSONL + references | Confirmed |
| Analytics | import runs, paired metrics, trace UI | `apps/vlm-analytics/main.py:main` | local SQLite | Confirmed |

## Критический поток AgentRag

1. `runner.load_tasks()` валидирует каждую строку через `Task`.
2. `SOLVERS[condition]` создаёт solver с настройками из `.env`/окружения.
3. `AgentRag` для изображения извлекает структурированные факты и формирует
   тематический запрос без чисел и вариантов ответа.
4. `LocalTextbookSearchClient` лениво вызывает `textbook_retrieve_checked()` в
   том же процессе.
5. Retrieval pipeline возвращает ранжированные чанки и relevance verdict;
   слабая выдача скрывается, один отличный rewrite разрешён, общий лимит — два
   tool calls.
6. Для изображения конфликтующие чанки удаляются, затем ответ повторно
   проверяется по исходному screenshot.
7. Runner сохраняет `SolveResult` и trace в JSONL. Judge получает ответ и эталон,
   после чего paired evaluation считает fixed/regressed и accuracy.

## Где менять

| Задача | Сначала читать | Обычно менять | Проверить |
| --- | --- | --- | --- |
| Добавить solver | `solvers/base.py`, `solvers/__init__.py` | новый файл в `solvers/` и registry | `tests/test_mla_runner.py`, новый unit test |
| Изменить prompt | `mla_baseline/prompts.py` | новая prompt version, не переписывать старую | `tests/test_schema_and_prompts.py` |
| Изменить tool policy | `docs/retrieval_tool_contract.md`, `agent_rag.py` | solver + trace schema | `tests/test_agent_rag.py` |
| Изменить retrieval | `retrieve/service.py`, `retrieve/pipeline.py` | ranker/config/persistence | retrieval unit tests + fixed qrels eval |
| Изменить схемы | `mla_baseline/contracts.py`, `src/schemas/`, `vlm_judge/schema.py` | минимальный владелец контракта | schema + adapter tests |
| Изменить judge | `vlm_judge/prompts.py`, `runner.py`, `text_judge.py` | versioned prompt/schema | judge audit + calibration tests |
| Добавить эксперимент | `scripts/README.md`, `configs/experiment.example.json` | config + runner + isolated tests | smoke, frozen config, paired audit |
| Изменить analytics | `apps/vlm-analytics/README.md` | только `apps/vlm-analytics/` | app-local pytest |

## Данные и внешние системы

| Объект | Владелец | Инвариант / failure mode |
| --- | --- | --- |
| `Task` | `mla_baseline/contracts.py` | `task_id` уникален в одном dataset |
| `SolveResult` | `mla_baseline/schemas.py` | ошибки и пустые ответы не удаляются из знаменателя |
| Учебные чанки | `data/chunks/jsonl/` + `retrieve` | корпус локальный; provenance и `chunk_id` сохраняются |
| FAISS index | `data/cache/index/` | должен соответствовать corpus/embedder manifest |
| OpenRouter | `mla_baseline/config.py` | ключ только из окружения; live calls платные |
| Judge output | `vlm_judge/judge_audit.py` | до метрик нужны полнота, уникальные IDs и валидный verdict |
| Analytics DB | `apps/vlm-analytics/` | локальная SQLite, не источник первичных результатов |

## Соглашения и опасные места

- **Confirmed:** `pyproject.toml` — источник зависимостей и CLI entrypoints.
- **Confirmed:** основной retrieval агента — прямой вызов; HTTP опционален.
- **Confirmed:** `apps/vlm-analytics/` — каноническое приложение;
  `apps/vlm_analytics/` — legacy snapshot.
- **Confirmed:** `results/` и `outputs/` локальные; публикуемая компактная сводка
  живёт в `reports/`.
- **Confirmed:** frozen-пути в `artifacts/`, `reports/` и `experiments/` входят в
  manifests/tests; массовое переименование ломает воспроизводимость.
- **Likely:** часть ранних документов и baseline plan описывает устаревшие
  deployment-предположения; текущие настройки проверяйте по коду и `.env.example`.
- **Unknown:** единая CI-команда для всех исторических experiment packages не
  определена; некоторые пакеты должны тестироваться отдельными процессами.

## Команды

| Цель | Команда | Статус |
| --- | --- | --- |
| Установка | `python -m pip install -e ".[sources,dev]"` | подтверждена manifest, в этом проходе не переустанавливалась |
| Agent dry-run | `python -m mla_baseline.runner --tasks data/tasks.sample.jsonl --condition b0_no_tools --dry-run` | verified 2026-08-10 |
| Retrieval inventory | `python -m retrieve.build_index --dry-run` | verified 2026-08-10: 42,981 local chunks |
| Agent/RAG unit tests | `python -m pytest -q tests/test_dryrun.py tests/test_mla_runner.py tests/test_agent_rag.py tests/test_textbook_search_tool.py` | verified 2026-08-10: 31 passed |
| Полный test suite | `python -m pytest -q` | not run; содержит тяжёлые исторические пакеты |
| Analytics | `python apps/vlm-analytics/main.py` | требует app dependencies и GUI |

## Связанные документы

- [`../README.md`](../README.md) — человеческое введение и quick start;
- [`getting-started.md`](getting-started.md) — runbook;
- [`judge-and-data-runbook.md`](judge-and-data-runbook.md) — judge/data recipes;
- [`retrieval_tool_contract.md`](retrieval_tool_contract.md) — RAG tool;
- [`evaluation_protocol.md`](evaluation_protocol.md) — методика оценки;
- [`../scripts/README.md`](../scripts/README.md) — entrypoint-скрипты;
- [`../apps/README.md`](../apps/README.md) — приложения.

## Свежесть

Карта проверена для commit `586c95b517af4209b4399425a8665d81c4e8b555` на
dirty working tree. При изменении entrypoints, контрактов или структуры обновите
затронутые строки и metadata, не пересканируя без необходимости весь репозиторий.
