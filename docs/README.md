# Документация проекта

Начните с [`project-map.md`](project-map.md), если нужно быстро понять код, или с
[`getting-started.md`](getting-started.md), если нужно запустить проект.

## Основной путь

- [`project-map.md`](project-map.md) — компоненты, точки входа, критический поток
  и таблица «что менять → что проверять»;
- [`getting-started.md`](getting-started.md) — установка, smoke-прогоны, RAG,
  judge и аналитика;
- [`judge-and-data-runbook.md`](judge-and-data-runbook.md) — подготовка данных,
  judge CLI, калибровка и агрегация;
- [`current_status.md`](current_status.md) — исторический handoff состояния
  judge/data pipeline; перед использованием сверяйте с текущим кодом.

## Контракты

- [`data_contract.md`](data_contract.md) — форматы задач, ответов и прогонов;
- [`retrieval_tool_contract.md`](retrieval_tool_contract.md) — интерфейс и
  политика textbook retrieval tool;
- [`evaluation_protocol.md`](evaluation_protocol.md) — метрики и правила честного
  сравнения;
- [`judge_acceptance_criteria.md`](judge_acceptance_criteria.md) — критерии
  приёмки LLM-as-a-Judge;
- [`text_binary_judge_contract.md`](text_binary_judge_contract.md) — контракт
  бинарного text judge.

## Retrieval и данные

- [`data_and_retrieval_strategy.md`](data_and_retrieval_strategy.md) — стратегия
  подготовки корпуса;
- [`dense_mmr_retrieval_evaluation.md`](dense_mmr_retrieval_evaluation.md) —
  Recall/MAP/MRR и Dense→MMR evaluation;
- [`bge_m3_semantic_candidates.md`](bge_m3_semantic_candidates.md) —
  экспериментальный BGE-M3 candidate arm;
- [`knowledge_graph.md`](knowledge_graph.md) — графовые расширения retrieval;
- [`source_audit.md`](source_audit.md) — аудит источников.

## Интерфейсы и запуск

- [`interface_design.md`](interface_design.md) — интерфейс ручной оценки;
- [`compute_handoff.md`](compute_handoff.md) — исторические инструкции для
  GPU/vLLM; основной текущий backend — OpenRouter;
- [`../apps/vlm-analytics/README.md`](../apps/vlm-analytics/README.md) — desktop
  analytics и импорт прогонов.

## Исторические исследования

Документы `MAXIM_*`, каталог [`maxim_ru/`](maxim_ru/) и отдельные протоколы
сохраняют доказательства конкретных экспериментов. Они полезны для аудита, но не
являются общей инструкцией запуска. Новую общую документацию добавляйте в один из
разделов выше, а описание конкретного frozen run — рядом с его manifest в
`experiments/` или `reports/`.
