# Скрипты

`scripts/` содержит два разных класса файлов. Не смешивайте их при навигации.

## Стабильные entrypoints

| Скрипт | Назначение |
| --- | --- |
| `run_rag_evaluation.sh` | text-only B0 против AgentRag с одинаковым judge |
| `run_image_rag_evaluation.sh` | photo-only B0 против AgentRag |
| `run_openrouter_routed_experiment.ps1` | E0/E3/E4 и импорт в analytics |
| `run_openrouter_mmr_experiment.ps1` | изолированная MMR-абляция |
| `run_openrouter_context_order_experiment.ps1` | frozen-context order-only абляция |
| `run_openrouter_reranker_accuracy_experiment.ps1` | end-to-end сравнение reranker arms |

Перед полным прогоном сначала используйте `--limit`/smoke-конфигурацию и отдельный
`RunId`. Скрипты, использующие OpenRouter, требуют `OPENROUTER_API_KEY` только в
окружении.

## Исследовательские утилиты

Файлы `build_*`, `prepare_*`, `audit_*`, `analyze_*`, `compose_*` и многие
`maxim_*` воспроизводят конкретные исторические исследования. Их пути могут быть
зашиты в frozen manifests, reports и tests, поэтому существующие файлы не нужно
массово переименовывать ради группировки.

Для нового общего workflow предпочитайте модуль в `src/` и тонкий скрипт здесь.
Для одноразового frozen-эксперимента храните config, README и тест рядом в
`experiments/<experiment_id>/`.

