# Навигация для разработчиков и кодовых агентов

Перед изменениями прочитайте [`docs/project-map.md`](docs/project-map.md), затем
откройте только файлы компонента из таблицы «Где менять».

## Постоянные правила

- `src/` — источник рабочего Python-кода; не создавайте вторую реализацию пакета
  в корне репозитория.
- Основной агентский retrieval — прямой вызов
  `retrieve.service.textbook_retrieve_checked`; HTTP используется только для
  отдельного deployment-сценария.
- Новые функции desktop analytics добавляются в `apps/vlm-analytics/`, не в
  legacy-каталог `apps/vlm_analytics/`.
- Не коммитьте `.env`, API-ключи, локальные корпуса/индексы, `results/`,
  `outputs/`, SQLite-базы или скачанные модели.
- Не переименовывайте массово frozen-файлы в `artifacts/`, `reports/`,
  `experiments/` и исторические scripts: их пути входят в manifests и tests.
- Изменение prompt, judge contract или experiment policy требует новой версии и
  сопоставимого smoke/paired test; старые результаты не переписываются.

## Минимальная проверка

```powershell
.\.venv\Scripts\python.exe -m mla_baseline.runner `
  --tasks data/eval/tasks.sample.jsonl `
  --condition b0_no_tools `
  --dry-run

.\.venv\Scripts\python.exe -m pytest -q <затронутые тесты>
```

Полные live-прогоны через OpenRouter выполняйте только по явной необходимости,
начиная с `--limit` и отдельного output/RunId.
