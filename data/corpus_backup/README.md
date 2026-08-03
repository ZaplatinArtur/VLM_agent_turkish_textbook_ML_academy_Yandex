# Бэкап ретрив-стека с a100-2 (2026-07-31)

Забрано с `mla-yac-a100-2:~/mla_baseline/` перед потерей доступа к машине
(её забронировала другая команда).

| Файл | Что это | В git |
|---|---|---|
| `odevjet_corpus.jsonl.gz` | корпус ÖdevJet: 42 553 страницы, 198 книг, классы 1–8; 157 МБ в распакованном виде | нет (38 МБ, в .gitignore) |
| `report.json` | отчёт `vlm-judge prepare-corpus` по этому корпусу | да |
| `validation.bridge.jsonl` | валидация с подставленными транскрипциями эталонов-картинок (вход судьи) | да |
| `run_arag.sh`, `run_aragr.sh` | пайплайн прогона agent_rag: run → prepare-mla-judge-input (base+bridge) → delta → run-text-judge ×2 | да |

Корпус — единственный невосстановимый артефакт: пересбор это сутки скрейпинга
(`scripts/scrape_odevjet_text.py`). Целостность проверена после скачивания:
распаковывается без ошибок, 42 553 записи, совпадает с серверным размером.

Не забиралось, потому что выводится из корпуса за минуты:
`data/corpus/chunks.jsonl` (200 МБ), `data/corpus/pages.jsonl` (170 МБ),
`data/bm25.sqlite` (396 МБ). Восстановление:

```
vlm-judge prepare-corpus --input odevjet_corpus.jsonl --out-dir data/corpus
vlm-judge build-bm25 --chunks data/corpus/chunks.jsonl --db data/bm25.sqlite
python -m vlm_judge.retrieval_server --db data/bm25.sqlite --port 8770
```

## Замечание к качеству корпуса

`report.json` показывает `low_information_pages: 14435` и
`boilerplate_pages_text_suppressed: 14435` — конвейер сам признал
малоинформативными **34% страниц** (14 435 из 42 553). При этом в прогонах
agent_rag 16% выдач всё равно возвращали верхним чанком навигационный шаблон
ÖdevJet: подавление отработало по страницам, целиком состоящим из шаблона, но
не по смешанным, где шаблон соседствует с парой строк текста. Фильтровать надо
на этапе скрейпинга, построчно (см. reports/tool_errors_analysis.md).
