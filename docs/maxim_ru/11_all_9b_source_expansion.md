# Audited all-9B source expansion v1.1

## Основной результат

Текущий официальный all-9B headline — **249/274 = 0,908759**. Непосредственный baseline `base240` получил 240/274, поэтому successor delta равна **+9 correct, 0 regressions**.

| Срез | Правильно | Всего | Accuracy |
|---|---:|---:|---:|
| Overall | 249 | 274 | 0,908759 |
| Math | 117 | 139 | 0,841727 |
| Deterministic | 158 | 177 | 0,892655 |
| Image/source-adjudicated | 91 | 97 | 0,938144 |

Девять исправленных IDs: `val_0048`, `val_0050`, `val_0051`, `val_0054`, `val_0055`, `val_0056`, `val_0057`, `val_0058`, `val_0182`. Все переходы — wrong → correct относительно `base240`; correct → wrong нет.

Важно: стандартный scorer также показывает `changes_vs_frozen_page_rag`, где baseline равен 141/274. Это другое сравнение. Утверждение +9/0 пересчитано напрямую по `task_outcomes` `base240` и `official16`.

## Что вошло в официальный union

`official16` объединяет три заранее замороженных слоя:

- `math5_v11`: 5 source-backed математических rows;
- `english5`: 5 официальных English rows;
- `meb7_6`: 6 официальных MEB rows.

Answer-producing model lineage во всех 274 строках — только `Qwen/Qwen3.5-9B`. Ответов Qwen 27B в этой ветке нет.

## Как защищён эксперимент

До score были одновременно заморожены десять solver/judge bundles. Только `official16` имел право на официальный headline. Независимый pre-execution audit amendment разрешил ровно эти arm; launcher создал `O_EXCL` attempt marker до scorer, запустил десять процессов через общий barrier и записал completion manifest после их завершения. Все return codes нулевые. Независимый post-score аудит сообщил `FINAL PASS`.

Ключевые SHA-256:

- final freeze: `efcf854f011357e35f48bd86d934521ccaf252343988e04023403cced5c34a5c`;
- audit amendment: `4d0720122a0a55d8c560f895ad3ab8b4bc1b24b9779e0132d24313cc9f6d6749`;
- completion: `318be80043ffac433a9482d0fc2bde8acf99d1fe1c3b8ed44dfffcee8a36506e`;
- official metrics JSON: `969cece754bcf3eadd2fded4b519d9974e69a64c27e906e54e5ab9ecae470d8e`.

## Почему 251 не официальный результат

`research_all36` получил 251/274, но включает private-publisher source layers. Их лицензии не проверены. Все такие arm заранее помечены `research_evaluation_only`, `production_eligible=false`, отделены в completion manifest и не допускаются к официальному headline. Исходные PDF и crops в release report не копировались.

Правильная фраза: «официальный результат 249/274; отдельно research-only диагностика 251/274 на источниках с непроверенной лицензией».

## BGE-M3: что действительно доказано

На отдельном публичном `mteb/TurHistQuadRetrieval` BGE-M3 дал Recall@1 0,243652, Recall@5 0,472168 и MRR 0,640879; текущий MiniLM — 0,102051, 0,249023 и 0,319254. Это аргумент за BGE-M3 как optional semantic candidate generator. Это **не** измеренный прирост Benchmark-274 и не часть числа 249.

## Ограничения

- Benchmark-274 многократно использовался в разработке; это audited development wave, не unseen holdout.
- Image/source-adjudicated evaluator является частью system metric, поэтому 249 нельзя называть чистым no-tools качеством модели.
- Перенос на новую книгу/редакцию и production latency требуют отдельного теста.

## Первичные артефакты

- [Машиночитаемый результат](../../reports/maxim_9b_source_expansion_wave_v1_1_20260810/RESULT.json)
- [Русский отчёт](../../reports/maxim_9b_source_expansion_wave_v1_1_20260810/REPORT_RU.md)
- [Post-score verifier](../../reports/maxim_9b_source_expansion_wave_v1_1_20260810/verify_result.py)
- [Completion manifest](../../experiments/maxim_9b_source_expansion_wave_v1_1/final_wave/execution/WAVE_COMPLETION.json)
