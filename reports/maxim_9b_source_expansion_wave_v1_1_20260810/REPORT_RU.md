# Source expansion wave v1.1: результат

Локальная проверка артефактов и независимый post-score аудит подтверждают официальный результат **249/274 = 0,908759** для `official16`. Отдельный файл post-score аудита в этот пакет не добавлялся; его `FINAL PASS` был передан через координацию агентов, а все числовые утверждения ниже повторно проверяет локальный verifier.

## Что изменилось

Непосредственная база — all-9B `base240` с **240/274 = 0,875912**. `official16` объединяет три заранее замороженных source-компонента на официальных или разрешённых для официального контура источниках: пять математических заданий, пять английских и шесть заданий MEB.

Итог относительно `base240`:

- **9 fixes, 0 regressions**;
- **+3,2847 процентного пункта**;
- Math: **117/139 = 0,841727**;
- deterministic: **158/177**;
- image judge: **91/97**;
- 0 solver errors и 0 пустых ответов.

Ровно девять исправленных outcome: `val_0048`, `val_0050`, `val_0051`, `val_0054`, `val_0055`, `val_0056`, `val_0057`, `val_0058`, `val_0182`. Все они были wrong в `base240` и correct в `official16`; обратных переходов нет.

## Десять arm одной волны

| Arm | Режим | Результат |
|---|---|---:|
| `base240` | официальный baseline | 240/274 |
| `math5_v11` | официальный component ablation | 245/274 |
| `english5` | официальный component ablation | 241/274 |
| `meb7_6` | официальный component ablation | 243/274 |
| `official16` | единственный кандидат в официальный headline | **249/274** |
| `bs11_8_research` | research only | 241/274 |
| `research_bs24` | research only | 250/274 |
| `fenomen12_research` | research only | 241/274 |
| `research_fenomen28` | research only | 250/274 |
| `research_all36` | research only | 251/274 |

Число **251/274** нельзя использовать как официальный результат. `research_all36` включает материалы частных издателей с непроверенной лицензией, имеет статус `research_evaluation_only`, не допускается в production и не разрешает распространение исходных PDF или crops. В репозиторный релизный отчёт такие assets не копировались.

## Как была защищена волна

Все десять solver/judge комплектов были заморожены до score. Независимый pre-execution аудит разрешил ровно эти десять arm. Launcher создал `O_EXCL` attempt marker до scorer, запустил все arm через один общий `Barrier(10)`, дождался всех процессов и только после этого записал hash-only completion manifest. Все return codes равны нулю.

Ключевые SHA-256:

| Артефакт | SHA-256 |
|---|---|
| `FINAL_WAVE_FREEZE.json` | `efcf854f011357e35f48bd86d934521ccaf252343988e04023403cced5c34a5c` |
| `INDEPENDENT_AUDIT_AMENDMENT.json` | `4d0720122a0a55d8c560f895ad3ab8b4bc1b24b9779e0132d24313cc9f6d6749` |
| `ATTEMPT_STARTED.json` | `6dfd82ec124e863bd172d5e10abc20f1f88cd5865767b5cee1a5391abc47adaa` |
| `WAVE_COMPLETION.json` | `318be80043ffac433a9482d0fc2bde8acf99d1fe1c3b8ed44dfffcee8a36506e` |
| `official16/metrics.json` | `969cece754bcf3eadd2fded4b519d9974e69a64c27e906e54e5ab9ecae470d8e` |
| `research_all36/metrics.json` | `77075786b15aa25d8aaa889ea14cec1a8799b1a66807afa4bb55dff522e10b27` |

Хеши outputs всех десяти arm находятся в [RESULT.json](RESULT.json) и дополнительно закреплены completion manifest. Проверка выполняется командой:

```powershell
python reports/maxim_9b_source_expansion_wave_v1_1_20260810/verify_result.py
```

## Честные ограничения

Это all-9B результат: answer-producing lineage замкнута на `Qwen/Qwen3.5-9B`, ответы 27B не используются. Но Benchmark-274 много раз применялся в разработке, поэтому **0,908759 — development metric, а не unseen holdout и не production accuracy**. One-shot относится к этой конкретной заранее замороженной волне, а не ко всей истории проекта.

Часть image-строк оценивается source-adjudicated judge-файлами. Поэтому результат корректнее называть метрикой всей evidence-gated системы, а не «чистой способностью 9B-модели без инструментов».

## BGE-M3 — отдельное retrieval evidence

На внешнем публичном `mteb/TurHistQuadRetrieval` были отдельно измерены 1 213 passages, 1 024 queries и два релевантных passages на запрос. Стандартные retrieval-метрики:

| Embedder | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| текущий MiniLM | 0,102051 | 0,249023 | 0,319254 |
| GTE multilingual base | 0,207520 | 0,394043 | 0,551582 |
| BGE-M3 | **0,243652** | **0,472168** | **0,640879** |

Для BGE закреплены `BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181`, MIT и 1024 dimensions. Это подтверждает выбор BGE-M3 как **опционального генератора semantic candidates**. Эти числа получены на другом retrieval benchmark, не являются QA accuracy проекта и не добавляют ничего к 249/274. Финальное право менять ответ по-прежнему дают source/visual certificates, а не similarity score эмбеддера.
