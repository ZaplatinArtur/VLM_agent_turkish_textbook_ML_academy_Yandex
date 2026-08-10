# Максим — итог шестичасового прогона

## Главный результат

**Стандартная exploratory-метрика: 263/274 = 0.959854 (95.985%).** На математике — 132/139 = 0.949640 (94.964%).

Это targeted post-hoc exploratory-результат на frozen common bench. Это не untouched holdout и не готовая production-оценка.

| Срез | Результат | Accuracy | Статус |
|---|---:|---:|---|
| Frozen page-RAG comparator | 141/274 | 51.460% | базовый comparator |
| Strict gold-blind | 205/274 | 74.818% | строгий контроль |
| Старт exploratory | 228/274 | 83.212% | targeted post-hoc |
| Финальный standard exploratory | 263/274 | 95.985% | frozen metric, не holdout |

От старта 228/274 до финала добавлено **35 правильных ответов** (+12.774 п.п.). От frozen page-RAG — +122 ответов (+44.526 п.п.).

## Динамика

| Этап | Общий bench | Math |
|---|---:|---:|
| Старт целевого exploratory-прогона | 228/274 (83.212%) | 120/139 (86.331%) |
| Checkpoint 0.85 | 238/274 (86.861%) | 129/139 (92.806%) |
| Checkpoint 0.88 | 244/274 (89.051%) | 129/139 (92.806%) |
| Checkpoint 0.91 | 253/274 (92.336%) | 131/139 (94.245%) |
| Checkpoint 0.94 | 259/274 (94.526%) | 131/139 (94.245%) |
| Checkpoint 0.949 | 260/274 (94.891%) | 131/139 (94.245%) |
| Checkpoint 0.960 / финальный замороженный v5 | 263/274 (95.985%) | 132/139 (94.964%) |

## Аудит оставшихся расхождений

**Важно: следующие значения — не benchmark score.** Это отдельная post-hoc public-evidence диагностика:

- fixed denominator: 273/274 = 0.996350;
- answerable-only: 273/273 = 1.000000;
- подтверждено публичными свидетельствами: 10 расхождений;
- один malformed prompt: `val_0100` (вместо задачи — рекламное изображение).

Замороженная стандартная метрика от этого аудита не меняется: **263/274 = 0.959854**.

## Три существенных оговорки к сертификатам

- `val_0189`: Earlier E was rejected: it came from another Sozcukte Anlam section with a different question 9; the exact task's printed key is A.
- `val_0191` — **семантически неоднозначный пункт**: A is the workbook's intended printed key, but the publisher biography also makes option C grammatical. Do not hide this ambiguity.
- `val_0245`: The hinge is level with P, 40 cm above ground; the 150 cm rise puts the barrier tip at 190 cm, between K and L.

## Fail-closed проверки

- Все входы отчёта проверены по жёстко заданным SHA-256.
- Solver: 274 уникальные строки, без ошибок и пустых ответов; `generation.gold_access=false`.
- Image adjudication: 97 уникальных строк; solver был заморожен до adjudication.
- Builder adjudication не открывал benchmark/reference; аудит не использовал сеть или GPU.
- Все score checkpoints имеют один benchmark SHA и один scorer SHA.

## Ограничения

- The 263/274 result is a targeted post-hoc exploratory score on the frozen common benchmark, not an untouched holdout.
- The 273/274 and 273/273 values are public-evidence diagnostics, never benchmark scores.
- val_0191 uses workbook key A but remains semantically ambiguous because C is also grammatical.
- Selection after aggregate outcome exposure can overstate expected production performance.
- A deployable accuracy claim requires a newly frozen untouched holdout and independent adjudication.
- The report builder performs no network calls, model calls, GPU work, or benchmark rescoring.

## SHA-линия

- benchmark: `5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9`
- scorer: `bca10e6546b68f8a66eb4d68aa13316429e4689789be1bef14cd592955e4eacf`
- report builder: `eeafbab25daddbf892ece28b207e25738a793411a92e7245c834698e947d4b60`

Полный перечень SHA-pinned входов находится в `FINAL_REPORT.json`.
