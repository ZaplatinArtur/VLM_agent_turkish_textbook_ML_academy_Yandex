# Документация экспериментов Максима

Этот каталог — русскоязычная точка входа в исследовательскую ветку VLM-агента для турецких учебников. Здесь отдельно описаны измеренные результаты, устройство финального source-native пайплайна, ограничения оценки и материал для защиты проекта.

## Результат в одной таблице

| Что именно измерено | Результат | Как это правильно называть |
|---|---:|---|
| Frozen page-RAG | 141/274 = **0,5146** | исходный контрольный RAG |
| Historical no-tools reasoning | 191/274 = **0,6971** | сильный исторический baseline; другая lineage судьи |
| Matched clean no-tools 9B | 193/274 = **0,7044** | exact all-9B anchor, пересчитанный текущим judge-v2 |
| Query Active Crop 9B | 194/274 = **0,7080** | all-9B tuned development anchor |
| Clean 9B source rebase | 237/274 = **0,8650** | post-hoc deterministic development replay; Math 107/139 |
| Active Crop 9B source rebase | 238/274 = **0,8686** | tuned post-hoc development replay; Math 108/139 |
| Audited all-9B selector v1.2 primary | 240/274 = **0,8759** | one-shot four-arm dev wave; +2 fixes, 0 regressions против tuned 238; Math 109/139 |
| Audited all-9B official source expansion | 249/274 = **0,9088** | frozen ten-arm wave; +9 fixes, 0 regressions против 240; Math 117/139 |
| Gold-blind V2.1 + conservative V3.1 repair | 205/274 = **0,7482** | dev-selected anchor |
| Source-native V6 | 238/274 = **0,8686** | one-shot development replay; 4 изменения ответа относительно V5 |
| Source-adjudicated V7 | 242/274 = **0,8832** | one-shot development replay; 1 новый ответ + 3 исправления оценки |
| V7, математика | 112/139 = **0,8058** | frozen dev metric; без прироста относительно V6 |
| Holdout80 raw source evidence | 71/80 = **0,8875** | same-book task-disjoint; неоднородный source composite, не QA accuracy |
| Holdout80 после принятого protocol erratum | 79/80 = **0,9875**; valid-only 79/79 | 8 переставленных Physics labels и один open-response подтверждены отдельно |

Главная оговорка: **0,908759 — текущая all-9B system metric на development replay, а 0,8875/0,9875 — source evidence на same-book Holdout80**. Это разные метрики, их нельзя сравнивать как две версии одной accuracy. Holdout проверяет точную адресацию новых заданий внутри уже известных книг, но не end-to-end reasoning и не перенос на новые книги. Historical V7 0,8832 также имеет другую model/evaluator lineage и показывается отдельно.

Ещё одна важная оговорка относится к модели. Historical no-tools и более поздняя source-native ladder имеют разную model lineage. Историческая цепочка V1–V7 наследует meta-verifier anchor, в котором 272 из 274 строк были сгенерированы `Qwen/Qwen3.5-27B`, а две строки fail-closed сохранили 9B-ответ. Поэтому historical V6/V7 нельзя называть результатом «9B + RAG».

Отдельный all-9B replay прошёл независимый read-only аудит. Clean rebase дал 237/274, а Active Crop rebase — 238/274. В обоих вариантах source union содержит 156 задач: 45 ответов заменены, 111 подтверждены источником, 118 строк прошли из anchor без source certificate. Относительно каждого собственного anchor получено 44 fixes и 0 regressions. Разница clean/tuned по correctness состоит только в `val_0084`, которая находится вне source union; поэтому 237/274 — более консервативная цифра, а 238/274 — post-hoc tuned development result. Записанные latency и token usage унаследованы от 9B anchor и не являются end-to-end временем source replay.

Поверх tuned all-9B source anchor был проведён отдельный frozen four-arm selector audit. `v1.2 primary` изменил две deterministic строки (`val_0089` и `val_0251`), обе wrong → correct, и получил 240/274 без regressions относительно 238 anchor. Все 156 source rows и 97 image rows остались byte-identical. Это one-shot волна новых outputs, но всё ещё результат на многократно использованном dev benchmark. Параллельный source-calibrated selector дал честный null result: 0 безопасных override вне source coverage и поэтому не оценивался.

Последующий post-score answer-contract repair v1.1 оставил score 240/274: `val_0223` был очищен из `} }16` в `16`, но outcome уже был correct; `val_0248` fail-closed сохранил wrong anchor. Итого 0 fixes, 0 regressions и 0 outcome diffs. Три generic canonicalization arm дали 0/0/0 изменений.

Следующая frozen source-expansion wave добавила к `base240` шестнадцать заранее определённых официальных source rows: 5 Math, 5 English и 6 MEB. Единственный официальный headline arm `official16` получил **249/274 = 0,908759**, Math 117/139, deterministic 158/177 и image/source-adjudicated 91/97. Относительно `base240` это ровно 9 fixes и 0 regressions. Независимый post-score аудит сообщил `FINAL PASS`. Лучший численно arm `research_all36` дал 251/274, но остаётся только `research_evaluation_only`: лицензии частных источников не проверены, production и официальный headline запрещены.

## Что читать

1. [Проект и границы результата](01_проект_и_результат.md) — задача, benchmark и короткая формулировка вклада.
2. [История экспериментов](02_история_экспериментов.md) — положительные, отрицательные и невалидные результаты.
3. [Финальная архитектура](03_финальная_архитектура.md) — Evidence OS, exact-source, fail-closed и certificates.
4. [Честность оценки](04_оценка_утечки_и_воспроизводимость.md) — dev/holdout, leakage, V6/V7 и воспроизводимость.
5. [Запуск и структура репозитория](05_запуск_и_структура.md) — установка, smoke test, judge, UI и граница source-native replay.
6. [Постер и демонстрация](06_постер_и_демо.md) — композиция постера, сценарий доклада и trace-demo.
7. [Жёсткие вопросы](07_жесткие_вопросы.md) — короткие честные ответы руководителям.
8. [Статьи и заимствованные идеи](08_статьи_и_статус_реализации.md) — что было только inspiration, а что реально проверялось.
9. [Первичные локальные источники](09_первичные_источники.md) — отчёты, SHA и границы текущего checkout.
10. [Финальный публичный Holdout80 source-evidence report](../../reports/maxim_holdout80_final_source_evidence_20260809/REPORT_RU.md) — frozen raw, erratum, хронология и ограничения blind-процедуры.
11. [Audited all-9B selector](10_audited_all_9b_selector.md) — four-arm one-shot wave, 240/274 и source-calibrated null result.
12. [All-9B source expansion](11_all_9b_source_expansion.md) — ten-arm frozen wave, официальный 249/274 и отдельно research-only 251/274.

## Короткая формулировка для защиты

> Сильная reasoning-модель часто лучше обычного RAG, потому что нерелевантный контекст портит уже правильный ответ. Поэтому мы сделали retrieval не источником безусловной истины, а претендентом на изменение ответа. Изменение разрешается только после точной привязки к официальному PDF и детерминированной проверки сертификата. Если доказательство неполное или противоречивое, система сохраняет исходный ответ.

## Что можно и нельзя утверждать

Можно:

- на фиксированном development benchmark V6 получил 238/274, а V7 — 242/274;
- V6 был заморожен и оценён одним запуском без post-score rollback;
- у V7 один прямой solver-answer gain и три source-backed исправления прежней оценки;
- clean all-9B source rebase воспроизводится как 237/274, tuned Active Crop rebase — как 238/274; оба результата имеют 44 fixes и 0 regressions относительно своих 9B anchors;
- frozen all-9B selector `v1.2 primary` получил 240/274: две замены относительно tuned 238 anchor, обе fixes, при 0 regressions; все четыре arm были досчитаны до раскрытия результатов;
- frozen all-9B `official16` получил 249/274: девять wrong → correct и ни одного correct → wrong относительно `base240`; все десять arm завершились до чтения outputs;
- source-native этапы не используют `task_id` как признак поиска или выбора ответа;
- ранние task-ID keyed результаты 0,766/0,832/0,960 были выявлены и исключены из production claim.
- frozen raw Holdout80 source-evidence composite равен 71/80; после явно опубликованного erratum — 79/80, а на 79 валидных строках — 79/79.

Нельзя:

- называть 0,8832 качеством на новых книгах или production accuracy;
- называть all-9B 0,875912 blind holdout или доказанным переносом на новые книги;
- называть all-9B 0,908759 unseen holdout или production accuracy;
- использовать research-only 251/274 как официальный headline: у частных источников не подтверждена лицензия;
- приписывать source-calibrated selector прирост: он сделал 0 uncovered overrides и не оценивался;
- говорить, что V7 научил модель решать ещё четыре задачи;
- приписывать element-level proxy результат 0,4489 методу ColPali;
- считать post-hoc 0,832 или 0,960 переносимым результатом;
- утверждать, что clean-room replay из чистого клона уже доказан.
- называть Holdout80 0,9875 end-to-end QA accuracy или доказательством переноса на новые книги.
