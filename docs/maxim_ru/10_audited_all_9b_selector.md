# Audited all-9B baseline selector

Этот эксперимент отвечает на узкий вопрос: можно ли безопасно улучшить уже пересчитанный all-9B source rebase, не возвращаясь к историческим 27B-ответам и не открывая результаты новой волны по одному.

## Основной результат

`v1.2 primary` получил **240/274 = 0,875912**. Его непосредственный anchor — tuned Active Crop 9B source rebase с **238/274 = 0,868613**. Selector изменил только две строки:

| ID | Предмет | Изменение относительно anchor |
|---|---|---|
| `val_0089` | History | wrong → correct |
| `val_0251` | Math | wrong → correct |

Итоговая дельта относительно 238-строчного anchor: **2 fixes, 0 regressions**. Срезы результата:

| Срез | Правильно | Всего | Accuracy |
|---|---:|---:|---:|
| Overall | 240 | 274 | 0,875912 |
| Math | 109 | 139 | 0,784173 |
| History | 10 | 10 | 1,000000 |
| Deterministic | 158 | 177 | 0,892655 |
| Image judge | 82 | 97 | 0,845361 |

Image-часть не менялась. Все **156 source-union rows** и **97 image-judge rows** сохранены byte-identical. Из 274 строк 272 прошли как точная копия anchor, а две получили structural replacement. Единственная model lineage в этой ветке — `Qwen/Qwen3.5-9B`.

Важно не путать два сравнения. Утверждение «2 fixes, 0 regressions» относится к разнице с all-9B source anchor 238/274. В стандартном score report есть отдельное сравнение с frozen page-RAG 141/274; оно отвечает на другой вопрос.

## Что делал selector

Anchor оставался default. Замена разрешалась только тогда, когда три независимо сформированных evidence group совпадали на одном структурном challenger и этот ответ отличался от anchor:

1. structural strict group;
2. native group, причём его ответ считался определённым только при согласии V4 и V5;
3. parallel group, причём его ответ считался определённым только при согласии двух parallel8 финалов.

Source union выполнял роль safety veto, а image route был полностью защищён от изменения. Runtime selector не получал gold, reference answer, correctness, score, judge verdict или прежний outcome конкретной строки. `task_id` использовался только для выравнивания уже существующих артефактов.

Это не означает, что вся идея была придумана вслепую. Benchmark многократно использовался до этой волны, исторические агрегаты и результаты прежних экспериментов были известны. Корректная формулировка — **frozen one-shot multi-arm development wave для новых outputs**, а не blind holdout.

## Все четыре frozen arm

До открытия любого нового score были зафиксированы четыре arm, после чего они были запущены через общий barrier. Same-wave retuning был запрещён.

| Arm | Overall | Math | History | Deterministic | Image judge |
|---|---:|---:|---:|---:|---:|
| `v1.1 primary` | 239/274 = 0,872263 | 108/139 | 10/10 | 157/177 | 82/97 |
| `v1.1 secondary` | 237/274 = 0,864964 | 106/139 | 10/10 | 155/177 | 82/97 |
| `v1.2 exploratory` | 239/274 = 0,872263 | 108/139 | 10/10 | 157/177 | 82/97 |
| `v1.2 primary` | **240/274 = 0,875912** | **109/139** | **10/10** | **158/177** | **82/97** |

Completion manifest SHA-256:

`ea32a839ccf8dc256e69f3b994332ce32bae552c120d6a9d98bd7691cd950973`

Он подтверждает завершение всех четырёх score-процессов с нулевым return code до раскрытия результатов волны. SHA-256 финального `v1.2 primary` solver:

`09aa8d69e7de3a02bbc9b28b2b269b845a0dee1a40ef2d6aa55f7e966a779bef`

## Source-calibrated selector: честный нулевой результат

Параллельно был заморожен другой, outcome-free selector. Его веса калибровались только на 156 source-backed строках по шести существующим all-9B solver artifacts. На 118 строках вне source coverage он не нашёл ни одной замены, которая прошла бы консервативный порог: **0 uncovered overrides**.

Эта ветка **не оценивалась**, поэтому у неё нет QA accuracy и её нельзя записывать как 237, 238 или 240 из 274. Это полезный null result: при заданном safety contract имеющихся source-only сигналов оказалось недостаточно, и selector корректно abstain. Все 97 image rows в primary candidate были сохранены byte-identical.

## Что можно утверждать

- all-9B `v1.2 primary` воспроизводимо записан как 240/274 на известном development benchmark;
- относительно tuned all-9B source anchor он сделал две замены, обе правильные, без regressions;
- все четыре arm были заморожены и досчитаны до раскрытия результатов волны;
- source и image safety regions не изменялись;
- в артефактах этой ветки нет 27B-ответов.

## Что нельзя утверждать

- что 0,875912 является unseen holdout или production accuracy;
- что selector доказанно переносит +2 на новые книги, языки или распределения;
- что лучший из четырёх arm был выбран без знания прежнего dev benchmark;
- что source-calibrated selector дал прирост: он не сделал override и не был оценён.

## Post-score answer-contract repair

Repair v1.1 был оценён отдельно и сохранил **240/274**. Он сделал одну форматную замену: `val_0223` изменился с `} }16` на `16`. Поскольку scorer уже засчитывал исходную строку, correctness осталась correct → correct. `val_0248` не прошёл строгий parser contract, система abstain и сохранила wrong anchor.

Итог относительно v1.2 primary: **0 fixes, 0 regressions, 0 outcome diffs**. Score SHA-256:

`453970038673fb29b97d754b4ef980e19850a930499e89ccccfb9f9b8e6c9dc8`

Три более общих canonicalization arm не изменили ни одной строки: 0/0/0 canonicalized rows. Они не оценивались. На момент этого эксперимента headline оставался **240/274**, а repair следовало показывать как безопасную нормализацию формата, не как metric gain. Более поздняя отдельная source-expansion wave подняла текущий audited all-9B headline до [249/274](11_all_9b_source_expansion.md).

## Строка для общей таблицы

| Автор | Часть пайплайна | Идея | Accuracy |
|---|---|---|---:|
| Максим | Агент/Ансамбль | all-9B source anchor + консервативный selector: замена только при согласии structural/native/parallel; source и image зоны защищены | **0,875912** |

## Первичные артефакты

- [Профиль v1.2](../../experiments/maxim_9b_baseline_selector_v1/profile_v1_2.json)
- [Composition manifest v1.2](../../experiments/maxim_9b_baseline_selector_v1/compositor_output_v1_2/composition_manifest_v1_2.json)
- [Score v1.2 primary](../../experiments/maxim_9b_baseline_selector_v1/evaluation_wave_v1/results/v1_2_primary_score.json)
- [Completion manifest волны](../../experiments/maxim_9b_baseline_selector_v1/evaluation_wave_v1/WAVE_COMPLETION_MANIFEST.json)
- [Source-calibrated selector README](../../experiments/maxim_9b_source_calibrated_selector_v1/README.md)
- [Source-calibrated candidate manifest](../../experiments/maxim_9b_source_calibrated_selector_v1/candidate_manifest.json)
- [Source-calibrated freeze](../../experiments/maxim_9b_source_calibrated_selector_v1/FREEZE.json)
- [Answer-contract repair score](../../experiments/maxim_9b_answer_contract_repair_v1_1/evaluation_on_v1_2_primary_240/score.json)
- [Answer canonicalization manifest](../../experiments/maxim_9b_answer_canonicalization_v1/candidate_output/candidate_manifest.json)
