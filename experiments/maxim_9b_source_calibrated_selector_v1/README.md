# source-calibrated 9B selector v1

это отдельный candidate, он не меняет frozen V7. исторические агрегаты и прошлые результаты
проекта уже были известны до этого эксперимента, поэтому я не называю всю работу глобально
gold-blind. более узкая и проверяемая граница такая: runtime этого selector не читает benchmark
gold, score, judge correctness или outcomes, а rules фиксируются до оценки нового output.

здесь официальные ответы 156 строк
используются только как calibration labels для шести уже существующих 9B solver. benchmark
gold, score, judge correctness и память ошибок не читаются.

для override веса agreement с ActiveCrop не считается доказательством. на 156 source-backed
строках берутся только случаи, где кандидат отличается от anchor: `fix` означает candidate
верен, anchor нет; `regression` — наоборот. Beta(1,1)-smoothed fix rate по
`fixes + regressions` превращается в global logit weight. строки agreement и случаи, где оба
ответа неверны, не увеличивают override weight. затем та же conditional calibration пять раз
повторяется с исключением одного `owner_stage`. кандидат получает ненулевой вес только при
минимум трёх global decisive случаях, не менее 20 total training rows в каждом fold и ни одном
fold posterior ниже 0.50. leave-one-family-out — robustness gate, а не лучший fold после score.

V4/V5 и две parallel8 версии коррелированы. поэтому их веса не складываются как веса четырёх
независимых моделей: суммарный вес каждой пары ограничивается весом её лучшего участника.
ActiveCrop остаётся default anchor. заменить его на одной из 118 uncovered строк можно только
при margin не ниже 0.10, top share не ниже 0.50 и поддержке минимум двух разных candidate
groups. при tie, missing, invalid, low margin или неудачной calibration сохраняются исходные
байты строки ActiveCrop.

если `final_answer` anchor malformed, код не считает его автоматически нормальным. сначала
разрешён только очень строгий generic JSON salvage: весь `raw_response` должен быть одним JSON
объектом с одним разрешённым ключом на уровень и одним scalar leaf. без task ID и reference.
если salvage не прошёл, challenger допускается лишь при том же cross-fitted precision gate и
agreement минимум двух correlation groups; иначе malformed anchor сохраняется fail-closed, а
это явно отмечается в decision artifact.

на 156 covered строках selector ничего не выбирает: он копирует байты frozen source solver.
на 118 uncovered он копирует целую исходную строку выбранного 9B donor, а не синтезирует новый
ответ. `task_id` нужен только для exact join и membership source-union; он не входит в score.

primary output дополнительно ограничен outcome-free route authority из frozen input package v1.1.
это только safety veto, а не quality feature: все 97 `image_judge` строк в
`candidate_solver.jsonl` копируются байт-в-байт из ActiveCrop. поэтому изменения primary
возможны только на 177 deterministic строках. полный source-overlay сохраняется отдельно в
`full_candidate_solver.jsonl`, но его нельзя оценивать с переиспользованием старого image judge:
если его запускать, нужен новый judge для изменённых image rows.

первый pre-route draft успел получить freeze до появления этой route authority, но ни разу не
оценивался и был сразу superseded. его exact хэши сохранены в
`PRE_ROUTE_DRAFT_PROVENANCE.json`; финальным считается только новый `FREEZE.json`.

воспроизведение без GPU и сети:

```powershell
python experiments/maxim_9b_source_calibrated_selector_v1/test_selector.py
python experiments/maxim_9b_source_calibrated_selector_v1/selector.py --build
python experiments/maxim_9b_source_calibrated_selector_v1/selector.py --verify-freeze
```

после `--build` появляются `calibration.json`, `decisions.jsonl`,
`candidate_solver.jsonl`, `full_candidate_solver.jsonl`, `candidate_manifest.json`,
`FREEZE.json` и `FREEZE_PINS.json`.
никакой QA accuracy эти файлы не содержат. последующая оценка допускается только отдельным
процессом после публикации exact SHA `FREEZE.json`/`FREEZE_PINS.json`.
input identity не берётся из self-labelled pool: она обязана совпасть сразу у шести exact
SHA-pinned solver, а membership 156 строк приходит из отдельно pinned source aggregate/freeze.
route и task-specific answer type вообще не используются как quality features.
route читается только из independently pinned outcome-free authority и только запрещает менять
image rows в primary. candidate pool из чужого эксперимента проходит только SHA integrity check,
но его content не парсится и selection outcomes оттуда не копируются.
