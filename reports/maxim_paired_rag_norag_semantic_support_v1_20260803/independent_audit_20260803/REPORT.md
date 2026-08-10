# Независимый аудит результата 193/274

Дата аудита: 2026-08-03. Аудит был read-only относительно исходных артефактов; GPU и удалённые endpoints не использовались.

## Вердикт

**Целостность метрики: PASS.** Артефакты действительно дают `193/274 = 0.704379562...` (`0.7044` при округлении). Все проверенные хэши совпадают, итог содержит ровно 274 уникальных outcome, разбиение `177 deterministic + 97 image judge` полное и непересекающееся, а повторный подсчёт даёт `132 + 61 = 193`.

**Атрибуция результата новому paired RAG/no-RAG подходу: FAIL.** Композитор выбрал RAG в `0/274` строк. Все 274 ответа взяты из ранее существовавшего frozen `b0_no_tools` / no-RAG источника. Корректное название результата: **frozen no-tools v2_cot, заново оценённый по frozen matched judge-v2**. Называть `0.7044` приростом semantic-support RAG нельзя.

**Прямой утечки gold/reference в verifier generation не найдено.** Все 100 queue-строк прошли рекурсивную проверку запрещённых полей, их request hashes восстановлены без расхождений, а profile/queue/code были зафиксированы до verifier run. Однако для более старого исходного `b0_no_tools_raw.jsonl` нет полноценного immutable run manifest с хэшами команды, runtime-конфига и request transcript. Текущий tracked solver-код не передаёт модели `reference_answer`/`reference_solution`, а сам raw-артефакт не содержит gold-полей; это сильное свидетельство отсутствия утечки, но не криптографическое доказательство исторического remote run.

## Что проверено

- Benchmark: 274 строки, 274 уникальных `task_id`, SHA-256 `5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9`.
- Solver: 274 строки в точном benchmark order, 274 уникальных `task_id`, SHA-256 `41e8ee9c191a312377b4997798509cd89d4fb39e6e84eff2da1f829005ce5c90`.
- Score outcomes: 274 уникальные строки в точном benchmark order.
- Deterministic branch: 177 уникальных задач, 132 correct.
- Image-judge branch: 97 уникальных задач, 61 correct; пересечение с deterministic branch равно нулю, объединение равно всем 274 задачам.
- Все 97 fresh judge outputs семантически идентичны строкам итогового `matched_image97_judge.jsonl`, order сохранён.
- У всех 97 judge rows: `prompt_version=judge-v2`, model `Qwen/Qwen3.5-9B`, `cache_hit=false`, `judge.error=null`, уникальные request IDs и cache keys.
- Все 97 request IDs независимо восстановлены из frozen judge adapter без расхождений.
- Все 97 backend hashes равны точному frozen значению `e3f71b4af7fa8ad8a6db755d43bdf4a895d087b701436c105da8c5416804fbd9`.
- Frozen judge adapter hashes совпали: `prompts.py=84a134...`, `pipeline.py=d61a24...`, `schema.py=d86fed...`.
- Рекурсивная проверка 35 записанных path/SHA пар в evaluation/orchestration/finalization manifests не нашла ни одного расхождения.
- In-memory повторный запуск frozen composer дал байт-семантически те же 274 JSON objects и решения `no_rag=274`, `rag=0`.

## Происхождение итогового solver

Маршруты композитора:

| Маршрут | Строк |
|---|---:|
| same answer, default no-RAG | 141 |
| semantic verifier, gate default no-RAG | 100 |
| unsafe/missing context, default no-RAG | 33 |
| **RAG выбран** | **0** |
| **no-RAG выбран** | **274** |

Для 100 verifier rows: 99 валидных verdict были отрицательными (`rag_answer_supported=false`), одна строка (`val_0197`) завершилась runner error и fail-closed ушла в no-RAG. Это соответствует preregistered fallback.

Во всех 274 строках solver семантически точно совпадают с frozen no-RAG источником следующие поля: `task_id`, `model`, `final_answer`, `solution_steps`, `reasoning`, `forced_answer`, `raw_response`, `tool_calls`, все scalar-поля `usage`, `error`. Различаются только композиционные метаданные (`condition`, `prompt_version`, добавленный `generation.semantic_support_composition`) и порядок JSON-ключей. Все 274 записанных `source_row_sha256` для no-RAG и page-RAG независимо восстановлены без расхождений.

## Сравнение с frozen Subject Router 182/274

| Срез | Новый solver | Subject Router | Разница | Fixed / regressed |
|---|---:|---:|---:|---:|
| Math, 139 | 103 (0.7410) | 103 (0.7410) | 0 | 0 / 0 |
| Non-Math, 135 | 90 (0.6667) | 79 (0.5852) | +11 | 19 / 8 |
| **Все 274** | **193 (0.7044)** | **182 (0.6642)** | **+11** | **19 / 8** |

На Math все 139 answer-bearing payload полностью идентичны Router. На Non-Math Router использует page-RAG, поэтому полностью идентичных payload нет (`0/135`); точный `final_answer` совпадает на 88 строках и различается на 47. Во всех 27 изменившихся outcome (`19 fixed + 8 regressed`) изменился и `final_answer`, то есть прирост относительно Router не вызван повторной случайностью judge на одинаковом ответе.

Fixed vs Router: `val_0025`, `val_0035`, `val_0036`, `val_0090`, `val_0096`, `val_0097`, `val_0114`, `val_0119`, `val_0120`, `val_0127`, `val_0147`, `val_0148`, `val_0154`, `val_0155`, `val_0166`, `val_0168`, `val_0177`, `val_0184`, `val_0191`.

Regressed vs Router: `val_0089`, `val_0124`, `val_0126`, `val_0129`, `val_0132`, `val_0150`, `val_0196`, `val_0197`.

## Важные оговорки

1. `0.7044` валидно как score конкретного frozen no-RAG артефакта на этом benchmark и по этому frozen judge protocol. Оно **не валидно как score нового RAG-подхода**, поскольку RAG не выбран ни разу.
2. No-RAG источник уже был создан и оценён до preregistration этого эксперимента; его старый manifest содержит `191/274` по другой, presentation-hybrid judge lineage. Свежий matched judge дал `193/274`; между старой и новой оценками той же модели перевернулись 24 image verdict (`13` в плюс, `11` в минус, net `+2`). Поэтому эти `+2` — эффект смены judge lineage, а не улучшение solver.
3. Benchmark многократно использовался для сравнения и проектирования вариантов. Поэтому `0.7044` нельзя представлять как незатронутую адаптацией held-out оценку или как доказательство ожидаемой generalization на новом наборе.
4. Frozen backend hash фиксирует request config, endpoint, модельное имя, seed и decoding, но не содержит model-weight revision/container digest. Это не нарушает текущий внутренний протокол, однако ограничивает внешнюю криптографическую воспроизводимость judge.
5. Исторический no-RAG raw source не имеет полного run manifest. Никаких наблюдаемых признаков gold leakage не найдено, текущий solver-код явно исключает reference fields из model messages, но абсолютное доказательство происхождения старого remote run недоступно.

## Ключевые SHA-256

| Артефакт | SHA-256 |
|---|---|
| benchmark | `5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9` |
| frozen no-RAG source | `496236da966ed68aa81af3d33da1c40b85c5a11b342de253ada244f97320de8f` |
| preregistered profile | `67fb456fde408581c2af77a9aca6d0033b82d1459c45db106dcd8b0c72b60f03` |
| preparation manifest | `534ee1028b18a44069ad3d0b306a2c37ee197e68ed19823ba3bb19015a435d0f` |
| verifier queue | `ada53eecbdbab157a12ec838b1d76d3f6d4dd3f5f265777a028c693eeffde387` |
| verifier results | `4ca5c66d2f49d7404b4e9a328411b5355332597188f95dfe4cdb399db9043b24` |
| composition manifest | `74d79c4e7659f1b1088444ecccc3e4563a554cb74e10784c8fe01dc674c32472` |
| composed solver | `41e8ee9c191a312377b4997798509cd89d4fb39e6e84eff2da1f829005ce5c90` |
| evaluation manifest | `36a008c2738467269d43cca78389478e51b82c0f7d9c7d948d56e37dc62fac6c` |
| fresh judge input | `64c7a71332f73fc2c8c326e3b12b3bbf61c40a2db6544e2ab2143cbb60754dea` |
| fresh judge result | `f089f49edc174c39e61e118624f2ec384da4ef3034698e4622aefd3abe7ca34e` |
| matched image judge | `28f3d107a0840970e0f82c46157cd85c0f6200f82e1cf105ff39409e58c636b5` |
| orchestration manifest | `f57cda8d9b6198016154b6d80c8fbc3b8008792f6b37a1d062c49ccac932f4ad` |
| finalization manifest | `e3d778972d9657380ccdc717156b407f7c111dd6cb3f0d13bb1056ad1edc5554` |
| score.json | `77932f8ef6af9e10df1d79b607902ff54a3144b3c1c17be300ce406a35f2a462` |
| frozen judge backend config | `e3f71b4af7fa8ad8a6db755d43bdf4a895d087b701436c105da8c5416804fbd9` |
