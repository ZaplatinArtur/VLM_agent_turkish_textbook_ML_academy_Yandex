# Holdout80: финальный отчёт по source evidence

Дата фиксации отчёта: 2026-08-09.

## Короткий итог

Holdout80 проверяет новые задания из уже доступных системе официальных учебников. Это `same-book task-disjoint` срез, а не новые книги и не новый домен. В нём измерялось точное связывание изображения с заданием и официальным источником: для Biology/Physics — извлечение буквы из ключа, для Math12 — определение номера activity. Поэтому числа ниже нельзя называть end-to-end accuracy математического reasoning или полной QA-метрикой модели.

| Срез | Frozen raw | После принятого erratum | Что измерено |
|---|---:|---:|---|
| Biology9 MCQ | 30/30 = **1,0000** | 30/30 = **1,0000** | точный ответ из официального ключа |
| Physics12 MCQ | 21/30 = **0,7000**; coverage 29/30 | 29/30 = **0,9667** | 8 дефектов sealed gold исправлены; один невалидный open-response остаётся ошибкой |
| Math12 clean | 20/20 = **1,0000** | без изменений | правильный номер source activity, не решение математики |
| MCQ вместе | 51/60 = **0,8500** | 59/60 = **0,9833** | frozen raw сохранён; erratum считается отдельно |
| Protocol-inclusive composite | 71/80 = **0,8875** | 79/80 = **0,9875** | неоднородный composite: 60 source answers + 20 source bindings |
| Только валидные задания | — | 79/79 = **1,0000** | исключён Physics Unit 2 q24, который не является MCQ |

У всех 59 принятых MCQ-ответов результат resolver совпал с ячейкой официального ключа PDF. Единственный abstain пришёлся на Physics Unit 2 q24: source adapter корректно отказался придумывать A–E для открытого задания.

**Главное число неизменяемого запуска — 71/80 = 0,8875.** Значение 79/80 = 0,9875 — не переписанный score, а отдельный protocol erratum после независимого подтверждения восьми ошибок frozen gold. На валидном подмножестве получается 79/79, но этот знаменатель сформирован только из-за обнаруженного дефекта протокола и всегда должен публиковаться рядом с исходными 71/80.

### Это не V7

V7 остаётся отдельным результатом: 242/274 = **0,883212** на inspected development QA benchmark. Он включает reasoning answer и source-aware adjudication. Holdout80 source-evidence composite 0,8875/0,9875 проверяет адресацию источника внутри тех же книг и не заменяет V7, не подтверждает QA accuracy 0,9875 и не доказывает перенос на unseen books.

## Что произошло с Physics

После запечатывания outputs и открытия alignment/gold восемь принятых Physics-ответов не совпали с sealed gold. Повторная проверка официальной страницы ключа показала систематическую ошибку транскрипции: для выбранных вопросов Unit 2 в sealed gold стояла буква из строки Unit 3 с тем же номером вопроса, а для Unit 3 — буква из Unit 2. Для каждой строки frozen label действительно совпадает с соседним unit, а ответ resolver — с нужным unit.

Независимая проверка повторно подтвердила все восемь ячеек по официальному PDF. Frozen score при этом не менялся.

| Задание | Frozen sealed gold | Официальная ячейка нужного unit | Откуда ошибочно взята frozen-буква |
|---|:---:|:---:|---|
| Physics U2 q31 | C | D | Unit 3 q31 |
| Physics U2 q32 | B | C | Unit 3 q32 |
| Physics U2 q34 | E | C | Unit 3 q34 |
| Physics U2 q38 | C | A | Unit 3 q38 |
| Physics U3 q28 | D | A | Unit 2 q28 |
| Physics U3 q30 | B | A | Unit 2 q30 |
| Physics U3 q31 | D | C | Unit 2 q31 |
| Physics U3 q37 | A | B | Unit 2 q37 |

Ещё одна ошибка была в типе задания. Physics Unit 2 q24 на странице вопроса и в официальном ключе является открытым заданием, но frozen protocol записал его как `exact_choice` с буквой A. Source inventory ещё до opaque запуска относил вопросы 24–27 к `unsupported_open_response`; resolver fail-closed выдал abstain. В protocol-inclusive метрике q24 остаётся ошибкой, а в valid-only срезе исключается как невалидная benchmark-строка.

## Хронология без переписывания результата

1. 8 августа selection был построен без benchmark outcomes и gold. Manifest из 80 строк получил SHA-256 `7a6ecbe1579790541bc0c0c36a9141941c2c0cf8f65dd3cd5b6a687bae8c5fa5` и статус `FROZEN_BEFORE_GOLD`.
2. Отдельной командой был создан sealed gold: SHA-256 `55e58007a1f30f4ee1c7f10c6f9c4bdd6b87be861046c13c7496a709a4fb7bcc`. Его payload, key pages и alignment map не публикуются.
3. Math12 resolver, opaque runner и evaluator были заморожены отдельными commit до private map: `cb13b66`, `6f2be06`, `c7efa65`. Clean outputs были включены в общий pre-map seal.
4. MCQ v1.0 не допустили к запуску после аудита trust boundary. Исправленная v1.1 была заморожена в commit `99963dc0225e2f5ff855d404004cceafadc3ddd5`; 56 adversarial tests и повторная public-source сборка прошли до чтения opaque input.
5. MCQ запуск выполнялся из clean detached checkout указанного commit, локально на CPU, без сети, GPU и shared compute. Launch attestation был записан до чтения opaque MCQ.
6. Source run атомарно завершился: 59 accepted, 1 abstain. Outputs, 60 сертификатов и evaluator bytes были рекурсивно запечатаны 8 августа в 21:03:44 UTC. Pre-map seal имеет SHA-256 `4441255a7aac6bdcf83e3f4af3f348eb9ad7e78afb199bf1c4f4a321ddb5fe9b`.
7. Только после seal были сопоставлены opaque ID и task ID и открыт sealed gold. Получен неизменяемый raw MCQ score 51/60; вместе с Math source binding — 71/80.
8. После оценки была диагностирована перестановка Unit 2/Unit 3. Повторная независимая проверка официального PDF подтвердила восемь errata и невалидность q24. Raw score оставлен без изменений; скорректированные числа опубликованы отдельными полями.

Во время MCQ запуска случайно стартовал второй локальный процесс в другой staging-каталог. Его command line был проверен, процесс остановлен, staging не переиспользовался и не финализировался. Основной процесс и его output не менялись; оба процесса видели только opaque input и frozen public source bundle. На blindness и итоговые байты основного запуска этот incident не повлиял.

## Цепочка хешей и pins

### Protocol и официальные PDF

| Объект | SHA-256 |
|---|---|
| selection manifest | `7a6ecbe1579790541bc0c0c36a9141941c2c0cf8f65dd3cd5b6a687bae8c5fa5` |
| sealed gold payload | `55e58007a1f30f4ee1c7f10c6f9c4bdd6b87be861046c13c7496a709a4fb7bcc` |
| Math12 PDF | `16d650177e62dc04b9a8b42fd7aafc3c1a8a38ec8c7040f92d5a26b120cde548` |
| Biology9 PDF | `717548090c5bece21242fab41a3dad26aa43031f5a73d4191538903ab3ec4ea0` |
| Physics12 PDF | `0957cb2a74ed46d6b7c3a3165863e03b5a7206cdf444f6ad8ecf6a13179a6307` |
| Physics official key index | `1ef19fe8e56b0ba97307d8c22abe8a8c8d3a72baf42eb5f7a22eb069b6a5e8e0` |
| Physics key page 264 render | `59ff5d209b353f1f1ac879a5b3b516ba65122672eb30e95890d255fd4473a3f8` |

### Math12

| Объект | Commit / SHA-256 |
|---|---|
| resolver code | `cb13b66156126b7a4ffab74450b62128b7b2ec92` |
| resolver freeze projection | `2f9068d4822351fb8ea7183bfb816d6bf5f01e0aa548754127ae3f2426f0b24e` |
| opaque runner code | `6f2be0624bdfa825c2f08a828bf9c83e290a11f6` |
| runner freeze projection | `5cde2115ea4b01e8fff7380fe39a49effff4dc1613bd22d20c69363d7c1e34e5` |
| evaluator code | `c7efa65ca8518c7ad918eb2f9bad7804e854dd73` |
| evaluator freeze projection | `3bb06d9f162a0397e8352516018f3fd5f31916974214c6ea87505f257e54f36d` |
| clean opaque input | `e0ee22d58187fbe11c951ef8153ad825734f83d50e127a1746f4e38649f11960` |
| clean run manifest | `5b7f807cec193cf709ae5988567188ec8a96075e737faec1dbb6822d6a303955` |
| clean artifact projection | `7b5e52ea9d3aa5f23d3aa0e5e66a6301611a20c53e7823e01506d4c502f1a36c` |
| Math clean+stress output seal file | `b7419a76dbffbd1e45daffb6f4476bf13cb4e8c4099dcd1c4be74fcb1ccc60bc` |
| Math output seal projection | `93fe423ee744db667002b93ffb8d652abc855f471116f86a9fa40b51d2955f59` |

### Biology/Physics MCQ

| Объект | Commit / SHA-256 |
|---|---|
| clean launch commit | `99963dc0225e2f5ff855d404004cceafadc3ddd5` |
| v1.1 freeze manifest | `ceab4465cd2aeb36603471a5c053e43d42a633f8ddb9d8a6864f36eec1229f16` |
| v1.1 freeze projection | `4d20ce0a35bc4d5ff494b0916f33a4e4c70e0535a4ddb414d133394087e98bb0` |
| v1.1 code projection, 16 files | `f9f63eb50a2087b95385bd4c9644e547e9ae1698cc6811712b692aae26779828` |
| external pins file | `6a00d94919b1af317ee7dabf41c861aa68c1bd93955d71c9e935e0ca096d888d` |
| opaque MCQ input JSONL | `b719426c45896d6fe90b22a75940b305b9ba840c1de66a666af7bfda3387b053` |
| source run manifest | `6ea0feb3a24c8766d200f2b45e77dc0b6f463be995e90ec49454314d0eb3afcb` |
| source results | `2b03610c385b2def0612d075145c366860e0aa909714e71c93ac7b2a81862d9e` |
| sealed predictions | `c568be738844730674e3d9e20f40a53ae47372aeca289f16fdbc0fd1861d97a5` |
| pre-map artifacts projection | `76359e69dc0dd01b8af56cc5f29af849b656838b6a47634e7b3b983017ba455f` |
| pre-map output seal file | `4441255a7aac6bdcf83e3f4af3f348eb9ad7e78afb199bf1c4f4a321ddb5fe9b` |

Runtime запуска: Python 3.12.13, pdfplumber 0.11.9, NumPy 2.5.1, OpenCV 5.0.0, Poppler 26.05.0. `pdftoppm.exe` был закреплён SHA-256 `742cbbd9a00931ad16c6618410bc40471375d639a45c61c1d86f3dcfc54b6388`.

## Post-map compatibility и граница blind-доказательства

Первый MCQ evaluator ожидал prefix `h80-physics12`, а frozen protocol использовал `h80-phys12`. После map был добавлен узкий compatibility wrapper только для subject dispatch. Он не изменял predictions, gold comparison или формулу accuracy. Это post-map изменение evaluator plumbing и поэтому явно записано в отчёте; байты source outputs и pre-map evaluator были уже включены в seal.

Stress Math12 также оценивался post-map compatibility adapter: его outputs были запечатаны до map, но сам adapter не был preregistered. Clean Math12 20/20 оценён заранее frozen evaluator и именно clean-результат входит в composite. Stress 20/20 — отдельная synthetic-robustness проверка, не новая строка Holdout80.

Криптографические хеши доказывают, что опубликованные input/output и код не были незаметно заменены после фиксации. Они сами по себе не доказывают отрицательный факт «private map/gold никто никогда раньше не открывал». На рабочем Windows-хосте не был заранее включён полный OS-level file-access audit и запуск не выполнялся в отдельной аппаратно или организационно изолированной среде. Поэтому blindness здесь подтверждается хронологией, clean checkout, launch attestation, process discipline и pre-map seals, но не независимым ретроспективным журналом каждого файлового чтения. Для следующего claim уровня внешнего benchmark нужен isolated runner, deny-by-construction доступ к gold и append-only audit log.

## Границы результата

Можно утверждать:

- на 80 новых заданиях из трёх уже известных учебников raw source-evidence composite равен 71/80;
- 59/59 принятых валидных MCQ совпали с официальными PDF-ключами;
- Math12 resolver правильно связал 20/20 clean и 20/20 synthetic-stress inputs с activity;
- независимая проверка подтвердила восемь переставленных Unit 2/Unit 3 labels и open-response q24;
- после явного erratum получается 79/80 protocol-inclusive и 79/79 valid-only.

Нельзя утверждать:

- что end-to-end QA accuracy системы стала 0,9875 или 1,0;
- что модель решила 20 математических заданий правильно;
- что проверены новые книги, языки, редакции или реальные фотографии;
- что исправленный score заменяет frozen raw 0,8875;
- что отсутствие прежнего доступа к gold доказано полным OS access log;
- что V7 QA 0,883212 и Holdout80 source evidence являются одной метрикой.

## Что опубликовано

Этот каталог содержит только агрегированный отчёт и обезличенную машинно-читаемую сводку. Здесь нет изображений holdout, source-address map, sealed-gold JSONL, key-page payloads, predictions, per-input certificates или строк evaluation. Публичный protocol остаётся в [`../maxim_holdout80_protocol_v1_20260808`](../maxim_holdout80_protocol_v1_20260808), а замороженные source-компоненты — в соседних versioned report directories.
