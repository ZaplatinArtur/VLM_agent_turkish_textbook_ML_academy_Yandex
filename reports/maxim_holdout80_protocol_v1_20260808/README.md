# Holdout80: честная проверка на новых вопросах из тех же книг

Этот набор собран для отдельной проверки текущего пайплайна на 80 ранее не использованных заданиях. Это **same-book new-question holdout**: вопросы новые, но книги уже были доступны проекту. Поэтому результат нельзя называть `book-disjoint`, переносом на новую книгу или проверкой на новом домене.

**Статус на момент заморозки этого protocol README: protocol-ready / no model score.** Ниже сохранено описание pre-run процедуры. Поздний task-ID-free source-evidence запуск завершён: frozen raw 71/80, отдельно принятый protocol erratum 79/80, valid-only 79/79. Это source lookup/binding, а не end-to-end QA. Финальный публичный отчёт с хронологией и ограничениями: [`../maxim_holdout80_final_source_evidence_20260809/REPORT_RU.md`](../maxim_holdout80_final_source_evidence_20260809/REPORT_RU.md).

Главный принцип: сначала выбираются страницы и номера заданий без ответов, выборка фиксируется хешем, и только после этого отдельный скрипт открывает официальные страницы с ключами. Результаты исходного бенчмарка при выборе не читались.

## Что вошло

| Источник | Формат | Количество | Как выбиралось |
|---|---:|---:|---|
| Matematik 12 Beceri Temelli Etkinlik Kitabı | целая многочастная активность | 20 | фиксированные квоты по 7 разделам, затем SHA-256-ранг |
| 9. Sınıf Biyoloji Ders Kitabı | ABCDE | 30 | все тестовые вопросы трёх разделов |
| 12. Sınıf Fizik Ders Kitabı | ABCDE | 30 | по 5 SHA-256-ранжированных вопросов из каждого из 6 разделов |

У Math12 одна запись — это целая активность со всеми её страницами и всеми пронумерованными подпунктами. Это сложнее одного ABCDE-вопроса. Для неё нужен слепой ручной разбор по официальному решению; нельзя посчитать итоговую accuracy только по 60 автоматически проверяемым вопросам и назвать её результатом на всех 80.

Выбранные Math12 activity ID: `12, 14, 15, 18, 23, 26, 30, 32, 36, 37, 39, 53, 57, 62, 73, 77, 87, 90, 91, 93`.

Из этой книги заранее исключены пять активностей, привязанных к исходному бенчмарку:

| benchmark task | activity | страницы вопроса |
|---|---:|---:|
| val_0054 | 3 | 7 |
| val_0055 | 17 | 24 |
| val_0056 | 88 | 119 |
| val_0057 | 43 | 60 |
| val_0058 | 31 | 40–41 |

Полный машинно-читаемый индекс всех 95 активностей лежит в `math12_family_question_index.jsonl`. Он включает диапазоны страниц, хеш текста и флаг исключения. `source_inventory.json` описывает все три источника и выбранные подмножества.

## Freeze и sealed gold

Публичная часть:

- `selection_manifest.jsonl` — 80 заданий, пути к изображениям, страницы, структурные страты и dedup-метрики; ответов там нет;
- `freeze.json` — хеш manifest, хеши трёх PDF, seed и точное описание типа split;
- `assets/questions/` — только страницы вопросов; они создаются локально и исключены из Git;
- `math12_family_question_index.jsonl` — все 95 question-side activity ranges;
- `known_math12_benchmark_bindings.jsonl` — пять явных исключений.

Закрытая часть, которая не входит в публичный Git bundle:

- `sealed/sealed_gold.jsonl` — буквы для 60 ABCDE и официальные решения для 20 Math12;
- `sealed/key_pages/` — изображения официальных страниц ключа;
- `sealed/gold_seal.json` — хеш sealed gold и хеш manifest, к которому он относится.

В этой директории опубликован только `sealed/gold_seal.json` с hash/count. Сам `sealed_gold.jsonl`, страницы ключей, predictions и evaluation закрыты правилами `.gitignore`. Это сделано специально, чтобы resolver можно было запускать без доступа к ответам.

Зафиксированный manifest SHA-256:

`7a6ecbe1579790541bc0c0c36a9141941c2c0cf8f65dd3cd5b6a687bae8c5fa5`

Текущий sealed gold SHA-256:

`55e58007a1f30f4ee1c7f10c6f9c4bdd6b87be861046c13c7496a709a4fb7bcc`

У activity 73 официальный PDF отдаёт многоколоночный текст в неправильном порядке. Для неё в sealed gold сохранена вся официальная страница ключа, а не подозрительно короткий автоматически вырезанный фрагмент. У activity 90 решение продолжается на следующей странице; обе страницы включены. Это отмечено в поле `reference_extraction_mode`.

## Как проверялось отсутствие пересечения

Для всех 80 записей текст question-side страниц сравнивался с OCR каждого из 274 исходных benchmark input. Использованы Unicode-нормализованные триграммы слов, containment и Jaccard. Задание проходило только при `containment < 0.65` и `jaccard < 0.50`. Максимальный наблюдавшийся containment после отбора — `0.395833`. Пять точно сопоставленных Math12 activities исключены независимо от текстовой метрики.

Хеш каждого нового question asset также сравнивался с 274 исходными image SHA-256: точных совпадений нет. Полноценный perceptual crop audit не заявляется, потому что исходные 274 bitmap-файла не сохранены под стабильными task path. Это ограничение записано в `dedup_audit.json`; его нельзя замалчивать.

## Что уже измерено, а что нет

`certificate_audit.json` фиксирует локальную проверку целостности до отделения публичного bundle: 80/80 question assets, 80/80 official-key records, соответствие manifest ↔ sealed gold и их хеши. Это не качество модели.

`retrieval_roundtrip.json` запускает существующий локальный BM25 для 60 заданий из Biology9 и Physics12. Запрос берётся из чистого PDF-текста выбранного вопроса, поэтому это оптимистичная проверка «может ли индекс вернуть исходную страницу», а не end-to-end OCR/RAG/QA. Math12 Beceri в текущем BM25 отсутствует и честно считается `unindexed`, а не промахом модели.

Ни одна accuracy текущего solver на Holdout80 пока не измерена. В частности, числа `0.88`, `0.90` или старые `0.854` к этому набору не относятся. Отчётным результатом станет только frozen run с prediction JSONL и слепой ручной оценкой всех 20 Math12 activities.

## Воспроизведение

Проверка публичных frozen metadata из корня репозитория `VLM_agent_turkish_textbook_basic_rag`:

```powershell
python -m pytest reports/maxim_holdout80_protocol_v1_20260808/tests/test_protocol.py -q
```

В публичном checkout эта команда проверяет только tracked metadata; проверки изображений, opaque resolver inputs и sealed gold будут явно помечены `skipped`, потому что эти данные намеренно не входят в Git. Полная проверка выполняется после локального rebuild с PDF и закрытым gold.

Пути внутри `freeze.json` и исходного `selection_manifest.jsonl` сохраняют logical root первого замороженного запуска `reports/holdout80_20260808`. Переписывать их нельзя: это изменит bytes manifest и сломает freeze SHA. portable launcher создаёт новый согласованный report root; внешнему коду не следует механически присоединять старый logical path к каталогу публичного bundle.

Для полного rebuild нужен локальный Math12 PDF в `tmp/remaining_official_source_audit/pdfs`, OCR 274 исходных input и корпус `full_2026_07` с Biology9/Physics12. Если корпус лежит в соседнем проекте `VLM`, portable launcher найдёт его сам; иначе задаётся `VLM_HOLDOUT_TEXTBOOK_ROOT`.

```powershell
$env:VLM_HOLDOUT_REPORT_DIR = (Resolve-Path .).Path + '\reports\maxim_holdout80_protocol_v1_rebuild'
python reports/maxim_holdout80_protocol_v1_20260808/tools/build_selection.py build
python reports/maxim_holdout80_protocol_v1_20260808/tools/seal_gold.py
python reports/maxim_holdout80_protocol_v1_20260808/tools/run_diagnostics.py
python -m pytest reports/maxim_holdout80_protocol_v1_20260808/tests/test_protocol.py -q
```

`tools/build_selection_frozen.py` — точная реализация, хеш которой записан в исходном `freeze.json`. `tools/build_selection.py` — portable launcher: он только находит локальные корни данных и вызывает frozen implementation. Для исторической v1 повторный `build` поверх опубликованной директории запрещён; rebuild идёт в новую директорию.

Prediction JSONL содержит одну строку на task:

```json
{"task_id":"h80-bio9-u1-q01","answer":"D"}
```

Для Math12 слепой проверяющий после сравнения всех подпунктов с официальным решением добавляет `manual_score: 0` или `1`. Пустая форма находится в `manual_scoring_form.jsonl`.

После полного прогона:

```powershell
python reports/maxim_holdout80_protocol_v1_20260808/tools/evaluate.py path/to/predictions.jsonl --output $env:VLM_HOLDOUT_REPORT_DIR\evaluation.json
```

Evaluator отдельно показывает accuracy на 60 MCQ и не выпускает общую accuracy, пока нет 20 ручных оценок и полного набора task ID.
При missing, duplicate, unknown task ID или неполной ручной разметке поля `overall.correct`, `overall.total` и `overall.accuracy` остаются `null`, `overall.reportable=false`, а процесс завершается с кодом `2`. Это сделано fail-closed: потребитель не должен случайно опубликовать частичную цифру, проигнорировав статус.
