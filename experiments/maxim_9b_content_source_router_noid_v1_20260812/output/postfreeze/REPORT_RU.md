# Content-only / no-ID quick win для Maxim274

## Итог

Строгая ветка B, начатая от архивного `base240`, получила **251/274 =
0.916058**. Парное сравнение с `base240`: **11 исправлений, 0 регрессий**.
Это честный прирост no-ID overlay: функция выбора видит только OCR-текст,
`answer_type` и `input_mode`; `task_id`, имя файла и SHA входной картинки в
неё передать невозможно.

Ветка A тоже получила 251/274, но это только диагностический артефакт. Её
база 249 уже была собрана старым exact-task-ID component routing, поэтому
`249 + 2` нельзя называть end-to-end no-ID результатом.

## Что импортировано

Новых книг в этой волне не скачивалось. Повторно использованы три уже
закреплённых официальных источника:

- MEB 7 Matematik, PDF SHA
  `dee64189589ba60431680f552edcb9613e620bdf8138c234daf16c7b02450219`;
- MEB Matematik 12 Beceri Temelli Etkinlik Kitabı, PDF SHA
  `16d650177e62dc04b9a8b42fd7aafc3c1a8a38ec8c7040f92d5a26b120cde548`;
- MEB English 10 Activity Workbook, PDF SHA
  `b495bf857f155bb6488de82b5d874a9e4df6307ffe50fe126266678be39cfdb1`.

В no-ID runtime source DB импортировано 17 content records: 7 MEB7, 5
Math12 и 5 English10. Из них 16 имеют официальный answer binding. SHA базы:
`c1307ba10d98287295b7f81bf7b415406bb32ff68c59aae34d57511d8eb1eae6`.
Частные книги и research records не использовались.

## Как выбирается ответ

- MEB7: global IDF top-1 внутри официальной книги, наблюдаемый номер
  упражнения и полный набор content/formula anchors из официальной страницы.
- Math12 и English10: IDF coverage по тексту официальной страницы с
  fail-closed порогами `score >= 0.65`, `margin >= 0.50`.
- Два общих CPU-инструмента: арифметика знаковых температур и LCM для
  целочисленных длин блоков.

Замороженный `decisions.jsonl` содержит 274 уникальных ordinal и ровно одно
действие на вход. В нём нет `task_id`, `controller_id`, benchmark ID,
input filename, image/content SHA или ответа. После решения ID используется
только для выравнивания с форматом solver/judge.

## Изменения и метрики

Материализовано 18 solver replacements: 16 official-source candidates и 2
tool candidates. Семь из 16 source rows уже были верны в base240, поэтому
это не 18 исправлений.

Строгая B исправила:

`val_0048`, `val_0050`, `val_0051`, `val_0054`, `val_0055`, `val_0056`,
`val_0057`, `val_0058`, `val_0182`, `val_0213`, `val_0216`.

Регрессий относительно base240: **0**.

- overall: 251/274 = 0.916058;
- Math: 119/139 = 0.856115;
- deterministic: 160/177;
- image judge: 91/97.

Ветка A относительно архивных 249 исправила только `val_0213` и
`val_0216`, регрессий нет.

## SHA

- candidate freeze file:
  `76a09995b1104b4b5fec67bb737e73e4a5b21032916f37a24a563118802a8a7c`;
- freeze projection:
  `ca463c2fd99512edda0931fe55d2d6e7ad3c0af52252d4940d9794625f65d46c`;
- decisions:
  `fd2b43204fe027369418c18ffb96bbf93be0fddadbc7d6ddd3fbad1c09f40db6`;
- strict-B solver:
  `f87f6ad41817c3d55fde5630781cd6f9f958350bfde72bdebeb8567b454c832a`;
- strict-B image judge:
  `f2375bf3cea7492e3947ea285ca5db9262a50a5c5abfc92f2dc68cc1386095bf`;
- strict-B metrics JSON:
  `8d55d663850873b8a493fc2507097ee9c94e3807e071b041c27344125861d674`;
- A metrics JSON:
  `a175f4a72b0b88304c6574d5d8e99a77d52c7402a651431d0c42455e46056c1d`.

## Что мешает 0.95

Осталось 23 ошибки: 20 Math, 2 Biology и 1 Chemistry; 17 deterministic и
6 image-judge. До 261/274 нужны ещё 10 чистых исправлений. Ровно десять из
оставшихся ошибок уже имеют проверенные CPU-kernels из прежней tool wave:
`val_0067`, `val_0086`, `val_0204`, `val_0205`, `val_0218`, `val_0230`,
`val_0232`, `val_0245`, `val_0253`, `val_0267`. Но прежняя волна выбирала их
по ID/SHA; перенос возможен только после реализации observable visual/OCR
parsers. Эти десять нельзя прибавлять к 251 до отдельной заморозки и прогона.

Независимый аудит текущего freeze запрошен у отдельного агента; его статус
будет закреплён отдельным post-freeze артефактом и не изменит frozen code.
