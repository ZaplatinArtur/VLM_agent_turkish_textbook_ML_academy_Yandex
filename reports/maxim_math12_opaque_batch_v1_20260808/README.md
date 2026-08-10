# Math12 opaque batch runner v1

Это служебный batch-слой над Math12 resolver v1.1 из commit
`cb13b66156126b7a4ffab74450b62128b7b2ec92`. Resolver и его source artifacts
заморожены до любого unseen run. Runner ничего не подбирает по
результатам benchmark и не меняет пороги resolver. Его задача проще: безопасно
прогнать каждый opaque image, сохранить проверяемые source-certificate и
official-source solution record, а затем собрать решение на уровне одного
многостраничного входа.

## честный scope

Runner не принимает task id, ожидаемый activity, benchmark answer, gold,
correctness или score. Из opaque JSONL используются только `input_id`,
`schema_version` и список пар `path`/`sha256`. Prompt, язык и формат ответа не
участвуют в source matching. Текст official solution извлекается frozen
resolver из закрепленного PDF после принятого визуального сертификата; это
источник учебника, а не benchmark gold и не измерение QA accuracy.

Этот commit фиксирует код и тесты до полного запуска. Clean- и stress-batch на
20 входах каждый при подготовке этой заморозки **не запускались**. Поэтому в
этой папке нет результата, accuracy или заявления о переносе качества.

Stress JSONL содержит синтетически ухудшенные версии тех же изображений. Его
будущий результат можно интерпретировать только как synthetic robustness
source-resolution check, а не как независимый benchmark.

## fail-closed правила

1. JSONL сначала полностью проверяется, до первого вызова resolver.
2. Absolute path, drive path, `..`, symlink escape за `asset-root`, отсутствующий
   файл и несовпадение SHA-256 отклоняют весь запуск.
3. Дубли `input_id`, resolved path или SHA-256 изображения отклоняются, чтобы
   один пример нельзя было посчитать дважды.
4. Байты всех перечисленных assets читаются один раз после проверки hash; эти
   же байты передаются resolver, поэтому между check и use нет подмены файла.
5. Обрабатываются все images каждого input. Исключение на любом image дает
   input-level `abstain_incomplete_image_processing`.
6. Каждый созданный certificate до агрегации проходит публичный
   `verify_math12_source_certificate`: replay всех evidence, точное совпадение
   frozen thresholds/runtime profile, inventory/render pins и hashes страниц.
7. Input принимается только при наличии хотя бы одного accepted certificate и
   единственном activity среди всех accepted certificates. Ноль accepted дает
   `abstain_no_accepted_certificate`, разные activity дают
   `abstain_conflicting_accepted_activities`.
8. Abstained images не голосуют и не создают ложный конфликт. Accepted и
   abstained pages могут дать accepted input, если все реально accepted pages
   указывают на один activity.
9. Ошибка extraction отдельно фиксируется в `solution_record_status`. Она не
   переписывает source-binding certificate и не превращается в иной activity.
10. Output directory должен не существовать. Запись идет во временную соседнюю
   папку и публикуется одним rename, чтобы не смешивать два запуска.
11. В output нет времени запуска, latency и абсолютных путей, поэтому одинаковые
    входы и одинаковые функции дают побайтно одинаковый bundle.

## выходные файлы

- `results.jsonl` — один input-level record, внутри ordered image records с
  `input_id`, image SHA-256, resolver reason, certificate status/path/hash,
  выбранной source page/activity и solution-record status/path/hash;
- `certificates/<input_id>/image-*.json` — полный frozen resolver certificate
  для каждого image, если resolver завершился;
- `solution_records/<input_id>/image-*.json` — официальный source record только
  для accepted certificate;
- `run_manifest.json` — pins исходного JSONL, inventory, render manifest, PDF и
  SHA-256 всех созданных artifacts. Здесь есть counts accepted/abstained, но нет
  correctness и нет score.

При конфликте per-image solution records сохраняются для аудита, однако
input-level `selected_activity_number` остается `null`: runner не выбирает
удобный для результата вариант.

## запуск после независимой проверки freeze

Из корня репозитория:

```powershell
python scripts\run_math12_opaque_batch_v1.py `
  --input-jsonl <OPAQUE_JSONL> `
  --asset-root <ROOT_FOR_RELATIVE_ASSET_PATHS> `
  --inventory reports\maxim_math12_activity_source_v1_20260808\inventory.json `
  --render-manifest reports\maxim_math12_activity_source_v1_20260808\render_manifest.json `
  --render-page-root <DIRECTORY_WITH_PINNED_PAGE_PNGS> `
  --pdf <PINNED_OFFICIAL_MATH12_PDF> `
  --output-dir <NEW_OUTPUT_DIRECTORY>
```

Clean и stress следует запускать в разные новые output directories, не меняя
inventory, render manifest, PDF, resolver или этот runner между прогонами.
Сначала нужно зафиксировать SHA-256 output bundle, и только после этого можно
передавать его независимому evaluator. Сам runner evaluator не вызывает.

## проверки

```powershell
python -m pytest tests\test_math12_opaque_batch.py -q
```

Unit tests покрывают согласие accepted pages, конфликт activity, отсутствие
accepted certificate, path traversal, duplicate input IDs/assets, hash mismatch
и побайтную детерминированность двух независимых output directories. Отдельный
тест доказывает, что ошибка strict verifier сохраняет сырой certificate для
аудита, но принудительно дает input-level abstain. Тесты используют маленький
fake resolver и не открывают holdout assets.

Frozen v1.1 требует Python 3.12.13. Проверенная локальная команда:

```powershell
$p1=(Resolve-Path 'tmp\maxim_math12_test_pkgs').Path
$p2=(Resolve-Path 'tmp\portfolio_official_sources\python_pkgs').Path
$env:PYTHONPATH="$p1;$p2"
$py='C:\Users\kmaxc\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -m pytest tests\test_math12_opaque_batch.py tests\test_math12_activity_source.py -q
```

Фактический результат: `25 passed, 1 skipped`. Из них runner unit tests:
`11 passed`. Skip — уже существующая opt-in real-PDF integration; реальный
dev-5 rebuild и extraction отдельно закреплены resolver freeze v1.1. Обычный
локальный `python` сейчас имеет версию 3.13.13 и правильно отклоняется runtime
preflight, поэтому для реального запуска нужно использовать указанный pinned
3.12.13 executable.

`freeze.json` связывает SHA-256 runner module, CLI, tests и этого README с
frozen resolver commit. `FREEZE_SHA256.txt` содержит SHA-256 самого
`freeze.json`.
