# Math12 official-source adapter v1.1

## коротко

это исправленная и повторно собранная версия source-only компонента для официального турецкого учебника `Matematik 12 Beceri Temelli Etkinlik Kitabı`. компонент принимает изображение, сравнивает его со всеми 127 содержательными страницами книги, при достаточной геометрической уверенности определяет activity и извлекает только относящийся к ней фрагмент официального решения.

v1.1 появился после независимого аудита freeze-коммита `9db67f2`. аудит нашёл не подгонку метрики, а четыре инженерные проблемы в границе официального ответа и воспроизводимости:

1. координата заголовка `Etkinlik No.: N` бралась из первого слова и округлялась. из-за дробных координат PDF собственный заголовок иногда терялся, а заголовок следующей activity иногда попадал в текущий ответ;
2. JSON-loader проверял собственные SHA, но его название создавало впечатление полного replay. сохранённое решение не пересчитывалось из всех evidences в отдельном authoritative API;
3. Math12 допускал изменение части runtime profile. даже более строгая настройка после freeze была бы уже другой политикой;
4. версия `pdfplumber` и точное Python-окружение не были включены в обязательный CLI preflight.

в v1.1 все четыре проблемы исправлены. пять dev-изображений заново прогнаны по всем 127 страницам, сертификаты не переупаковывались из старых решений. source binding остался 5/5. все пять official solution records извлечены заново; в каждом есть собственный заголовок и ни в одном нет заголовка следующей activity.

это по-прежнему не accuracy result. Math12-семейство было выбрано после просмотра пяти dev-примеров. поэтому 5/5 является dev replay и проверкой адресации источника, а не оценкой переноса. holdout, unseen inputs, gold answers и scorer в этой пересборке не читались и не запускались.

## источник и inventory

- официальный PDF: 182 физические страницы;
- SHA-256 PDF: `16d650177e62dc04b9a8b42fd7aafc3c1a8a38ec8c7040f92d5a26b120cde548`;
- размер PDF: 36 032 401 байт;
- оглавление activities: страницы 2–3;
- задания: страницы 4–130;
- официальный ключ: страницы 131–179;
- activities: ровно 95;
- диапазоны заданий без дыр и пересечений покрывают 127 страниц;
- inventory projection SHA-256: `2bf810400de721bb47f02974df47be44e93f87f6796eca09c7375d42c0398b2c`;
- inventory file SHA-256: `878ddb53cbf1f6ddfe5b332dc9f27a882caea557dccfbab3453546e106637ac5`.

inventory строится только из PDF-native слов оглавления и официального ключа. benchmark task id, ожидаемый номер activity, ответ и correctness при построении не используются.

## исправленная граница официального решения

у ключа двухколоночная вёрстка. логический порядок идёт по левой колонке, затем по правой, затем по следующей странице. поэтому каждый ответ задаётся полуинтервалом:

```text
[key_start, key_end_exclusive)
```

заголовок состоит из трёх PDF-слов: `Etkinlik`, `No.:`, `N`. теперь его верхняя координата вычисляется так:

```text
marker_top = floor(min(top_Etkinlik, top_No, top_N), 4 decimals)
```

используется именно нижнее округление минимума, а не `round(first.top, 4)`. поэтому locator гарантированно расположен не ниже любого слова собственного заголовка. та же координата следующего заголовка используется как исключающая правая граница и гарантированно расположена не выше его слов. regression test отдельно проверяет три условия: собственные три слова включены, следующий заголовок исключён, его body также исключён.

на реальном PDF результат такой:

| input | activity | собственный header | header следующей activity | длина текста |
|---|---:|---|---|---:|
| `val_0054` | 3 | есть | отсутствует | 1 875 |
| `val_0055` | 17 | есть | отсутствует | 1 950 |
| `val_0056` | 88 | есть | отсутствует | 2 288 |
| `val_0057` | 43 | есть | отсутствует | 751 |
| `val_0058` | 31 | есть | отсутствует | 930 |

## strict replay

`load_math12_source_certificate(path)` теперь явно является parse/self-pin loader. он проверяет JSON schema, evidence projection и certificate projection, но один он не делает сертификат доверенным. это специально записано в docstring.

authoritative API:

```python
verify_math12_source_certificate(inventory, render_manifest, certificate)
```

он заново проверяет:

1. projection всего inventory;
2. source identity: document id, PDF SHA и inventory SHA;
3. projection portable render manifest;
4. точный render profile;
5. присутствие всех 127 страниц ровно один раз;
6. размер и SHA каждого внешнего PNG;
7. связь `evidence.rendered_page_sha256` с конкретным PNG manifest;
8. SHA полного списка из 127 evidences;
9. точное равенство thresholds и SIFT runtime frozen-профилю;
10. повторный вызов decision policy на всех evidences;
11. полное равенство replayed decision сохранённому decision;
12. итоговую certificate projection.

самосогласованно переподписанный post-hoc decision verifier отвергает. аналогично отвергается evidence, перепривязанный к другому render SHA. `extract_official_solution(...)`, команда `extract-solution`, dev audit и команда `verify-certificate` используют strict verifier, а не доверяют loader.

## portable render manifest

в git хранится `render_manifest.json`, но не 127 PNG и не PDF. в manifest находятся только переносимые имена `page-NNN.png`, размеры и SHA. внешняя директория с payload передаётся через `--page-root`; её расположение не входит в projection и не меняет freeze.

- render DPI: 144;
- режим: grayscale PNG;
- Poppler: 26.05.0;
- страниц: 127;
- render manifest projection SHA-256: `d8a7c55d11a5d5affb0368d39ebda5e3d4c6f5fd79d1b6ae8f367af827846b66`;
- tracked manifest file SHA-256: `903039725bb5c9e3894a928fa124153b0c2b5ec4b81fb06ae615344f49eff26a`.

loader принимает tracked manifest и `--page-root`, после чего всё равно читает и хеширует каждый PNG. отсутствие payload или несовпадение хотя бы одного байта приводит к fail closed.

## exact frozen runtime

разрешён только один профиль:

| часть | версия или значение |
|---|---|
| Python | 3.12.13 |
| pdfplumber | 0.11.9 |
| NumPy | 2.5.1 |
| OpenCV | 5.0.0 |
| Poppler | 26.05.0 |
| render | 144 dpi, grayscale PNG |
| SIFT nfeatures | 12 000 |
| SIFT contrast / edge | 0.02 / 12.0 |
| Lowe ratio | 0.72 |
| RANSAC reprojection | 4 px |
| RANSAC max iterations | 5 000 |
| RANSAC confidence | 0.999 |
| RNG seed | 19 870 511 |

thresholds также равны frozen profile, а не просто не слабее его:

| gate | значение |
|---|---:|
| min good matches | 50 |
| min inliers | 40 |
| min inlier ratio | 0.65 |
| min task hull fraction | 0.30 |
| max median reprojection error | 1.0 |
| min mapped-inside fraction | 0.98 |
| max scale anisotropy | 1.15 |
| min rank-score margin | 10.0 |
| min rank-score ratio | 5.0 |

любой override, в том числе более строгий, отвергается direct API. если нужен другой профиль, это должна быть новая версия и новый freeze до просмотра новых результатов.

рабочий preflight:

```powershell
$env:PYTHONPATH=(Resolve-Path 'tmp\portfolio_official_sources\python_pkgs').Path+';'+(Resolve-Path 'src').Path
& 'C:\Users\kmaxc\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  scripts\math12_official_source_adapter.py preflight `
  --pdftoppm 'C:\Users\kmaxc\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe'
```

фактический результат preflight: все пять версий совпали с pins.

## повторный dev-5 прогон

каждый сертификат заново вычислялся из исходного PNG, SIFT/RANSAC и всех 127 content pages. expected activity использовался только позднее в отдельном audit-скрипте.

| input | page | activity | key span | best / runner rank | checks |
|---|---:|---:|---|---:|---:|
| `val_0054` | 7 | 3 | `131/right/216.0852` → `132/left/382.8293` | 389.703 / 4.206 | 16/16 |
| `val_0055` | 24 | 17 | `139/left/78.2717` → `139/right/78.2717` | 486.347 / 1.910 | 16/16 |
| `val_0056` | 119 | 88 | `174/left/77.4737` → `175/left/78.3347` | 415.434 / 4.143 | 16/16 |
| `val_0057` | 60 | 43 | `152/right/431.4961` → `153/left/78.1048` | 422.585 / 4.701 | 16/16 |
| `val_0058` | 41 | 31 | `145/right/430.7505` → `146/right/78.1048` | 200.172 / 38.638 | 16/16 |

последний случай остаётся честно отмеченным как наиболее близкий к ratio floor: примерно 5.18 при минимуме 5.0.

dev audit projection SHA-256: `cbb043f453e65dce9a5ec93b466277b1e80fbd9484f8d07d239354f8540d39d8`.

## official solution records

| input | activity | solution text SHA-256 | answer-bound projection SHA-256 |
|---|---:|---|---|
| `val_0054` | 3 | `e1016f8034a48b57db2d526f201357e9802b5e1b241ffd9068265131dafa80fe` | `3286182fa5505d0b82d0457d06d213e83f83a4a21f0387ceeb373867095ad4dd` |
| `val_0055` | 17 | `ec80ed6b09d2820acca3cbb8a98253e7f2b15dde506b31ced9a01aa9a66bf297` | `351014c357e707b2c6431d70334585359ba75d3a260cd1908da0f034a87ea548` |
| `val_0056` | 88 | `a3e17117e9adebd42d9ba0dd79a1b48d4bb7eb71f4df7f824cf3def38295e8bd` | `b188926886b9000b3f6b5fd204ad80ff81cc67ae83759076d76804d82e71fcf0` |
| `val_0057` | 43 | `7bb9fbc31b4816aa5cba7dfc5d0a2b1fd44c3de5fcb0ca0fb315763c93b14721` | `9c855646b84ad7326bbb322bc3aa0b10062ce76468fd588ebf3295c5ee5ba2f5` |
| `val_0058` | 31 | `0f4bfb5434238a2995354069cada50e02fee186a658c3f657892b0041c151ef4` | `2a9fe7877fc8d0c6ba19e3902157d8c57d1ca1c5c36cbfb97dc0623a464d4617` |

это PDF-native evidence для downstream solver/judge, а не автоматически правильный короткий ответ. формулы в PDF могут извлекаться текстовым движком неидеально. component scope не содержит gold, correctness или score.

## команды

```powershell
# resolve arbitrary image; ожидаемого activity здесь нет
python scripts\math12_official_source_adapter.py resolve `
  --inventory reports\maxim_math12_activity_source_v1_20260808\inventory.json `
  --render-manifest reports\maxim_math12_activity_source_v1_20260808\render_manifest.json `
  --page-root <directory-with-page-NNN.png> `
  --image <image.png> `
  --output <certificate.json>

# отдельно проверить сохранённый сертификат полным replay
python scripts\math12_official_source_adapter.py verify-certificate `
  --inventory reports\maxim_math12_activity_source_v1_20260808\inventory.json `
  --render-manifest reports\maxim_math12_activity_source_v1_20260808\render_manifest.json `
  --page-root <directory-with-page-NNN.png> `
  --certificate <certificate.json>

# извлечь официальный span только после strict replay
python scripts\math12_official_source_adapter.py extract-solution `
  --pdf <math12.pdf> `
  --inventory reports\maxim_math12_activity_source_v1_20260808\inventory.json `
  --render-manifest reports\maxim_math12_activity_source_v1_20260808\render_manifest.json `
  --page-root <directory-with-page-NNN.png> `
  --certificate <certificate.json> `
  --output <official-solution.json>
```

## проверки v1.1

- unit/fail-closed/tamper/runtime/marker: `14 passed, 1 skipped`;
- opt-in real-PDF inventory integration: `1 passed, 14 deselected`, 149.32 s;
- полный inventory: 95/95 activities;
- portable render manifest: 127/127 page pins;
- full visual recomputation: 5/5 accepted;
- dev source-address alignment: 5/5;
- official extraction: 5/5 projection hashes;
- own marker fully present: 5/5;
- next marker absent: 5/5;
- Python compile и CLI help/preflight: passed;
- GPU: не использовался;
- сеть: не использовалась;
- holdout/unseen/gold/scorer: не читались и не запускались.

старый `freeze_candidate_manifest.json` относится к v1 и сохранён как исторический артефакт commit `9db67f2`. актуальные code/artifact/runtime pins находятся в новом `freeze_manifest_v1_1.json`. до отдельного frozen transfer run нельзя называть этот dev replay переносом или новым benchmark score.
