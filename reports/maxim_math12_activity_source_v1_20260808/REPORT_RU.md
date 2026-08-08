# честный отчёт по Math12 official-source adapter

## коротко

я сделал отдельный source-only компонент для одной официальной семьи учебников — турецкого `Matematik 12 Beceri Temelli Etkinlik Kitabı`. он умеет принять произвольную картинку, визуально сопоставить её со всеми 127 страницами содержательной части книги, определить activity и вернуть только привязанный к ней фрагмент официального решения. если геометрия или отрыв от второго кандидата недостаточны, компонент обязан отказаться от ответа.

на пяти уже известных dev-картинках компонент правильно восстановил пять заданных source bindings: 3, 17, 88, 43 и 31. это хороший инженерный sanity check, но не новый accuracy result. семейство Math12 было выбрано после просмотра этих dev-примеров, поэтому результат 5/5 нельзя выдавать за перенос на новые данные. gold answers, correctness, scorer и benchmark outcome в resolver не передавались.

## что именно является источником

- PDF: `tmp/remaining_official_source_audit/pdfs/matematik 12*.pdf`;
- SHA-256 PDF: `16d650177e62dc04b9a8b42fd7aafc3c1a8a38ec8c7040f92d5a26b120cde548`;
- размер: 36 032 401 байт;
- физические страницы: 182;
- оглавление activities: страницы 2–3;
- содержательная часть: страницы 4–130;
- официальные ответы: страницы 131–179;
- найдено ровно 95 activities и ровно 95 уникальных заголовков `Etkinlik No.: N`.

инвентарь лежит в `inventory.json`. его projection SHA-256: `154924c9495d47c2ff5b04da0cb5b420d61d22a13c7081ce1e48ca4d6be9aee2`.

## почему нельзя было хранить только одну страницу ответа

у книги двухколоночная вёрстка. логический порядок ответа иногда выглядит так: остаток левой колонки текущей страницы, затем правая колонка, затем начало следующей страницы до следующего заголовка activity. обычное `extract_text()` может перемешать этот порядок.

поэтому для каждой activity хранится не только грубый диапазон физических страниц, но и точный полуинтервал:

`key_start = (physical page, left/right column, top)`

`key_end_exclusive = (physical page, left/right column, top)`

слова входят в решение только тогда, когда их логический адрес находится внутри `[key_start, key_end_exclusive)`. страница в `key_page_end` может содержать заголовок следующего ответа: этот заголовок уже не входит в текущий фрагмент, потому что правая граница исключающая.

для каждого элемента записаны четыре независимых пина:

1. projection строки оглавления;
2. projection всех содержательных страниц activity;
3. projection точного фрагмента официального ключа;
4. joint binding projection, связывающий PDF, номер activity, диапазоны и три предыдущих пина.

контентные диапазоны всех 95 элементов без дыр и пересечений покрывают страницы 4–130.

## как работает resolver

```text
image bytes
  -> SHA-256 входного изображения
  -> проверка inventory + PDF identity
  -> проверка render manifest и SHA каждого PNG
  -> SIFT/RANSAC для каждой страницы 4..130 без shortlist
  -> сортировка только по source-image geometry
  -> строгие geometry gates
  -> строгий margin/ratio против второго кандидата
  -> выбранная страница попадает ровно в один content range
  -> activity record имеет все source projection pins
  -> accepted source-binding certificate
  -> точное извлечение [key_start, key_end_exclusive)
  -> answer-bound official-solution certificate
```

в production API нет `task_id`, ожидаемого номера activity, gold answer, score или outcome. ожидаемые номера пяти dev-примеров появляются только в отдельном audit-скрипте после того, как resolver уже записал сертификаты.

вызов `resolve_math12_image_bytes(image_bytes, inventory, render_manifest, ...)` обязательно перебирает все 127 содержательных страниц. неполный или дублированный sweep приводит к abstention.

## зафиксированный visual profile

использован существующий строгий safety floor проекта, без ослабления под эти пять строк:

| параметр | значение |
|---|---:|
| `min_good_matches` | 50 |
| `min_inliers` | 40 |
| `min_inlier_ratio` | 0.65 |
| `min_task_hull_fraction` | 0.30 |
| `max_median_reprojection_error` | 1.0 |
| `min_mapped_inside_fraction` | 0.98 |
| `max_scale_anisotropy` | 1.15 |
| `min_rank_score_margin` | 10.0 |
| `min_rank_score_ratio` | 5.0 |

runtime profile:

| параметр | значение |
|---|---:|
| render | Poppler 26.05.0, 144 dpi, grayscale PNG |
| OpenCV | 5.0.0 |
| SIFT `nfeatures` | 12 000 |
| contrast / edge | 0.02 / 12.0 |
| Lowe ratio | 0.72 |
| RANSAC reprojection | 4 px |
| RANSAC max iterations | 5 000 |
| RANSAC confidence | 0.999 |
| RNG seed | 19 870 511 |

render manifest содержит 127 PNG и имеет projection SHA-256 `6f6e406a1eaf20e43d1384bbceda157602dd81258081e9c6eebb066dd8aa708c`.

## dev-5 source-binding audit

| input | ожидаемая source activity, только для audit | выбранная content page | activity | точный key span | best / runner rank | результат |
|---|---:|---:|---:|---|---:|---|
| `val_0054` | 3 | 7 | 3 | `131/right/216.0853` → `132/left/382.8293` | 389.703 / 4.206 | accepted, match |
| `val_0055` | 17 | 24 | 17 | `139/left/78.2717` → `139/right/78.2717` | 486.347 / 1.910 | accepted, match |
| `val_0056` | 88 | 119 | 88 | `174/left/77.4738` → `175/left/78.3347` | 415.434 / 4.143 | accepted, match |
| `val_0057` | 43 | 60 | 43 | `152/right/431.4961` → `153/left/78.1049` | 422.585 / 4.701 | accepted, match |
| `val_0058` | 31 | 41 | 31 | `145/right/430.7505` → `146/right/78.1049` | 200.172 / 38.638 | accepted, match |

последняя строка тоже проходит frozen ratio floor, но она заметно ближе к границе: `200.172 / 38.638 = 5.18`, при минимуме 5.0. это надо сохранить как честный caveat, а не округлять до «огромного запаса».

projection SHA-256 итогового dev audit: `cff121df9c3c72c96595574a7165485e51787603559ee2c1b739fea032447ed1`.

## официальный solution record

`extract_official_solution(pdf, inventory, accepted_certificate)` сначала полностью переигрывает visual decision из всех 127 evidences, проверяет certificate SHA, PDF SHA, inventory SHA и binding projection. только после этого читается точный key span.

функция не пытается угадать canonical short answer. она возвращает PDF-native текст официального решения и его SHA. математические формулы в PDF иногда извлекаются текстовым движком неидеально, поэтому downstream-модель должна видеть этот текст как evidence, а не как гарантированно чистую символьную запись. при нарушении span hash или certificate replay функция падает fail-closed.

| input | activity | длина извлечённого текста, символов | solution text SHA-256 |
|---|---:|---:|---|
| `val_0054` | 3 | 1 859 | `9c1daf61de3423710542e9fb77f865b0e7736e799091678c98abc21ce9ec4b6c` |
| `val_0055` | 17 | 1 950 | `ec80ed6b09d2820acca3cbb8a98253e7f2b15dde506b31ced9a01aa9a66bf297` |
| `val_0056` | 88 | 2 271 | `6698dd4bda37a9553a62e29ee5fcff5039e5a6f4b1bbbd57167d9bb767e9632e` |
| `val_0057` | 43 | 768 | `a1e815a4aa822a1cfcdb553e22a933c7ad51c2b228fff8018a1a885f6f0552a1` |
| `val_0058` | 31 | 947 | `0ef8878007abcbde90ffff035e0a44d2f7278b02fbbf4f903bfd077c57475ec1` |

это уже usable evidence component: официальный solution можно передать solver/judge. но сам по себе этот audit всё равно ничего не говорит о том, улучшился ли final benchmark answer.

## где здесь возможен самообман

главный риск не в SIFT thresholds. главный риск — post-hoc выбор семьи источника. пять dev-картинок сначала позволили понять, что они относятся к одной Math12 книге, и только потом для неё был построен полный adapter. поэтому 5/5 отвечает на вопрос «можем ли мы воспроизвести source address для уже рассмотренной семьи», но не отвечает на вопросы:

- узнает ли frozen resolver новые картинки этой книги;
- как часто он будет abstain;
- не выберет ли уверенно неправильную страницу на новых crop/layout;
- улучшит ли официальный solution конечный ответ модели;
- как компонент поведёт себя на другой книге или другом языке.

в отчёте намеренно нет `accuracy=...`, `0.9` или заявления о победе над baseline. correctness не вычислялась.

## что нужно сделать до валидного результата

1. зафиксировать одним commit код, тесты, inventory, render manifest и frozen profile SHA до любого нового запуска;
2. один раз применить этот commit к независимо замороженным непросмотренным изображениям Math12;
3. сначала раскрыть только transfer source-binding audit: coverage, abstentions, wrong bindings и confidence margins;
4. не менять thresholds и family rules после просмотра результата; любые изменения — новая версия и новый holdout;
5. отдельно preregister downstream policy: когда официальный solution разрешает заменить ответ модели, когда только добавляется как context, когда надо abstain;
6. только после этого считать correctness/accuracy тем же неизменённым scorer;
7. для заявлений о мультиязычности повторить source inventory и hidden transfer audit минимум на независимых книгах каждого языка.

если frozen transfer audit провалится, честный вывод будет не «SIFT плохой», а «dev family replay не перенёсся». если source binding перенесётся, но accuracy не вырастет, проблема уже находится в extraction-to-solver handoff или в интерпретации решения, а не в retrieval.

## воспроизводимость

основной CLI: `scripts/math12_official_source_adapter.py`.

```powershell
# 1. построить inventory только из PDF
python scripts/math12_official_source_adapter.py build-inventory `
  --pdf <math12.pdf> `
  --output reports/maxim_math12_activity_source_v1_20260808/inventory.json

# 2. один раз отрендерить все 127 content pages
python scripts/math12_official_source_adapter.py render-pages `
  --pdf <math12.pdf> `
  --inventory reports/maxim_math12_activity_source_v1_20260808/inventory.json `
  --pdftoppm <pinned-pdftoppm.exe> `
  --output-dir <render-dir> `
  --manifest <render-dir>/render_manifest.json

# 3. generic resolve; ожидаемый номер activity здесь отсутствует
python scripts/math12_official_source_adapter.py resolve `
  --inventory reports/maxim_math12_activity_source_v1_20260808/inventory.json `
  --render-manifest <render-dir>/render_manifest.json `
  --image <arbitrary-image.png> `
  --output <source-certificate.json>

# 4. извлечь только pinned official solution span
python scripts/math12_official_source_adapter.py extract-solution `
  --pdf <math12.pdf> `
  --inventory reports/maxim_math12_activity_source_v1_20260808/inventory.json `
  --certificate <source-certificate.json> `
  --output <official-solution.json>
```

использование GPU и сети не требуется. dev-5 был прогнан последовательно на локальном CPU.

## проверки

- unit/fail-closed suite: 9 passed;
- opt-in real-PDF inventory integration: 1 passed за 121.86 s;
- полный inventory build: 95/95 activities;
- official key marker uniqueness: 95/95;
- content partition: страницы 4–130 покрыты ровно один раз;
- visual dev audit: 5 accepted, 5 source-address matches;
- official solution extraction: 5/5 projection hashes совпали;
- holdout, gold answers, benchmark scorer и correctness в этой работе не читались и не запускались.

финальные хеши кода и профиля записываются отдельно в `freeze_candidate_manifest.json`; именно этот manifest должен попасть в freeze commit до transfer run.

