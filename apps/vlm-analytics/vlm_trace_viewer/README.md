# VLM Trace

> Важно: существующий V7 trace — archived/reference development replay. Его
> provenance: `META-27B anchor + deterministic source layers`. `base row model`
> и `final origin` показываются отдельно; source replacements не приписываются
> META-27B. Latency и tokens — recorded inherited-anchor usage, не end-to-end.

Локальный экран для демонстрации финального пайплайна V7. Он построен на той же
PySide6-основе, что и существующий `VLM Analytics`, но запускается отдельно и не
меняет его SQLite-базу.

Интерфейс читает уже сохранённые артефакты: финальный solver на 274 задачах,
composition decisions, V6 anchor, source candidates, сертификаты, OCR-блоки с
координатами и итоговый `score.json`. Никакие модели, API, GPU или SSH при
просмотре не вызываются.

## Быстрый запуск

```powershell
cd C:\Users\kmaxc\PycharmProjects\VLM_Analytics_App
.\.venv\Scripts\python.exe trace_viewer.py
```

По умолчанию приложение само находит соседний проект
`VLM_agent_turkish_textbook_basic_rag`. Явный путь можно передать так:

```powershell
.\.venv\Scripts\python.exe trace_viewer.py `
  --artifact-root C:\Users\kmaxc\PycharmProjects\VLM_agent_turkish_textbook_basic_rag
```

Для переносимого демо явный `--artifact-root` остаётся каноническим вариантом: он не
зависит от текущей рабочей папки. Автопоиск также поддерживает копию приложения внутри
`apps/vlm-analytics` и ищет проект с артефактами среди ограниченного числа родительских
папок и их соседей.

Перед показом руководителям полезно выполнить две проверки:

```powershell
.\.venv\Scripts\python.exe trace_viewer.py --validate-only
.\.venv\Scripts\python.exe trace_viewer.py `
  --task val_0196 `
  --screenshot artifacts\trace-viewer-smoke.png
```

По умолчанию приложение открывает новую вкладку `Holdout80 · source evidence`.
Для снимка trace explorer используйте `--screenshot-tab 0`, для Holdout80 —
`--screenshot-tab 1`, для прежней страницы V7 — `--screenshot-tab 2`, для
канонической семёрки 9B — `--screenshot-tab 3`, для audited selector —
`--screenshot-tab 4`.
Для снимка маршрута или сертификата используйте соответственно
`--detail-tab 1` или `--detail-tab 2`.

Полный новый 9B comparison подключается отдельно:

```powershell
.\.venv\Scripts\python.exe trace_viewer.py `
  --nine-b-comparison C:\path\to\comparison.json `
  --dataset auto
```

`auto` переключает task trace на новый source-adjudicated 9B V7 только после
успешной проверки всех семи SHA-pinned milestones. `--dataset nine-b-v7` без
полного валидного manifest завершается ошибкой. Формат и все fail-closed проверки
описаны в [`../NINE_B_COMPARISON_CONTRACT_RU.md`](../NINE_B_COMPARISON_CONTRACT_RU.md).

При наличии полного comparison loader также требует frozen selector-wave bundle.
На пятой вкладке семь canonical milestone остаются неизменными и заканчиваются
Source V7 `238/274`; Baseline Selector v1.2 показан отдельным восьмым слоем
`240/274 = 0.875912`. Это Qwen3.5-9B-only development wave на известном наборе,
не blind holdout. Две замены (`val_0089: A→D`, `val_0251: A→B`) разрешены только
при согласии structural, native и parallel групп. Интерфейс показывает их
decision/proposal/source row hashes, но не имитирует скрытый chain-of-thought.
Post-score answer-contract repair изменил строку `val_0223`, однако повторный score
остался `240/274`; он подписан как non-blind, non-preregistered null-result, а не
как новый milestone или 241.

После успешной hash-проверки selector становится активной проекцией на первых двух
вкладках: headline и task filters считают `240/274`, а `val_0089` и `val_0251`
показывают выбранные selector-ответы. Исходный Source V7 trace не мутируется:
записанное reasoning этих строк маркируется как lineage evidence, selector добавляет
только проверяемую provenance выбора. Вкладка семи milestone продолжает заканчиваться
на Source V7 `238/274`.

Для image evaluator интерфейс показывает cumulative split: сколько финальных
verdict уже source-adjudicated и сколько осталось byte-identical исходному
ActiveCrop 9B judge. Число строк, скопированных только из immediate base, не
называется «original 9B»: после первого source stage такая копия может уже не быть
модельным verdict.

Локальные изображения для 9B trace подключаются только как display-only assets из
ограниченного каталога внутри artifact root. Loader сверяет task id, запрещает
absolute/`..` locators и не читает из archived V7 ни solver answers, ни judge
verdicts, ни provenance rows. На текущем bundle доступны 27 таких изображений;
это число выводится отдельно в `--validate-only` и не участвует в score.

## Что видно в интерфейсе

- список всех 274 задач с фильтрами по предмету, correctness, сертификату и
  действию composer;
- исходное локальное изображение, если оно осталось в кеше, иначе точная
  OCR-реконструкция из сохранённых блоков и bbox;
- записанные `solution_steps` и `reasoning` с пошаговым воспроизведением;
- путь ответа через anchor, router, exact-source lookup, PDF/page/key binding,
  certificate, composer и evaluation;
- официальный PDF, страницу вопроса, страницу ключа, bbox, coverage, margin,
  fingerprint и все детерминированные проверки сертификата;
- сравнение V6 anchor, source challenger и финального V7;
- archived/reference итог `242/274 = 0.8832`, математику `112/139 = 0.8058`,
  предметные срезы и recorded inherited-anchor latency (не E2E).
- source-first профиль из отдельного artifact replay: `131/274 = 47.8%`
  сильных source shortcuts, эквивалентных финальным V7-ответам; потенциально
  устраняется `44.84%` записанной latency reasoning-модели и `46.98%` её input
  tokens.

Отдельная вкладка Holdout80 показывает только замороженный публичный aggregate:

- raw frozen-protocol source-evidence: `71/80 = 0.8875`;
- official-key erratum с исходным знаменателем: `79/80 = 0.9875`;
- срез по 79 валидным заданиям: `79/79 = 1.0000`;
- Math source-activity binding `20/20`, Biology lookup `30/30`, Physics raw
  `21/30` и valid `29/29`;
- хронологию freeze → blind run → output seal → raw score → independent audit →
  отдельный erratum;
- два дефекта протокола: 8 перепутанных строк sealed gold и одно открытое задание,
  ошибочно объявленное тестом A–E.

Эта вкладка не загружает приватные строки holdout. Публичный JSON содержит только
агрегаты и SHA-256 связей, а loader сверяет его каноническую проекцию fail-closed.
Raw не заменяется исправленным числом. На экране отдельно стоит V7 QA
`242/274 = 0.883212`: это development replay и другая метрика. Holdout80 здесь
измеряет source lookup/source binding, а не QA или правильность рассуждения.

Подсветка OCR-блоков не называется attention модели. Это честная визуализация:
сохранённый шаг решения эвристически сопоставляется с сохранённым OCR-блоком.
Интерфейс также явно маркирует V7 как ранее изученный development replay, а не
как unseen holdout или production accuracy.

Speed-профиль тоже маркирован консервативно: это не измеренный online speedup.
Стоимость source lookup в отчёт не включена, cold/warm-cache wall-clock ещё не
замерен. Поэтому интерфейс показывает только потенциально устранимую работу
reasoning-модели на сохранённом прогоне, а не обещание ускорения в проде.

## Тесты

```powershell
python -m pytest -q tests\test_trace_viewer_adapter.py
```

Тестовый fixture содержит только маленькую искусственную структуру файлов. Он
проверяет join, fail-closed ветку, source override и внутреннюю согласованность
метрики; реальные ответы benchmark в репозиторий приложения не копируются.
