# VLM Analytics

Отдельное desktop-приложение для хранения и анализа прогонов VLM-проекта.

Внутри:

- SQLite-история всех импортированных прогонов;
- метрики по режимам, предметам и датам;
- карточки отдельных задач с сырыми ответами агента и judge;
- статистика reasoning, tools, retrieval и ошибок;
- реестр факапов с ответственным компонентом, статусом и повторяемостью;
- ручной импорт JSONL и синхронизация результатов по SSH.

## Запуск из исходников

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

## Демонстрация trace и provenance

Отдельный read-only экран показывает 274 joined-задачи, сохранённые шаги решения,
маршрут, official-source certificates, сравнение anchor/challenger/final и честные
границы метрик. Он не запускает модель и не изменяет SQLite-базу приложения. Старый
V7 явно подписан как archived/reference: это META-27B anchor с последующими
детерминированными source layers, а не цельный output одной 27B модели. Показанные
latency/tokens относятся только к сохранённому inherited anchor и не являются E2E.

```powershell
.\.venv\Scripts\python.exe trace_viewer.py `
  --artifact-root C:\path\to\VLM_agent_turkish_textbook_basic_rag
```

Перед демонстрацией проверьте bundle командой `trace_viewer.py --validate-only`.
Полное описание, режим скриншотов и трактовка source-first профиля находятся в
[`vlm_trace_viewer/README.md`](vlm_trace_viewer/README.md). Готовый Windows launcher и
трёхминутный сценарий показа описаны в [`DEMO_GUIDE_RU.md`](DEMO_GUIDE_RU.md).

Новый 9B-only режим включается только с полным SHA-замкнутым manifest:

```powershell
.\.venv\Scripts\python.exe trace_viewer.py `
  --nine-b-comparison C:\path\to\comparison.json `
  --dataset auto
```

При успешной проверке он становится default trace. При отсутствии manifest старый
27B экран остаётся только явно помеченным reference; 27B task trace и 9B score не
смешиваются. Exact wrapper/native contract описан в
[`NINE_B_COMPARISON_CONTRACT_RU.md`](NINE_B_COMPARISON_CONTRACT_RU.md).

В отдельной вкладке Holdout80 показан честный source-evidence aggregate: неизменный
raw `71/80`, отдельный official-key erratum `79/80` и `79/79` по валидным строкам.
Он явно отделён от V7 QA `242/274`: source lookup/binding не выдаётся за качество
ответа или reasoning. В приложение встроена только fail-closed публичная сводка с
хешами, без приватных строк holdout.

Отдельная вкладка `9B · selector v1.2` продолжает каноническую 9B-историю, не
переписывая её: семь milestone остаются завершёнными на Source V7 `238/274`, а
selector показан самостоятельным восьмым development-слоем `240/274 = 0.875912`.
Loader сверяет completion manifest, score, composition, solver, все четыре arm и
provenance двух замен (`val_0089`, `val_0251`) по SHA-256. На экране явно отмечены
`+2 fixes`, `0 regressions`, `272/274` byte-exact passthrough, сохранение 156 source
и 97 image строк. Post-score repair `val_0223` вынесен в отдельный null-result:
content изменился, но score остался `240 → 240`; число 241 не заявляется.

Когда этот bundle прошёл проверку, основные вкладки задач и аналитики показывают
selector как активный all-9B результат: `240/274 = 87.5912%`, Math `109/139`,
History `10/10`, deterministic `158/177`, image `82/97`. Две замены проецируются
на task-level view, поэтому фильтры и aggregate согласованы. Их reasoning при этом
остаётся явно подписанным как записанный Source V7 trace; интерфейс не сочиняет
для selector новый chain-of-thought. Лестница семи этапов по-прежнему честно
заканчивается на `238/274`, а archived QA V7 `242/274` остаётся отдельным reference.

![VLM Trace — сохранённый V7 trace](docs/assets/trace-viewer-final.png)

![VLM Trace — Holdout80 source evidence](docs/assets/holdout80-source-evidence.png)

![VLM Trace — audited Baseline Selector v1.2](docs/assets/selector-wave-v1.2-audited.png)

![VLM Trace — active selector task view](docs/assets/selector-active-tasks.png)

![VLM Trace — active selector analytics](docs/assets/selector-active-analytics.png)

## Синхронизация без интерфейса

Перед первой синхронизацией задайте адрес источника через настройки приложения
или переменные окружения. Приватные адреса и логины в репозитории не хранятся.

```powershell
$env:VLM_ANALYTICS_SERVER = "example.org"
$env:VLM_ANALYTICS_USER = "username"
$env:VLM_ANALYTICS_REMOTE_ROOT = "/path/to/v2_274/app"
```

```powershell
.\.venv\Scripts\python.exe main.py --sync-once
```

База `vlm_analytics.db` создается рядом с программой. При запуске собранного EXE
это позволяет переносить приложение вместе со всей историей одним каталогом.

## Тесты

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

## Сборка Windows EXE

```powershell
.\build.ps1
```

Отдельная SHA-проверяемая 9B trace-сборка создаётся командой:

```powershell
.\build_trace_viewer.ps1 -CopyToDesktop
```

Она собирает `VLM Analytics 9B.exe`; при наличии canonical frozen comparison под
обнаруженным artifact root приложение валидирует все семь milestones и открывает
9B V7 как default dataset. При отсутствии comparison остаётся только явно
помеченный archived 27B reference.

В Git хранится только код приложения. Локальные SQLite-базы, синхронизированные
JSONL, judge-кэши, изображения учебников и собранные EXE исключены через
`.gitignore`.
