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

## Демонстрация финального V7 trace

Отдельный read-only экран показывает 274 joined-задачи, сохранённые шаги решения,
маршрут, official-source certificates, сравнение anchor/challenger/final и честные
границы метрик. Он не запускает модель и не изменяет SQLite-базу приложения.

```powershell
.\.venv\Scripts\python.exe trace_viewer.py `
  --artifact-root C:\path\to\VLM_agent_turkish_textbook_basic_rag
```

Перед демонстрацией проверьте bundle командой `trace_viewer.py --validate-only`.
Полное описание, режим скриншотов и трактовка source-first профиля находятся в
[`vlm_trace_viewer/README.md`](vlm_trace_viewer/README.md). Готовый Windows launcher и
трёхминутный сценарий показа описаны в [`DEMO_GUIDE_RU.md`](DEMO_GUIDE_RU.md).

В отдельной вкладке Holdout80 показан честный source-evidence aggregate: неизменный
raw `71/80`, отдельный official-key erratum `79/80` и `79/79` по валидным строкам.
Он явно отделён от V7 QA `242/274`: source lookup/binding не выдаётся за качество
ответа или reasoning. В приложение встроена только fail-closed публичная сводка с
хешами, без приватных строк holdout.

![VLM Trace — сохранённый V7 trace](docs/assets/trace-viewer-final.png)

![VLM Trace — Holdout80 source evidence](docs/assets/holdout80-source-evidence.png)

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

В Git хранится только код приложения. Локальные SQLite-базы, синхронизированные
JSONL, judge-кэши, изображения учебников и собранные EXE исключены через
`.gitignore`.
