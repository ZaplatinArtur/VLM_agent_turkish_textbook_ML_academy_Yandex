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
