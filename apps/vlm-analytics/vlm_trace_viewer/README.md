# VLM Trace

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

Для снимка страницы с общими метриками добавьте `--screenshot-tab 1`.
Для снимка маршрута или сертификата используйте соответственно
`--detail-tab 1` или `--detail-tab 2`.

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
- итог `242/274 = 0.8832`, математику `112/139 = 0.8058`, предметные срезы и
  latency сохранённого запуска.
- source-first профиль из отдельного artifact replay: `131/274 = 47.8%`
  сильных source shortcuts, эквивалентных финальным V7-ответам; потенциально
  устраняется `44.84%` записанной latency reasoning-модели и `46.98%` её input
  tokens.

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
