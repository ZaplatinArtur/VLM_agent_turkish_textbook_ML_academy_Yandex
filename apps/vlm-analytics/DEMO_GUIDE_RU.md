# VLM Trace: демо за 3 минуты

## Подготовка один раз

Откройте PowerShell в этой папке и создайте локальное окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Канонический запуск — с явным путём к проекту, в котором лежат frozen V7 artifacts:

```powershell
.\run_trace_viewer.ps1 `
  -ArtifactRoot C:\path\to\VLM_agent_turkish_textbook_basic_rag
```

Для запуска из `cmd.exe` или двойным кликом используйте обёртку:

```bat
run_trace_viewer.cmd "C:\path\to\VLM_agent_turkish_textbook_basic_rag"
```

Launcher сначала выполняет `--validate-only` и открывает интерфейс только после успешной
проверки. Если проекты лежат рядом, путь обычно находится автоматически, но для показа
лучше передавать его явно. Полезные варианты:

```powershell
# только проверить bundle, интерфейс не открывать
.\run_trace_viewer.ps1 -ArtifactRoot C:\path\to\basic_rag -ValidateOnly

# сразу открыть страницу метрик
.\run_trace_viewer.ps1 -ArtifactRoot C:\path\to\basic_rag -Metrics

# явно открыть новый честный экран Holdout80 (он же открыт по умолчанию)
.\run_trace_viewer.ps1 -ArtifactRoot C:\path\to\basic_rag -Holdout

# начать с другой задачи
.\run_trace_viewer.ps1 -ArtifactRoot C:\path\to\basic_rag -Task val_0196
```

## Что показать

**0:00–0:35. Holdout80.** По умолчанию открывается экран source evidence. Сначала
покажите три числа рядом: raw `71/80`, erratum-inclusive `79/80` и valid `79/79`.
Raw остаётся неизменяемым. Исправленные срезы показаны отдельно после независимой
проверки официальных PDF: в sealed gold перепутаны 8 строк Physics, ещё одно
открытое задание ошибочно объявлено тестом A–E. Сразу проговорите: это метрика
поиска/привязки источника, не QA и не математический reasoning.

**0:00–0:30. Общая рамка.** Покажите верхние карточки и бейджи `OFFLINE`, `DEV REPLAY`,
`V7`. Сразу проговорите: `242/274 = 0.8832` — результат зафиксированного development
replay, а не оценка на unseen holdout и не обещание production accuracy.

**0:30–1:15. Задача `val_0178`.** Это удобный стартовый пример с локальным изображением
и реальной заменой anchor. На вкладке «Рассуждение» запустите пошаговое воспроизведение.
Показываются сохранённые `solution_steps` и объяснение ответа. Подсветка блока —
эвристическое сопоставление текста шага с OCR, а не neural attention и не скрытый
chain-of-thought.

**1:15–1:50. Маршрут.** Откройте вкладку «Маршрут»: reasoning anchor, router,
exact-source lookup, привязка PDF/page/key, source certificate, composer и evaluation.
Серые блоки означают fail-closed ветку, которая не получила права менять anchor.

**1:50–2:20. Доказательство.** На вкладке «Доказательство» покажите официальный документ,
страницу задания, страницу и bbox ключа, coverage, margin, deterministic checks и trace
fingerprint. Это объясняет, почему конкретному challenger разрешена замена.

**2:20–2:45. Сравнение.** Сопоставьте `Anchor · V6`, `Source challenger` и `Final · V7`.
Полные длинные ответы прокручиваются внутри карточек. Для второго source replacement можно
перейти к `val_0196`.

**2:45–3:00. Метрики и границы.** Откройте «Метрики и границы». Подчеркните предметные
срезы, реальный состав прироста относительно V6 и осторожную трактовку source-first.

## Что нельзя обещать

- Набор уже изучался при разработке: это development replay, не независимая проверка
  обобщения.
- Из четырёх дополнительных correct относительно V6 только один связан с новым
  solver-answer; ещё три — исправления ошибочных verdict с опорой на точный официальный
  источник.
- Сильных source certificates 131, но composer заменил anchor только в двух задачах. В
  остальных сертифицированный ответ совпал с anchor.
- Source-first replay показывает 131/274 безопасных shortcut и потенциальное устранение
  44.84% записанной latency reasoning anchor. Это не измеренный online wall-clock speedup:
  стоимость source lookup исключена, cold/warm cache не измерялись.
- Локальные исходные изображения сохранились для 27 задач. Для остальных интерфейс честно
  строит реконструкцию по сохранённым OCR-блокам и bbox.
- Интерфейс ничего не пересчитывает: при просмотре не вызываются модель, API, GPU, SSH или
  web search.

Если preflight не проходит, не показывайте старый скрин как актуальный запуск. Проверьте
`-ArtifactRoot`, наличие `V7_POST_SCORE_RESULT.json` и повторите `-ValidateOnly`.
