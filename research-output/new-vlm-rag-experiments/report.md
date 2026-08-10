# Новые эксперименты для textbook VLM-RAG

Дата исследования: 2026-08-03.

## Что исключено

Не предлагались уже исследованные направления: tool-wrapper для RAG, multilingual retrieval, page-level retrieval, element/ColPali chunking, смена embedding model и hybrid search, декомпозиция, parallel reasoning и выбор судьёй, Self-RAG, preprocessing найденного чанка, knowledge graph, image crops для retrieval, Solver–Critic–Repair, параллельные RAG/no-RAG ответы, task/subject router, answer voting/clustering, calculator/SymPy, structured transcription, answer canonicalization и память типовых ошибок. Также не выдавались за новые уже реализованные subject/grade filters, один query rewrite, relevance gate и image-first conflict check.

## Рекомендуемый порядок

| Приоритет | Эксперимент | Почему сейчас | Стоимость |
|---:|---|---|---|
| 1 | MMR-диверсификация | Точечно исправляет дубли top-k; текущий DenseRanker уже получает широкий пул кандидатов | низкая |
| 2 | Порядок evidence в контексте | Нулевой риск для retrieval: меняется только порядок тех же chunk_id | очень низкая |
| 3 | Option-aware retrieval | Хорошо изолированный эксперимент на choice subset | средняя |
| 4 | Индекс синтетических вопросов | Может сократить разрыв между языком задания и учебника без online query rewrite | средняя/высокая |
| 5 | Metric-driven prompt/tool policy optimization | Может оптимизировать весь цикл, но дорог и опасен переобучением | средняя/высокая |

## Единый протокол оценки

1. Заморозить один список task_id, модель, seed/temperature, judge и версии prompts.
2. Для retrieval-only ablations кэшировать исходный пул кандидатов: это исключит сетевую и генеративную вариативность.
3. Считать не только абсолютный accuracy, а paired delta, wrong-to-right, right-to-wrong и 95% paired bootstrap CI. В проекте уже есть paired aggregation и bootstrap CI.
4. Отдельно показывать overall, by-subject, choice-only, corpus-covered и tasks-with-confident-retrieval.
5. Для каждой идеи менять одну ось. Не совмещать MMR, новый порядок, новый prompt и новый индекс в одном первом прогоне.
6. Первые два эксперимента можно отсеять на dev-срезе, затем один раз подтвердить победителя на frozen holdout.

## Практическая первая серия

Запустить на одинаковых retrieval candidates четыре варианта: current dense top-5, MMR lambda 0.3, 0.5 и 0.7. Для лучшего MMR-варианта отдельно сравнить current order и edge-interleaving. Это даёт причинно интерпретируемый результат максимум за семь arms с повторным использованием retrieval-кэша, а не ещё одну большую агентную стратегию.

Детальные карточки экспериментов находятся в `items/`.
