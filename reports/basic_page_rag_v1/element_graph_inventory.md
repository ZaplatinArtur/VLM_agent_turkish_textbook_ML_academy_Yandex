# Инвентаризация element-level knowledge graph

Дата проверки: 2026-07-31.

## Краткий ответ

Граф знаний с разбиением учебника на смысловые элементы существует. Он не
входит в зафиксированный `basic_page_rag_v1` и должен подключаться только как
отдельный A/B-кандидат.

Типы элементов:

- `theory`;
- `worked_example`;
- `exercise`;
- `solution`;
- `answer_key`;
- `instruction`;
- `other`.

Отдельного узла `condition` нет: условие, варианты и относящиеся к задаче
визуальные/OCR-фрагменты остаются внутри `exercise`.

Типы сохраняемых связей:

- `theory_for`;
- `worked_example_for`;
- `solution_of`.

## Реально доступный локальный граф

Исходные element-level chunks:
`C:/Users/kmaxc/PycharmProjects/VLM_agent_turkish_textbook_ML_academy_Yandex/artifacts/hybrid_chunks_restored14_v1`.

Граф:
`C:/Users/kmaxc/PycharmProjects/VLM_agent_turkish_textbook_ML_academy_Yandex/artifacts/knowledge_graph_restored14_v1`.

- 14 книг и 2 486 страниц;
- 12 981 узел;
- 7 165 поисковых узлов;
- 6 510 рёбер;
- 3 821 `exercise`;
- 3 060 `theory`;
- 284 `worked_example`;
- 5 808 `solution`;
- 8 `instruction`.

SHA256:

- `nodes.jsonl`: `87ed1801efc08018032311e88fb7a42ae14c555d05dd6cbb68de3d362bc98247`;
- `edges.jsonl`: `e0a5eb048005c0e70ac36c47ff28b4b2db893062a4184dce55bfe9454fb74968`;
- `manifest.json`: `5d33945ab46ff83db7fe71af7cc72cb42d8961f77ac76fd20e84aa74fb9a7d13`.

Штатный loader прочитал граф как `12 981 nodes / 6 510 edges / 7 165
searchable nodes`. Целевые тесты chunker/graph/knowledge base: `21 passed`.

Ограничение: в этих 14 локальных книгах нет математики, поэтому этот артефакт
не подходит для Math A/B без восстановления полного корпуса.

## Полная сборка, сохранившаяся в отчётах

Отчёты о 200-книжной сборке фиксируют:

- 151 576 element-level узлов;
- 137 253 поисковых узла;
- 114 594 ребра;
- 59 875 задач;
- 90,19% задач связаны с теорией;
- 16,25% — с разобранным примером;
- 3,26% — с каким-либо решением;
- 2,72% — с решением, безопасным для выдачи агенту при confidence >= 0.8.

Полные `nodes.jsonl`, `edges.jsonl` и FAISS-индекс этой сборки локально
отсутствуют. Сохранились отчёты, а серверная 253-книжная сборка описана как
212 317 узлов, 185 462 поисковых узла и 157 390 рёбер.

## Качество и статус

Сегментация в основном эвристическая; selective Qwen refinement был только
пилотом. Human-labeled precision/recall для типов элементов нет. Oversized
элемент только маркируется и не разбивается повторно.

Graph-RAG пока нельзя считать улучшением математики: в последнем полном
Math139 A/B прямой Graph-RAG дал `63/139`, а fresh B0 — `85/139`. Поэтому
базовый page-RAG остаётся неизменяемым control, а element graph — отдельным
экспериментальным retriever-вариантом.

## Код и отчёты

- Chunker: `src/retrieve/chunking/educational.py` в основном репозитории.
- Graph: `src/retrieve/graph.py`.
- Runtime integration: `src/retrieve/service.py`.
- Full graph report: `reports/knowledge_graph_v1.json`.
- Coverage report: `reports/knowledge_graph_v1_analysis.json`.
- Local graph report: `reports/knowledge_graph_restored14_v1.json`.
