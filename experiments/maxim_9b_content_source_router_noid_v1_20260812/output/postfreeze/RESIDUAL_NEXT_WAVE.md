# Следующая no-ID волна (не включена в 251/274)

Самый короткий потенциальный путь к 261 — десять оставшихся ошибок с уже
доказанными CPU proof kernels:

| Группа | Задачи | Готовое ядро | Недостающий честный слой |
|---|---|---|---|
| text + visual fact | 0067, 0086 | digit inequality; integer quadratic inequality | извлечь формулу/длины из картинки без ID/SHA |
| structured OCR | 0204, 0205, 0218 | column arithmetic; mixed radix; rationality | надёжно распознать символы/операнды в in-image crops |
| diagram semantics | 0230, 0232, 0245, 0253 | bead equations; semicircle geometry; barrier; GCD thickness | общий image-to-structure parser с proof closure |
| propositional logic | 0267 | truth-table evaluator | определить p/q/r из наблюдаемой схемы и распознать все формулы |

Kernels повторно использовать можно; прежние route maps, task IDs, image SHA
и готовые task-specific answers использовать для выбора нельзя. Новая волна
должна иметь отдельные parser fixtures, counterfactual re-ID tests, candidate
freeze и paired regression guard относительно strict B=251.
