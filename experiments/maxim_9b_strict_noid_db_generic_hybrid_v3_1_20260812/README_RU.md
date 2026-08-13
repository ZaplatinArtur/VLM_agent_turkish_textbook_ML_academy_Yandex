# Strict no-ID DB + generic Qwen hybrid v3.1

Headline certified-ветка — не широкий fuzzy matcher v2, а неизменённый и
независимо проверенный `content_source_router_noid_v1`: 18 действий (16 официальных
source bindings и 2 детерминированных инструмента). На Maxim274 этот policy уже
показал относительно честного base240 результат 251/274, +11 исправлений и 0
регрессий. Это development evidence, а не unseen generalization.

Селектор получает ровно `ocr_text`, `answer_type`, `input_mode`. ID читается внешним
адаптером только после возврата action и нужен лишь для сохранения порядка и join с
generic prediction. При abstain задача передаётся точной `Qwen/Qwen3.5-9B`.
Транспорт, провайдер и квантование не участвуют в выборе ветки. Перед compose
обязательны внешние SHA candidate freeze и независимого PASS-аудита.

`base249/official16` запрещён как fallback. `maxim_base240_control` принимает только
замороженный SHA base240 и нужен для парной контрольной проверки certified-ветки.
