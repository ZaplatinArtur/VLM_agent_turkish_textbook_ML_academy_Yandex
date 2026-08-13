# Статус работ на 2026-08-12

## Что уже подтверждено

### 1. Generic-система без DB: YKS DEV

Получен агрегат **166/185 = 0.8972972973**. Целевой порог 0.8 пройден.

Failover был зафиксирован до появления атомарного V6-результата: строгий валидный ответ V6.2 используется первым, иначе берётся заранее замороженный V5 theory fallback. Итоговый состав — 135 ответов V6.2 и 50 fallback-ответов.

Важно: результат механически валиден, но не является полностью слепой DEV-оценкой. Обязательная маркировка:

> Pre-V6-atomic-outcome fixed-rule DEV failover; V5 per-row outcome exposure to the builder disclosed; not a fully blind DEV evaluation.

Артефакты:

- [Post-score audit](../maxim_9b_ykslop_v6_2_v5_failover_dev_postscore_audit_20260812/REPORT_RU.md), SHA-256 `3cd92890ccac27a62a28a08948f12c4843e498de05e6915cfc2755713e850b27`.
- Failover completion: `experiments/maxim_9b_ykslop_generic_v6_2_v5_theory_failover_dev_v1_20260812/DEV_WAVE_COMPLETION.json`, SHA-256 `a92ec4d6d4b42396ea23b786371e6bec2875bbbe62cadeb90c055ddc32fa4f04`.
- Aggregate private result: `experiments/maxim_9b_ykslop_generic_v6_2_v5_theory_failover_dev_v1_20260812/DEV_RESULT_PRIVATE.json`, SHA-256 `764da75325bf10a8ba669ccd739993ea7cb8e992f3dcfbca6be5e391868e54cd`.

### 2. Задачи, присутствующие в DB, без роутинга по ID

Получен строгий результат **251/274 = 0.9160583942** против исходных 240/274: **+11 исправлений, 0 регрессий**. Цель выше 0.91 пройдена. Роутинг использует только содержательные признаки OCR/answer type/input mode и не принимает task ID или hash задачи.

Артефакты:

- [Post-freeze report](../../experiments/maxim_9b_content_source_router_noid_v1_20260812/output/postfreeze/REPORT_RU.md), SHA-256 `ba540bb892727270ddae3030566e239eca9ba3a36f2fedfd641c9c88afef18f9`.
- Result SHA-256 `94698b0163e370e1397057c59ecd253e108f51f0c199450f97318b758fa39a5b`.
- Freeze SHA-256 `76a09995b1104b4b5fec67bb737e73e4a5b21032916f37a24a563118802a8a7c`.
- Independent audit SHA-256 `c1d67ea44fa88f487aa4e65c7c78c3b1e13cbce0564ac1b06f50d009a4e45d82`.

### 3. Аудит старого результата 95.3%

Старые **261/274 = 95.3%** нельзя считать общим качеством модели без ID-роутинга: результат создавался benchmark-specific overlay через task ID и SHA PNG. На контрольных преобразованиях он не обобщился: re-ID 0/12 и re-encode 0/12. Честный ближайший базовый результат без этого overlay — **240/274 = 0.876**.

Артефакты:

- [Generalization audit](../maxim_9b_source95_generalization_audit_v1_20260811/REPORT_RU.md), SHA-256 `e6dec07cf2538159af29de732ec810ec47b92bd3eaaaa96d9b0c73f1a13e4980`.
- Audit result SHA-256 `c50c05f2215c3d7ae8c5bd4726a64ba5e28228a8e8521893d1584e6e0a86238f`.

### 4. Базово-задачная + generic: строгий hybrid-контракт

Граница объединения заморожена и проаудирована без task ID: на Maxim DB-ветка покрывает 18/274, generic-ветка — 256/274; на YKS DB-ветка не срабатывает (0/185), generic-ветка покрывает все 185 задач.

- Hybrid rule freeze SHA-256 `c904f1ea7151513cb83757cc80e21e8dd1cdbd8c7eb4fbf47a40ee40e35ac177`.
- Independent audit SHA-256 `412809bf7dd33b25e582f425ac04eded14ed9ac919f6f0ec59ae781be3301128`.

## Что ещё не завершено

Финальная оценка Maxim hybrid с новым generic-ответчиком ещё выполняется. До появления атомарного результата и независимого агрегатного аудита для неё не заявляется новый итоговый score.
