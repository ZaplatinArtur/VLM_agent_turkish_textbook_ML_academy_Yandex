# Независимый post-score аудит YKS DEV failover

Дата аудита: 2026-08-12.

## Итог

Агрегатный результат механически подтверждён: **166/185 = 0.8972972973**. Предварительно зафиксированный порог **148/185 (0.8)** пройден.

Состав результата: 135 строк выбраны из строгого успешного ответа V6.2, 50 строк — из заранее замороженного V5 theory fallback; сумма равна полному знаменателю 185.

## Проверенная цепочка

- V6 completion SHA-256: `55ec3e8c0ca00f6fbf6fb2423a9e41d5639e37394a7c851f3e036516e9fd5497`.
- Failover execution freeze SHA-256: `0b32ccaae77f947015656a16db68ed59b90868cf3a604f4753c33c434d5c08c2`.
- Failover private-score freeze SHA-256: `bc1fe65ad728465b31328c1927b4c67b50ce6cb222485a5c874bffc06bcb08ec`.
- Failover completion SHA-256: `a92ec4d6d4b42396ea23b786371e6bec2875bbbe62cadeb90c055ddc32fa4f04`.
- Aggregate private result SHA-256: `764da75325bf10a8ba669ccd739993ea7cb8e992f3dcfbca6be5e391868e54cd`.
- Pre-run independent audit SHA-256: `e0832d8dfc491a9115d8b6a234e10a628548c0259f101018ee3ae7ad6832de68`.

Проверены точные schema/version, SHA-связи completion → execution freeze → V6 completion, result → completion/execution freeze/private freeze, полный знаменатель, порог и отсутствие task ID, gold-ответов и построчных outcomes в агрегатном результате. Все проверки пройдены.

## Ограничение заявления

Обязательная маркировка результата:

> Pre-V6-atomic-outcome fixed-rule DEV failover; V5 per-row outcome exposure to the builder disclosed; not a fully blind DEV evaluation.

То есть результат пригоден как механическая оценка заранее продиктованного правила без настраиваемых порогов и без содержательного арбитража, но не должен называться полностью слепой DEV-оценкой. Он также сам по себе не доказывает качество на FINAL80 или на новом unseen holdout.

## Границы аудита

Аудитор не читал построчные ответы, gold-содержимое или построчные outcomes. Проверялись только агрегатный result, completion-метаданные, дескрипторы и криптографическая provenance-цепочка.
