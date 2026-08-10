# Source-first cascade: V7 artifact replay

Это проверка эквивалентности на уже замороженных артефактах V7, а не новый score и не online latency benchmark.

## Результат

- строк: `274`;
- задач с единственным принятым source-answer: `131` (`47.81%`);
- задач, где reasoning anchor всё ещё нужен: `143`;
- source-answer совпал с финальным V7: `131/131`;
- потенциально исключаемая записанная model latency: `17700.906` из `39477.954` секунд (`44.84%`);
- потенциально исключаемые input tokens: `434584` из `924999` (`46.98%`);
- потенциально исключаемые output tokens: `171177` из `383557` (`44.63%`).

## Интерпретация

В production source resolver запускается первым. Если найден ровно один сильный input-bound и answer-bound сертификат, его ответ можно вернуть без вызова reasoning-модели. При отсутствии сертификата или конфликте запускается прежний anchor и полный fail-closed composer.

Качество в этом replay не меняется: все shortcut-ответы дословно совпадают с финальным V7. Реальная задержка source resolver здесь не измерена, поэтому проценты выше описывают устранённую model work, а не обещанное wall-clock ускорение сервиса.

## Границы честного утверждения

- gold answers, judge verdicts и correctness не читались;
- task_id применялся только для выравнивания строк;
- replay использует previously inspected development artifacts;
- перед production нужен online benchmark с cold/warm cache, p50/p95 и стоимостью source lookup.
