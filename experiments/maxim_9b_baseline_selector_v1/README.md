# baseline-selector v1.1

первая версия freeze получила independent audit FAIL и никогда не запускалась. ее exact SHA
сохранен в `SUPERSEDED_FREEZE_v1.json` со статусом `superseded_not_executed`. основная причина
была не в самом пороге, а в provenance: описание делало слишком сильное утверждение о моменте
freeze, а runtime доверял `opaque_id` и `evaluation_route` из candidate pool.

v1.1 исправляет обе проблемы.

исторический aggregate 238/274 и prior task outcomes были известны до проектирования и freeze.
это нельзя называть экспериментом, полностью preregistered до любых прошлых результатов.
более узкое и проверяемое утверждение такое: rules/profile фиксируются до генерации и любой
gold/score/correctness/judge evaluation новых selector-arm outputs. runtime selector-а не читает
gold, reference, correctness, outcomes или judge verdicts. в его input package этих полей быть
не может, а strict schemas и recursive denylist закрывают запуск при их появлении.

до окончательного v1.1 freeze также был проведен структурный аудит новых candidate answers,
agreement counts и provenance. это раскрыто явно. V4/V5 — две коррелированные native-thinking
ablations на одном routed86/model/seeds, но с разными treatment и max tokens. parallel8 v1/v2 —
два отдельно выполненных 8-route batches на той же модели и routes; v2 считается core, v1 —
diversity donor. это не четыре полностью независимые системы. у parallel batches нет immutable
endpoint revision manifest.

правила arms не изменились:

- anchor — Query Active Crop V2;
- primary предлагает challenger, только когда V4 = V5 != anchor и за challenger есть минимум
  13 из 16 raw parallel votes;
- secondary exploratory предлагает challenger, только когда V4 = V5 = final parallel8 v1 =
  final parallel8 reasoning-first v2;
- invalid, missing, disagreement или любой failed binding сохраняет anchor либо закрывает весь
  запуск fail-closed;
- source-union и image-judge строки всегда сохраняют anchor.

v1.1 не принимает identity или evaluator route из pool. candidate pool содержит только
`row_index` и candidate projections. authoritative task ID берется из отдельно SHA-pinned списка
274 benchmark IDs. authoritative evaluator route берется из отдельно SHA-pinned outcome-free
format map. оба artifacts привязаны к exact benchmark SHA
`5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9`.

каждый из пяти upstream artifacts получает row-level binding: для каждого authoritative
row index хранится SHA compact canonical projection кандидата или всего parallel8 batch.
перестановка кандидатов между задачами, подмена protected ID, relabel route, пропуск, дубликат,
лишняя строка или другой порядок обнаруживаются до selection.

gold-bearing benchmark bytes не копируются в input package и не открываются runtime. benchmark
SHA используется только как внешний identity pin. runtime также не парсит V7 aggregate, потому
что этот JSON содержит исторический score. вместо этого input package включает отдельную
outcome-free projection из 156 task IDs. она фиксирует aggregate SHA и внутренний source-union
projection SHA, имеет собственный preregistered file SHA и используется только как safety veto,
не как признак качества.

`profile_v1_1.json` теперь locked на exact SHA package/order/route/source-membership/pool/bindings.
strict structural load подтвердил 274 unique IDs, 177 deterministic routes, 97 image-judge
routes и 156 protected IDs без запуска selector rules. `PREREGISTERED_FREEZE.json` pin-ит profile,
код, tests и supersession record. selection и evaluation все равно запрещены, пока independent
re-audit не даст PASS.

после будущего PASS команды будут такими:

```powershell
python selector_v1_1.py --verify-freeze

python selector_v1_1.py `
  --input-package input\frozen\input_package_v1_1.json `
  --output-dir C:\path\to\new_empty_output_dir
```

выход — patch proposals со статусом `new_selector_arm_outputs_frozen_not_evaluated`, а не QA
accuracy и не готовый scored solver.
