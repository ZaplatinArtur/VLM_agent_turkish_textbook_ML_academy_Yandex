# Math12 opaque source-binding evaluator v1

это отдельный fail-closed evaluator для уже завершённого opaque run. он считает
только совпадение найденной `activity` с запечатанным source address. это не
оценка качества математического рассуждения и не проверка текста решения.

evaluator заморожен до открытия private map. он:

- заново хеширует каждый artifact frozen run;
- запрещает временные `.tmp-*` output directories;
- сверяет SHA public opaque JSONL и private map с исходным seal;
- принимает в private map только `input_id` и `task_id` вида
  `h80-math12-aNNN`;
- считает abstain ошибкой;
- отдельно пишет coverage, source-binding accuracy и conditional precision;
- не читает official answer, gold, manual score или solution text как truth.

запуск после того, как clean/stress output полностью завершён и его
`run_manifest.json` зафиксирован:

```powershell
python scripts\evaluate_math12_binding_v1.py `
  --run-dir <FINAL_RUN_DIR> `
  --input-seal <math12.seal.json> `
  --private-map <resolver_input_map_math12.jsonl> `
  --output <NEW_EVALUATION_JSON>
```

проверка:

```powershell
python -m pytest -q tests\test_math12_binding_eval.py
```

frozen unit result: `6 passed`. evaluator не запускался на private map до
создания этого freeze.

