# Math12 stress source-binding: post-seal compatibility evaluation v1

## главный статус

`prediction_sealed_before_map; adapter_extended_after_map; not_preregistered_evaluator`

предсказания clean/stress и общий output seal были зафиксированы до первого чтения private map. сам compatibility adapter, вычисление метрики, этот отчёт и freeze сделаны уже после чтения private map. поэтому результат нельзя называть результатом заранее зарегистрированного evaluator. это прозрачное исправление несовместимости: старый evaluator умел связать run только с clean `math12.seal.json`, а stress JSONL был зафиксирован отдельной preregistration/freeze-цепочкой.

исходные predictions, `results.jsonl`, run manifests, clean seal, stress preregistration/freeze и pre-map output seal не менялись и не перезаписывались.

## результат

| метрика | значение |
|---|---:|
| входов | 20 |
| изображений | 28 |
| accepted | 20 |
| abstained | 0 |
| correct source activity | 20 |
| incorrect | 0 |
| coverage | 1.000 |
| source-binding accuracy | 1.000 |
| conditional precision | 1.000 |

это **не QA accuracy и не точность математического рассуждения**. проверялось только то, что резолвер на детерминированно ухудшенных копиях тех же 20 opaque-входов нашёл правильный номер activity в том же учебнике. stress включает небольшую перспективу, поворот, обрезку внешних полей, resize, JPEG, слабый blur и сдвиг яркости/контраста. это узкая проверка synthetic robustness, а не новый независимый benchmark и не проверка переноса на реальные фотографии, другой учебник или другой язык.

## как сохранена честность результата

adapter fail-closed проверяет всю цепочку:

1. исходный clean seal и реальный clean JSONL;
2. SHA private map, заранее записанный в clean seal;
3. байты stress builder и его preregistration;
4. связь preregistration с SHA clean JSONL;
5. каждый clean image SHA, каждый stress image SHA и их пару в transform manifest;
6. все stress artifacts, assets Merkle и точные counts из stress freeze;
7. полный run по его artifact manifest;
8. точный stress run manifest и его artifact projection;
9. точный общий output seal, созданный до чтения private map;
10. запись stress run внутри этого output seal.

для подсчёта adapter вызывает исходный frozen evaluator с неизменённым правилом:

`correct = accepted AND predicted_activity == expected_activity`

любой abstain считается ошибкой. файл исходного metric engine также проверяется по SHA-256, поэтому незаметно заменить формулу нельзя. transient compatibility bridge создаётся только во временной директории и нужен, чтобы frozen evaluator принял уже зафиксированный SHA stress JSONL. исходный seal на диске не изменяется.

## основные SHA-256

| объект | SHA-256 |
|---|---|
| stress freeze | `6a2568b05c479eb47ae44812478d7348552052d2ebbb77040437b068633e927c` |
| stress builder | `f94d64117b1c036cfc29aa886a48467d600dcf6e12b863530bd1430cf7a3b714` |
| stress preregistration | `08a350e9f6d48b8e7f7c475f224385f0df52c898ffdf5ea11bd6e7f850e2bd27` |
| clean input seal | `7ae72fd90de09bc863a868bc03ec31bda2021795808b27771a0dc077406ede93` |
| clean opaque JSONL | `e0ee22d58187fbe11c951ef8153ad825734f83d50e127a1746f4e38649f11960` |
| stress opaque JSONL | `f4ed135db6c046485efad8c5fc0f67a8195b0d5735e3ddc469705d64d76acf5b` |
| stress run manifest | `f74b4901797e412556c45496660f8829826529e750442eff7d7581d45e18a128` |
| stress run artifacts projection | `c81634adc8c87426296ab8c4f9e6e09317ce6bab1920b119437da84b3badded9` |
| pre-map output seal | `b7419a76dbffbd1e45daffb6f4476bf13cb4e8c4099dcd1c4be74fcb1ccc60bc` |
| pre-map output seal projection | `93fe423ee744db667002b93ffb8d652abc855f471116f86a9fa40b51d2955f59` |
| frozen evaluator source | `e4ff8329ab9ebce90e6fed180c395091048ce92ec79455bbfa53a849fa16c8d1` |
| private evaluation artifact | `af5fbf3a7979ecc246ef75aaf2a3288d8c39e3f1bf47c8464fcf65a997401865` |
| evaluation projection | `02b550a591de9e6cec7bea5fa2d8e1ba9b8f95396c67b83440edc3cac3f589d1` |
| rows projection | `b0be20620c21967db99e07249cc3e73175a541c2e3ad3d696b5395e739b06d8f` |

детальные строки evaluation лежат в private holdout workspace и не добавляются в репозиторий, чтобы не публиковать sealed source-address map.

## запуск

```powershell
python scripts\evaluate_math12_stress_binding_compat_v1.py `
  --run-dir <math12_blind_run>\stress `
  --stress-dir <holdout80>\resolver_inputs_stress_v1 `
  --clean-input-seal <holdout80>\resolver_inputs\math12.seal.json `
  --clean-input-jsonl <holdout80>\resolver_inputs\math12.jsonl `
  --private-map <holdout80>\sealed\resolver_input_map_math12.jsonl `
  --output-seal <math12_blind_run>\output_seal_before_map.json `
  --output <private_new_evaluation.json>
```

unit-проверка:

```powershell
python -m pytest -q tests\test_math12_stress_binding_compat_eval.py
```

зафиксированный результат: `9 passed`. тесты отдельно проверяют wrong/abstain, подмену результата, alternate freeze, разрыв preregistration -> clean input, разрыв transform chain, подмену pre-map seal и create-only запись evaluation.
