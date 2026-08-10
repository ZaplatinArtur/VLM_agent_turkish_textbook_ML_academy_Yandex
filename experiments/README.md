# Проверка пакета экспериментов

Каталоги в `experiments/` являются изолированными воспроизводимыми пакетами. Некоторые
из них намеренно содержат сохранённые pre-run тесты, которые утверждают, что score-файлы
ещё не существуют. После завершённой one-shot оценки эти исторические тесты не входят в
актуальный post-run набор. Кроме того, два независимых пакета используют локальное имя
модуля `selector.py`, поэтому их тесты запускаются отдельными процессами.

Актуальная проверка опубликованного результата `240/274`:

```powershell
cd experiments/maxim_9b_baseline_selector_v1
python -m pytest -q test_selector_v1_1.py test_selector_v1_2.py test_compositor_v1_1.py test_compositor_v1_2.py input/test_build_pool.py input/v1_2/test_build_pool_v1_2.py test_result.py

cd ../maxim_9b_source_calibrated_selector_v1
python -m pytest -q test_selector.py

cd ../maxim_9b_answer_canonicalization_v1
python -m pytest -q test_answer_canonicalization_v1.py

cd ../maxim_9b_answer_contract_repair_v1
python -m pytest -q test_answer_contract_repair_v1.py

cd ../maxim_9b_answer_contract_repair_v1_1
python -m pytest -q test_answer_contract_repair_v1_1.py
```

Первый набор проверяет обе замороженные версии селектора, композицию, outcome-free
authority-пакеты и итоговый `RESULT.json`. Остальные команды проверяют отрицательные
контроли и безопасный post-score repair. Совокупно актуальный набор содержит 220 тестов.
