# visrag2

Корректный Turkish visual page retriever: `query -> PNG page` на основе готового
retrieval-checkpoint `athrael-soju/colqwen3.5-4.5B-v3` и LoRA.

- late-interaction MaxSim, без pooling в обучающей цели;
- батчи состоят из страниц одного предмета;
- страницы из `visrag_relevance_groups_v3_reviewed.json` считаются взаимозаменяемыми positives;
- целые relevance-группы попадают либо в train, либо в validation;
- validation ранжирует 120 запросов среди одних и тех же 120 страниц предмета;
- W&B: Hit@1/2/3/5/10/15/20/25/30, MRR@10 и nDCG@10 в целом и по предметам;
- автоматическая остановка и checkpoint за 8 минут до 11:00 Europe/Moscow;
- checkpoint содержит LoRA, optimizer, scheduler, mixed-precision state и RNG;
- `--resume auto` продолжает обучение с последнего checkpoint.

Запуск на GPU 0 и 1:

```bash
bash src/visrag2/run_a100.sh
```

Возобновление:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 -m visrag2.train \
  ...те же аргументы... --resume auto
```

Основа решения — официальный ColPali/ColQwen multi-vector подход и ColBERT-style
late interaction. В отличие от старого trainer метрики не усредняются по маленьким
in-batch задачам.
