# visrag2: subject-local SigLIP continuation

Continues the already trained `visrag_siglip_e5_v3` checkpoint. It trains the
last three blocks of both SigLIP towers and their projection heads; LoRA is not
used because this base-size dual encoder fits comfortably on two A100 80GB GPUs
and direct block tuning adapts both image and Turkish text representations.

## Objective and batching

- physical batch: 64 query/page pairs per GPU, no gradient accumulation;
- global differentiable pool: 128 pairs across GPU 0 and 1;
- every global optimizer step contains one subject only;
- reviewed relevance groups and exact duplicate-query links form positive sets;
- grouped pages are packed together in a batch when possible;
- symmetric multi-positive supervised contrastive loss averages log-probability
  across all relevant pages, so a relevant sibling is never a false negative.

## Validation

The split holds out complete relevance groups. Every subject has exactly 120
page images and 120 deterministic queries, ranked only inside that subject.
Online W&B logs Hit@1/3/5/10/20/30, Recall@k, Precision@k, MRR@10 and NDCG@10
for each subject plus macro and micro averages.

## Run and resume

```bash
cd /home/d.teslov/VLM_agent_turkish_textbook_ML_academy_Yandex
bash src/visrag2/run_a100.sh

# Resume model, optimizer, scheduler, RNG, epoch and dataloader position:
bash src/visrag2/run_a100.sh --resume auto
```

Checkpoints are written to
`/mnt/storage-1/d.teslov/visrag_training/checkpoints/visrag2_siglip_e5_v3_a100/checkpoint-N`.
`best_model` is selected by macro Hit@3; `final_model` is always exported in
Hugging Face format. The run stops ten minutes before the next 11:00 Moscow.
