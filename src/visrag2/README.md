# visrag2: subject-local SigLIP continuation

Continues the already trained `visrag_siglip_e5_v3` checkpoint. It trains the
last three blocks of both SigLIP towers and their projection heads; LoRA is not
used because this base-size dual encoder fits without adapters and direct block
tuning adapts both image and Turkish text representations.

## Objective and batching

Numbers below are the defaults in `run_a100.sh`; every one of them is a flag.

- physical batch: 32 query/page pairs per GPU, no gradient accumulation;
- global differentiable pool: 128 pairs across four GPUs;
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

The script finds the repository itself, so it runs from anywhere:

```bash
bash src/visrag2/run_a100.sh

# Resume model, optimizer, scheduler, RNG, epoch and dataloader position:
bash src/visrag2/run_a100.sh --resume auto
```

It launches four processes on `CUDA_VISIBLE_DEVICES=0,1,2,3` and expects
`.venv/bin/accelerate` in the repository root. `start_when_ready.sh` is the same
run, delayed until the query manifest is complete and all four GPUs are free.

Query manifests are not in git and the two hosts disagree on where they live, so
`VISRAG_CATALOG_DIR` selects the directory — `catalog` on the training server,
`data/visual_retrive/catalog` by the repository convention:

```bash
VISRAG_CATALOG_DIR=data/visual_retrive/catalog bash src/visrag2/run_a100.sh
```

`best_model` is selected by macro Hit@3; `final_model` is always exported in
Hugging Face format, both under `--output-dir`. The run stops ten minutes before
the next 12:00 Moscow.

The base checkpoint, `--output-dir` and the paths inside `start_when_ready.sh`
are host-specific and point at the training server, not at this repository.
