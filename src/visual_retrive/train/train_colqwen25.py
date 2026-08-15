"""ColQwen2.5 LoRA training for Turkish textbook visual retrieval."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments

from colpali_engine.collators import VisualRetrieverCollator
from colpali_engine.loss.late_interaction_losses import ColbertLoss
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer


def read_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fixed_subject_split(items, root: Path, pages_per_subject: int, seed: int):
    by_subject = defaultdict(lambda: defaultdict(list))
    for row in items:
        query = str(row.get("query") or "").strip()
        image = root / str(row.get("positive_image") or "")
        if query and image.is_file():
            by_subject[str(row.get("subject") or "unknown")][str(row["positive_page_id"])].append(row)
    val_pages, stats = set(), {}
    for subject, pages in sorted(by_subject.items()):
        page_ids = sorted(pages)
        random.Random(f"{seed}:{subject}").shuffle(page_ids)
        if len(page_ids) <= pages_per_subject:
            raise ValueError(f"{subject}: {len(page_ids)} usable pages; need > {pages_per_subject}")
        selected = page_ids[:pages_per_subject]
        val_pages.update(selected)
        stats[subject] = {"total_pages": len(page_ids), "train_pages": len(page_ids)-pages_per_subject,
                          "validation_pages": pages_per_subject}
    train, val = [], []
    for pages in by_subject.values():
        for page_id, page_rows in pages.items():
            (val if page_id in val_pages else train).extend(page_rows)
    return train, val, stats


class PageDataset(Dataset):
    def __init__(self, records): self.records=list(records)
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec=self.records[idx]
        with Image.open(rec["image"]) as im: image=im.convert("RGB").copy()
        return {"query":rec["queries"],"pos_target":image,"answer_texts":rec["answers"]}


class ImageAnswerCollator(VisualRetrieverCollator):
    """Encode page pixels and answer text as one multimodal document."""
    def __call__(self, examples):
        queries=[]; images=[]; prompts=[]
        for example in examples:
            query=example["query"]
            queries.append(random.choice(query) if isinstance(query,list) else query)
            images.append(example["pos_target"])
            answers=[x for x in example.get("answer_texts",[]) if str(x).strip()]
            # Bound text tokens so a long scraped solution cannot crowd out all
            # visual tokens or cause an unexpected memory spike.
            answer=(random.choice(answers)[:2000] if answers else "")
            suffix=("\nAnswer text:\n"+answer) if answer else "Describe the image."
            prompts.append("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"+suffix+"<|im_end|><|endoftext|>")
        query_texts=[self.processor.query_prefix+q+self.processor.query_augmentation_token*10 for q in queries]
        batch_query=self.auto_collate(query_texts,key_prefix=self.query_prefix)
        batch_doc=self.processor(text=prompts,images=[x.convert("RGB") for x in images],padding="longest",return_tensors="pt")
        offsets=batch_doc["image_grid_thw"][:,1]*batch_doc["image_grid_thw"][:,2]
        pixels=list(torch.split(batch_doc["pixel_values"],offsets.tolist()))
        batch_doc["pixel_values"]=torch.nn.utils.rnn.pad_sequence(pixels,batch_first=True)
        return {**batch_query,**{f"{self.pos_doc_prefix}{k}":v for k,v in batch_doc.items()}}


def collapse_by_page(items, root: Path):
    """Keep every query variant, but represent each page once in the dataset.

    VisualRetrieverCollator samples one query from the list on every access. Thus
    paraphrases all train the same positive page, while that page can never be an
    in-batch false negative for another paraphrase.
    """
    pages = {}
    for row in items:
        page_id = str(row["positive_page_id"])
        rec = pages.setdefault(page_id, {"queries": [], "answers": [], "image": str(root / row["positive_image"]),
                                         "subject": str(row.get("subject") or "unknown")})
        query = str(row["query"]).strip()
        if query not in rec["queries"]:
            rec["queries"].append(query)
        answer=str(row.get("positive_answer_text") or "").strip()
        if answer and answer not in rec["answers"]: rec["answers"].append(answer)
    return PageDataset(pages.values()), pages


def ranking_metrics(scores: torch.Tensor):
    """Metrics for the page-distinct in-batch candidate set."""
    order = scores.argsort(dim=1, descending=True)
    target = torch.arange(scores.size(0), device=scores.device)[:, None]
    ranks = (order == target).nonzero()[:, 1] + 1
    out = {}
    cutoffs=(1,2,3,5,10,20,30)
    for k in cutoffs: out[f"hit@{k}"] = (ranks <= k).float().mean().item()
    for k in cutoffs:
        out[f"recall@{k}"] = (ranks <= k).float().mean().item()
        out[f"precision@{k}"] = ((ranks <= k).float()/k).mean().item()
    out["mrr@10"] = torch.where(ranks <= 10, 1/ranks.float(), 0).mean().item()
    out["ndcg@10"] = torch.where(ranks <= 10, 1/torch.log2(ranks.float()+1), 0).mean().item()
    return out


class MetricTrainer(ContrastiveTrainer):
    """Official ColPali trainer plus the metrics used by the previous trainer."""
    def __init__(self, *args, subject_keys=(), **kwargs):
        super().__init__(*args, **kwargs); self._metric_buffer = []; self.subject_keys=set(subject_keys); self._subject_metrics={}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        with torch.no_grad():
            q, docs = outputs
            scores = self.loss_func._inbatch_scores(q, docs)
            idx = torch.arange(scores.size(0), device=scores.device)
            pos = scores[idx, idx]
            neg = scores.masked_fill(torch.eye(scores.size(0), device=scores.device, dtype=torch.bool), -torch.inf).max(1).values
            m = ranking_metrics(scores)
            m.update({"loss": loss.detach().item(), "positive_similarity": pos.mean().item(),
                      "hardest_negative_similarity": neg.mean().item(), "similarity_margin": (pos-neg).mean().item()})
            self._metric_buffer.append(m)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # The upstream ContrastiveTrainer bypasses compute_loss during eval;
        # route through it so validation receives the same ranking metrics.
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)
        return loss.detach(), None, None

    def log(self, logs, *args, **kwargs):
        logs["train/global_step"]=self.state.global_step
        if self._metric_buffer:
            eval_key=next((k[5:-5] for k in logs if k.startswith("eval_") and k.endswith("_loss")),None)
            prefix = ("validation/overall" if eval_key=="overall" else f"validation/subject/{eval_key}") if eval_key else "train"
            keys = self._metric_buffer[0]
            train_keys={"loss","positive_similarity","hardest_negative_similarity","similarity_margin","hit@1","recall@1","precision@1"}
            log_keys=keys if prefix.startswith("validation/") else (key for key in keys if key in train_keys)
            for key in log_keys: logs[f"{prefix}/{key}"] = statistics.fmean(x[key] for x in self._metric_buffer)
            if eval_key in self.subject_keys:
                self._subject_metrics[eval_key]={key:statistics.fmean(x[key] for x in self._metric_buffer) for key in keys}
                if self.subject_keys.issubset(self._subject_metrics):
                    for key in keys: logs[f"validation/macro/{key}"]=statistics.fmean(self._subject_metrics[s][key] for s in self.subject_keys)
                    self._subject_metrics.clear()
            self._metric_buffer.clear()
            if prefix == "train" and "train_runtime" not in logs:
                bs = self.args.per_device_train_batch_size*self.args.world_size
                logs["train/throughput_queries_s"] = bs/max(float(logs.get("step_time", 0) or 0), 1e-9) if "step_time" in logs else 0.0
                logs["train/learning_rate"] = float(logs.get("learning_rate", 0))
        return super().log(logs, *args, **kwargs)


class WandbDatasetCallback(TrainerCallback):
    def __init__(self, train_pages, val_pages, stats): self.payload = {
        "dataset/train_pages": len(train_pages), "dataset/validation_pages": len(val_pages),
        "dataset/train_query_variants": sum(len(x["queries"]) for x in train_pages.values()),
        "dataset/validation_query_variants": sum(len(x["queries"]) for x in val_pages.values()),
        "dataset/mean_queries_per_train_page": statistics.fmean(len(x["queries"]) for x in train_pages.values()),
        "dataset/max_queries_per_train_page": max(len(x["queries"]) for x in train_pages.values()),
        "dataset/train_pages_with_answer": sum(bool(x["answers"]) for x in train_pages.values()),
        "dataset/validation_pages_with_answer": sum(bool(x["answers"]) for x in val_pages.values()),
        "dataset/train_answer_coverage": statistics.fmean(bool(x["answers"]) for x in train_pages.values()),
        "dataset/mean_answers_per_train_page": statistics.fmean(len(x["answers"]) for x in train_pages.values()),
        **{f"dataset/subject/{s}/validation_pages": v["validation_pages"] for s,v in stats.items()}}
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            import wandb
            if wandb.run:
                # Force scalar histories to render as line plots sharing the
                # optimizer-step x axis; no histogram objects are logged.
                wandb.define_metric("train/global_step")
                wandb.define_metric("train/*",step_metric="train/global_step")
                wandb.define_metric("validation/*",step_metric="train/global_step")
                wandb.define_metric("dataset/*",step_metric="train/global_step")
                wandb.log({"train/global_step":state.global_step,**self.payload},step=state.global_step)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pairs",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--resume-from-checkpoint",type=Path)
    p.add_argument("--model",default="vidore/colqwen2.5-base"); p.add_argument("--pages-per-subject",type=int,default=120)
    p.add_argument("--seed",type=int,default=17); p.add_argument("--epochs",type=float,default=3)
    p.add_argument("--batch-size",type=int,default=2); p.add_argument("--grad-accum",type=int,default=8)
    p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--eval-steps",type=int,default=250); p.add_argument("--save-steps",type=int,default=250)
    p.add_argument("--wandb-project",default="turkish-colqwen25"); p.add_argument("--run-name")
    a=p.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    train_rows,val_rows,stats=fixed_subject_split(read_rows(a.pairs),a.data_root,a.pages_per_subject,a.seed)
    train_ds,train_pages=collapse_by_page(train_rows,a.data_root); val_ds,val_pages=collapse_by_page(val_rows,a.data_root)
    val_by_subject=defaultdict(list)
    for row in val_rows: val_by_subject[str(row.get("subject") or "unknown")].append(row)
    subject_eval={s.replace("/","_").replace(" ","_"):collapse_by_page(v,a.data_root)[0] for s,v in val_by_subject.items()}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/"split_stats.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
    processor=ColQwen2_5_Processor.from_pretrained(a.model,max_num_visual_tokens=768)
    attention="flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    model=ColQwen2_5.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation=attention)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=32,lora_alpha=32,lora_dropout=.05,bias="none",task_type="FEATURE_EXTRACTION",
        target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)"))
    tr=TrainingArguments(output_dir=str(a.output_dir),num_train_epochs=a.epochs,per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,gradient_accumulation_steps=a.grad_accum,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},bf16=True,learning_rate=a.lr,warmup_steps=100,logging_steps=10,
        eval_strategy="steps",eval_steps=a.eval_steps,save_strategy="steps",save_steps=a.save_steps,save_total_limit=3,
        report_to=["wandb"],run_name=a.run_name,dataloader_num_workers=8,dataloader_drop_last=True,remove_unused_columns=False,
        ddp_find_unused_parameters=False,seed=a.seed,data_seed=a.seed)
    trainer=MetricTrainer(model=model,train_dataset=train_ds,eval_dataset={"overall":val_ds,**subject_eval},args=tr,
        data_collator=ImageAnswerCollator(processor=processor,max_length=256),
        loss_func=ColbertLoss(temperature=.02,normalize_scores=True),is_vision_model=True,
        callbacks=[WandbDatasetCallback(train_pages,val_pages,stats)],subject_keys=subject_eval)
    trainer.train(resume_from_checkpoint=str(a.resume_from_checkpoint) if a.resume_from_checkpoint else None)
    trainer.save_model(str(a.output_dir)); processor.save_pretrained(a.output_dir)

if __name__=="__main__": main()
