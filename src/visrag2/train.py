from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

from .data import PageDataset, SubjectBatchSampler, build_records, split_by_group
from .evaluate import evaluate_corpus
from .loss import maxsim_scores, multi_positive_infonce


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="athrael-soju/colqwen3.5-4.5B-v3")
    p.add_argument("--resume", help="checkpoint path or 'auto' for the latest checkpoint")
    p.add_argument("--pages-per-subject", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=4.57e-5)
    p.add_argument("--temperature", type=float, default=.02)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--deadline", default="11:00", help="Europe/Moscow HH:MM; next occurrence")
    p.add_argument("--stop-buffer-minutes", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--wandb-project", default="turkish-visrag2")
    p.add_argument("--wandb-entity", default="mr-magno-hse")
    p.add_argument("--run-name", default="colqwen25-v02-multipositive")
    return p.parse_args()


def deadline_timestamp(value: str, buffer_minutes: int) -> float:
    tz = ZoneInfo("Europe/Moscow"); now = datetime.now(tz)
    hour, minute = map(int, value.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now: target += timedelta(days=1)
    return (target - timedelta(minutes=buffer_minutes)).timestamp()


def collate(records): return records


def embeddings(output):
    return output.embeddings if hasattr(output, "embeddings") else output


def encode_batch(model, processor, records, device):
    queries = [random.choice(record.queries) for record in records]
    images = []
    for record in records:
        with Image.open(record.image) as image: images.append(image.convert("RGB").copy())
    query_batch = processor.process_queries(queries)
    document_batch = processor.process_images(images=images)
    query_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in query_batch.items()}
    document_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in document_batch.items()}
    return embeddings(model(**query_batch)), embeddings(model(**document_batch))


def checkpoint_step(path: Path) -> int:
    try: return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError): return -1


def resolve_resume(value: str | None, output: Path) -> Path | None:
    if not value: return None
    if value != "auto":
        path = Path(value)
        if not path.is_dir(): raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    checkpoints = [path for path in output.glob("checkpoint-*") if (path/"accelerate_state").is_dir()]
    return max(checkpoints, key=checkpoint_step) if checkpoints else None


def save_checkpoint(accelerator, model, processor, output: Path, step: int):
    accelerator.wait_for_everyone()
    target = output / f"checkpoint-{step}"
    state_dir = target / "accelerate_state"
    accelerator.save_state(str(state_dir))
    if accelerator.is_main_process:
        target.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(target)
        processor.save_pretrained(target)
        (target/"step.json").write_text(json.dumps({"step": step}), encoding="utf-8")
        (output/"latest_checkpoint.txt").write_text(str(target.resolve()), encoding="utf-8")
    accelerator.wait_for_everyone()


def flatten_metrics(metrics):
    out = {f"validation/macro/{k}": v for k, v in metrics["macro"].items()}
    out.update({f"validation/micro/{k}": v for k, v in metrics["micro"].items() if k not in {"pages", "queries"}})
    for subject, values in metrics["subjects"].items():
        name = subject.replace("/", "_")
        out.update({f"validation/subject/{name}/{k}": v for k, v in values.items()})
    return out


def main():
    a = parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    accelerator = Accelerator(gradient_accumulation_steps=a.grad_accum, mixed_precision="bf16", log_with="wandb")
    records = build_records(a.pairs, a.groups, a.data_root)
    train_records, val_records, stats = split_by_group(records, a.pages_per_subject, a.seed)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        (a.output_dir/"split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    processor = ColQwen3_5Processor.from_pretrained(a.model, max_num_visual_tokens=768)
    # SDPA is available in stock PyTorch and avoids a fragile compiled
    # flash-attn dependency. On A100 it is still memory efficient.
    model = ColQwen3_5.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                       attn_implementation="sdpa")
    model.config.use_cache = False
    resume_path = resolve_resume(a.resume, a.output_dir)
    if resume_path:
        model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=64, lora_dropout=.197, bias="none", task_type="FEATURE_EXTRACTION",
            target_modules=r"(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*custom_text_proj.*)",
        ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    dataset = PageDataset(train_records, a.seed)
    sampler = SubjectBatchSampler(train_records, a.batch_size, a.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=8, pin_memory=True)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=a.lr, weight_decay=.02)
    total_steps = max(1, len(loader)*a.epochs//a.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(total_steps*.08)), total_steps)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
    accelerator.init_trackers(a.wandb_project, config=config,
                              init_kwargs={"wandb": {"name": a.run_name, "entity": a.wandb_entity}})
    stop_at = deadline_timestamp(a.deadline, a.stop_buffer_minutes)
    step = 0
    if resume_path:
        accelerator.load_state(str(resume_path/"accelerate_state"))
        step_file = resume_path/"step.json"
        step = int(json.loads(step_file.read_text(encoding="utf-8"))["step"]) if step_file.is_file() else checkpoint_step(resume_path)
        accelerator.print(f"Resumed complete training state from {resume_path} at step {step}")

    should_stop = False
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch); model.train()
        for records_batch in loader:
            if datetime.now().timestamp() >= stop_at: should_stop = True; break
            with accelerator.accumulate(model):
                query_embeddings, doc_embeddings = encode_batch(model, processor, records_batch, accelerator.device)
                # Keep negatives subject-local. Accelerate may assign different
                # subject batches to different ranks on the same optimizer step,
                # so cross-rank gathering would introduce forbidden negatives.
                scores = maxsim_scores(query_embeddings, doc_embeddings)
                groups = [record.group_id for record in records_batch]
                positive_mask = torch.tensor([[left == right for right in groups] for left in groups], device=scores.device)
                loss = multi_positive_infonce(scores, positive_mask, a.temperature)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if accelerator.sync_gradients:
                step += 1
                accelerator.log({"train/loss": loss.detach().float().item(), "train/lr": scheduler.get_last_lr()[0]}, step=step)
                if step % a.save_every == 0: save_checkpoint(accelerator, model, processor, a.output_dir, step)
                if step % a.eval_every == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        metrics = evaluate_corpus(accelerator.unwrap_model(model), processor, val_records, accelerator.device)
                        accelerator.log(flatten_metrics(metrics), step=step)
                        (a.output_dir/f"metrics-{step}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                    model.train(); accelerator.wait_for_everyone()
        if should_stop: break

    save_checkpoint(accelerator, model, processor, a.output_dir, step)
    if accelerator.is_main_process and datetime.now().timestamp() < stop_at:
        metrics = evaluate_corpus(accelerator.unwrap_model(model), processor, val_records, accelerator.device)
        accelerator.log(flatten_metrics(metrics), step=step)
        (a.output_dir/"metrics-final.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    accelerator.end_training()


if __name__ == "__main__": main()
