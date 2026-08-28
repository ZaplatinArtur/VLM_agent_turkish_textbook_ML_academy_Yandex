from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from PIL import Image
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .data import PageDataset, SubjectGlobalBatchSampler, build_records, split_by_group
from .evaluate import evaluate_corpus
from .loss import symmetric_multi_positive_loss
from .model import configure_last_blocks, encode_images, encode_texts, load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, required=True); p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="models/visrag_siglip_e5_v3")
    p.add_argument("--resume", help="checkpoint directory or 'auto'")
    p.add_argument("--resume-model-only", action="store_true", help="load model weights but reset optimizer/scheduler")
    p.add_argument("--pages-per-subject", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=64, help="physical queries/images per GPU")
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--max-steps", type=int)
    p.add_argument("--lr", type=float, default=5e-6); p.add_argument("--temperature", type=float, default=.05)
    p.add_argument("--unfreeze-blocks", type=int, default=3); p.add_argument("--weight-decay", type=float, default=.02)
    p.add_argument("--eval-every", type=int, default=200); p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--deadline", default="11:00"); p.add_argument("--stop-buffer-minutes", type=int, default=10)
    p.add_argument("--seed", type=int, default=17); p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--wandb-project", default="turkish-visrag-v3-a100")
    p.add_argument("--wandb-entity", default="mr-magno-hse"); p.add_argument("--run-name", default="siglip-e5-v3-3blocks-bs128")
    return p.parse_args()


def deadline_timestamp(value, buffer_minutes):
    now = datetime.now(ZoneInfo("Europe/Moscow")); hour, minute = map(int, value.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now: target += timedelta(days=1)
    return (target-timedelta(minutes=buffer_minutes)).timestamp()


def stable_hash(text): return int.from_bytes(hashlib.blake2b(text.encode(),digest_size=8).digest(),"big") & (2**63-1)
def collate(records): return records


def gather_tensor_with_grad(tensor):
    if not dist.is_initialized(): return tensor
    return torch.cat(differentiable_all_gather(tensor), dim=0)


def gather_ids(values, device):
    local = torch.tensor([stable_hash(x) for x in values], dtype=torch.long, device=device)
    if not dist.is_initialized(): return local
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return torch.cat(gathered)


def encode_batch(model, processor, records, device, epoch):
    queries = []
    for record in records:
        rng = random.Random(f"{epoch}:{record.page_id}")
        queries.append(record.queries[rng.randrange(len(record.queries))])
    images = []
    for record in records:
        with Image.open(record.image) as image: images.append(image.convert("RGB").copy())
    return encode_texts(model,processor,queries,device), encode_images(model,processor,images,device)


def checkpoint_step(path):
    try: return int(path.name.rsplit("-",1)[1])
    except Exception: return -1


def resolve_resume(value, output):
    if not value: return None
    if value != "auto":
        path = Path(value)
        if not (path/"accelerate_state").is_dir(): raise FileNotFoundError(path)
        return path
    paths = [p for p in output.glob("checkpoint-*") if (p/"accelerate_state").is_dir() and (p/"trainer_state.json").is_file()]
    return max(paths,key=checkpoint_step) if paths else None


def save_checkpoint(accelerator, model, processor, output, state):
    accelerator.wait_for_everyone(); target = output/f"checkpoint-{state['step']}"
    accelerator.save_state(str(target/"accelerate_state"))
    if accelerator.is_main_process:
        processor.save_pretrained(output/"processor")
        (target/"trainer_state.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
        (output/"latest_checkpoint.txt").write_text(str(target.resolve()),encoding="utf-8")
    accelerator.wait_for_everyone()


def save_hf(accelerator, model, processor, target, meta):
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        target.mkdir(parents=True,exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(target,state_dict=state,safe_serialization=True)
        processor.save_pretrained(target)
        (target/"visrag_meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    accelerator.wait_for_everyone()


def flatten_metrics(metrics):
    out = {f"validation/macro/{k}":v for k,v in metrics["macro"].items()}
    out.update({f"validation/micro/{k}":v for k,v in metrics["micro"].items()})
    for subject, values in metrics["subjects"].items():
        safe = subject.replace("/","_").replace(" ","_")
        out.update({f"validation/subject/{safe}/{k}":v for k,v in values.items()})
    return out


def evaluate_and_log(accelerator, model, processor, validation, output, step, name):
    accelerator.wait_for_everyone(); metrics = None
    if accelerator.is_main_process:
        metrics = evaluate_corpus(accelerator.unwrap_model(model),processor,validation,accelerator.device,batch_size=64)
        values = flatten_metrics(metrics)
        accelerator.log(values,step=step)
        # Explicit Comet call: validation is evaluated only on rank zero, so
        # send these scalar values directly to guarantee `metric vs step` panels.
        if os.environ.get("TRACKER") == "comet_ml":
            accelerator.get_tracker("comet_ml", unwrap=True).log_metrics(values, step=step)
        (output/f"metrics-{name}.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    accelerator.wait_for_everyone(); model.train()
    values = [metrics["macro"]["hit@3"] if metrics else None]
    broadcast_object_list(values)
    return values[0], metrics


def main():
    a=parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    tracker = os.environ.get("TRACKER", "wandb")
    accelerator=Accelerator(mixed_precision=os.environ.get("MIXED_PRECISION", "bf16"),log_with=tracker)
    if accelerator.num_processes < 2: raise RuntimeError("launch with at least two GPU processes")
    records=build_records(a.pairs,a.groups,a.data_root); train,val,stats=split_by_group(records,a.pages_per_subject,a.seed)
    if any(v["validation_pages"] != a.pages_per_subject for v in stats.values()): raise ValueError(f"not 120 pages per subject: {stats}")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    resume=resolve_resume(a.resume,a.output_dir)
    model,processor=load_model(a.model); trainable=configure_last_blocks(model,a.unfreeze_blocks)
    if hasattr(model,"gradient_checkpointing_enable"): model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    dataset=PageDataset(train); global_batch=a.batch_size*accelerator.num_processes
    sampler=SubjectGlobalBatchSampler(train,global_batch,a.seed,drop_last=True)
    loader=DataLoader(dataset,batch_sampler=sampler,collate_fn=collate,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
    optimizer=AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr,weight_decay=a.weight_decay)
    steps_per_epoch=len(loader); total_steps=a.max_steps or steps_per_epoch*a.epochs
    scheduler=get_cosine_schedule_with_warmup(optimizer,max(1,int(total_steps*.05)),total_steps)
    model,optimizer,scheduler=accelerator.prepare(model,optimizer,scheduler)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update({"physical_batch_per_gpu":a.batch_size,"global_contrastive_batch":global_batch,
                   "gradient_accumulation":1,"trainable_parameters":trainable,"split_stats":stats,
                   "loss":"symmetric_multi_positive_supcon","base_is_trained_siglip":True})
    if tracker == "comet_ml":
        accelerator.init_trackers(a.wandb_project, config=config)
    else:
        run_id_file=a.output_dir/"wandb_run_id.txt"; ids=[None]
        if accelerator.is_main_process:
            import wandb
            ids[0]=run_id_file.read_text().strip() if resume and run_id_file.is_file() else wandb.util.generate_id()
            run_id_file.write_text(ids[0],encoding="utf-8")
        broadcast_object_list(ids)
        accelerator.init_trackers(a.wandb_project,config=config,init_kwargs={"wandb":{
            "name":a.run_name,"entity":a.wandb_entity,"id":ids[0],"resume":"allow","mode":os.environ.get("WANDB_MODE", "online")}})
    state={"step":0,"epoch":0,"next_batch":0,"best_macro_hit3":-1.0,"base_model":a.model}
    if resume:
        if a.resume_model_only:
            from safetensors.torch import load_file
            weights = load_file(str(resume/"accelerate_state"/"model.safetensors"), device="cpu")
            accelerator.unwrap_model(model).load_state_dict(weights, strict=True)
        else:
            accelerator.load_state(str(resume/"accelerate_state"))
        state.update(json.loads((resume/"trainer_state.json").read_text()))
        accelerator.print(f"resumed={resume} model_only={a.resume_model_only} state={state}")
    stop_at=deadline_timestamp(a.deadline,a.stop_buffer_minutes); started=time.time(); stop=False
    for epoch in range(state["epoch"],a.epochs):
        sampler.set_epoch(epoch); active_loader=loader
        skip=state["next_batch"] if epoch==state["epoch"] else 0
        if skip: active_loader=accelerator.skip_first_batches(loader,skip)
        for local_index, batch in enumerate(active_loader,start=skip):
            if time.time()>=stop_at or state["step"]>=total_steps: stop=True; break
            start = accelerator.process_index*a.batch_size
            batch = batch[start:start+a.batch_size]
            if len(batch) != a.batch_size: continue
            subjects=gather_ids([r.subject for r in batch],accelerator.device)
            if torch.unique(subjects).numel()!=1: raise RuntimeError("cross-subject negatives detected")
            queries,documents=encode_batch(model,processor,batch,accelerator.device,epoch)
            global_queries=gather_tensor_with_grad(queries); global_documents=gather_tensor_with_grad(documents)
            group_ids=gather_ids([r.group_id for r in batch],accelerator.device)
            positive_mask=group_ids[:,None].eq(group_ids[None,:])
            loss,scores=symmetric_multi_positive_loss(global_queries,global_documents,positive_mask,a.temperature)
            accelerator.backward(loss)
            if dist.is_initialized():
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad.div_(dist.get_world_size())
            accelerator.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            state.update({"step":state["step"]+1,"epoch":epoch,"next_batch":local_index+1})
            with torch.no_grad():
                pos=scores[positive_mask].mean(); neg=scores.masked_fill(positive_mask,-torch.inf).max(1).values.mean()
            accelerator.log({"train/loss":loss.item(),"train/lr":scheduler.get_last_lr()[0],
                             "train/positive_similarity":pos.item(),"train/hardest_negative_similarity":neg.item(),
                             "train/similarity_margin":(pos-neg).item(),"train/physical_batch_per_gpu":len(batch),
                             "train/global_contrastive_pool":len(group_ids)},step=state["step"])
            if state["step"]%a.save_every==0: save_checkpoint(accelerator,model,processor,a.output_dir,state)
            if state["step"]%a.eval_every==0:
                score,metrics=evaluate_and_log(accelerator,model,processor,val,a.output_dir,state["step"],str(state["step"]))
                if score>state["best_macro_hit3"]:
                    state["best_macro_hit3"]=score
                    save_hf(accelerator,model,processor,a.output_dir/"best_model",{"step":state["step"],"metrics":metrics,"config":config})
        state.update({"epoch":epoch+1,"next_batch":0})
        if stop: break
    save_checkpoint(accelerator,model,processor,a.output_dir,state)
    score,metrics=evaluate_and_log(accelerator,model,processor,val,a.output_dir,state["step"],"final")
    save_hf(accelerator,model,processor,a.output_dir/"final_model",{
        "step":state["step"],"seconds":time.time()-started,"metrics":metrics,"config":config})
    accelerator.end_training()


if __name__=="__main__": main()
