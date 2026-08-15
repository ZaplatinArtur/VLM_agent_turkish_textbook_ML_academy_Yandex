from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from .data import dataset_fingerprint, page_id_to_image, read_jsonl, split_by_subject_pages, usable_rows, write_jsonl
from .groups import load_relevance_groups, relevant_pages
from .model import DEFAULT_MODEL, configure_trainable, encode_images, encode_text, load_encoder, save_checkpoint


class PairDataset(Dataset):
    def __init__(self, rows, data_root: Path, seed: int, relevance=None):
        self.rows, self.root, self.seed, self.relevance, self.epoch = rows, data_root, seed, relevance or {}, 0
    def set_epoch(self, epoch: int): self.epoch = epoch
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        page_id=str(row["positive_page_id"]); positives=relevant_pages(row,self.relevance)
        positives={x for x in positives if (self.root/page_id_to_image(x)).is_file()}
        negs = [str(x) for x in (row.get("hard_negative_page_ids") or []) if str(x) not in positives]
        rng = random.Random(self.seed + self.epoch * len(self.rows) + idx)
        rng.shuffle(negs)
        neg_rel = next((page_id_to_image(x) for x in negs if (self.root / page_id_to_image(x)).is_file()), row["positive_image"])
        # SupCon uses all positives present in a mini-batch; it does not require
        # materializing every globally known positive for every anchor. Always
        # include the canonical page and rotate one additional positive by epoch.
        alternatives=sorted(positives-{page_id}); rng.shuffle(alternatives)
        sampled_positives=[page_id] if page_id in positives else []
        if alternatives: sampled_positives.append(alternatives[0])
        candidates=[(x,str(self.root/page_id_to_image(x))) for x in sampled_positives]
        neg_id=next((str(x) for x in negs if (self.root/page_id_to_image(str(x))).is_file()),page_id)
        if neg_id not in positives: candidates.append((neg_id,str(self.root/page_id_to_image(neg_id))))
        return row["query"], candidates, positives, page_id

class SubjectBatchSampler:
    """Same-subject batches without duplicate positive pages in a batch."""
    def __init__(self, rows, batch_size: int, seed: int):
        self.rows,self.batch_size,self.seed,self.epoch=rows,batch_size,seed,0
    def __len__(self): return len(self.rows)//self.batch_size
    def __iter__(self):
        from collections import defaultdict, deque
        rng=random.Random(self.seed+self.epoch); self.epoch+=1; by_subject=defaultdict(lambda:defaultdict(list))
        for idx,row in enumerate(self.rows):
            by_subject[str(row.get("subject") or "unknown")][str(row["positive_page_id"])].append(idx)
        batches=[]
        for pages in by_subject.values():
            for values in pages.values(): rng.shuffle(values)
            queues={page:deque(values) for page,values in pages.items()}
            while queues:
                page_ids=list(queues); rng.shuffle(page_ids); batch=[]
                for page in page_ids[:self.batch_size]:
                    batch.append(queues[page].popleft())
                    if not queues[page]: del queues[page]
                if len(batch)==self.batch_size: batches.append(batch)
        rng.shuffle(batches); yield from batches


def collate(items):
    queries, candidate_lists, relevant_sets, ids = zip(*items)
    paths={pid:path for values in candidate_lists for pid,path in values}
    page_ids=list(paths); images=[Image.open(paths[x]).convert("RGB") for x in page_ids]
    positive_mask=torch.tensor([[pid in relevant for pid in page_ids] for relevant in relevant_sets],dtype=torch.bool)
    return list(queries),images,page_ids,positive_mask,list(ids)


def multi_positive_loss(logits, positive_mask):
    """SupCon-style mean log-probability across every relevant page."""
    if not positive_mask.any(dim=1).all(): raise ValueError("every query needs at least one positive")
    log_probs=F.log_softmax(logits,dim=1)
    return -((log_probs*positive_mask).sum(1)/positive_mask.sum(1)).mean()


def multi_relevance_metrics(order, relevant_idx):
    ranks=[rank for rank,idx in enumerate(order,1) if idx in relevant_idx]
    first=min(ranks)
    metrics={}
    for k in tuple(range(1,11))+(20,30):
        found=sum(rank<=k for rank in ranks)
        metrics[f"hit@{k}"]=float(found>0)
    for k in (1,5,10,20,30):
        found=sum(rank<=k for rank in ranks)
        metrics[f"recall@{k}"]=found/max(1,len(relevant_idx))
        metrics[f"precision@{k}"]=found/k
    mrr=1/first if first<=10 else 0.
    dcg=sum(1/math.log2(rank+1) for rank in ranks if rank<=10)
    ideal=sum(1/math.log2(rank+1) for rank in range(1,min(10,len(relevant_idx))+1))
    return metrics,mrr,dcg/max(ideal,1e-12)


@torch.no_grad()
def evaluate(model, processor, rows, root, device, limit=0, batch=8, relevance=None):
    relevance=relevance or {}
    model.eval(); sample=list(rows)
    if limit and len(sample)>limit:
        # Preserve subject balance when a debugging limit is explicitly used.
        by_subject={}
        for row in sample: by_subject.setdefault(str(row.get("subject") or "unknown"),[]).append(row)
        rng=random.Random(101); per=max(1,limit//max(1,len(by_subject)))
        sample=[row for values in by_subject.values() for row in rng.sample(values,min(per,len(values)))]
    pages = {}; page_subject = {}; validation_pages={str(r["positive_page_id"]) for r in sample}
    for r in sample:
        canonical=str(r["positive_page_id"]); rel=str(r["positive_image"])
        if (root/rel).is_file(): pages[canonical]=str(root/rel); page_subject[canonical]=str(r.get("subject") or "unknown")
    page_ids, image_vecs = list(pages), []
    for i in range(0, len(page_ids), batch):
        ims = [Image.open(pages[x]).convert("RGB") for x in page_ids[i:i+batch]]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            image_vecs.append(encode_images(model, processor, ims, device).cpu())
    corpus = torch.cat(image_vecs)
    keys=[f"hit@{k}" for k in tuple(range(1,11))+(20,30)]+[f"{name}@{k}" for k in (1,5,10,20,30) for name in ("recall","precision")]+["mrr@10","ndcg@10"]
    sums={key:0. for key in keys}; subject_sums={}; subject_counts={}
    losses=[]
    for i in range(0, len(sample), batch):
        part=sample[i:i+batch]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            q=encode_text(model,processor,[x["query"] for x in part],device).cpu()
        scores=q@corpus.T; filtered_scores=scores.clone()
        for j,row in enumerate(part):
            # Runtime retrieval filters by subject; validation must match it.
            allowed=torch.tensor([page_subject[x] == str(row.get("subject") or "unknown") for x in page_ids])
            filtered=scores[j].clone(); filtered[~allowed]=-1e9; filtered_scores[j]=filtered
            canonical=str(row["positive_page_id"]); relevant=relevant_pages(row,relevance)&validation_pages
            relevant_idx={page_ids.index(x) for x in relevant if x in pages}
            order=torch.argsort(filtered,descending=True)
            values,mrr,ndcg=multi_relevance_metrics(order.tolist(),relevant_idx)
            values={**values,"mrr@10":mrr,"ndcg@10":ndcg}
            subject=str(row.get("subject") or "unknown")
            subject_sums.setdefault(subject,{key:0. for key in keys}); subject_counts[subject]=subject_counts.get(subject,0)+1
            for key,value in values.items(): sums[key]+=value; subject_sums[subject][key]+=value
        positive_mask=torch.zeros_like(filtered_scores,dtype=torch.bool)
        for j,row in enumerate(part):
            canonical=str(row["positive_page_id"]); relevant=relevant_pages(row,relevance)&validation_pages
            for x in relevant:
                if x in pages: positive_mask[j,page_ids.index(x)]=True
        losses.append(multi_positive_loss(filtered_scores/0.02,positive_mask).item())
    n=max(1,len(sample)); result={f"validation/overall/{k}":v/n for k,v in sums.items()}
    for subject,values in subject_sums.items():
        safe=subject.replace("/","_").replace(" ","_")
        for key,value in values.items(): result[f"validation/subject/{safe}/{key}"]=value/max(1,subject_counts[subject])
        result[f"validation/subject/{safe}/queries"]=subject_counts[subject]
    result["validation/overall/loss"]=sum(losses)/max(1,len(losses)); result["validation/overall/queries"]=len(sample)
    # Macro average gives every subject equal weight irrespective of size.
    for key in keys:
        result[f"validation/macro/{key}"]=sum(values[key]/max(1,subject_counts[s]) for s,values in subject_sums.items())/max(1,len(subject_sums))
    model.train(); return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pairs",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--splits-dir",type=Path,required=True); p.add_argument("--model",default=DEFAULT_MODEL); p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--grad-accum",type=int,default=4); p.add_argument("--epochs",type=float,default=1.0); p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--temperature",type=float,default=.02); p.add_argument("--max-steps",type=int); p.add_argument("--val-limit",type=int,default=0)
    p.add_argument("--val-ratio-per-subject",type=float,default=.08)
    p.add_argument("--log-every",type=int,default=10); p.add_argument("--seed",type=int,default=17); p.add_argument("--wandb-project",default="turkish-visrag")
    p.add_argument("--eval-every",type=int,default=750); p.add_argument("--subject-batches",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--relevance-groups",type=Path)
    p.add_argument("--unfreeze-blocks",type=int,default=5)
    a=p.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    rows=usable_rows(read_jsonl(a.pairs),a.data_root); splits,split_stats=split_by_subject_pages(rows,seed=a.seed,val_ratio_per_subject=a.val_ratio_per_subject)
    for name,part in splits.items(): write_jsonl(a.splits_dir/f"{name}.jsonl",part)
    print(json.dumps({"usable":len(rows),**{k:len(v) for k,v in splits.items()},"subject_page_splits":split_stats,"fingerprint":dataset_fingerprint(rows)},ensure_ascii=False),flush=True)
    device=torch.device("cuda"); model,processor=load_encoder(a.model,device=device); trainable=configure_trainable(model,a.unfreeze_blocks); model.train()
    # Adam updates fp16 parameters unreliably on V100. Keep fp32 master weights
    # for the small trainable subset and use autocast + GradScaler for compute.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    relevance=load_relevance_groups(a.relevance_groups)
    dataset=PairDataset(splits["train"],a.data_root,a.seed,relevance)
    if a.subject_batches:
        loader=DataLoader(dataset,batch_sampler=SubjectBatchSampler(splits["train"],a.batch_size,a.seed),num_workers=4,pin_memory=True,collate_fn=collate)
    else:
        loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=True,num_workers=4,pin_memory=True,collate_fn=collate,drop_last=True)
    max_steps=a.max_steps or math.ceil(len(loader)*a.epochs/a.grad_accum); opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=a.lr,weight_decay=.01)
    sched=get_cosine_schedule_with_warmup(opt,max(1,int(max_steps*.05)),max_steps)
    scaler=torch.amp.GradScaler("cuda")
    import wandb
    run=wandb.init(project=a.wandb_project,config={**vars(a),"pairs":str(a.pairs),"output":str(a.output),"trainable":trainable,"subject_page_splits":split_stats,"grades":"1-12","checkpoint_selection_metric":"validation/macro/hit@3"})
    wandb.define_metric("train/step")
    wandb.define_metric("train/*",step_metric="train/step")
    wandb.define_metric("validation/*",step_metric="train/step")
    print(f"trainable={trainable} max_steps={max_steps} wandb_run={run.id}",flush=True)
    opt.zero_grad(set_to_none=True); step=micro=0; started=time.time(); rolling=[]; best_selection_metric=-1.0
    required_epochs=(math.ceil(max_steps*a.grad_accum/max(1,len(loader))) if a.max_steps else math.ceil(a.epochs))
    for epoch in range(required_epochs):
      dataset.set_epoch(epoch)
      for queries,images,page_ids,positive_mask,_ in loader:
        t0=time.time()
        with torch.autocast(device_type="cuda",dtype=torch.float16):
          q=encode_text(model,processor,queries,device); im=encode_images(model,processor,images,device)
          logits=q@im.T/a.temperature; positive_mask=positive_mask.to(device)
          loss=multi_positive_loss(logits,positive_mask)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at microstep {micro+1}: {loss.item()}")
        scaler.scale(loss/a.grad_accum).backward(); micro+=1
        similarities=logits*a.temperature
        pos=(similarities.masked_fill(~positive_mask,0).sum(-1)/positive_mask.sum(-1)).mean()
        masked=similarities.masked_fill(positive_mask,torch.finfo(similarities.dtype).min); hardest=masked.max(-1).values
        rolling.append((loss.item(),pos.item(),hardest.mean().item(),len(queries)/(time.time()-t0)))
        if micro%a.grad_accum: continue
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True); step+=1
        if step%a.log_every==0 or step==1:
          vals=[sum(x[i] for x in rolling)/len(rolling) for i in range(4)]; rolling=[]
          log={"train/loss":vals[0],"train/positive_similarity":vals[1],"train/hardest_negative_similarity":vals[2],"train/similarity_margin":vals[1]-vals[2],"train/learning_rate":sched.get_last_lr()[0],"train/throughput_queries_s":vals[3],"train/step":step}
          wandb.log(log,step=step); print(json.dumps(log),flush=True)
        if a.eval_every and step%a.eval_every==0:
          metrics=evaluate(model,processor,splits["val"],a.data_root,device,a.val_limit,a.batch_size,relevance); metrics["train/step"]=step
          wandb.log(metrics,step=step); print(json.dumps(metrics),flush=True)
          if metrics["validation/macro/hit@3"]>best_selection_metric:
            best_selection_metric=metrics["validation/macro/hit@3"]
            save_checkpoint(model,processor,a.output/"best",{"step":step,"metrics":metrics,"split":"subject_stratified_pages_8pct"})
        if step>=max_steps: break
      if step>=max_steps: break
    metrics=evaluate(model,processor,splits["val"],a.data_root,device,a.val_limit,a.batch_size,relevance); wandb.log(metrics,step=step); print(json.dumps(metrics),flush=True)
    meta={"base_model":a.model,"train_rows":len(splits["train"]),"val_rows":len(splits["val"]),"steps":step,"seconds":time.time()-started,"metrics":metrics,"split":"subject_stratified_pages_8pct","subject_page_splits":split_stats,"hard_negatives":True,"unfreeze_blocks":a.unfreeze_blocks,"checkpoint_selection_metric":"validation/macro/hit@3"}
    save_checkpoint(model,processor,a.output,meta); run.finish(); print(f"saved={a.output}",flush=True)
if __name__=="__main__": main()
