from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import page_id_to_image, read_jsonl
from .model import encode_images, encode_text, load_encoder


def select_top_indices(
    scores: np.ndarray,
    subjects: np.ndarray,
    grades: np.ndarray,
    *,
    top_k: int,
    subject=None,
    grade=None,
    min_score: float | None = None,
) -> list[int]:
    """Exact filtered top-k without sorting the complete page corpus."""
    if top_k < 1 or scores.size == 0:
        return []
    mask = np.ones(scores.shape[0], dtype=bool)
    if subject is not None:
        mask &= subjects == str(subject)
    if grade is not None:
        mask &= grades == str(grade)
    if min_score is not None:
        mask &= scores >= min_score
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        return []

    if candidates.size > top_k:
        local = np.argpartition(-scores[candidates], top_k - 1)[:top_k]
        candidates = candidates[local]
    # Deterministic exact ordering for the small selected set.
    order = np.lexsort((candidates, -scores[candidates]))
    return [int(index) for index in candidates[order]]


class VisRAGSiglipIndex:
    def __init__(self, index_dir: Path, *, load_model_now: bool = True):
        self.index_dir = Path(index_dir)
        self.meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        self.pages = read_jsonl(self.index_dir / "pages.jsonl")
        self.embeddings = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
        if len(self.pages) != len(self.embeddings):
            raise ValueError("pages.jsonl and embeddings.npy have different lengths")
        self.subjects = np.asarray([str(page.get("subject")) for page in self.pages])
        self.grades = np.asarray([str(page.get("grade")) for page in self.pages])
        self.model = self.processor = self.device = None
        if load_model_now: self.ensure_model()
    def ensure_model(self):
        if self.model is None:
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype=torch.float16 if self.device.type=="cuda" else torch.float32
            self.model,self.processor=load_encoder(self.meta["model"],dtype=dtype,device=self.device); self.model.eval()
    @torch.inference_mode()
    def search(self, query: str, top_k: int = 5, subject=None, grade=None, min_score: float | None = None):
        if len(query.strip()) < 3: return []
        self.ensure_model()
        context=torch.autocast("cuda",dtype=torch.float16) if self.device.type=="cuda" else torch.autocast("cpu",enabled=False)
        with context: q=encode_text(self.model,self.processor,[query],self.device)[0].float().cpu().numpy()
        scores=np.asarray(self.embeddings)@q
        order=select_top_indices(
            scores,self.subjects,self.grades,top_k=top_k,
            subject=subject,grade=grade,min_score=min_score,
        )
        hits=[]
        for idx in order:
            page=self.pages[int(idx)]; score=float(scores[int(idx)])
            hits.append({**page,"score":score,"rank":len(hits)+1})
        return hits


def build(model_dir: Path, pairs: Path, root: Path, output: Path, batch: int, max_pages: int | None):
    rows=read_jsonl(pairs); unique={}
    for r in rows:
        canonical=str(r.get("positive_page_id") or "")
        page_ids=[canonical,*[str(x) for x in (r.get("same_source_page_ids") or [])]]
        for pid in dict.fromkeys(x for x in page_ids if x):
            rel=str(r.get("positive_image") or "") if pid==canonical else page_id_to_image(pid)
            if rel and (root/rel).is_file() and pid not in unique:
                unique[pid]={"page_id":pid,"page_image":rel,"book_slug":r.get("book_slug"),"grade":r.get("grade"),"subject":r.get("subject")}
                if max_pages and len(unique)>=max_pages: break
        if max_pages and len(unique)>=max_pages: break
    pages=list(unique.values()); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,processor=load_encoder(model_dir,dtype=torch.float16 if device.type=="cuda" else torch.float32,device=device); model.eval(); vectors=[]; started=time.time()
    for i in range(0,len(pages),batch):
        ims=[Image.open(root/x["page_image"]).convert("RGB") for x in pages[i:i+batch]]
        with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16,enabled=device.type=="cuda"):
            vectors.append(encode_images(model,processor,ims,device).float().cpu().numpy())
        if (i//batch)%25==0: print(f"indexed={min(i+batch,len(pages))}/{len(pages)}",flush=True)
    output.mkdir(parents=True,exist_ok=True); np.save(output/"embeddings.npy",np.concatenate(vectors))
    with (output/"pages.jsonl").open("w",encoding="utf-8") as f:
        for p in pages: f.write(json.dumps(p,ensure_ascii=False)+"\n")
    (output/"meta.json").write_text(json.dumps({"model":str(model_dir.resolve()),"pages":len(pages),"seconds":time.time()-started,"scoring":"cosine"},indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",type=Path,required=True); p.add_argument("--pairs",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--max-pages",type=int)
    a=p.parse_args(); build(a.model,a.pairs,a.data_root,a.output,a.batch_size,a.max_pages)
if __name__=="__main__": main()
