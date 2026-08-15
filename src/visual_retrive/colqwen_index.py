"""Two-stage ColQwen2.5 page index: pooled shortlist, then exact MaxSim."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .page_index import PageRecord, load_page_records
from .paths import PROJECT_ROOT, VISUAL_RETRIVE_DIR

DEFAULT_COLQWEN_MODEL = "vidore/colqwen2.5-base"
DEFAULT_COLQWEN_ADAPTER = PROJECT_ROOT / "models" / "colqwen25_turkish"
DEFAULT_COLQWEN_INDEX_DIR = PROJECT_ROOT / "models" / "colqwen25_turkish_full_index"

def _data_root() -> Path:
    return VISUAL_RETRIVE_DIR if (VISUAL_RETRIVE_DIR / "books").is_dir() else PROJECT_ROOT

def load_colqwen_pages(bundles_path=None, *, require_solution=False, max_pages=None):
    root=_data_root(); path=Path(bundles_path) if bundles_path else root/"catalog"/"page_bundles.cleaned.jsonl"
    if not path.is_file(): path=root/"catalog"/"page_bundles.jsonl"
    rows=[]
    for line in path.open(encoding="utf-8"):
        if not line.strip(): continue
        r=json.loads(line); rel=str(r.get("page_image") or "")
        if not rel or not (root/rel).is_file() or (require_solution and not r.get("has_solution")): continue
        rows.append(PageRecord(str(r["page_id"]),str(r.get("book_slug") or ""),int(r.get("page_number") or 0),
            r.get("grade"),r.get("subject"),rel,str(r.get("answer_text") or "")[:4000],bool(r.get("has_solution"))))
        if max_pages and len(rows)>=max_pages: break
    return rows


def _document_prompt(answer: str) -> str:
    answer = str(answer or "").strip()[:2000]
    suffix = "\nAnswer text:\n" + answer if answer else "Describe the image."
    return (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        + suffix + "<|im_end|><|endoftext|>"
    )


def _load_encoder(model_name: str, adapter: Path | None):
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from peft import PeftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = ColQwen2_5.from_pretrained(model_name, torch_dtype=dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    model = model.eval().to(device)
    processor = ColQwen2_5_Processor.from_pretrained(str(adapter or model_name))
    return model, processor, device


def _pool(tokens: torch.Tensor) -> torch.Tensor:
    return F.normalize(tokens.float().mean(0), p=2, dim=-1)


@dataclass
class ColQwenCascadeIndex:
    pages: list[PageRecord]
    offsets: np.ndarray
    tokens: np.ndarray
    pooled: np.ndarray
    meta: dict[str, Any]
    model: Any | None = None
    processor: Any | None = None
    device: torch.device | None = None

    @classmethod
    def load(cls, index_dir: Path, *, load_model: bool = False):
        root = Path(index_dir)
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        pages = []
        for line in (root / "pages.jsonl").open(encoding="utf-8"):
            if not line.strip(): continue
            r = json.loads(line)
            pages.append(PageRecord(str(r["page_id"]), str(r.get("book_slug") or ""),
                int(r.get("page_number") or 0), r.get("grade"), r.get("subject"),
                str(r.get("page_image") or ""), str(r.get("answer_text") or ""),
                bool(r.get("has_solution"))))
        obj = cls(pages, np.load(root / "offsets.npy"),
            np.load(root / "tokens.f16.npy", mmap_mode="r"),
            np.load(root / "pooled.f16.npy", mmap_mode="r"), meta)
        if load_model: obj.ensure_encoder()
        return obj

    def ensure_encoder(self):
        if self.model is None:
            self.model, self.processor, self.device = _load_encoder(
                str(self.meta["model"]), Path(self.meta["adapter"]))

    @torch.inference_mode()
    def encode_query(self, query: str) -> torch.Tensor:
        self.ensure_encoder()
        text = self.processor.query_prefix + query + self.processor.query_augmentation_token * 10
        batch = self.processor.process_texts([text])
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
        emb = self.model(**batch)[0].float()
        return F.normalize(emb, p=2, dim=-1).cpu()

    def _ids(self, subject, grade):
        sn = str(subject).strip().lower() if subject is not None else None
        gn = str(grade).strip().lower() if grade is not None else None
        return [i for i,p in enumerate(self.pages)
            if (sn is None or str(p.subject or "").strip().lower() == sn)
            and (gn is None or str(p.grade or "").strip().lower() == gn)]

    def search(self, query: str, *, top_k=5, subject=None, grade=None,
               candidate_ids=None, shortlist_size=None):
        if len((query or "").strip()) < 2: return []
        q = self.encode_query(query.strip())
        ids = candidate_ids if candidate_ids is not None else self._ids(subject, grade)
        if not ids: return []
        limit = int(shortlist_size or self.meta.get("shortlist_size") or 500)
        q_pool = _pool(q).numpy()
        pool_scores = np.asarray(self.pooled[np.asarray(ids)], dtype=np.float32) @ q_pool
        take = min(max(limit, top_k), len(ids))
        local = np.argpartition(-pool_scores, take - 1)[:take]
        shortlist = [ids[int(j)] for j in local]
        dev = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qg = q.to(dev, dtype=torch.float16)
        scored = []
        for start in range(0, len(shortlist), 64):
            chunk = shortlist[start:start+64]
            lengths = [int(self.offsets[i+1]-self.offsets[i]) for i in chunk]
            mx = max(lengths)
            docs = torch.zeros((len(chunk), mx, q.shape[-1]), dtype=torch.float16)
            mask = torch.zeros((len(chunk), mx), dtype=torch.bool)
            for j,(i,n) in enumerate(zip(chunk,lengths)):
                docs[j,:n] = torch.from_numpy(np.asarray(self.tokens[self.offsets[i]:self.offsets[i+1]]))
                mask[j,:n] = True
            docs, mask = docs.to(dev), mask.to(dev)
            sim = torch.einsum("qd,bsd->qbs", qg, docs).masked_fill(~mask.unsqueeze(0), -1e4)
            scores = sim.amax(-1).sum(0).float().cpu().numpy()
            scored.extend((float(s), i) for s,i in zip(scores,chunk))
        scored.sort(reverse=True)
        hits=[]
        for rank,(score,i) in enumerate(scored[:top_k],1):
            p=self.pages[i]; row=p.as_dict(); row.update(score=score, rank=rank)
            hits.append(row)
        return hits


@torch.inference_mode()
def encode_shard(pages, shard_dir: Path, *, model_name, adapter, batch_size=2):
    shard_dir.mkdir(parents=True, exist_ok=True)
    chunks = shard_dir / "chunks"; chunks.mkdir(exist_ok=True)
    model, processor, device = _load_encoder(model_name, adapter)
    for start in range(0, len(pages), batch_size):
        path = chunks / f"{start:08d}.npz"
        if path.is_file(): continue
        part=pages[start:start+batch_size]
        images=[]
        for p in part:
            with Image.open(_data_root() / p.page_image) as im: images.append(im.convert("RGB").copy())
        prompts=[_document_prompt(p.answer_text) for p in part]
        batch=processor(text=prompts, images=images, padding="longest", return_tensors="pt")
        offsets=batch["image_grid_thw"][:,1]*batch["image_grid_thw"][:,2]
        pixels=list(torch.split(batch["pixel_values"], offsets.tolist()))
        batch["pixel_values"]=torch.nn.utils.rnn.pad_sequence(pixels,batch_first=True)
        batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        out=model(**batch).float().cpu()
        mask=batch["attention_mask"].cpu().bool()
        arrays=[]; lens=[]; pools=[]
        for i in range(len(part)):
            tok=F.normalize(out[i][mask[i]],p=2,dim=-1)
            arrays.append(tok.numpy().astype(np.float16)); lens.append(len(tok)); pools.append(_pool(tok).numpy())
        np.savez(path, tokens=np.concatenate(arrays), lengths=np.asarray(lens), pooled=np.asarray(pools,dtype=np.float16))
        if (start//batch_size)%50==0: print(f"[colqwen-index] {start+len(part)}/{len(pages)}",flush=True)
    (shard_dir/"complete").write_text("ok\n")


def merge_shards(output_dir: Path, pages, num_shards: int, model_name: str, adapter: Path):
    from .page_index import _shard_bounds
    lengths=[]; pools=[]; chunk_paths=[]
    for sid in range(num_shards):
        root=output_dir/".partial"/"shards"/str(sid); a,b=_shard_bounds(len(pages),sid,num_shards)
        if not (root/"complete").is_file(): raise RuntimeError(f"shard {sid} incomplete")
        for path in sorted((root/"chunks").glob("*.npz")):
            z=np.load(path); lengths.extend(z["lengths"].tolist()); pools.append(z["pooled"]); chunk_paths.append(path)
    if len(lengths)!=len(pages): raise RuntimeError(f"expected {len(pages)} pages, got {len(lengths)}")
    offsets=np.concatenate(([0],np.cumsum(lengths,dtype=np.int64))); np.save(output_dir/"offsets.npy",offsets)
    mm=np.lib.format.open_memmap(output_dir/"tokens.f16.npy",mode="w+",dtype=np.float16,shape=(int(offsets[-1]),128))
    pos=0
    for path in chunk_paths:
        tok=np.load(path)["tokens"]; mm[pos:pos+len(tok)]=tok; pos+=len(tok)
    del mm; np.save(output_dir/"pooled.f16.npy",np.concatenate(pools).astype(np.float16))
    with (output_dir/"pages.jsonl").open("w",encoding="utf-8") as f:
        for p in pages: f.write(json.dumps(p.as_dict(),ensure_ascii=False)+"\n")
    meta={"model":model_name,"adapter":str(adapter),"scoring":"pooled_then_maxsim",
          "shortlist_size":500,"num_pages":len(pages),"num_token_rows":int(offsets[-1]),"dim":128,"created_unix":int(time.time())}
    (output_dir/"meta.json").write_text(json.dumps(meta,indent=2)+"\n")
    return ColQwenCascadeIndex.load(output_dir)
