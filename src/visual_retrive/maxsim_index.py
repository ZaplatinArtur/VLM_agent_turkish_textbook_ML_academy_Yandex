"""ColModern MaxSim (late-interaction) page index — the correct search for LoRA v2.

Mean-pooled cosine is *not* compatible with ColBERT-style training (hit@1≈0 on
the full corpus). This module stores per-page token vectors (float16, length-
compressed) and scores with MaxSim, matching ``train_colmodern_contrastive`` eval.
"""

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

from .page_index import DEFAULT_ADAPTER, PageRecord, _shard_bounds, load_page_records
from .paths import VISUAL_RETRIVE_DIR, ensure_visual_retrive_dirs

DEFAULT_MAXSIM_INDEX_DIR = (
    VISUAL_RETRIVE_DIR / "indexes" / "colmodern_v2_best_maxsim64"
)


def compress_token_embeddings(
    emb: torch.Tensor,
    *,
    max_tokens: int = 64,
) -> torch.Tensor:
    """Pool variable-length [S,D] → [<=max_tokens,D], L2-normalize each token."""
    if emb.ndim != 2:
        raise ValueError(f"expected [S,D], got {tuple(emb.shape)}")
    seq = emb.float()
    if seq.size(0) > max_tokens:
        # Even windows along the sequence (keeps layout, shrinks MaxSim cost).
        edges = torch.linspace(0, seq.size(0), max_tokens + 1).long()
        chunks = []
        for i in range(max_tokens):
            a, b = int(edges[i]), int(edges[i + 1])
            if b <= a:
                b = min(a + 1, seq.size(0))
            chunks.append(seq[a:b].mean(dim=0))
        seq = torch.stack(chunks, dim=0)
    return F.normalize(seq, p=2, dim=-1)


def maxsim_score(query: torch.Tensor, doc: torch.Tensor) -> float:
    """query [Q,D], doc [S,D] → scalar MaxSim (sum of max over doc tokens)."""
    return float(torch.einsum("qd,sd->qs", query.float(), doc.float()).amax(dim=1).sum())


@dataclass
class MaxSimPageIndex:
    pages: list[PageRecord]
    offsets: np.ndarray  # int64 [N+1]
    tokens: np.ndarray  # float16 [sum_len, D]
    meta: dict[str, Any]
    model: Any | None = None
    processor: Any | None = None
    device: torch.device | None = None
    _gpu_docs: torch.Tensor | None = None  # [N, max_tok, D] float16
    _gpu_mask: torch.Tensor | None = None  # [N, max_tok] bool

    @property
    def dim(self) -> int:
        return int(self.tokens.shape[1]) if self.tokens.ndim == 2 else int(self.meta["dim"])

    def doc_tokens(self, i: int) -> np.ndarray:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.tokens[a:b]

    @classmethod
    def load(cls, index_dir: Path, *, load_model: bool = False) -> "MaxSimPageIndex":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        offsets = np.load(index_dir / "offsets.npy")
        tokens = np.load(index_dir / "tokens.f16.npy", mmap_mode="r")
        pages: list[PageRecord] = []
        with (index_dir / "pages.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                pages.append(
                    PageRecord(
                        page_id=str(row["page_id"]),
                        book_slug=str(row.get("book_slug") or ""),
                        page_number=int(row.get("page_number") or 0),
                        grade=row.get("grade"),
                        subject=row.get("subject"),
                        page_image=str(row.get("page_image") or ""),
                        answer_text=str(row.get("answer_text") or ""),
                        has_solution=bool(row.get("has_solution")),
                    )
                )
        model = processor = device = None
        if load_model:
            from .page_index import ColModernPageIndex

            model, processor, device = ColModernPageIndex._load_encoder(
                model_name=str(meta.get("model") or "ModernVBERT/colmodernvbert-merged"),
                adapter=Path(meta["adapter"]) if meta.get("adapter") else None,
            )
        return cls(
            pages=pages,
            offsets=offsets,
            tokens=np.asarray(tokens),
            meta=meta,
            model=model,
            processor=processor,
            device=device,
        )

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "offsets.npy", self.offsets)
        np.save(index_dir / "tokens.f16.npy", self.tokens.astype(np.float16))
        with (index_dir / "pages.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for page in self.pages:
                handle.write(json.dumps(page.as_dict(), ensure_ascii=False) + "\n")
        (index_dir / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def ensure_encoder(self) -> None:
        if self.model is not None:
            return
        from .page_index import ColModernPageIndex

        self.model, self.processor, self.device = ColModernPageIndex._load_encoder(
            model_name=str(self.meta.get("model") or "ModernVBERT/colmodernvbert-merged"),
            adapter=Path(self.meta["adapter"]) if self.meta.get("adapter") else None,
        )

    @torch.inference_mode()
    def encode_query(self, query: str) -> torch.Tensor:
        self.ensure_encoder()
        assert self.model is not None and self.processor is not None
        texts = [
            self.processor.query_prefix
            + query
            + self.processor.query_augmentation_token * 10
        ]
        batch = self.processor.process_texts(texts)
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        emb = self.model(**batch)[0]
        return compress_token_embeddings(
            emb, max_tokens=int(self.meta.get("max_tokens") or 64)
        ).cpu()

    def _candidate_ids(
        self,
        *,
        subject: str | None,
        grade: int | str | None,
        candidate_ids: list[int] | None,
    ) -> list[int]:
        if candidate_ids is None:
            candidate_ids = list(range(len(self.pages)))
        if subject is None and grade is None:
            return candidate_ids
        subject_n = str(subject).strip().lower() if subject is not None else None
        grade_n = str(grade).strip().lower() if grade is not None else None
        filtered: list[int] = []
        for i in candidate_ids:
            page = self.pages[i]
            if subject_n is not None and str(page.subject or "").strip().lower() != subject_n:
                continue
            if grade_n is not None and str(page.grade or "").strip().lower() != grade_n:
                continue
            filtered.append(i)
        return filtered

    def prepare_gpu_corpus(
        self,
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack all page token matrices onto GPU once (≈450MB fp16 @ 64×128)."""
        if self._gpu_docs is not None and self._gpu_mask is not None:
            return self._gpu_docs, self._gpu_mask
        max_tok = int(self.meta.get("max_tokens") or 64)
        dim = self.dim
        n = len(self.pages)
        docs_np = np.zeros((n, max_tok, dim), dtype=np.float16)
        mask_np = np.zeros((n, max_tok), dtype=np.bool_)
        for i in range(n):
            tok = np.asarray(self.doc_tokens(i), dtype=np.float16)
            L = min(int(tok.shape[0]), max_tok)
            if L <= 0:
                continue
            docs_np[i, :L] = tok[:L]
            mask_np[i, :L] = True
        dev = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._gpu_docs = torch.from_numpy(docs_np).to(dev)
        self._gpu_mask = torch.from_numpy(mask_np).to(dev)
        return self._gpu_docs, self._gpu_mask

    def score_all(
        self,
        q_emb: torch.Tensor,
        *,
        candidate_ids: list[int] | None = None,
        chunk_size: int = 4096,
        device: torch.device | None = None,
    ) -> np.ndarray:
        """Batched GPU MaxSim scores for every page (or a candidate subset)."""
        ids = candidate_ids if candidate_ids is not None else list(range(len(self.pages)))
        n = len(ids)
        scores = np.full(n, -1e9, dtype=np.float32)
        if n == 0:
            return scores

        docs, mask = self.prepare_gpu_corpus(device=device)
        q = q_emb.to(device=docs.device, dtype=torch.float16)
        if q.ndim != 2:
            raise ValueError(f"query must be [Q,D], got {tuple(q.shape)}")

        # Fast path: score full corpus, then gather candidates.
        if candidate_ids is None or (
            len(ids) == len(self.pages) and ids[0] == 0 and ids[-1] == len(self.pages) - 1
        ):
            full = np.empty(len(self.pages), dtype=np.float32)
            for start in range(0, len(self.pages), chunk_size):
                end = min(start + chunk_size, len(self.pages))
                sim = torch.einsum(
                    "qd,bsd->qbs", q, docs[start:end]
                )  # [Q,B,S]
                sim = sim.masked_fill(~mask[start:end].unsqueeze(0), -1e4)
                full[start:end] = sim.amax(dim=-1).sum(dim=0).float().cpu().numpy()
            if candidate_ids is None:
                return full
            return full[np.asarray(ids, dtype=np.int64)]

        for start in range(0, n, chunk_size):
            batch_ids = ids[start : start + chunk_size]
            idx = torch.as_tensor(batch_ids, device=docs.device, dtype=torch.long)
            sim = torch.einsum("qd,bsd->qbs", q, docs.index_select(0, idx))
            sim = sim.masked_fill(~mask.index_select(0, idx).unsqueeze(0), -1e4)
            scores[start : start + len(batch_ids)] = (
                sim.amax(dim=-1).sum(dim=0).float().cpu().numpy()
            )
        return scores

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        subject: str | None = None,
        grade: int | str | None = None,
        candidate_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if len(q) < 2:
            return []
        q_emb = self.encode_query(q)
        ids = self._candidate_ids(
            subject=subject, grade=grade, candidate_ids=candidate_ids
        )
        if not ids:
            return []

        # Full-corpus / large candidate sets → batched GPU MaxSim.
        if len(ids) >= 256 and (
            torch.cuda.is_available()
            or (self.device is not None and self.device.type == "cuda")
        ):
            scores = self.score_all(q_emb, candidate_ids=ids)
            order = np.argsort(-scores)[:top_k]
            ranked = [(float(scores[j]), ids[int(j)]) for j in order]
        else:
            ranked_local: list[tuple[float, int]] = []
            for i in ids:
                doc = torch.from_numpy(np.asarray(self.doc_tokens(i), dtype=np.float32))
                ranked_local.append((maxsim_score(q_emb, doc), i))
            ranked_local.sort(key=lambda x: x[0], reverse=True)
            ranked = ranked_local[:top_k]

        hits: list[dict[str, Any]] = []
        for rank, (score, i) in enumerate(ranked, start=1):
            page = self.pages[i]
            hits.append(
                {
                    "page_id": page.page_id,
                    "book_slug": page.book_slug,
                    "page_number": page.page_number,
                    "grade": page.grade,
                    "subject": page.subject,
                    "page_image": page.page_image,
                    "answer_text": page.answer_text,
                    "score": float(score),
                    "rank": rank,
                    "has_solution": page.has_solution,
                }
            )
        return hits


@torch.inference_mode()
def encode_maxsim_shard(
    pages: list[PageRecord],
    shard_dir: Path,
    *,
    model_name: str = "ModernVBERT/colmodernvbert-merged",
    adapter: Path | None = DEFAULT_ADAPTER,
    batch_size: int = 4,
    max_tokens: int = 64,
    resume: bool = True,
    log_prefix: str = "[maxsim]",
) -> dict[str, Any]:
    """Encode one shard to token buffers under ``shard_dir``."""
    from .page_index import ColModernPageIndex

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    meta_path = shard_dir / "meta.json"
    pages_path = shard_dir / "pages.jsonl"
    offsets_path = shard_dir / "offsets.npy"
    tokens_path = shard_dir / "tokens.f16.npy"

    start_at = 0
    token_chunks: list[np.ndarray] = []
    offsets = [0]
    if resume and meta_path.is_file() and pages_path.is_file() and offsets_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        done = int(meta.get("done") or 0)
        if done > 0 and done <= len(pages) and tokens_path.is_file():
            prev_off = np.load(offsets_path)
            prev_tok = np.load(tokens_path)
            if int(prev_off[-1]) == prev_tok.shape[0] and len(prev_off) == done + 1:
                start_at = done
                token_chunks = [np.asarray(prev_tok)]
                offsets = [int(x) for x in prev_off.tolist()]
                print(f"{log_prefix} resume from {start_at}/{len(pages)}", flush=True)

    model, processor, device = ColModernPageIndex._load_encoder(
        model_name=model_name,
        adapter=adapter if adapter and Path(adapter).is_dir() else None,
    )

    if start_at == 0:
        pages_path.write_text("", encoding="utf-8")
        token_chunks = []
        offsets = [0]

    t0 = time.time()
    with pages_path.open("a", encoding="utf-8", newline="\n") as pages_handle:
        for start in range(start_at, len(pages), batch_size):
            chunk = pages[start : start + batch_size]
            imgs = [
                Image.open(VISUAL_RETRIVE_DIR / p.page_image).convert("RGB")
                for p in chunk
            ]
            batch = processor.process_images(imgs)
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            emb = model(**batch)
            for i in range(emb.size(0)):
                tok = compress_token_embeddings(emb[i], max_tokens=max_tokens)
                arr = tok.detach().float().cpu().numpy().astype(np.float16)
                token_chunks.append(arr)
                offsets.append(offsets[-1] + arr.shape[0])
                pages_handle.write(
                    json.dumps(chunk[i].as_dict(), ensure_ascii=False) + "\n"
                )
            pages_handle.flush()
            done = start + len(chunk)
            if done % 100 == 0 or done == len(pages):
                flat = np.concatenate(token_chunks, axis=0)
                np.save(tokens_path, flat)
                np.save(offsets_path, np.asarray(offsets, dtype=np.int64))
                meta_path.write_text(
                    json.dumps(
                        {
                            "done": done,
                            "total": len(pages),
                            "dim": int(flat.shape[1]),
                            "max_tokens": max_tokens,
                            "complete": done == len(pages),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                rate = max(done - start_at, 1) / max(time.time() - t0, 1e-6)
                print(
                    f"{log_prefix} {done}/{len(pages)} ({rate:.2f} pages/s)",
                    flush=True,
                )

    return {"done": len(pages), "total": len(pages), "complete": True}


def merge_maxsim_shards(
    output_dir: Path,
    *,
    num_shards: int,
    pages: list[PageRecord],
    model_name: str,
    adapter: Path | None,
    max_tokens: int,
) -> MaxSimPageIndex:
    output_dir = Path(output_dir)
    shard_root = output_dir / ".partial" / "shards"
    all_tokens: list[np.ndarray] = []
    offsets = [0]
    merged_pages: list[PageRecord] = []

    for shard_id in range(num_shards):
        shard_dir = shard_root / str(shard_id)
        meta = json.loads((shard_dir / "meta.json").read_text(encoding="utf-8"))
        if not meta.get("complete"):
            raise RuntimeError(f"shard {shard_id} incomplete: {meta}")
        start, end = _shard_bounds(len(pages), shard_id, num_shards)
        expected = pages[start:end]
        got_ids = []
        with (shard_dir / "pages.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    got_ids.append(json.loads(line)["page_id"])
        if got_ids != [p.page_id for p in expected]:
            raise RuntimeError(f"shard {shard_id} page_id mismatch")
        tok = np.load(shard_dir / "tokens.f16.npy")
        off = np.load(shard_dir / "offsets.npy")
        if len(off) != len(expected) + 1:
            raise RuntimeError(f"shard {shard_id} offsets length mismatch")
        for j in range(len(expected)):
            offsets.append(offsets[-1] + int(off[j + 1] - off[j]))
        all_tokens.append(tok)
        merged_pages.extend(expected)

    flat = np.concatenate(all_tokens, axis=0)
    if offsets[-1] != flat.shape[0]:
        raise RuntimeError(
            f"offset/token mismatch: offsets[-1]={offsets[-1]} tokens={flat.shape[0]}"
        )

    meta = {
        "model": model_name,
        "adapter": str(adapter) if adapter else None,
        "scoring": "maxsim",
        "max_tokens": max_tokens,
        "num_pages": len(merged_pages),
        "dim": int(flat.shape[1]),
        "num_token_rows": int(flat.shape[0]),
        "created_unix": int(time.time()),
        "recipe": "colmodern_canonical_v2_best_maxsim64",
        "num_shards": num_shards,
    }
    index = MaxSimPageIndex(
        pages=merged_pages,
        offsets=np.asarray(offsets, dtype=np.int64),
        tokens=flat.astype(np.float16),
        meta=meta,
    )
    index.save(output_dir)
    print(
        f"[maxsim] saved → {output_dir} pages={len(merged_pages)} "
        f"tokens={flat.shape[0]} dim={flat.shape[1]}",
        flush=True,
    )
    return index


def build_maxsim_index_parallel(
    *,
    output_dir: Path = DEFAULT_MAXSIM_INDEX_DIR,
    bundles_path: Path | None = None,
    model_name: str = "ModernVBERT/colmodernvbert-merged",
    adapter: Path | None = DEFAULT_ADAPTER,
    batch_size: int = 4,
    max_tokens: int = 64,
    require_solution: bool = True,
    max_pages: int | None = None,
    devices: list[str],
    resume: bool = True,
) -> MaxSimPageIndex:
    """Shard MaxSim encoding across GPUs, then merge."""
    import os
    import subprocess
    import sys

    ensure_visual_retrive_dirs()
    pages = load_page_records(
        bundles_path, require_solution=require_solution, max_pages=max_pages
    )
    if not pages:
        raise RuntimeError("no indexable pages")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_shards = len(devices)
    print(
        f"[maxsim] parallel {len(pages)} pages on {devices} (max_tokens={max_tokens})",
        flush=True,
    )

    env_base = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1])
    env_base["PYTHONPATH"] = src + (
        os.pathsep + env_base["PYTHONPATH"] if env_base.get("PYTHONPATH") else ""
    )

    procs: list[subprocess.Popen] = []
    logs: list[Path] = []
    for shard_id, gpu in enumerate(devices):
        log_path = output_dir / ".partial" / f"shard_{shard_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logs.append(log_path)
        cmd = [
            sys.executable,
            "-m",
            "visual_retrive.scripts.build_maxsim_index",
            "--output-dir",
            str(output_dir),
            "--adapter",
            str(adapter) if adapter else "",
            "--model",
            model_name,
            "--batch-size",
            str(batch_size),
            "--max-tokens",
            str(max_tokens),
            "--cuda-device",
            str(gpu),
            "--shard-id",
            str(shard_id),
            "--num-shards",
            str(num_shards),
            "--worker",
        ]
        if not require_solution:
            cmd.append("--all-pages")
        if max_pages is not None:
            cmd.extend(["--max-pages", str(max_pages)])
        if not resume:
            cmd.append("--no-resume")
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_f = log_path.open("w", encoding="utf-8")
        print(f"[maxsim] launch shard {shard_id} GPU {gpu}", flush=True)
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        proc.log_f = log_f  # type: ignore[attr-defined]
        procs.append(proc)

    rc = 0
    try:
        for i, proc in enumerate(procs):
            code = proc.wait()
            getattr(proc, "log_f").close()
            if code != 0:
                rc = code
                print(f"[maxsim] shard {i} FAILED exit={code} see {logs[i]}", flush=True)
            else:
                print(f"[maxsim] shard {i} OK", flush=True)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()

    if rc != 0:
        raise RuntimeError(f"maxsim shard failure exit={rc}")

    return merge_maxsim_shards(
        output_dir,
        num_shards=num_shards,
        pages=pages,
        model_name=model_name,
        adapter=adapter,
        max_tokens=max_tokens,
    )
