"""Build / load a fast ColModern page-image index (pooled single-vector).

Stores L2-normalized page embeddings for cosine search. Late-interaction
MaxSim remains available via ``visual_retrive.search.maxsim_rerank`` for
offline quality checks; the agent tool uses the pooled index for latency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .paths import CATALOG_DIR, VISUAL_RETRIVE_DIR, ensure_visual_retrive_dirs

DEFAULT_ADAPTER = (
    VISUAL_RETRIVE_DIR / "models" / "colmodern_canonical_lora_v2" / "best"
)
DEFAULT_INDEX_DIR = VISUAL_RETRIVE_DIR / "indexes" / "colmodern_v2_best_pooled"


@dataclass
class PageRecord:
    page_id: str
    book_slug: str
    page_number: int
    grade: Any
    subject: Any
    page_image: str
    answer_text: str
    has_solution: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "book_slug": self.book_slug,
            "page_number": self.page_number,
            "grade": self.grade,
            "subject": self.subject,
            "page_image": self.page_image,
            "answer_text": self.answer_text,
            "has_solution": self.has_solution,
        }


def _iter_indexable_pages(
    bundles_path: Path,
    *,
    require_solution: bool = True,
) -> Iterator[PageRecord]:
    with bundles_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rel = row.get("page_image")
            if not rel:
                continue
            abs_path = VISUAL_RETRIVE_DIR / str(rel)
            if not abs_path.is_file():
                continue
            if require_solution and not row.get("has_solution"):
                continue
            yield PageRecord(
                page_id=str(row["page_id"]),
                book_slug=str(row.get("book_slug") or ""),
                page_number=int(row.get("page_number") or 0),
                grade=row.get("grade"),
                subject=row.get("subject"),
                page_image=str(rel),
                answer_text=str(row.get("answer_text") or "")[:4_000],
                has_solution=bool(row.get("has_solution")),
            )


def load_page_records(
    bundles_path: Path | None = None,
    *,
    require_solution: bool = True,
    max_pages: int | None = None,
) -> list[PageRecord]:
    path = bundles_path or (CATALOG_DIR / "page_bundles.cleaned.jsonl")
    if not path.is_file():
        path = CATALOG_DIR / "page_bundles.jsonl"
    rows: list[PageRecord] = []
    for rec in _iter_indexable_pages(path, require_solution=require_solution):
        rows.append(rec)
        if max_pages is not None and len(rows) >= max_pages:
            break
    return rows


def _pool(emb: torch.Tensor) -> torch.Tensor:
    if emb.ndim == 3:
        vec = emb.float().mean(dim=1)
    elif emb.ndim == 2:
        vec = emb.float()
    else:
        raise ValueError(f"unexpected emb shape {tuple(emb.shape)}")
    return F.normalize(vec, p=2, dim=-1)


class ColModernPageIndex:
    """In-memory cosine index over pooled ColModern page embeddings."""

    def __init__(
        self,
        *,
        embeddings: np.ndarray,
        pages: list[PageRecord],
        meta: dict[str, Any],
        model: Any | None = None,
        processor: Any | None = None,
        device: torch.device | None = None,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be [N, D]")
        if len(pages) != embeddings.shape[0]:
            raise ValueError("pages/embeddings length mismatch")
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.pages = pages
        self.meta = meta
        self.model = model
        self.processor = processor
        self.device = device

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @classmethod
    def load(cls, index_dir: Path, *, load_model: bool = False) -> "ColModernPageIndex":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        emb = np.load(index_dir / "embeddings.f32.npy", mmap_mode="r")
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
            model, processor, device = cls._load_encoder(
                model_name=str(meta.get("model") or "ModernVBERT/colmodernvbert-merged"),
                adapter=Path(meta["adapter"]) if meta.get("adapter") else None,
            )
        return cls(
            embeddings=np.asarray(emb),
            pages=pages,
            meta=meta,
            model=model,
            processor=processor,
            device=device,
        )

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.f32.npy", self.embeddings)
        with (index_dir / "pages.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for page in self.pages:
                handle.write(json.dumps(page.as_dict(), ensure_ascii=False) + "\n")
        (index_dir / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _load_encoder(
        *,
        model_name: str,
        adapter: Path | None,
    ) -> tuple[Any, Any, torch.device]:
        """Load ColModern (+ optional LoRA) with vocab/freeze/weight-key patches."""
        from visual_retrive.colmodern_load import load_model_and_processor

        return load_model_and_processor(
            model_name,
            adapter=adapter,
        )

    def ensure_encoder(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        self.model, self.processor, self.device = self._load_encoder(
            model_name=str(
                self.meta.get("model") or "ModernVBERT/colmodernvbert-merged"
            ),
            adapter=Path(self.meta["adapter"]) if self.meta.get("adapter") else None,
        )

    @torch.inference_mode()
    def encode_queries(self, queries: list[str]) -> np.ndarray:
        self.ensure_encoder()
        assert self.model is not None and self.processor is not None
        texts = [
            self.processor.query_prefix
            + q
            + self.processor.query_augmentation_token * 10
            for q in queries
        ]
        batch = self.processor.process_texts(texts)
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        emb = self.model(**batch)
        return _pool(emb).detach().float().cpu().numpy().astype(np.float32)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        subject: str | None = None,
        grade: int | str | None = None,
        candidate_multiplier: int = 5,
    ) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if len(q) < 2:
            return []
        fetch = top_k if subject is None and grade is None else top_k * candidate_multiplier
        fetch = max(fetch, top_k)
        q_vec = self.encode_queries([q])[0]
        scores = self.embeddings @ q_vec
        if scores.size == 0:
            return []
        # argpartition for speed on large corpora
        k = min(fetch, scores.shape[0])
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        hits: list[dict[str, Any]] = []
        for rank_i, di in enumerate(idx, start=1):
            page = self.pages[int(di)]
            if subject is not None and str(page.subject) != str(subject):
                continue
            if grade is not None and str(page.grade) != str(grade):
                continue
            hits.append(
                {
                    "page_id": page.page_id,
                    "book_slug": page.book_slug,
                    "page_number": page.page_number,
                    "grade": page.grade,
                    "subject": page.subject,
                    "page_image": page.page_image,
                    "answer_text": page.answer_text,
                    "score": float(scores[int(di)]),
                    "rank": len(hits) + 1,
                    "has_solution": page.has_solution,
                }
            )
            if len(hits) >= top_k:
                break
        return hits


def _shard_bounds(n: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id {shard_id} out of range for {num_shards} shards")
    start = (n * shard_id) // num_shards
    end = (n * (shard_id + 1)) // num_shards
    return start, end


@torch.inference_mode()
def encode_pages_shard(
    pages: list[PageRecord],
    shard_dir: Path,
    *,
    model_name: str = "ModernVBERT/colmodernvbert-merged",
    adapter: Path | None = DEFAULT_ADAPTER,
    batch_size: int = 4,
    resume: bool = True,
    checkpoint_every: int = 500,
    log_prefix: str = "[index]",
) -> dict[str, Any]:
    """Encode ``pages`` into ``shard_dir`` with resume support. Returns shard meta."""
    if not pages:
        raise RuntimeError("empty shard")

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    emb_path = shard_dir / "embeddings.f32.npy"
    pages_path = shard_dir / "pages.jsonl"
    meta_partial = shard_dir / "meta.json"

    start_at = 0
    embeddings: np.ndarray | None = None
    if resume and emb_path.is_file() and pages_path.is_file():
        done_ids: list[str] = []
        with pages_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done_ids.append(json.loads(line)["page_id"])
        loaded = np.load(emb_path, mmap_mode=None)
        if len(done_ids) <= len(pages) and done_ids == [
            p.page_id for p in pages[: len(done_ids)]
        ]:
            # Accept either exact-length memmap or full preallocated shard file.
            if loaded.shape[0] >= len(done_ids):
                start_at = len(done_ids)
                embeddings = loaded
                print(
                    f"{log_prefix} resume from page {start_at}/{len(pages)}",
                    flush=True,
                )
        if embeddings is None:
            print(f"{log_prefix} partial mismatch — rebuilding shard", flush=True)
            emb_path.unlink(missing_ok=True)
            pages_path.unlink(missing_ok=True)

    model, processor, torch_device = ColModernPageIndex._load_encoder(
        model_name=model_name,
        adapter=adapter if adapter and Path(adapter).is_dir() else None,
    )

    if embeddings is None:
        probe_img = Image.open(VISUAL_RETRIVE_DIR / pages[0].page_image).convert("RGB")
        probe_batch = processor.process_images([probe_img])
        probe_batch = {
            k: v.to(torch_device) if isinstance(v, torch.Tensor) else v
            for k, v in probe_batch.items()
        }
        dim = int(_pool(model(**probe_batch)).shape[-1])
        embeddings = np.lib.format.open_memmap(
            emb_path, mode="w+", dtype=np.float32, shape=(len(pages), dim)
        )
        pages_path.write_text("", encoding="utf-8")
        start_at = 0
    else:
        dim = int(embeddings.shape[1])
        embeddings = np.lib.format.open_memmap(
            emb_path, mode="r+", dtype=np.float32, shape=(len(pages), dim)
        )

    t0 = time.time()
    encoded_this_run = 0
    with pages_path.open("a", encoding="utf-8", newline="\n") as pages_handle:
        for start in range(start_at, len(pages), batch_size):
            chunk = pages[start : start + batch_size]
            imgs = [
                Image.open(VISUAL_RETRIVE_DIR / p.page_image).convert("RGB")
                for p in chunk
            ]
            batch = processor.process_images(imgs)
            batch = {
                k: v.to(torch_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            emb = model(**batch)
            pooled = _pool(emb).detach().float().cpu().numpy().astype(np.float32)
            end = start + len(chunk)
            embeddings[start:end] = pooled
            for page in chunk:
                pages_handle.write(json.dumps(page.as_dict(), ensure_ascii=False) + "\n")
            pages_handle.flush()
            encoded_this_run += len(chunk)
            done = end
            if (
                done % 200 == 0
                or done == len(pages)
                or encoded_this_run % checkpoint_every == 0
            ):
                embeddings.flush()
                meta_partial.write_text(
                    json.dumps(
                        {
                            "done": done,
                            "total": len(pages),
                            "dim": dim,
                            "model": model_name,
                            "adapter": str(adapter) if adapter else None,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                rate = max(done - start_at, 1) / max(time.time() - t0, 1e-6)
                print(
                    f"{log_prefix} {done}/{len(pages)} pages ({rate:.2f} pages/s)",
                    flush=True,
                )

    embeddings.flush()
    meta = {
        "done": len(pages),
        "total": len(pages),
        "dim": dim,
        "model": model_name,
        "adapter": str(adapter) if adapter else None,
        "complete": True,
    }
    meta_partial.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{log_prefix} shard complete ({len(pages)}×{dim})", flush=True)
    return meta


def merge_page_shards(
    output_dir: Path,
    *,
    num_shards: int,
    pages: list[PageRecord],
    model_name: str,
    adapter: Path | None,
    require_solution: bool,
) -> ColModernPageIndex:
    """Merge ``.partial/shards/{i}`` into the final index under ``output_dir``."""
    output_dir = Path(output_dir)
    shard_root = output_dir / ".partial" / "shards"
    dim: int | None = None
    vectors: list[np.ndarray] = []
    merged_pages: list[PageRecord] = []

    for shard_id in range(num_shards):
        shard_dir = shard_root / str(shard_id)
        emb_path = shard_dir / "embeddings.f32.npy"
        pages_path = shard_dir / "pages.jsonl"
        meta_path = shard_dir / "meta.json"
        if not emb_path.is_file() or not pages_path.is_file():
            raise FileNotFoundError(f"missing shard artifacts: {shard_dir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        if not meta.get("complete") and int(meta.get("done") or 0) != int(
            meta.get("total") or -1
        ):
            raise RuntimeError(f"shard {shard_id} incomplete: {meta}")

        start, end = _shard_bounds(len(pages), shard_id, num_shards)
        expected = pages[start:end]
        got: list[PageRecord] = []
        with pages_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                got.append(
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
        if [p.page_id for p in got] != [p.page_id for p in expected]:
            raise RuntimeError(
                f"shard {shard_id} page_id mismatch "
                f"(got {len(got)}, expected {len(expected)})"
            )
        emb = np.load(emb_path)
        if emb.shape[0] < len(got):
            raise RuntimeError(f"shard {shard_id} embeddings shorter than pages")
        emb = np.asarray(emb[: len(got)], dtype=np.float32)
        if dim is None:
            dim = int(emb.shape[1])
        elif int(emb.shape[1]) != dim:
            raise RuntimeError(f"shard {shard_id} dim mismatch")
        vectors.append(emb)
        merged_pages.extend(got)

    if len(merged_pages) != len(pages):
        raise RuntimeError(
            f"merged {len(merged_pages)} pages but catalog has {len(pages)}"
        )
    embeddings = np.concatenate(vectors, axis=0)
    meta = {
        "model": model_name,
        "adapter": str(adapter) if adapter else None,
        "pooling": "mean_l2",
        "num_pages": len(merged_pages),
        "dim": int(embeddings.shape[1]),
        "require_solution": require_solution,
        "created_unix": int(time.time()),
        "recipe": "colmodern_canonical_v2_best_pooled",
        "num_shards": num_shards,
    }
    index = ColModernPageIndex(
        embeddings=embeddings,
        pages=merged_pages,
        meta=meta,
    )
    index.save(output_dir)
    print(
        f"[index] merged → {output_dir} ({len(merged_pages)}×{embeddings.shape[1]})",
        flush=True,
    )
    return index


def build_page_index(
    *,
    output_dir: Path = DEFAULT_INDEX_DIR,
    bundles_path: Path | None = None,
    model_name: str = "ModernVBERT/colmodernvbert-merged",
    adapter: Path | None = DEFAULT_ADAPTER,
    batch_size: int = 4,
    require_solution: bool = True,
    max_pages: int | None = None,
    device: str | None = None,
    resume: bool = True,
    checkpoint_every: int = 500,
    shard_id: int = 0,
    num_shards: int = 1,
    finalize: bool = True,
) -> ColModernPageIndex | dict[str, Any]:
    """Encode page images into a pooled cosine index (optionally one shard)."""
    ensure_visual_retrive_dirs()
    pages = load_page_records(
        bundles_path, require_solution=require_solution, max_pages=max_pages
    )
    if not pages:
        raise RuntimeError("no indexable pages found")

    if device:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    output_dir = Path(output_dir)
    start, end = _shard_bounds(len(pages), shard_id, num_shards)
    shard_pages = pages[start:end]
    if num_shards == 1:
        shard_dir = output_dir / ".partial"
    else:
        shard_dir = output_dir / ".partial" / "shards" / str(shard_id)

    print(
        f"[index] shard {shard_id}/{num_shards} pages [{start}:{end}) "
        f"({len(shard_pages)} pages)",
        flush=True,
    )
    shard_meta = encode_pages_shard(
        shard_pages,
        shard_dir,
        model_name=model_name,
        adapter=adapter,
        batch_size=batch_size,
        resume=resume,
        checkpoint_every=checkpoint_every,
        log_prefix=f"[index shard{shard_id}]",
    )

    if num_shards > 1 and not finalize:
        return shard_meta

    if num_shards > 1:
        return merge_page_shards(
            output_dir,
            num_shards=num_shards,
            pages=pages,
            model_name=model_name,
            adapter=adapter,
            require_solution=require_solution,
        )

    # Single-shard: promote partial → final index.
    emb = np.load(shard_dir / "embeddings.f32.npy")
    emb = np.asarray(emb[: len(shard_pages)], dtype=np.float32)
    meta = {
        "model": model_name,
        "adapter": str(adapter) if adapter else None,
        "pooling": "mean_l2",
        "num_pages": len(shard_pages),
        "dim": int(emb.shape[1]),
        "require_solution": require_solution,
        "created_unix": int(time.time()),
        "recipe": "colmodern_canonical_v2_best_pooled",
    }
    index = ColModernPageIndex(embeddings=emb, pages=shard_pages, meta=meta)
    index.save(output_dir)
    for p in (
        shard_dir / "embeddings.f32.npy",
        shard_dir / "pages.jsonl",
        shard_dir / "meta.json",
    ):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        shard_dir.rmdir()
    except OSError:
        pass
    print(f"[index] saved → {output_dir} ({len(shard_pages)}×{emb.shape[1]})", flush=True)
    return index


def build_page_index_parallel(
    *,
    output_dir: Path = DEFAULT_INDEX_DIR,
    bundles_path: Path | None = None,
    model_name: str = "ModernVBERT/colmodernvbert-merged",
    adapter: Path | None = DEFAULT_ADAPTER,
    batch_size: int = 8,
    require_solution: bool = True,
    max_pages: int | None = None,
    devices: list[int] | list[str],
    resume: bool = True,
) -> ColModernPageIndex:
    """Shard the corpus across GPUs via subprocesses, then merge."""
    import os
    import subprocess
    import sys

    if not devices:
        raise ValueError("devices must be non-empty")
    devices_s = [str(d) for d in devices]
    num_shards = len(devices_s)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Drop legacy single-process partial (incompatible layout).
    legacy_emb = output_dir / ".partial" / "embeddings.f32.npy"
    if legacy_emb.is_file():
        print("[index] removing legacy single-GPU partial", flush=True)
        for name in ("embeddings.f32.npy", "pages.jsonl", "meta.json"):
            (output_dir / ".partial" / name).unlink(missing_ok=True)

    pages = load_page_records(
        bundles_path, require_solution=require_solution, max_pages=max_pages
    )
    print(
        f"[index] parallel build: {len(pages)} pages on GPUs {devices_s} "
        f"({num_shards} shards)",
        flush=True,
    )

    env_base = os.environ.copy()
    # Ensure src is importable in workers.
    src = str(Path(__file__).resolve().parents[1])
    env_base["PYTHONPATH"] = (
        src + (os.pathsep + env_base["PYTHONPATH"] if env_base.get("PYTHONPATH") else "")
    )

    procs: list[subprocess.Popen] = []
    logs: list[Path] = []
    for shard_id, gpu in enumerate(devices_s):
        log_path = output_dir / ".partial" / f"shard_{shard_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logs.append(log_path)
        cmd = [
            sys.executable,
            "-m",
            "visual_retrive.scripts.build_page_index",
            "--output-dir",
            str(output_dir),
            "--adapter",
            str(adapter) if adapter else "",
            "--model",
            model_name,
            "--batch-size",
            str(batch_size),
            "--cuda-device",
            gpu,
            "--shard-id",
            str(shard_id),
            "--num-shards",
            str(num_shards),
            "--worker",
        ]
        if bundles_path is not None:
            cmd.extend(["--bundles", str(bundles_path)])
        if max_pages is not None:
            cmd.extend(["--max-pages", str(max_pages)])
        if not require_solution:
            cmd.append("--all-pages")
        if not resume:
            cmd.append("--no-resume")
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_f = log_path.open("w", encoding="utf-8")
        print(f"[index] launch shard {shard_id} on GPU {gpu} → {log_path}", flush=True)
        procs.append(
            subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        )
        # keep log_f open for process lifetime; close after wait
        procs[-1].log_f = log_f  # type: ignore[attr-defined]

    rc = 0
    try:
        for i, proc in enumerate(procs):
            code = proc.wait()
            getattr(proc, "log_f").close()
            if code != 0:
                rc = code
                print(
                    f"[index] shard {i} FAILED exit={code}; see {logs[i]}",
                    flush=True,
                )
            else:
                print(f"[index] shard {i} OK", flush=True)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            log_f = getattr(proc, "log_f", None)
            if log_f and not log_f.closed:
                log_f.close()

    if rc != 0:
        raise RuntimeError(f"one or more shards failed (last exit={rc})")

    return merge_page_shards(
        output_dir,
        num_shards=num_shards,
        pages=pages,
        model_name=model_name,
        adapter=adapter,
        require_solution=require_solution,
    )
