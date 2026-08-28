"""CLI: build ColModern MaxSim (late-interaction) page index."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--bundles", type=Path, default=None)
    p.add_argument("--model", default="ModernVBERT/colmodernvbert-merged")
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--all-pages", action="store_true")
    p.add_argument("--cuda-device", default=None)
    p.add_argument("--devices", default=None, help="e.g. 0,1,3")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    from visual_retrive.maxsim_index import (
        DEFAULT_MAXSIM_INDEX_DIR,
        build_maxsim_index_parallel,
        encode_maxsim_shard,
        merge_maxsim_shards,
    )
    from visual_retrive.page_index import DEFAULT_ADAPTER, _shard_bounds, load_page_records

    output_dir = args.output_dir or DEFAULT_MAXSIM_INDEX_DIR
    adapter = args.adapter or DEFAULT_ADAPTER
    require_solution = not args.all_pages
    resume = not args.no_resume

    if args.merge_only:
        pages = load_page_records(
            args.bundles, require_solution=require_solution, max_pages=args.max_pages
        )
        merge_maxsim_shards(
            output_dir,
            num_shards=args.num_shards,
            pages=pages,
            model_name=args.model,
            adapter=adapter,
            max_tokens=args.max_tokens,
        )
        return

    if args.devices and not args.worker:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        build_maxsim_index_parallel(
            output_dir=output_dir,
            bundles_path=args.bundles,
            model_name=args.model,
            adapter=adapter,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            require_solution=require_solution,
            max_pages=args.max_pages,
            devices=devices,
            resume=resume,
        )
        return

    pages = load_page_records(
        args.bundles, require_solution=require_solution, max_pages=args.max_pages
    )
    start, end = _shard_bounds(len(pages), args.shard_id, args.num_shards)
    shard_pages = pages[start:end]
    shard_dir = (
        output_dir / ".partial"
        if args.num_shards == 1
        else output_dir / ".partial" / "shards" / str(args.shard_id)
    )
    print(
        f"[maxsim] shard {args.shard_id}/{args.num_shards} [{start}:{end})",
        flush=True,
    )
    encode_maxsim_shard(
        shard_pages,
        shard_dir,
        model_name=args.model,
        adapter=adapter,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        resume=resume,
        log_prefix=f"[maxsim shard{args.shard_id}]",
    )
    if args.worker:
        return
    merge_maxsim_shards(
        output_dir,
        num_shards=args.num_shards,
        pages=pages,
        model_name=args.model,
        adapter=adapter,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
