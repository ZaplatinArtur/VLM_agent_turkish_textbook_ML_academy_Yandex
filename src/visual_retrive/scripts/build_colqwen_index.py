"""Build a resumable ColQwen2.5 pooled-to-MaxSim page index."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,default=None); p.add_argument("--bundles",type=Path)
    p.add_argument("--model",default="vidore/colqwen2.5-base"); p.add_argument("--adapter",type=Path,default=None)
    p.add_argument("--batch-size",type=int,default=2); p.add_argument("--devices",default="1,3")
    p.add_argument("--shard-id",type=int); p.add_argument("--num-shards",type=int); p.add_argument("--worker",action="store_true")
    p.add_argument("--solutions-only",action="store_true"); p.add_argument("--max-pages",type=int)
    a=p.parse_args()
    from visual_retrive.colqwen_index import DEFAULT_COLQWEN_ADAPTER,DEFAULT_COLQWEN_INDEX_DIR,encode_shard,load_colqwen_pages,merge_shards
    from visual_retrive.page_index import _shard_bounds
    out=a.output_dir or DEFAULT_COLQWEN_INDEX_DIR; adapter=a.adapter or DEFAULT_COLQWEN_ADAPTER
    pages=load_colqwen_pages(a.bundles,require_solution=a.solutions_only,max_pages=a.max_pages)
    if a.worker:
        lo,hi=_shard_bounds(len(pages),a.shard_id,a.num_shards)
        encode_shard(pages[lo:hi],out/".partial"/"shards"/str(a.shard_id),model_name=a.model,adapter=adapter,batch_size=a.batch_size)
        return
    devices=[x.strip() for x in a.devices.split(",") if x.strip()]; out.mkdir(parents=True,exist_ok=True)
    procs=[]
    for sid,gpu in enumerate(devices):
        log=out/".partial"/f"shard_{sid}.log"; log.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"-m","visual_retrive.scripts.build_colqwen_index","--output-dir",str(out),"--model",a.model,"--adapter",str(adapter),"--batch-size",str(a.batch_size),"--devices",gpu,"--shard-id",str(sid),"--num-shards",str(len(devices)),"--worker"]
        if a.bundles: cmd += ["--bundles",str(a.bundles)]
        if a.solutions_only: cmd.append("--solutions-only")
        if a.max_pages: cmd += ["--max-pages",str(a.max_pages)]
        env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpu
        fh=log.open("a",encoding="utf-8"); proc=subprocess.Popen(cmd,env=env,stdout=fh,stderr=subprocess.STDOUT); procs.append((proc,fh,log))
        print(f"[colqwen-index] shard {sid} -> GPU {gpu} log={log}",flush=True)
    failed=[]
    for proc,fh,log in procs:
        code=proc.wait(); fh.close()
        if code: failed.append((code,log))
    if failed: raise RuntimeError(f"failed shards: {failed}")
    merge_shards(out,pages,len(devices),a.model,adapter)
    print(f"[colqwen-index] complete: {out}",flush=True)

if __name__=="__main__": main()
