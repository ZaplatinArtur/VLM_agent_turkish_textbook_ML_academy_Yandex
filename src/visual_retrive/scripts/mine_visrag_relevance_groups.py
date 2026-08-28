"""Mine conservative multi-positive groups among adjacent textbook pages."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

import torch

from visual_retrive.visrag_siglip.data import read_jsonl, usable_rows
from visual_retrive.visrag_siglip.groups import build_query_relevance, relevance_key
from visual_retrive.visrag_siglip.model import DEFAULT_MODEL, encode_text, load_encoder


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--model',default=DEFAULT_MODEL)
    ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--semantic-threshold',type=float,default=.94)
    ap.add_argument('--lexical-threshold',type=float,default=.72); ap.add_argument('--max-group-size',type=int,default=4)
    ap.add_argument('--max-page-gap',type=int,default=1)
    ap.add_argument('--lexical-only',action='store_true'); ap.add_argument('--limit',type=int)
    a=ap.parse_args(); rows=usable_rows(read_jsonl(a.pairs),a.data_root)
    if a.limit: rows=rows[:a.limit]
    query_items={relevance_key(str(row['positive_page_id']),str(row['query'])):str(row['query']) for row in rows}
    vectors=None
    if not a.lexical_only:
        device=torch.device('cuda'); model,processor=load_encoder(a.model,device=device); model.eval()
        keys=list(query_items); texts=[query_items[k] for k in keys]; chunks=[]
        for i in range(0,len(texts),a.batch_size):
            with torch.no_grad(), torch.autocast(device_type='cuda',dtype=torch.float16):
                chunks.append(encode_text(model,processor,texts[i:i+a.batch_size],device).float().cpu())
            if i%(a.batch_size*20)==0: print(f'embedded={min(i+a.batch_size,len(texts))}/{len(texts)}',flush=True)
        matrix=torch.cat(chunks); vectors={k:v.tolist() for k,v in zip(keys,matrix)}
    result=build_query_relevance(rows,vectors,a.semantic_threshold,a.lexical_threshold,a.max_page_gap)
    result['stats']['unique_queries']=len(query_items)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['stats']),flush=True)
    for edge in sorted(result['edges'],key=lambda x:max(x['lexical'],x['semantic'] or 0),reverse=True)[:20]: print(json.dumps(edge),flush=True)

if __name__=='__main__': main()
