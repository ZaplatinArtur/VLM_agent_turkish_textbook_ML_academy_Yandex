"""Create a deterministic stratified sample of relevance edges by E5 score."""
from __future__ import annotations
import argparse, json, math, random
from collections import defaultdict
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--low',type=float,default=.94); ap.add_argument('--high',type=float,default=1.00001)
ap.add_argument('--bin-width',type=float,default=.005); ap.add_argument('--per-bin',type=int,default=100); ap.add_argument('--seed',type=int,default=42)
ap.add_argument('--min-lexical',type=float,default=.30); ap.add_argument('--max-lexical',type=float,default=.72)
a=ap.parse_args(); data=json.loads(a.input.read_text(encoding='utf-8')); bins=defaultdict(list)
for edge in data['edges']:
    sim=float(edge.get('semantic') or 0); lex=float(edge.get('lexical') or 0)
    if a.low<=sim<a.high and a.min_lexical<=lex<a.max_lexical:
        idx=int(math.floor((sim-a.low)/a.bin_width)); bins[idx].append(edge)
rng=random.Random(a.seed); selected=[]; summary=[]
for idx in range(int(math.ceil((a.high-a.low)/a.bin_width))):
    pool=bins[idx]; chosen=rng.sample(pool,min(a.per_bin,len(pool)))
    left=a.low+idx*a.bin_width; right=min(a.high,left+a.bin_width)
    for edge in chosen: edge['calibration_bin']=[left,right]
    selected.extend(chosen); summary.append({'left':left,'right':right,'available':len(pool),'sampled':len(chosen)})
data['edges']=selected; data['config']={**data['config'],'semantic_threshold':a.low,'calibration_sample':summary,
    'per_bin':a.per_bin,'bin_width':a.bin_width,'seed':a.seed}; data['stats']={**data['stats'],'edges':len(selected)}
a.output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'selected':len(selected),'bins':summary}))
