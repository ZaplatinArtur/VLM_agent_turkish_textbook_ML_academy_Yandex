"""Map resumable Qwen judgments from one edge ordering into another."""
from __future__ import annotations
import argparse, json
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('--source-groups',type=Path,required=True); ap.add_argument('--source-judgments',type=Path,required=True)
ap.add_argument('--target-groups',type=Path,required=True); ap.add_argument('--target-judgments',type=Path,required=True); a=ap.parse_args()
source=json.loads(a.source_groups.read_text(encoding='utf-8'))['edges']; target=json.loads(a.target_groups.read_text(encoding='utf-8'))['edges']
labels={}
for line in a.source_judgments.open(encoding='utf-8'):
    row=json.loads(line); edge=source[int(row['id'])]; key=tuple(sorted((edge['a_key'],edge['b_key']))); labels[key]=bool(row['same_intent'])
written=0
with a.target_judgments.open('w',encoding='utf-8') as out:
    for idx,edge in enumerate(target):
        key=tuple(sorted((edge['a_key'],edge['b_key'])))
        if key not in labels: continue
        out.write(json.dumps({'id':idx,'same_intent':labels[key],'semantic':edge.get('semantic'),'lexical':edge.get('lexical'),'subject':edge.get('subject')},ensure_ascii=False)+'\n'); written+=1
print(json.dumps({'source_labels':len(labels),'migrated':written,'target_edges':len(target)}))
