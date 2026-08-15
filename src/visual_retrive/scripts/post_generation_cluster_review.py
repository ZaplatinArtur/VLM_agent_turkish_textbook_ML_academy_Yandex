"""Run block-pair build, 1-12 clustering and Qwen review with W&B telemetry."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
import wandb

ROOT=Path(__file__).resolve().parents[3]
CAT=ROOT/'data/visual_retrive/catalog'; DATA=ROOT/'data/visual_retrive'
GEN=ROOT/'work_resume/evvel_all_visual_queries_openrouter.jsonl'
HIGH=CAT/'train_queries_grades_9_12_blocks.jsonl'; ALL=CAT/'train_queries_grades_1_12_blocks.cleaned.jsonl'
RAW=CAT/'visrag_query_relevance_e5_v6_1_12_blocks_raw.json'; REVIEWED=CAT/'visrag_query_relevance_e5_v6_1_12_blocks_reviewed.json'

run=wandb.init(project='turkish-visrag-1-12-pipeline',group='grades-1-12-post-generation',name='cluster-qwen-review-v6',job_type='post-generation',id='cluster-qwen-review-v6',resume='allow',config={'semantic_threshold':.90,'lexical_threshold':.72,'top_k':12,'judge':'openrouter:qwen/qwen3.5-9b'})

def log(stage, **values):
    run.log({'pipeline/stage':stage,**{f'{stage}/{k}':v for k,v in values.items()}})

def command(stage,args,env=None):
    log(stage,status=1); started=time.time(); process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        print(line,end='',flush=True)
        try:
            row=json.loads(line)
            numeric={k:v for k,v in row.items() if isinstance(v,(int,float,bool))}
            if numeric: log(stage,**numeric)
        except Exception: pass
    code=process.wait(); log(stage,status=2 if code==0 else -1,duration_s=time.time()-started)
    if code: raise subprocess.CalledProcessError(code,args)

try:
    command('build_pairs',['.venv/bin/python','-u','src/visual_retrive/scripts/build_evvel_visual_block_pairs.py','--input',str(GEN),'--data-root',str(DATA),'--output',str(HIGH)])
    seen=set(); count=0; grades={}
    with ALL.open('w',encoding='utf-8') as out:
        for source in (CAT/'train_queries.cleaned.jsonl',HIGH):
            for line in source.open(encoding='utf-8'):
                row=json.loads(line); key=(str(row['positive_page_id']),str(row['query']).casefold().strip())
                if key in seen: continue
                seen.add(key); out.write(json.dumps(row,ensure_ascii=False)+'\n'); count+=1; grades[int(row['grade'])]=grades.get(int(row['grade']),0)+1
    missing=[g for g in range(1,13) if not grades.get(g)]
    if missing: raise RuntimeError(f'missing grades: {missing}')
    log('merge',rows=count,grades_present=len(grades))
    gpu=''
    while not gpu:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        gpu=next((x.split(',')[0].strip() for x in rows if int(x.split(',')[1])>=6000),'')
        if not gpu: log('cluster',waiting_for_gpu=True); time.sleep(300)
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gpu; env['PYTHONPATH']='src'
    command('cluster',['.venv/bin/python','-u','src/visual_retrive/scripts/cluster_queries_e5.py','--pairs',str(ALL),'--data-root',str(DATA),'--output',str(RAW),'--model','intfloat/multilingual-e5-base','--batch-size','256','--top-k','12','--similarity-threshold','.90','--lexical-threshold','.72'],env)
    judgments=Path(str(REVIEWED)+'.judgments.jsonl')
    old_raw=CAT/'visrag_query_relevance_e5_v4_090_raw.json'; old_j=CAT/'visrag_query_relevance_e5_v4_090_reviewed.json.judgments.jsonl'
    if old_raw.exists() and old_j.exists():
        command('migrate',['.venv/bin/python','-u','src/visual_retrive/scripts/migrate_qwen_judgments.py','--source-groups',str(old_raw),'--source-judgments',str(old_j),'--target-groups',str(RAW),'--target-judgments',str(judgments)])
    key=next(x for x in (ROOT/'.env.openrouter').read_text().splitlines() if x.startswith('OPENROUTER_API_KEY=')).split('=',1)[1].strip().strip('"\'')
    if key.startswith('v1-'): key='sk-or-'+key
    env=os.environ.copy(); env['PYTHONPATH']='src'; env['OPENROUTER_API_KEY']=key
    command('qwen_review',['.venv/bin/python','-u','src/visual_retrive/scripts/review_semantic_group_edges_qwen.py','--pairs',str(ALL),'--groups',str(RAW),'--output',str(REVIEWED),'--workers','6','--batch-size','10','--min-candidate-lexical','.30','--max-positives','4','--endpoint','https://openrouter.ai/api/v1','--model','qwen/qwen3.5-9b','--api-key-env','OPENROUTER_API_KEY'],env)
    result=json.loads(REVIEWED.read_text()); judge=result['judge']
    if judge['labeled'] != judge['candidates']: raise RuntimeError(f'incomplete Qwen review: {judge}')
    log('complete',candidates=judge['candidates'],labeled=judge['labeled'],approved=judge['approved_semantic'])
    run.summary.update({'status':'complete','qwen_candidates':judge['candidates'],'qwen_labeled':judge['labeled']})
except Exception as exc:
    run.summary.update({'status':'failed','error':f'{type(exc).__name__}: {exc}'})
    raise
finally:
    run.finish()
