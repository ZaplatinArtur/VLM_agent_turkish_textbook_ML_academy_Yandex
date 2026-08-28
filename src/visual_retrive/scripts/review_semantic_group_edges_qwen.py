"""Strictly judge embedding-only adjacent-page relevance edges with Qwen."""
from __future__ import annotations
import argparse, concurrent.futures, json, os, random, re, time
from collections import defaultdict
from pathlib import Path
import requests

ENDPOINTS=['http://127.0.0.1:8010/v1','http://127.0.0.1:8011/v1','http://127.0.0.1:8012/v1']
MODEL='Qwen/Qwen3.5-9B'
API_KEY=None

def judge(batch,endpoint,queries):
    items=[]
    for i,e in batch:
        items.append({'id':i,'subject':e.get('subject'),'page_a':e['a_page'],'page_b':e['b_page'],
                      'query_a':e['a_query'],'query_b':e['b_query']})
    prompt=("Türkçe ders kitabı arama sorgularında çok katı bir alaka denetçisi ol. İki komşu sayfa ancak "
            "aynı öğrenci sorgusu, kişi/nesne/kavram/sayı değiştirilmeden iki sayfayı da doğru sonuç yapıyorsa "
            "multi-positive olabilir. Yalnızca aynı ders veya geniş konu yeterli değildir. Örn. Dalton ve Thomson, "
            "dil sağlığı ve deri sağlığı farklıdır. JSON döndür: [{\"id\":0,\"same_intent\":true/false}].\n"+json.dumps(items,ensure_ascii=False))
    payload={'model':MODEL,'messages':[{'role':'user','content':prompt}],
             'temperature':0,'max_tokens':700,'chat_template_kwargs':{'enable_thinking':False}}
    headers={'Authorization':f'Bearer {API_KEY}'} if API_KEY else None
    last_error=None
    for attempt in range(5):
        try:
            with requests.post(endpoint.rstrip('/')+'/chat/completions',json=payload,headers=headers,timeout=180) as response:
                response.raise_for_status(); content=response.json()['choices'][0]['message']['content'] or ''
            parsed={int(i):v=='true' for i,v in re.findall(r'"id"\s*:\s*(\d+).*?"same_intent"\s*:\s*(true|false)',content,re.S|re.I)}
            if not parsed: raise ValueError('Qwen returned no parseable labels')
            return parsed
        except Exception as exc:
            last_error=exc
            if attempt<4: time.sleep(min(30,2**attempt+random.random()))
    raise last_error

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--groups',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--batch-size',type=int,default=10)
    ap.add_argument('--min-candidate-lexical',type=float,default=.30)
    ap.add_argument('--max-positives',type=int,default=4)
    ap.add_argument('--endpoint',action='append',dest='endpoints')
    ap.add_argument('--model',default='Qwen/Qwen3.5-9B'); ap.add_argument('--api-key-env')
    a=ap.parse_args(); queries=defaultdict(list); page_grades={}
    global ENDPOINTS, MODEL, API_KEY
    if a.endpoints: ENDPOINTS=a.endpoints
    MODEL=a.model; API_KEY=os.environ.get(a.api_key_env) if a.api_key_env else None
    for line in a.pairs.open(encoding='utf-8'):
        row=json.loads(line); page=str(row['positive_page_id']); queries[page].append(str(row['query'])); page_grades[page]=int(row.get('grade') or 0)
    data=json.loads(a.groups.read_text(encoding='utf-8')); lexical=float(data['config']['lexical_threshold'])
    def involves_high_school(edge):
        return max(int(edge.get('a_grade') or page_grades.get(edge['a_page'],0)),
                   int(edge.get('b_grade') or page_grades.get(edge['b_page'],0)))>=9
    forced_ids={i for i,e in enumerate(data['edges']) if involves_high_school(e)}
    candidates=[(i,e) for i,e in enumerate(data['edges']) if i in forced_ids or
                (e.get('semantic') is not None and e['semantic']>=data['config']['semantic_threshold'] and a.min_candidate_lexical<=e['lexical']<lexical)]
    checkpoint=a.output.with_suffix(a.output.suffix+'.judgments.jsonl'); labels={}
    if checkpoint.exists():
        for line in checkpoint.open(encoding='utf-8'):
            try:
                row=json.loads(line); labels[int(row['id'])]=bool(row['same_intent'])
            except Exception: pass
    pending=[x for x in candidates if x[0] not in labels]
    batches=[pending[i:i+a.batch_size] for i in range(0,len(pending),a.batch_size)]
    print(json.dumps({'candidates':len(candidates),'forced_high_school_candidates':len(forced_ids),
                      'resumed':len(labels),'pending':len(pending)}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        fs={ex.submit(judge,b,ENDPOINTS[i%len(ENDPOINTS)],queries):b for i,b in enumerate(batches)}
        for n,f in enumerate(concurrent.futures.as_completed(fs),1):
            try:
                new_labels=f.result(); labels.update(new_labels)
                with checkpoint.open('a',encoding='utf-8') as out:
                    for idx,value in new_labels.items():
                        edge=data['edges'][idx]
                        out.write(json.dumps({'id':idx,'same_intent':value,'semantic':edge.get('semantic'),
                            'lexical':edge.get('lexical'),'subject':edge.get('subject')},ensure_ascii=False)+'\n')
            except Exception as e: print(f'failed={type(e).__name__}:{e}',flush=True)
            print(f'reviewed_batches={n}/{len(batches)} labels={len(labels)}',flush=True)
    approved_edges=[e for i,e in enumerate(data['edges']) if labels.get(i,False) or (i not in forced_ids and e['lexical']>=lexical)]
    relevance={}
    kept=[]
    for e in sorted(approved_edges,key=lambda x:(x['lexical']>=lexical,max(x['lexical'],x.get('semantic') or 0)),reverse=True):
        aa=relevance.setdefault(e['a_key'],{e['a_page']});bb=relevance.setdefault(e['b_key'],{e['b_page']})
        if (e['b_page'] not in aa and len(aa)>=a.max_positives) or (e['a_page'] not in bb and len(bb)>=a.max_positives):continue
        aa.add(e['b_page']);bb.add(e['a_page']);kept.append(e)
    approved_edges=kept
    relevance={k:v for k,v in relevance.items() if len(v)>1}
    result={**data,'query_relevance':{k:sorted(v) for k,v in relevance.items()},'edges':approved_edges,
            'judge':{'model':MODEL,'candidates':len(candidates),'forced_high_school_candidates':len(forced_ids),
                     'labeled':len(labels),'approved_semantic':sum(labels.get(i,False) for i,_ in candidates),
                     'checkpoint':str(checkpoint)}}
    result['stats']={**data['stats'],'multi_positive_queries':len(relevance),'edges':len(approved_edges)}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result['judge']),flush=True);print(json.dumps(result['stats']),flush=True)

if __name__=='__main__':main()
