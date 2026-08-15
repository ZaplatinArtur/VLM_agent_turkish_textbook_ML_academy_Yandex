"""Generate resumable Turkish retrieval queries for useful Evvel pages via Qwen/vLLM."""
from __future__ import annotations
import argparse, concurrent.futures, json, re, threading
from pathlib import Path
import requests

lock = threading.Lock()

def chunks(xs, n):
    for i in range(0, len(xs), n): yield xs[i:i+n]

def request_batch(batch, endpoint, model):
    docs=[]
    for i,row in enumerate(batch):
        text=row['answer_text'][:2200].replace('\x00',' ')
        docs.append(f"ID={i}\nSINIF={row['grade']}\nBAŞLIK={row['title']}\nMETİN:\n{text}")
    prompt=("Aşağıdaki Türkçe ders kitabı çözüm sayfalarının her biri için tam 3 kısa ve doğal "
            "arama sorgusu üret. Öğrenci, sayfayı görmeden bu sorguları yazabilmelidir. Sorgular "
            "metindeki konu, soru veya kavramlara özgü olsun; cevapları kopyalama, sayfa URL'sini "
            "ve 'bu metin' ifadesini kullanma. Yalnızca JSON dizisi döndür: "
            '[{"id":0,"queries":["...","...","..."]}].\n\n'+"\n\n---\n\n".join(docs))
    payload={'model':model,'messages':[{'role':'user','content':prompt}],
             'temperature':0.35,'top_p':0.9,'max_tokens':1000,
             'chat_template_kwargs':{'enable_thinking':False}}
    r=requests.post(endpoint.rstrip('/')+'/chat/completions',json=payload,timeout=180)
    r.raise_for_status(); content=r.json()['choices'][0]['message']['content']
    m=re.search(r'\[.*\]',content,re.S)
    if not m: raise ValueError('no JSON array')
    try:
        parsed=json.loads(m.group())
    except json.JSONDecodeError:
        # Recover complete objects when the model leaves the outer array malformed.
        parsed=[]
        for obj in re.findall(r'\{\s*"id"\s*:\s*\d+\s*,\s*"queries"\s*:\s*\[[^\]]*\]\s*\}',m.group(),re.S):
            try: parsed.append(json.loads(obj))
            except json.JSONDecodeError: pass
    by_id={int(x['id']):x.get('queries',[]) for x in parsed if isinstance(x,dict) and 'id' in x}
    out=[]
    for i,row in enumerate(batch):
        seen=[]
        for q in by_id.get(i,[]):
            q=' '.join(str(q).split()).strip(' -"')
            key=q.casefold()
            if 8 <= len(q) <= 240 and key not in {x.casefold() for x in seen}: seen.append(q)
        if len(seen)>=2:
            out.append({**row,'synthetic_queries':seen[:3],'query_generator':model})
    # A small tail of pages repeatedly produces malformed batch JSON. For a
    # singleton, retry with a minimal schema and parse either JSON or lines.
    if len(batch)==1 and not out:
        row=batch[0]
        prompt=("Bu Türkçe ders kitabı çözümü için tam 3 farklı, kısa öğrenci arama sorgusu üret. "
                "Yalnızca üç satır döndür; numara, açıklama ve cevap yazma.\nBAŞLIK: "+str(row.get('title',''))+
                "\nMETİN:\n"+str(row.get('answer_text',''))[:1800])
        payload={'model':model,'messages':[{'role':'user','content':prompt}],
                 'temperature':0.25,'max_tokens':300,'chat_template_kwargs':{'enable_thinking':False}}
        rr=requests.post(endpoint.rstrip('/')+'/chat/completions',json=payload,timeout=180); rr.raise_for_status()
        content=rr.json()['choices'][0]['message']['content'] or ''
        candidates=[]
        try:
            value=json.loads(content[content.find('['):content.rfind(']')+1])
            if isinstance(value,list): candidates=[str(x) for x in value]
        except Exception: pass
        if not candidates:
            candidates=[re.sub(r'^\s*(?:[-*]|\d+[.)])\s*','',x).strip(' "') for x in content.splitlines()]
        seen=[]
        for q in candidates:
            q=' '.join(q.split())
            if 8<=len(q)<=240 and q.casefold() not in {x.casefold() for x in seen}:seen.append(q)
        if len(seen)>=2: out.append({**row,'synthetic_queries':seen[:3],'query_generator':model,'parse_fallback':'single_page_lines'})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--endpoint',action='append',dest='endpoints'); ap.add_argument('--model',default='Qwen/Qwen3.5-9B')
    ap.add_argument('--workers',type=int,default=4); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--max-pages',type=int)
    a=ap.parse_args(); rows=[json.loads(x) for x in a.input.open(encoding='utf-8')]
    rows=[x for x in rows if x.get('useful_answer')]
    done=set()
    if a.output.exists():
        for line in a.output.open(encoding='utf-8'):
            try: done.add(json.loads(line)['page_url'])
            except Exception: pass
    rows=[x for x in rows if x['page_url'] not in done]
    if a.max_pages: rows=rows[:a.max_pages]
    batches=list(chunks(rows,a.batch_size)); a.output.parent.mkdir(parents=True,exist_ok=True)
    endpoints=a.endpoints or ['http://127.0.0.1:8010/v1','http://127.0.0.1:8011/v1']
    print(f'endpoints={endpoints} workers={a.workers}',flush=True)
    ok=failed=0
    with a.output.open('a',encoding='utf-8') as f, concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(request_batch,b,endpoints[i%len(endpoints)],a.model):b for i,b in enumerate(batches)}
        for i,fut in enumerate(concurrent.futures.as_completed(futures),1):
            try:
                result=fut.result()
                with lock:
                    for row in result: f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    f.flush()
                ok+=len(result); failed+=len(futures[fut])-len(result)
            except Exception as e:
                failed+=len(futures[fut]); print(f'batch_failed={type(e).__name__}:{e}',flush=True)
            print(f'batches={i}/{len(batches)} generated={ok} failed={failed}',flush=True)
    print(json.dumps({'useful_input':len(rows),'generated':ok,'failed':failed,'output':str(a.output)}),flush=True)

if __name__=='__main__': main()
