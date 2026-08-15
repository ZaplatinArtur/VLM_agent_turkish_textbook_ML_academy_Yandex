"""Subject-wise mutual-kNN mining for query-specific multi-positive labels."""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from visual_retrive.visrag_siglip.data import read_jsonl, usable_rows
from visual_retrive.visrag_siglip.groups import relevance_key, token_jaccard

NUM=re.compile(r'\d+(?:[.,]\d+)?')

def subject_family(subject):
    value=str(subject or 'unknown')
    if value in {'physics','chemistry','biology','science'}: return 'science'
    if value in {'history','geography','social studies'}: return 'social studies'
    return value

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--model',default='intfloat/multilingual-e5-base')
    ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--top-k',type=int,default=12)
    ap.add_argument('--similarity-threshold',type=float,default=.94); ap.add_argument('--lexical-threshold',type=float,default=.72)
    a=ap.parse_args(); rows=usable_rows(read_jsonl(a.pairs),a.data_root)
    unique={}
    for row in rows:
        key=relevance_key(str(row['positive_page_id']),str(row['query']))
        unique[key]={'key':key,'page':str(row['positive_page_id']),'query':str(row['query']),
                     'subject':subject_family(row.get('subject')),'grade':int(row.get('grade') or 0)}
    items=list(unique.values()); model=SentenceTransformer(a.model,device='cuda')
    vec=model.encode(['query: '+x['query'] for x in items],batch_size=a.batch_size,normalize_embeddings=True,
                     convert_to_numpy=True,show_progress_bar=True).astype('float32')
    by_subject=defaultdict(list)
    for i,x in enumerate(items):by_subject[x['subject']].append(i)
    edges=[]
    for subject,global_ids in sorted(by_subject.items()):
        matrix=vec[global_ids]; index=faiss.IndexFlatIP(matrix.shape[1]); index.add(matrix)
        scores,neighbors=index.search(matrix,min(a.top_k+1,len(global_ids)))
        neighbor_sets=[set(int(x) for x in row[1:]) for row in neighbors]
        seen=set()
        for i in range(len(global_ids)):
            ai=items[global_ids[i]]
            for rank,j in enumerate(neighbors[i][1:],1):
                j=int(j)
                if i not in neighbor_sets[j] or ai['page']==items[global_ids[j]]['page']:continue
                pair=tuple(sorted((i,j)))
                if pair in seen:continue
                seen.add(pair); bi=items[global_ids[j]]; sim=float(matrix[i]@matrix[j]); lex=token_jaccard(ai['query'],bi['query'])
                nums_a,nums_b=set(NUM.findall(ai['query'])),set(NUM.findall(bi['query']))
                number_conflict=bool(nums_a and nums_b and nums_a!=nums_b)
                if not number_conflict and (sim>=a.similarity_threshold or lex>=a.lexical_threshold):
                    edges.append({'a_page':ai['page'],'a_key':ai['key'],'a_query':ai['query'],'a_grade':ai['grade'],
                                  'b_page':bi['page'],'b_key':bi['key'],'b_query':bi['query'],
                                  'b_grade':bi['grade'],'subject':subject,'semantic':sim,'lexical':lex,'mutual_rank_a':rank})
        print(json.dumps({'subject':subject,'queries':len(global_ids),'candidates':len(seen),
                          'accepted_total':sum(e['subject']==subject for e in edges)}),flush=True)
    result={'query_relevance':{},'edges':edges,'config':{'model':a.model,'top_k':a.top_k,
            'semantic_threshold':a.similarity_threshold,'lexical_threshold':a.lexical_threshold,'scope':'subject_mutual_knn'},
            'stats':{'rows':len(rows),'unique_queries':len(items),'edges':len(edges)}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['stats']),flush=True)

if __name__=='__main__':main()
