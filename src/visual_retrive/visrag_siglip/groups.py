from __future__ import annotations

import json
import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path


_WORD = re.compile(r"\w+", re.UNICODE)


def normalize_query(text: str) -> str:
    return " ".join(_WORD.findall(str(text).casefold()))


def token_jaccard(a: str, b: str) -> float:
    aa, bb = set(normalize_query(a).split()), set(normalize_query(b).split())
    return len(aa & bb) / max(1, len(aa | bb))


def relevance_key(page: str, query: str) -> str:
    value=f"{page}\0{normalize_query(query)}".encode('utf-8')
    return hashlib.sha1(value).hexdigest()


def relevant_pages(row, relevance: dict[str,set[str]]) -> set[str]:
    page=str(row['positive_page_id'])
    explicit = {str(value) for value in (row.get('same_source_page_ids') or [])}
    return set(relevance.get(relevance_key(page,str(row['query'])),{page})) | explicit | {page}


def load_relevance_groups(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if 'query_relevance' in payload:
        return {str(k):{str(x) for x in v} for k,v in payload['query_relevance'].items()}
    groups = payload.get("groups", payload)
    result: dict[str, set[str]] = {}
    if isinstance(groups, dict):
        for page, members in groups.items():
            values = {str(x) for x in members} | {str(page)}
            for member in values:
                result[member] = values
    else:
        for members in groups:
            values = {str(x) for x in members}
            for member in values:
                result[member] = values
    return result


def build_query_relevance(rows, query_vectors: dict[str,list[float]] | None=None,
                          semantic_threshold=.96, lexical_threshold=.72, max_page_gap=1):
    """Create per-query (not page-wide) relevance labels for adjacent pages."""
    by_page=defaultdict(list); books=defaultdict(list)
    for row in rows:
        page=str(row['positive_page_id']); query=str(row['query']); key=relevance_key(page,query)
        by_page[page].append((key,query));
    for page in by_page:
        try:book,num=page.rsplit(':',1);books[book].append((int(num),page))
        except ValueError:pass
    relevance={key:{page} for page,items in by_page.items() for key,_ in items}; edges=[]
    def cosine(a,b):
        dot=sum(x*y for x,y in zip(a,b));na=math.sqrt(sum(x*x for x in a));nb=math.sqrt(sum(x*x for x in b))
        return dot/max(1e-12,na*nb)
    for values in books.values():
        values.sort()
        for i,(num,a) in enumerate(values):
            for num2,b in values[i+1:]:
                if num2-num>max_page_gap:break
                for ka,qa in by_page[a]:
                    for kb,qb in by_page[b]:
                        lex=token_jaccard(qa,qb);sem=None
                        if query_vectors and ka in query_vectors and kb in query_vectors:sem=cosine(query_vectors[ka],query_vectors[kb])
                        if lex>=lexical_threshold or (sem is not None and sem>=semantic_threshold):
                            relevance[ka].add(b);relevance[kb].add(a)
                            edges.append({'a_page':a,'a_key':ka,'a_query':qa,'b_page':b,'b_key':kb,'b_query':qb,'lexical':lex,'semantic':sem})
    relevance={k:sorted(v) for k,v in relevance.items() if len(v)>1}
    return {'query_relevance':relevance,'edges':edges,'config':{'semantic_threshold':semantic_threshold,
            'lexical_threshold':lexical_threshold,'max_page_gap':max_page_gap},
            'stats':{'rows':len(rows),'multi_positive_queries':len(relevance),'edges':len(edges)}}


class _DSU:
    def __init__(self, values): self.parent={x:x for x in values}; self.members={x:{x} for x in values}
    def find(self,x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x=self.parent[x]
        return x
    def union(self,a,b,max_size=4):
        a,b=self.find(a),self.find(b)
        if a==b: return True
        if len(self.members[a])+len(self.members[b]) > max_size: return False
        if len(self.members[a]) < len(self.members[b]): a,b=b,a
        self.parent[b]=a; self.members[a] |= self.members.pop(b); return True


def build_adjacent_groups(rows, page_vectors: dict[str, list[float]] | None = None,
                          semantic_threshold: float = .94, lexical_threshold: float = .72,
                          max_page_gap: int = 1, max_group_size: int = 4):
    """Group only adjacent pages with strong query or embedding agreement.

    The lexical gate catches exact/reordered Turkish queries. Embeddings add semantic
    paraphrases, with a deliberately high threshold to avoid making broad chapter
    topics false positives.
    """
    queries=defaultdict(list); books=defaultdict(list)
    for row in rows:
        page=str(row["positive_page_id"]); queries[page].append(str(row["query"]))
    for page in queries:
        try: book,num=page.rsplit(":",1); books[book].append((int(num),page))
        except ValueError: continue
    dsu=_DSU(queries); edges=[]
    def cosine(a,b):
        dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
        return dot/max(1e-12,na*nb)
    for book, values in books.items():
        values.sort()
        for i,(num,a) in enumerate(values):
            for num2,b in values[i+1:]:
                if num2-num > max_page_gap: break
                lex=max(token_jaccard(x,y) for x in queries[a] for y in queries[b])
                sem=None
                if page_vectors and a in page_vectors and b in page_vectors: sem=cosine(page_vectors[a],page_vectors[b])
                matched=lex >= lexical_threshold or (sem is not None and sem >= semantic_threshold)
                if matched and dsu.union(a,b,max_group_size): edges.append({"a":a,"b":b,"lexical":lex,"semantic":sem})
    groups=[]
    for members in dsu.members.values():
        if len(members)>1: groups.append(sorted(members))
    mapping={p:g for g in groups for p in g}
    return {"groups":groups,"page_to_group":mapping,"edges":edges,
            "config":{"semantic_threshold":semantic_threshold,"lexical_threshold":lexical_threshold,
                      "max_page_gap":max_page_gap,"max_group_size":max_group_size}}
