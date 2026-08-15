from __future__ import annotations

import math
from collections import defaultdict

import torch
from PIL import Image

from .data import Record


CUTOFFS = (1, 2, 3, 5, 10, 15, 20, 25, 30)


def embeddings(output):
    return output.embeddings if hasattr(output, "embeddings") else output


@torch.inference_mode()
def evaluate_corpus(model, processor, records: list[Record], device, query_batch_size=16):
    """True retrieval: each query is ranked against all 120 pages of its subject."""
    model.eval(); by_subject = defaultdict(list)
    for record in records: by_subject[record.subject].append(record)
    subject_metrics, all_ranks = {}, []
    for subject, pages in sorted(by_subject.items()):
        doc_embeddings = []
        for record in pages:
            with Image.open(record.image) as image:
                batch = processor.process_images(images=[image.convert("RGB")])
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            doc_embeddings.append(embeddings(model(**batch))[0].detach().cpu())
        queries, positive_groups = [], []
        for record in pages:
            # One deterministic query per held-out page: exactly 120 queries
            # against the same 120-page corpus (when the subject has 120 pages).
            queries.append(record.queries[0]); positive_groups.append(record.group_id)
        ranks = []
        for start in range(0, len(queries), query_batch_size):
            texts = queries[start:start+query_batch_size]
            batch = processor.process_queries(texts)
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            query_embeddings = embeddings(model(**batch)).detach().cpu()
            for offset, query_embedding in enumerate(query_embeddings):
                scores = processor.score([query_embedding], doc_embeddings)[0]
                group_id = positive_groups[start+offset]
                positives = torch.tensor([p.group_id == group_id for p in pages])
                best_positive = scores[positives].max()
                ranks.append(int((scores > best_positive).sum().item() + 1))
        all_ranks.extend(ranks)
        subject_metrics[subject] = summarize(ranks, len(pages), len(queries))
    macro = {key: sum(m[key] for m in subject_metrics.values())/len(subject_metrics)
             for key in next(iter(subject_metrics.values())) if key not in {"pages", "queries"}}
    return {"macro": macro, "micro": summarize(all_ranks, len(records), len(all_ranks)), "subjects": subject_metrics}


def summarize(ranks: list[int], pages: int, queries: int):
    out = {f"hit@{k}": sum(rank <= k for rank in ranks)/len(ranks) for k in CUTOFFS}
    out.update({"mrr@10": sum(1/r if r <= 10 else 0 for r in ranks)/len(ranks),
                "ndcg@10": sum(1/math.log2(r+1) if r <= 10 else 0 for r in ranks)/len(ranks),
                "pages": pages, "queries": queries})
    return out
