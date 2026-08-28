from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image

from .data import Record
from .model import encode_images, encode_texts


CUTOFFS = (1, 3, 5, 10, 20, 30)


def summarize(rankings: list[list[int]], relevant_counts: list[int], pages: int):
    n = max(1, len(rankings)); out = {"pages": pages, "queries": len(rankings)}
    first_ranks = [min(ranks) for ranks in rankings]
    for k in CUTOFFS:
        found = [sum(rank <= k for rank in ranks) for ranks in rankings]
        out[f"hit@{k}"] = sum(x > 0 for x in found)/n
        out[f"recall@{k}"] = sum(x/max(1, total) for x, total in zip(found, relevant_counts))/n
        out[f"precision@{k}"] = sum(x/k for x in found)/n
    # Dynamic cutoff: each query is evaluated at the number of its relevant pages.
    found_at_r = [sum(rank <= total for rank in ranks) for ranks, total in zip(rankings, relevant_counts)]
    out["recall@R"] = sum(found/max(1, total) for found, total in zip(found_at_r, relevant_counts))/n
    out["precision@R"] = sum(found/max(1, total) for found, total in zip(found_at_r, relevant_counts))/n
    out["mrr@10"] = sum(1/r if r <= 10 else 0 for r in first_ranks)/n
    out["ndcg@10"] = sum(
        sum(1/math.log2(r+1) for r in ranks if r <= 10) /
        max(1e-12, sum(1/math.log2(r+1) for r in range(1, min(10,total)+1)))
        for ranks, total in zip(rankings, relevant_counts)
    )/n
    return out


@torch.inference_mode()
def evaluate_corpus(model, processor, records: list[Record], device, batch_size=64):
    """Each subject: exactly 120 page images and one deterministic query per page."""
    model.eval(); by_subject = defaultdict(list); subject_metrics = {}; all_rows = []
    for record in records: by_subject[record.subject].append(record)
    for subject, pages in sorted(by_subject.items()):
        doc_parts = []
        for start in range(0, len(pages), batch_size):
            images = []
            for record in pages[start:start+batch_size]:
                with Image.open(record.image) as image: images.append(image.convert("RGB").copy())
            doc_parts.append(encode_images(model, processor, images, device).cpu())
        docs = F.normalize(torch.cat(doc_parts), dim=-1)
        queries = [record.queries[0] for record in pages]
        query_parts = [encode_texts(model, processor, queries[i:i+batch_size], device).cpu()
                       for i in range(0, len(queries), batch_size)]
        scores = F.normalize(torch.cat(query_parts), dim=-1) @ docs.T
        rankings, counts = [], []
        for i, record in enumerate(pages):
            positives = torch.tensor([p.group_id == record.group_id for p in pages], dtype=torch.bool)
            order = torch.argsort(scores[i], descending=True)
            ranks = [rank for rank, index in enumerate(order.tolist(), 1) if positives[index]]
            rankings.append(ranks); counts.append(int(positives.sum()))
            all_rows.append((ranks, int(positives.sum())))
        subject_metrics[subject] = summarize(rankings, counts, len(pages))
    metric_keys = [key for key in next(iter(subject_metrics.values())) if key not in {"pages", "queries"}]
    macro = {key: sum(values[key] for values in subject_metrics.values())/len(subject_metrics) for key in metric_keys}
    micro = summarize([x[0] for x in all_rows], [x[1] for x in all_rows], len(records))
    return {"macro": macro, "micro": micro, "subjects": subject_metrics}
