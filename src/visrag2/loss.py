from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_supcon(scores: torch.Tensor, positive_mask: torch.Tensor, temperature: float = 0.05):
    """Mean log-probability of every relevant target; relevant pairs are never negatives."""
    if not positive_mask.any(dim=1).all():
        raise ValueError("every anchor must have at least one positive")
    log_prob = F.log_softmax(scores.float()/temperature, dim=1)
    weights = positive_mask.float()/positive_mask.sum(1, keepdim=True)
    return -(weights*log_prob).sum(1).mean()


def symmetric_multi_positive_loss(query_embeddings, document_embeddings, positive_mask, temperature=.05):
    scores = F.normalize(query_embeddings.float(), dim=-1) @ F.normalize(document_embeddings.float(), dim=-1).T
    q2d = multi_positive_supcon(scores, positive_mask, temperature)
    d2q = multi_positive_supcon(scores.T, positive_mask.T, temperature)
    return (q2d+d2q)/2, scores
