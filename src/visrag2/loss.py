from __future__ import annotations

import torch
import torch.nn.functional as F


def maxsim_scores(queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
    """ColBERT/ColQwen late interaction, normalized by query length."""
    # [query_batch, document_batch, query_tokens, document_tokens]
    similarity = torch.einsum("aqd,btd->abqt", queries, documents)
    return similarity.amax(dim=-1).sum(dim=-1) / queries.shape[1]


def multi_positive_infonce(scores: torch.Tensor, positive_mask: torch.Tensor, temperature: float = 0.02):
    """Probability mass of every equivalent page is positive; none is a false negative."""
    logits = scores / temperature
    if not positive_mask.any(dim=1).all():
        raise ValueError("every query must have at least one positive document")
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    return -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()


def stable_group_hash(group_id: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.blake2b(group_id.encode(), digest_size=8).digest(), "big") & (2**63-1)
