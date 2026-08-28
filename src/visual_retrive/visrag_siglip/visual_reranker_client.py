from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class VisualRerankerClient:
    def __init__(self, base_url: str, *, timeout_s: float = 120.0):
        self.url = base_url.rstrip("/") + "/rerank"
        self.timeout_s = timeout_s

    def rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not hits:
            return []
        payload = {
            "query": query,
            "top_k": top_k,
            "candidates": [
                {"page_id": str(hit["page_id"]), "page_image": str(hit["page_image"])}
                for hit in hits
            ],
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
        scores = {str(row["page_id"]): float(row["score"]) for row in result.get("results", [])}
        reranked = []
        for hit in hits:
            page_id = str(hit["page_id"])
            if page_id not in scores:
                continue
            row = dict(hit)
            row["retrieval_score"] = float(hit.get("score", 0.0))
            row["rerank_score"] = scores[page_id]
            row["score"] = scores[page_id]
            reranked.append(row)
        reranked.sort(key=lambda row: (-row["rerank_score"], row["page_id"]))
        for rank, row in enumerate(reranked[:top_k], 1):
            row["rank"] = rank
        return reranked[:top_k]
