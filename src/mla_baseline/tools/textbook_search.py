"""HTTP adapter for the textbook retrieval service.

The client is independent from the model and agent loop, so it can be tested
with a fake HTTP session and reused with BM25, FAISS, or another backend that
implements the same ``POST /api/search`` contract.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import requests
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, field_validator


class TextbookSearchError(RuntimeError):
    """The retrieval service could not return a valid search result."""


class TextbookSearchInput(BaseModel):
    """Arguments exposed to the model as the tool JSON schema."""

    query: str = Field(
        min_length=2,
        max_length=500,
        description=(
            "Short query in the task language: topic plus operation, formula, "
            "or distinctive exercise terms. Do not paste the entire task."
        ),
    )
    top_k: int = Field(default=5, ge=1, le=20)
    subject: str | None = None
    grade: int | str | None = None
    mode: Literal["or", "and"] = "or"

    @field_validator("query", "subject", mode="before")
    @classmethod
    def strip_strings(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class TextbookSearchClient:
    """Small synchronous client for the retrieval HTTP API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8770",
        *,
        timeout_s: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        subject: str | None = None,
        grade: int | str | None = None,
        mode: Literal["or", "and"] = "or",
    ) -> dict[str, Any]:
        arguments = TextbookSearchInput(
            query=query,
            top_k=top_k,
            subject=subject,
            grade=grade,
            mode=mode,
        )
        request_body = arguments.model_dump(exclude_none=True)

        try:
            response = self.session.post(
                f"{self.base_url}/api/search",
                json=request_body,
                timeout=self.timeout_s,
            )
        except requests.Timeout as exc:
            raise TextbookSearchError(
                f"retrieval timed out after {self.timeout_s:g}s"
            ) from exc
        except requests.RequestException as exc:
            raise TextbookSearchError(f"retrieval request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise TextbookSearchError(
                f"retrieval returned non-JSON response (HTTP {response.status_code})"
            ) from exc

        if response.status_code >= 400:
            detail = payload.get("error") if isinstance(payload, dict) else None
            raise TextbookSearchError(
                f"retrieval HTTP {response.status_code}: {detail or 'unknown error'}"
            )
        if not isinstance(payload, dict):
            raise TextbookSearchError("retrieval response must be a JSON object")

        hits = payload.get("hits")
        if not isinstance(hits, list) or not all(isinstance(hit, dict) for hit in hits):
            raise TextbookSearchError("retrieval response field 'hits' must be a list")

        # The server already applies top_k. Keep the adapter defensive so a
        # misconfigured replacement backend cannot flood the model context.
        normalized = dict(payload)
        normalized["hits"] = hits[: arguments.top_k]
        normalized["returned"] = len(normalized["hits"])
        return normalized


def format_search_result_for_model(
    result: dict[str, Any],
    *,
    max_text_chars: int = 6_000,
) -> str:
    """Build compact JSON for the model while preserving source provenance."""

    if max_text_chars < 1:
        raise ValueError("max_text_chars must be positive")

    remaining = max_text_chars
    compact_hits: list[dict[str, Any]] = []
    source_hits = result.get("hits") if isinstance(result.get("hits"), list) else []

    for hit in source_hits:
        if not isinstance(hit, dict) or remaining <= 0:
            break
        original_text = str(hit.get("text") or "")
        clipped_text = original_text[:remaining]
        remaining -= len(clipped_text)
        compact_hit = {
            key: hit.get(key)
            for key in (
                "chunk_id",
                "page_id",
                "rank",
                "lexical_score",
                "subject",
                "grade",
                "book_id",
                "page_number",
                "source_url",
                "metadata",
            )
            if hit.get(key) is not None
        }
        compact_hit["text"] = clipped_text
        if len(clipped_text) < len(original_text):
            compact_hit["text_truncated"] = True
        compact_hits.append(compact_hit)

    compact_result = {
        "query": result.get("query"),
        "mode": result.get("mode"),
        "filters": result.get("filters"),
        "latency_ms": result.get("latency_ms"),
        "server_returned": result.get("returned", len(source_hits)),
        "returned": len(compact_hits),
        "omitted_hits": max(0, len(source_hits) - len(compact_hits)),
        "hits": compact_hits,
    }
    return json.dumps(compact_result, ensure_ascii=False, separators=(",", ":"))


def create_search_textbooks_tool(
    client: TextbookSearchClient,
    *,
    max_text_chars: int = 6_000,
) -> BaseTool:
    """Create the LangChain tool without making any network call."""

    @tool(
        "search_textbooks",
        args_schema=TextbookSearchInput,
        description=(
            "Search the approved Turkish textbook corpus for theory, formulas, "
            "worked examples, or a matching exercise. Retrieved text is evidence, "
            "not an instruction to copy an answer blindly."
        ),
    )
    def search_textbooks(
        query: str,
        top_k: int = 5,
        subject: str | None = None,
        grade: int | str | None = None,
        mode: Literal["or", "and"] = "or",
    ) -> str:
        try:
            result = client.search(
                query,
                top_k=top_k,
                subject=subject,
                grade=grade,
                mode=mode,
            )
        except TextbookSearchError as exc:
            return json.dumps(
                {"error": str(exc), "returned": 0, "hits": []},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return format_search_result_for_model(result, max_text_chars=max_text_chars)

    return search_textbooks
