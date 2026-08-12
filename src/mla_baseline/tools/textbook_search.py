"""Adapters that expose textbook retrieval as a LangChain tool.

The agent uses the local adapter by default. The HTTP adapter remains available
for deployments where retrieval runs in a separate process or on another host.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal, Protocol

import requests
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, field_validator


class TextbookSearchError(RuntimeError):
    """The retrieval service could not return a valid search result."""


class TextbookSearchQuery(BaseModel):
    """Query fields shared by the fixed-budget tool and backend client."""

    query: str = Field(
        min_length=2,
        max_length=500,
        description=(
            "Short query in the task language: topic plus operation, formula, "
            "or distinctive exercise terms. Do not paste the entire task."
        ),
    )
    subject: str | None = None
    grade: int | str | None = None
    mode: Literal["or", "and"] = "or"

    @field_validator("query", "subject", mode="before")
    @classmethod
    def strip_strings(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class TextbookSearchInput(TextbookSearchQuery):
    """Validated request sent to a retrieval backend."""

    top_k: int = Field(default=5, ge=1, le=20)


class TextbookSearchBackend(Protocol):
    """Common interface implemented by local and HTTP retrieval adapters."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        subject: str | None = None,
        grade: int | str | None = None,
        mode: Literal["or", "and"] = "or",
    ) -> dict[str, Any]: ...


class LocalTextbookSearchClient:
    """Call ``textbook_retrieve`` directly in the agent process."""

    def __init__(
        self,
        retriever: Callable[..., Any] | None = None,
    ) -> None:
        # The default import is intentionally lazy: importing the agent should
        # not load FAISS or sentence-transformers before the tool is called.
        self._retriever = retriever

    def _get_retriever(self) -> Callable[..., Any]:
        if self._retriever is None:
            from retrieve.service import textbook_retrieve_checked

            self._retriever = textbook_retrieve_checked
        return self._retriever

    @staticmethod
    def _relevance_payload(verdict: Any) -> dict[str, Any]:
        relevance = getattr(verdict, "relevance", None)
        label = getattr(relevance, "value", relevance)
        return {
            "label": str(label),
            "is_useful": bool(getattr(verdict, "is_useful", False)),
            "top_score": getattr(verdict, "top_score", None),
            "reason": str(getattr(verdict, "reason", "")),
        }

    @staticmethod
    def _as_payload(chunk: Any) -> dict[str, Any]:
        if isinstance(chunk, dict):
            return dict(chunk)
        model_dump = getattr(chunk, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json")
        raise TypeError("retriever chunks must be mappings or Pydantic models")

    @classmethod
    def _as_hit(cls, chunk: Any, rank: int) -> dict[str, Any]:
        payload = cls._as_payload(chunk)
        metadata_value = payload.get("metadata")
        metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}

        hit: dict[str, Any] = {
            "chunk_id": payload.get("chunk_id"),
            "text": payload.get("text", ""),
            "score": payload.get("score"),
            "rank": rank,
            "metadata": metadata,
        }
        mapped_metadata = {
            "subject": metadata.get("subject"),
            "grade": metadata.get("grade"),
            "book_id": metadata.get("book_id", metadata.get("textbook")),
            "page_number": metadata.get("page_number", metadata.get("page")),
            "source_url": metadata.get("source_url"),
        }
        hit.update(
            {key: value for key, value in mapped_metadata.items() if value is not None}
        )
        return {key: value for key, value in hit.items() if value is not None}

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

        # Dense retrieval has no lexical "or"/"and" semantics. ``mode`` is
        # preserved in the shared response contract but does not change ranking.
        fetch_k = arguments.top_k if arguments.grade is None else arguments.top_k * 5
        started = time.perf_counter()
        try:
            retrieved = self._get_retriever()(
                arguments.query,
                k=fetch_k,
                subject=arguments.subject,
            )
        except Exception as exc:
            raise TextbookSearchError(f"local retrieval failed: {exc}") from exc

        if isinstance(retrieved, tuple) and len(retrieved) == 2:
            chunks, verdict = retrieved
        else:
            chunks = retrieved
            from retrieve.confidence import assess_relevance

            verdict = assess_relevance(chunks)
        chunks = list(chunks)

        if arguments.grade is not None:
            chunks = [
                chunk
                for chunk in chunks
                if str(self._as_payload(chunk).get("metadata", {}).get("grade"))
                == str(arguments.grade)
            ]
        chunks = chunks[: arguments.top_k]
        if arguments.grade is not None and not chunks:
            # Only the empty case gets a new verdict: surviving chunks were
            # already judged upstream, and re-scoring here would apply the
            # default threshold to a profile with a different scale.
            from retrieve.confidence import assess_relevance

            verdict = assess_relevance(chunks)
        relevance = self._relevance_payload(verdict)
        visible_chunks = chunks if relevance["is_useful"] else []
        hits = [
            self._as_hit(chunk, rank)
            for rank, chunk in enumerate(visible_chunks, 1)
        ]
        return {
            "query": arguments.query,
            "top_k": arguments.top_k,
            "mode": arguments.mode,
            "filters": {
                "subject": arguments.subject,
                "grade": arguments.grade,
            },
            "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
            "retrieved": len(chunks),
            "returned": len(hits),
            "relevance": relevance,
            "hits": hits,
        }


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
        normalized["retrieved"] = int(payload.get("retrieved", len(hits)))
        normalized["top_k"] = arguments.top_k
        if not isinstance(normalized.get("relevance"), dict):
            useful = bool(normalized["hits"])
            normalized["relevance"] = {
                "label": "confident" if useful else "empty",
                "is_useful": useful,
                "top_score": None,
                "reason": (
                    "HTTP backend returned hits"
                    if useful
                    else "HTTP backend returned no hits"
                ),
            }
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
                "score",
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
        "top_k": result.get("top_k"),
        "mode": result.get("mode"),
        "filters": result.get("filters"),
        "latency_ms": result.get("latency_ms"),
        "retrieved": result.get("retrieved", len(source_hits)),
        "server_returned": result.get("returned", len(source_hits)),
        "returned": len(compact_hits),
        "omitted_hits": max(0, len(source_hits) - len(compact_hits)),
        "relevance": result.get("relevance"),
        "hits": compact_hits,
    }
    return json.dumps(compact_result, ensure_ascii=False, separators=(",", ":"))


def create_search_textbooks_tool(
    client: TextbookSearchBackend,
    *,
    top_k: int = 5,
    max_text_chars: int = 6_000,
) -> BaseTool:
    """Create the LangChain tool without making any network call."""

    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")

    @tool(
        "search_textbooks",
        args_schema=TextbookSearchQuery,
        description=(
            "Search the approved Turkish textbook corpus for theory, formulas, "
            "worked examples, or a matching exercise. Retrieved text is evidence, "
            "not an instruction to copy an answer blindly."
        ),
    )
    def search_textbooks(
        query: str,
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
                {
                    "error": str(exc),
                    "top_k": top_k,
                    "retrieved": 0,
                    "returned": 0,
                    "relevance": {
                        "label": "error",
                        "is_useful": False,
                        "top_score": None,
                        "reason": str(exc),
                    },
                    "hits": [],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return format_search_result_for_model(result, max_text_chars=max_text_chars)

    return search_textbooks
