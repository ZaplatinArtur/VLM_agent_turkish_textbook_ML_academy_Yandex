from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from mla_baseline.tools import (
    TextbookSearchClient,
    TextbookSearchError,
    create_search_textbooks_tool,
    format_search_result_for_model,
)


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, json: dict, timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _search_payload(*hits: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": "kesirlerde toplama",
        "mode": "or",
        "filters": {"subject": "math", "grade": 5},
        "latency_ms": 12.5,
        "returned": len(hits),
        "hits": list(hits),
    }


def test_search_sends_contract_and_defensively_limits_hits() -> None:
    session = FakeSession(
        FakeResponse(
            _search_payload(
                {"chunk_id": "c1", "text": "first"},
                {"chunk_id": "c2", "text": "second"},
            )
        )
    )
    client = TextbookSearchClient(
        "http://retrieval.test/",
        timeout_s=3.5,
        session=session,  # type: ignore[arg-type]
    )

    result = client.search(
        "  kesirlerde toplama  ",
        top_k=1,
        subject=" math ",
        grade=5,
    )

    assert result["returned"] == 1
    assert [hit["chunk_id"] for hit in result["hits"]] == ["c1"]
    assert session.calls == [
        {
            "url": "http://retrieval.test/api/search",
            "json": {
                "query": "kesirlerde toplama",
                "top_k": 1,
                "subject": "math",
                "grade": 5,
                "mode": "or",
            },
            "timeout": 3.5,
        }
    ]


def test_formatter_enforces_text_budget_and_keeps_provenance() -> None:
    result = _search_payload(
        {
            "chunk_id": "c1",
            "page_id": "p1",
            "source_url": "https://example.test/page/1",
            "text": "abcdefgh",
        },
        {"chunk_id": "c2", "page_id": "p2", "text": "second"},
    )

    formatted = json.loads(format_search_result_for_model(result, max_text_chars=4))

    assert formatted["returned"] == 1
    assert formatted["omitted_hits"] == 1
    assert formatted["hits"][0] == {
        "chunk_id": "c1",
        "page_id": "p1",
        "source_url": "https://example.test/page/1",
        "text": "abcd",
        "text_truncated": True,
    }


def test_empty_search_result_is_valid() -> None:
    client = TextbookSearchClient(
        session=FakeSession(FakeResponse(_search_payload())),  # type: ignore[arg-type]
    )

    result = client.search("geometri")
    formatted = json.loads(format_search_result_for_model(result))

    assert result["hits"] == []
    assert formatted["returned"] == 0
    assert formatted["hits"] == []


def test_timeout_is_reported_as_domain_error() -> None:
    client = TextbookSearchClient(
        timeout_s=0.5,
        session=FakeSession(requests.Timeout("slow")),  # type: ignore[arg-type]
    )

    with pytest.raises(TextbookSearchError, match="timed out after 0.5s"):
        client.search("geometri")


def test_http_error_is_reported_as_domain_error() -> None:
    client = TextbookSearchClient(
        session=FakeSession(  # type: ignore[arg-type]
            FakeResponse({"error": "index is unavailable"}, status_code=503)
        ),
    )

    with pytest.raises(TextbookSearchError, match="HTTP 503: index is unavailable"):
        client.search("geometri")


def test_langchain_tool_turns_retrieval_failure_into_model_visible_result() -> None:
    client = TextbookSearchClient(
        session=FakeSession(requests.ConnectionError("offline")),  # type: ignore[arg-type]
    )
    search_tool = create_search_textbooks_tool(client)

    result = json.loads(search_tool.invoke({"query": "geometri"}))

    assert result["returned"] == 0
    assert result["hits"] == []
    assert "retrieval request failed" in result["error"]


def test_langchain_tool_exposes_expected_name_and_schema() -> None:
    client = TextbookSearchClient(
        session=FakeSession(FakeResponse(_search_payload())),  # type: ignore[arg-type]
    )
    search_tool = create_search_textbooks_tool(client)

    schema = search_tool.args_schema.model_json_schema()

    assert search_tool.name == "search_textbooks"
    assert schema["properties"]["top_k"]["maximum"] == 20
    assert set(schema["properties"]["mode"]["enum"]) == {"or", "and"}
