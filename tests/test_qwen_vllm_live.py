"""Opt-in smoke tests for a live Qwen3.5 vLLM server.

Run on the GPU server with ``MLA_RUN_LIVE_QWEN_TESTS=1``. These tests stay
skipped during normal unit-test runs and never download or start the model.
"""

from __future__ import annotations

import os

import pytest
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from mla_baseline.config import Settings


RUN_LIVE_TESTS = os.getenv("MLA_RUN_LIVE_QWEN_TESTS") == "1"

pytestmark = [
    pytest.mark.live_qwen,
    pytest.mark.skipif(
        not RUN_LIVE_TESTS,
        reason="set MLA_RUN_LIVE_QWEN_TESTS=1 to test a running Qwen server",
    ),
]


@pytest.fixture
def live_settings() -> Settings:
    return Settings(_env_file=None)


def test_vllm_exposes_configured_qwen_model(live_settings: Settings) -> None:
    response = requests.get(
        f"{live_settings.vllm_base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {live_settings.vllm_api_key}"},
        timeout=15,
    )
    response.raise_for_status()

    payload = response.json()
    model_ids = {
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict)
    }
    assert live_settings.model_name in model_ids


def test_qwen_returns_langchain_tool_call(live_settings: Settings) -> None:
    @tool
    def search_textbooks(query: str, top_k: int = 5) -> str:
        """Search Turkish textbooks for evidence relevant to a homework task."""

        return f"unused smoke-test result for {query!r}, top_k={top_k}"

    llm = ChatOpenAI(
        base_url=live_settings.vllm_base_url,
        api_key=live_settings.vllm_api_key,
        model=live_settings.model_name,
        max_tokens=256,
        temperature=0.0,
        timeout=live_settings.request_timeout_s,
        max_retries=0,
    ).bind_tools([search_textbooks])

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are testing tool calling. Call search_textbooks exactly "
                    "once and do not answer the homework problem yourself."
                )
            ),
            HumanMessage(
                content="Find the Turkish textbook formula for rectangle area."
            ),
        ]
    )

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call["name"] == "search_textbooks"
    assert isinstance(call["args"].get("query"), str)
    assert call["args"]["query"].strip()
