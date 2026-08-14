from __future__ import annotations

from unittest.mock import patch

import pytest

from mla_baseline.config import Settings
from mla_baseline.solvers.b0_no_tools import B0NoTools


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="openrouter",
        openrouter_api_key="test-key",
        **overrides,
    )


def test_openrouter_resolves_endpoint_model_and_secret() -> None:
    settings = _settings()

    assert settings.llm_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_model_name == "qwen/qwen3.5-9b"
    assert settings.llm_api_key == "test-key"
    assert "test-key" not in repr(settings)


def test_openrouter_is_the_default_llm_provider() -> None:
    settings = Settings(_env_file=None, openrouter_api_key="test-key")

    assert settings.llm_provider == "openrouter"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_model_name == "qwen/qwen3.5-9b"


def test_openrouter_key_is_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.llm_api_key == "test-key"


def test_openrouter_requires_api_key() -> None:
    settings = Settings(_env_file=None, llm_provider="openrouter")

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        _ = settings.llm_api_key


def test_openrouter_uses_reasoning_parameter() -> None:
    solver = B0NoTools(_settings(enable_thinking=True), llm=object())

    assert solver._generation_extra_body(think=True) == {
        "max_tokens": 16384,
        "top_k": 20,
        "reasoning": {"enabled": True},
    }
    assert solver._generation_extra_body(think=False, max_tokens=512) == {
        "max_tokens": 512,
        "top_k": 20,
        "reasoning": {"effort": "none"},
    }


def test_vllm_keeps_chat_template_thinking_parameter() -> None:
    settings = Settings(_env_file=None, llm_provider="vllm", enable_thinking=False)
    solver = B0NoTools(settings, llm=object())

    assert solver._generation_extra_body() == {
        "max_tokens": 16384,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_openrouter_chat_client_uses_resolved_configuration() -> None:
    settings = _settings(enable_thinking=False)

    with patch("mla_baseline.solvers.b0_no_tools.ChatOpenAI") as chat_openai:
        B0NoTools(settings)

    kwargs = chat_openai.call_args.kwargs
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["model"] == "qwen/qwen3.5-9b"
    assert kwargs["extra_body"]["reasoning"] == {"effort": "none"}
    assert "chat_template_kwargs" not in kwargs["extra_body"]
