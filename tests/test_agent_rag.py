from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from mla_baseline.config import Settings
from mla_baseline.contracts import ImageRef, Task
from mla_baseline.solvers.agent_rag import AgentRag
from mla_baseline.tools import LocalTextbookSearchClient


class FakeLlm:
    def __init__(self, responses: list[AIMessage | Exception]) -> None:
        self.responses = list(responses)
        self.invocations: list[list[Any]] = []
        self.invoke_kwargs: list[dict[str, Any]] = []
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "FakeLlm":
        self.bound_tools = tools
        return self

    def bind(self, **kwargs: Any) -> "FakeLlm":
        return self

    def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        self.invocations.append(list(messages))
        self.invoke_kwargs.append(kwargs)
        if not self.responses:
            raise AssertionError("fake LLM has no response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeSearchClient:
    def __init__(self, relevance: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.relevance = list(relevance or [])

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"query": query, **kwargs})
        label = self.relevance.pop(0) if self.relevance else "confident"
        useful = label == "confident"
        hits = [
            {
                "chunk_id": "book-1:0042",
                "page_id": "book-1:42",
                "rank": 1,
                "text": "Dikdörtgenin alanı uzun kenar ile kısa kenarın çarpımıdır.",
                "source_url": "https://example.test/book-1/page-42",
            }
        ] if useful else []
        return {
            "query": query,
            "top_k": kwargs.get("top_k"),
            "mode": kwargs.get("mode"),
            "filters": {
                "subject": kwargs.get("subject"),
                "grade": kwargs.get("grade"),
            },
            "retrieved": 1,
            "returned": len(hits),
            "latency_ms": 2.5,
            "relevance": {
                "label": label,
                "is_useful": useful,
                "top_score": 0.91 if useful else 0.2,
                "reason": f"test {label}",
            },
            "hits": hits,
        }


def _settings(**overrides: Any) -> Settings:
    values = {
        "model_name": "fake-qwen",
        "structured_mode": "none",
        **overrides,
    }
    return Settings(_env_file=None, **values)


def _task() -> Task:
    return Task(
        task_id="math-001",
        subject="math",
        grade=7,
        question="6 cm ve 4 cm kenarlı dikdörtgenin alanı nedir?",
        reference_answer="SECRET_REFERENCE_24",
        answer_type="numeric",
        reference_solution="SECRET_REFERENCE_SOLUTION",
    )


def _tool_call(query: str, *, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_textbooks",
                "args": {
                    "query": query,
                    "top_k": 5,
                    "subject": "math",
                    "grade": 7,
                    "mode": "or",
                },
                "id": call_id,
                "type": "tool_call",
            }
        ],
        usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    )


def _final_answer() -> AIMessage:
    return AIMessage(
        content=(
            '{"solution_steps":"Alan = 6 × 4 = 24 cm²",'
            '"final_answer":"24 cm²"}'
        ),
        usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
    )


def test_agent_uses_local_retrieval_by_default() -> None:
    solver = AgentRag(_settings(), llm=FakeLlm([_final_answer()]))

    assert isinstance(solver.search_client, LocalTextbookSearchClient)


def test_agent_forwards_mmr_experiment_settings_to_local_retrieval() -> None:
    solver = AgentRag(
        _settings(
            retrieval_fetch_k=20,
            retrieval_mmr_enabled=True,
            retrieval_mmr_lambda=0.3,
            retrieval_context_order="edge",
        ),
        llm=FakeLlm([_final_answer()]),
    )

    assert isinstance(solver.search_client, LocalTextbookSearchClient)
    assert solver.search_client.retrieval_fetch_k == 20
    assert solver.search_client.mmr_lambda == 0.3
    assert solver.search_client.context_order == "edge"


def test_agent_rejects_candidate_pool_smaller_than_top_k() -> None:
    with pytest.raises(ValueError, match="at least retrieval_top_k"):
        AgentRag(
            _settings(retrieval_top_k=5, retrieval_fetch_k=4),
            llm=FakeLlm([_final_answer()]),
        )


def test_retrieval_call_budget_cannot_exceed_two() -> None:
    assert _settings().retrieval_max_calls == 2
    with pytest.raises(ValidationError):
        _settings(retrieval_max_calls=3)


def test_agent_executes_tool_and_returns_traceable_final_answer() -> None:
    llm = FakeLlm([_tool_call("dikdörtgen alan formülü"), _final_answer()])
    search_client = FakeSearchClient()
    solver = AgentRag(
        _settings(),
        llm=llm,
        search_client=search_client,  # type: ignore[arg-type]
    )

    result = solver.solve(_task())

    assert result.error is None
    assert result.final_answer == "24 cm²"
    assert result.solution_steps == "Alan = 6 × 4 = 24 cm²"
    assert result.usage.input_tokens == 30
    assert result.usage.output_tokens == 11
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].returned_chunk_ids == ["book-1:0042"]
    assert result.tool_calls[0].latency_ms == 2.5
    assert result.tool_calls[0].relevance["label"] == "confident"
    assert result.exit_reason == "answered_with_retrieval"
    assert search_client.calls[0]["query"] == "dikdörtgen alan formülü"
    assert any(
        isinstance(message, ToolMessage)
        for message in llm.invocations[1]
    )


def test_agent_rejects_duplicate_tool_call_without_second_http_request() -> None:
    llm = FakeLlm(
        [
            _tool_call("dikdörtgen alan", call_id="call-1"),
            _tool_call("dikdörtgen alan", call_id="call-2"),
            _final_answer(),
        ]
    )
    search_client = FakeSearchClient()
    solver = AgentRag(
        _settings(),
        llm=llm,
        search_client=search_client,  # type: ignore[arg-type]
    )

    result = solver.solve(_task())

    assert result.error is None
    assert len(search_client.calls) == 1
    assert len(result.tool_calls) == 2
    assert result.tool_calls[1].error is not None
    assert "duplicate tool call" in result.tool_calls[1].error
    assert result.exit_reason == "tool_call_rejected"


def test_agent_allows_one_rewrite_only_after_weak_retrieval() -> None:
    llm = FakeLlm(
        [
            _tool_call("dikdörtgen alan", call_id="call-1"),
            _tool_call("dikdörtgen alan formülü", call_id="call-2"),
            _final_answer(),
        ]
    )


def _image_task() -> Task:
    return _task().model_copy(
        update={
            "question": "Soru görselde.",
            "question_images": [
                ImageRef(
                    image_id="rectangle",
                    format="base64",
                    data="AA==",
                    mime_type="image/png",
                )
            ],
        }
    )


def _image_evidence() -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "image_evidence": ["uzun kenar 6 cm", "kısa kenar 4 cm", "alanı bul"],
                "question": "6 cm ve 4 cm kenarlı dikdörtgenin alanı nedir?",
                "topic": "dikdörtgen",
                "unknown_concepts": ["alan formülü"],
            },
            ensure_ascii=False,
        )
    )


def _conflict_check(*chunk_ids: str) -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "conflicting_chunk_ids": list(chunk_ids),
                "reason": "test conflict check",
            }
        )
    )
    search_client = FakeSearchClient(relevance=["weak", "confident"])
    solver = AgentRag(_settings(), llm=llm, search_client=search_client)

    result = solver.solve(_task())

    assert result.error is None
    assert [call["query"] for call in search_client.calls] == [
        "dikdörtgen alan",
        "dikdörtgen alan formülü",
    ]
    assert [call.relevance["label"] for call in result.tool_calls] == [
        "weak",
        "confident",
    ]
    assert result.exit_reason == "forced_final_after_rewrite"


def test_agent_may_search_again_after_confident_retrieval() -> None:
    """Порог отсеивает выдачу не по теме, но не страницу по теме без нужного
    содержания — второй заход разрешён, потолок держит retrieval_max_calls."""
    llm = FakeLlm(
        [
            _tool_call("dikdörtgen alan", call_id="call-1"),
            _tool_call("geometri örnekleri", call_id="call-2"),
            _final_answer(),
        ]
    )
    search_client = FakeSearchClient(relevance=["confident"])
    solver = AgentRag(_settings(), llm=llm, search_client=search_client)

    result = solver.solve(_task())

    assert result.error is None
    assert len(search_client.calls) == 2
    assert result.tool_calls[1].error is None
    assert result.exit_reason == "forced_final_after_rewrite"


def test_agent_uses_configured_retrieval_top_k() -> None:
    llm = FakeLlm([_tool_call("dikdörtgen alan"), _final_answer()])
    search_client = FakeSearchClient()
    solver = AgentRag(
        _settings(retrieval_top_k=2),
        llm=llm,
        search_client=search_client,
    )

    result = solver.solve(_task())

    assert result.error is None
    assert search_client.calls[0]["top_k"] == 2


def test_agent_enforces_tool_call_limit_and_still_accepts_final_answer() -> None:
    llm = FakeLlm(
        [
            _tool_call("dikdörtgen alan", call_id="call-1"),
            _final_answer(),
        ]
    )
    search_client = FakeSearchClient()
    solver = AgentRag(
        _settings(retrieval_max_calls=1, structured_mode="response_format"),
        llm=llm,
        search_client=search_client,  # type: ignore[arg-type]
    )

    result = solver.solve(_task())

    assert result.error is None
    assert result.final_answer == "24 cm²"
    assert len(search_client.calls) == 1
    assert len(result.tool_calls) == 1
    assert result.forced_answer is True
    assert result.exit_reason == "tool_call_limit"
    response_format = llm.invoke_kwargs[-1]["response_format"]["json_schema"]
    assert set(response_format["schema"]["properties"]) == {
        "solution_steps",
        "final_answer",
    }


def test_agent_repairs_malformed_final_response_without_tools() -> None:
    llm = FakeLlm(
        [
            AIMessage(content="The answer is probably 24."),
            _final_answer(),
        ]
    )
    solver = AgentRag(
        _settings(),
        llm=llm,
        search_client=FakeSearchClient(),  # type: ignore[arg-type]
    )

    result = solver.solve(_task())

    assert result.error is None
    assert result.final_answer == "24 cm²"
    assert result.forced_answer is True
    assert result.exit_reason == "malformed_response"


def test_agent_falls_back_to_answer_only_after_compact_length_limit() -> None:
    class LengthFinishReasonError(Exception):
        pass

    llm = FakeLlm(
        [
            _tool_call("dikdörtgen alan", call_id="call-1"),
            LengthFinishReasonError("compact response reached its limit"),
            AIMessage(content='{"final_answer":"24 cm²"}'),
        ]
    )
    solver = AgentRag(
        _settings(retrieval_max_calls=1, structured_mode="response_format"),
        llm=llm,
        search_client=FakeSearchClient(),  # type: ignore[arg-type]
    )

    result = solver.solve(_task())

    assert result.error is None
    assert result.final_answer == "24 cm²"
    assert result.forced_answer is True
    assert result.exit_reason == "tool_call_limit"
    answer_schema = llm.invoke_kwargs[-1]["response_format"]["json_schema"]["schema"]
    assert set(answer_schema["properties"]) == {"final_answer"}


def test_agent_prompt_does_not_leak_reference_fields() -> None:
    llm = FakeLlm([_final_answer()])
    solver = AgentRag(
        _settings(),
        llm=llm,
        search_client=FakeSearchClient(),  # type: ignore[arg-type]
    )

    messages = solver.build_messages(_task())
    serialized = str(messages)

    assert "search_textbooks" in str(messages[0].content)
    assert "SECRET_REFERENCE_24" not in serialized
    assert "SECRET_REFERENCE_SOLUTION" not in serialized


def test_agent_can_answer_without_using_retrieval() -> None:
    llm = FakeLlm([_final_answer()])
    search_client = FakeSearchClient()
    solver = AgentRag(
        _settings(),
        llm=llm,
        search_client=search_client,  # type: ignore[arg-type]
    )

    result = solver.solve(_task())

    assert result.error is None
    assert result.final_answer == "24 cm²"
    assert result.tool_calls == []
    assert search_client.calls == []
    assert result.exit_reason == "answered_without_retrieval"


def test_text_only_mode_ignores_image_refs_and_sends_question_text() -> None:
    task = _task().model_copy(
        update={
            "question_images": [
                {
                    "image_id": "missing",
                    "format": "file_path",
                    "data": "images/does-not-exist.png",
                    "mime_type": "image/png",
                }
            ]
        }
    )
    solver = AgentRag(
        _settings(text_only=True),
        llm=FakeLlm([_final_answer()]),
        search_client=FakeSearchClient(),  # type: ignore[arg-type]
    )

    messages = solver.build_messages(task)
    blocks = messages[-1].content

    assert not any(block.get("type") == "image_url" for block in blocks)
    assert any(task.question in block.get("text", "") for block in blocks)


def test_image_first_pipeline_uses_evidence_query_and_verifies_final_answer() -> None:
    candidate = AIMessage(
        content='{"solution_steps":"6 × 4 = 25","final_answer":"25 cm²"}'
    )
    verified = AIMessage(
        content='{"solution_steps":"6 × 4 = 24","final_answer":"24 cm²"}'
    )
    llm = FakeLlm(
        [
            _image_evidence(),
            _tool_call("6 cm 4 cm sorunun tamamı"),
            _conflict_check(),
            candidate,
            verified,
        ]
    )
    search_client = FakeSearchClient()
    solver = AgentRag(_settings(), llm=llm, search_client=search_client)

    result = solver.solve(_image_task())

    assert result.error is None
    assert result.final_answer == "24 cm²"
    assert search_client.calls[0]["query"] == "dikdörtgen alan formülü"
    assert result.image_evidence == [
        "uzun kenar 6 cm",
        "kısa kenar 4 cm",
        "alanı bul",
    ]
    assert result.image_evidence_structured == {
        "image_evidence": [
            "uzun kenar 6 cm",
            "kısa kenar 4 cm",
            "alanı bul",
        ],
        "question": "6 cm ve 4 cm kenarlı dikdörtgenin alanı nedir?",
        "topic": "dikdörtgen",
        "unknown_concepts": ["alan formülü"],
    }
    assert result.retrieval_relevance == "confident"
    assert result.retrieval_conflict is False
    assert result.answer_source == "image_with_retrieval_support"
    assert any(
        isinstance(block, dict) and block.get("type") == "image_url"
        for block in llm.invocations[-1][1].content
    )


def test_image_first_pipeline_removes_conflicting_chunks_before_answer() -> None:
    llm = FakeLlm(
        [
            _image_evidence(),
            _tool_call("ignored original query"),
            _conflict_check("book-1:0042"),
            _final_answer(),
            _final_answer(),
        ]
    )
    solver = AgentRag(_settings(), llm=llm, search_client=FakeSearchClient())

    result = solver.solve(_image_task())

    tool_message = next(
        message
        for message in llm.invocations[3]
        if isinstance(message, ToolMessage)
    )
    payload = json.loads(tool_message.content)
    assert payload["hits"] == []
    assert payload["relevance"]["label"] == "conflict"
    assert result.tool_calls[0].returned_chunk_ids == []
    assert result.retrieval_relevance == "conflict"
    assert result.retrieval_conflict is True
    assert result.answer_source == "image_after_retrieval_rejected"
