from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from mla_baseline.config import Settings
from mla_baseline.contracts import Task
from mla_baseline.solvers.agent_rag import AgentRag
from mla_baseline.tools import LocalTextbookSearchClient


class FakeLlm:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.invocations: list[list[Any]] = []
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "FakeLlm":
        self.bound_tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(list(messages))
        if not self.responses:
            raise AssertionError("fake LLM has no response left")
        return self.responses.pop(0)


class FakeSearchClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"query": query, **kwargs})
        return {
            "query": query,
            "mode": kwargs.get("mode"),
            "filters": {
                "subject": kwargs.get("subject"),
                "grade": kwargs.get("grade"),
            },
            "returned": 1,
            "latency_ms": 2.5,
            "hits": [
                {
                    "chunk_id": "book-1:0042",
                    "page_id": "book-1:42",
                    "rank": 1,
                    "text": "Dikdörtgenin alanı uzun kenar ile kısa kenarın çarpımıdır.",
                    "source_url": "https://example.test/book-1/page-42",
                }
            ],
        }


def _settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        model_name="fake-qwen",
        structured_mode="none",
        **overrides,
    )


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


def test_agent_enforces_tool_call_limit_and_still_accepts_final_answer() -> None:
    llm = FakeLlm(
        [
            _tool_call("dikdörtgen alan", call_id="call-1"),
            _tool_call("geometri alan örnekleri", call_id="call-2"),
            _final_answer(),
        ]
    )
    search_client = FakeSearchClient()
    solver = AgentRag(
        _settings(retrieval_max_calls=1),
        llm=llm,
        search_client=search_client,  # type: ignore[arg-type]
    )

    result = solver.solve(_task())

    assert result.error is None
    assert result.final_answer == "24 cm²"
    assert len(search_client.calls) == 1
    assert "tool call limit reached" in (result.tool_calls[1].error or "")


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
