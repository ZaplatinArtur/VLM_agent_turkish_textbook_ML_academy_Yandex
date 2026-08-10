from __future__ import annotations

from langchain_core.messages import AIMessage

from mla_baseline.config import Settings
from mla_baseline.contracts import Task
from mla_baseline.solvers.agent_rag_routed import AgentRagRouted


class FakeLlm:
    def bind_tools(self, _tools):
        return self

    def bind(self, **_kwargs):
        return self

    def invoke(self, _messages, **_kwargs):
        return AIMessage(
            content='{"solution_steps":"Kısa çözüm.","final_answer":"A"}',
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )


class FailingLlm(FakeLlm):
    def invoke(self, _messages, **_kwargs):
        raise RuntimeError("backend unavailable")


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="vllm",
        rag_no_retrieval_subjects="Math,English",
        **overrides,
    )


def _task(subject: str) -> Task:
    return Task(
        task_id=f"task-{subject}",
        subject=subject,
        grade=None,
        question="Soru",
        reference_answer="A",
        answer_type="choice",
    )


def test_routed_rag_skips_retrieval_for_blocked_subject() -> None:
    result = AgentRagRouted(_settings(), llm=FakeLlm()).solve(_task("Math"))

    assert result.condition == "agent_rag_routed"
    assert result.final_answer == "A"
    assert result.tool_calls == []
    assert result.exit_reason == "router_no_retrieval"
    assert result.retrieval_relevance == "not_attempted"
    assert result.retrieval_conflict is False
    assert result.answer_source == "text_only_no_retrieval"
    assert result.generation["experiment_id"] == "e4_routed_image_first_rag_v1"
    assert result.generation["retrieval_route"] == "skip"
    assert result.generation["retrieval_route_reason"] == "subject_blocklist"


def test_routed_rag_allows_checked_retrieval_for_other_subjects() -> None:
    result = AgentRagRouted(_settings(), llm=FakeLlm()).solve(_task("Geography"))

    assert result.condition == "agent_rag_routed"
    assert result.final_answer == "A"
    assert result.exit_reason == "answered_without_retrieval"
    assert result.generation["experiment_id"] == "e4_routed_image_first_rag_v1"
    assert result.generation["retrieval_route"] == "allow"
    assert result.generation["retrieval_route_reason"] == "subject_allowed"


def test_routed_rag_preserves_failure_as_terminal_exit_reason() -> None:
    result = AgentRagRouted(_settings(), llm=FailingLlm()).solve(_task("Math"))

    assert result.error == "RuntimeError: backend unavailable"
    assert result.exit_reason == result.error
    assert result.retrieval_relevance == "not_attempted"
    assert result.generation["retrieval_route"] == "skip"
