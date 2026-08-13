from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from mla_baseline.config import Settings
from mla_baseline.contracts import Task
from mla_baseline.solvers import SOLVERS
from mla_baseline.solvers.agent_rag_sourced import AgentRagSourced
from source_router import Route

ROUTE = Route(
    family="meb7",
    record_id="yandex_7_matematik_meb_dee64189589b:p56:q3",
    answer="a: 3/10 < 3/7 < 3/5",
    answer_format="ordered_list",
    source_page=56,
    score=0.4283,
    margin=0.0,
    closure="global_idf_top1_plus_marker_plus_official_operand_anchors",
)


class FakeLlm:
    def __init__(self) -> None:
        self.invocations: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "FakeLlm":
        return self

    def invoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        self.invocations.append(messages)
        return AIMessage(content='{"final_answer": "24", "solution_steps": "s"}')


class FakeSearchClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"query": query, **kwargs})
        return {
            "query": query,
            "returned": 0,
            "retrieved": 0,
            "relevance": {"label": "confident", "is_useful": True, "reason": ""},
            "hits": [],
        }


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, model_name="fake-qwen",
                    structured_mode="none", **overrides)


def _task() -> Task:
    return Task(
        task_id="math-001",
        subject="math",
        grade=7,
        question="3. Rasyonel sayıları küçükten büyüğe sıralayınız.",
        reference_answer="SECRET_REFERENCE",
        answer_type="short_text",
    )


def _solver(route_result: Route | None, llm: FakeLlm, search: FakeSearchClient):
    return AgentRagSourced(
        _settings(),
        llm=llm,
        search_client=search,
        router=lambda _text: route_result,
    )


def test_registered_as_its_own_condition():
    assert SOLVERS["agent_rag_sourced"] is AgentRagSourced
    assert SOLVERS["agent_rag"] is not AgentRagSourced


def test_official_source_answers_without_calling_the_model():
    llm, search = FakeLlm(), FakeSearchClient()
    result = _solver(ROUTE, llm, search).solve(_task())

    assert result.final_answer == ROUTE.answer
    assert result.answer_source == "official_source"
    assert result.exit_reason == "answered_from_official_source"
    assert llm.invocations == []
    assert search.calls == []


def test_official_source_records_its_grounds():
    result = _solver(ROUTE, FakeLlm(), FakeSearchClient()).solve(_task())

    assert result.generation["route_decision"] == "official_source"
    assert result.generation["route_record_id"] == ROUTE.record_id
    assert result.generation["route_family"] == "meb7"
    assert result.generation["route_closure"] == ROUTE.closure


def test_abstain_falls_through_to_the_agent():
    llm, search = FakeLlm(), FakeSearchClient()
    result = _solver(None, llm, search).solve(_task())

    assert result.generation["route_decision"] == "abstain"
    assert llm.invocations, "агент должен был отработать"
    assert result.answer_source != "official_source"


def test_router_sees_only_the_question():
    seen: list[str] = []
    solver = AgentRagSourced(
        _settings(),
        llm=FakeLlm(),
        search_client=FakeSearchClient(),
        router=lambda text: seen.append(text) or None,
    )
    task = _task()
    solver.solve(task)

    assert seen == [task.question]
    assert task.reference_answer not in seen
