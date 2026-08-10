import json

from langchain_core.messages import AIMessage, ToolMessage

from mla_baseline.config import Settings
from mla_baseline.context_order_experiment import (
    FrozenContextOrderSolver,
    freeze_retrieval_contexts,
    order_hits,
    payload_for_order,
)
from mla_baseline.contracts import Task


class FakeSearchClient:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return {
            "query": query,
            "context_order": "score",
            "retrieved": len(self.hits),
            "returned": len(self.hits),
            "relevance": {"label": "confident", "is_useful": True},
            "hits": self.hits,
        }


class FakeLlm:
    def __init__(self):
        self.invocations = []

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        self.invocations.append(list(messages))
        return AIMessage(
            content='{"solution_steps":"used frozen context","final_answer":"A"}',
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )


def _task() -> Task:
    return Task(
        task_id="task-1",
        subject="math",
        grade=5,
        question="question in image",
        reference_answer="A",
        answer_type="choice",
    )


def _seed(chunk_ids):
    return {
        "task_id": "task-1",
        "image_evidence": ["6 cm", "4 cm"],
        "retrieval_conflict": False,
        "tool_calls": [
            {
                "tool": "search_textbooks",
                "args": {"query": "rectangle area"},
                "returned_chunk_ids": chunk_ids,
                "relevance": {"label": "confident", "is_useful": True},
                "error": None,
            }
        ],
    }


def _hits():
    return [
        {"chunk_id": f"c{index}", "rank": index, "text": f"text {index}"}
        for index in range(1, 6)
    ]


def test_order_hits_places_two_strongest_chunks_at_edges():
    ordered = order_hits(_hits(), "edge")

    assert [hit["chunk_id"] for hit in ordered] == ["c1", "c3", "c5", "c4", "c2"]
    assert [hit["context_position"] for hit in ordered] == [1, 2, 3, 4, 5]
    assert [hit["rank"] for hit in ordered] == [1, 3, 5, 4, 2]


def test_freeze_rehydrates_exact_visible_chunk_set_and_marks_affected_task():
    client = FakeSearchClient(_hits())

    records, affected = freeze_retrieval_contexts(
        tasks=[_task()],
        seed_rows=[_seed(["c1", "c2", "c4"])],
        search_client=client,
        top_k=5,
    )

    assert len(records) == 1
    assert [hit["chunk_id"] for hit in records[0]["payload"]["hits"]] == [
        "c1",
        "c2",
        "c4",
    ]
    assert [task.task_id for task in affected] == ["task-1"]
    assert client.calls == [
        (
            "rectangle area",
            {"top_k": 5, "subject": None, "grade": None, "mode": "or"},
        )
    ]


def test_payload_orders_same_frozen_hit_set_without_mutating_record():
    client = FakeSearchClient(_hits())
    records, _ = freeze_retrieval_contexts(
        tasks=[_task()],
        seed_rows=[_seed(["c1", "c2", "c3", "c4", "c5"])],
        search_client=client,
        top_k=5,
    )

    score = payload_for_order(records[0], "score")
    edge = payload_for_order(records[0], "edge")

    assert [hit["chunk_id"] for hit in score["hits"]] == [
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
    ]
    assert [hit["chunk_id"] for hit in edge["hits"]] == [
        "c1",
        "c3",
        "c5",
        "c4",
        "c2",
    ]
    assert {hit["chunk_id"] for hit in score["hits"]} == {
        hit["chunk_id"] for hit in edge["hits"]
    }
    assert records[0]["payload"]["context_order"] == "score"


def test_freeze_fails_if_seed_chunk_cannot_be_rehydrated():
    client = FakeSearchClient(_hits()[:2])

    try:
        freeze_retrieval_contexts(
            tasks=[_task()],
            seed_rows=[_seed(["c1", "missing"])],
            search_client=client,
            top_k=5,
        )
    except ValueError as exc:
        assert "could not rehydrate frozen chunks" in str(exc)
    else:
        raise AssertionError("missing frozen chunk must fail preparation")


def test_frozen_solver_changes_only_tool_payload_order():
    client = FakeSearchClient(_hits())
    records, _ = freeze_retrieval_contexts(
        tasks=[_task()],
        seed_rows=[_seed(["c1", "c2", "c3", "c4", "c5"])],
        search_client=client,
        top_k=5,
    )
    settings = Settings(
        _env_file=None,
        structured_mode="none",
        prompt_version="v1",
    )
    score_llm = FakeLlm()
    edge_llm = FakeLlm()
    score = FrozenContextOrderSolver(
        settings,
        records=records,
        order="score",
        llm=score_llm,
    ).solve(_task())
    edge = FrozenContextOrderSolver(
        settings,
        records=records,
        order="edge",
        llm=edge_llm,
    ).solve(_task())

    assert score.error is None
    assert edge.error is None
    assert score.tool_calls[0].returned_chunk_ids == ["c1", "c2", "c3", "c4", "c5"]
    assert edge.tool_calls[0].returned_chunk_ids == ["c1", "c3", "c5", "c4", "c2"]
    score_tool = next(
        message
        for message in score_llm.invocations[0]
        if isinstance(message, ToolMessage)
    )
    edge_tool = next(
        message
        for message in edge_llm.invocations[0]
        if isinstance(message, ToolMessage)
    )
    score_payload = json.loads(score_tool.content)
    edge_payload = json.loads(edge_tool.content)
    assert {hit["chunk_id"] for hit in score_payload["hits"]} == {
        hit["chunk_id"] for hit in edge_payload["hits"]
    }
