import pytest

pytest.importorskip("numpy")
pytest.importorskip("sentence_transformers")

from collections.abc import Callable
from retrieve.pipeline import RetrievalPipeline
from schemas.retrieve import RetrievedChunk


def make_chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=chunk_id, score=0.0, metadata={})

class StubRanker:
    def __init__(
        self,
        transform: Callable[[list[RetrievedChunk]], list[RetrievedChunk]],
    ) -> None:
        self.transform = transform
        self.received: list[RetrievedChunk] | None = None
        self.received_subject: str | None = None

    def rank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        subject: str | None = None,
    ) -> list[RetrievedChunk]:
        self.received = list(chunks)
        self.received_subject = subject
        return self.transform(list(chunks))


def test_first_ranker_seeded_with_empty_candidates():
    seed = [make_chunk(c) for c in ("a", "b", "c")]
    retriever = StubRanker(lambda chunks: seed)
    pipeline = RetrievalPipeline(rankers=[retriever])

    pipeline.run("q", k=10)
    assert retriever.received == []


def test_final_k_applied_once_at_the_end():
    seed = [make_chunk(c) for c in ("a", "b", "c", "d")]
    retriever = StubRanker(lambda chunks: seed)
    pipeline = RetrievalPipeline(rankers=[retriever])

    results = pipeline.run("q", k=2)
    assert [c.chunk_id for c in results] == ["a", "b"]


def test_reranker_receives_full_untruncated_output_of_previous_stage():
    seed = [make_chunk(c) for c in ("a", "b", "c", "d")]
    retriever = StubRanker(lambda chunks: seed)
    reranker = StubRanker(lambda chunks: list(reversed(chunks)))
    pipeline = RetrievalPipeline(rankers=[retriever, reranker])

    results = pipeline.run("q", k=2)
    assert [c.chunk_id for c in reranker.received] == ["a", "b", "c", "d"]
    assert [c.chunk_id for c in results] == ["d", "c"]


def test_subject_is_threaded_through_every_stage():
    retriever = StubRanker(lambda chunks: [make_chunk("a")])
    reranker = StubRanker(lambda chunks: chunks)
    pipeline = RetrievalPipeline(rankers=[retriever, reranker])

    pipeline.run("q", k=5, subject="physics")
    assert retriever.received_subject == "physics"
    assert reranker.received_subject == "physics"
