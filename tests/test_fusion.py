import pytest

from retrieve.rankers.fusion import DEFAULT_RRF_K, ReciprocalRankFusion
from schemas.retrieve import RetrievedChunk


def make_chunk(chunk_id: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=chunk_id, score=score, metadata={})


class StubRanker:
    """Всегда отдаёт заранее заданный порядок chunk_id."""

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.received: list[RetrievedChunk] | None = None
        self.received_subject: str | None = None

    def rank(self, query, chunks=None, subject=None):
        self.received = None if chunks is None else list(chunks)
        self.received_subject = subject
        return [make_chunk(chunk_id) for chunk_id in self.order]


def rrf(*positions: int) -> float:
    return sum(1.0 / (DEFAULT_RRF_K + position) for position in positions)


def test_agreement_between_rankers_wins():
    # "b" второй у обоих ранкеров, "a" и "c" — первые, но каждый лишь у одного:
    # 2/62 > 1/61, поэтому согласие источников важнее одиночного первого места.
    fusion = ReciprocalRankFusion([StubRanker(["a", "b"]), StubRanker(["c", "b"])])
    results = fusion.rank("q")
    assert [c.chunk_id for c in results] == ["b", "a", "c"]
    assert results[0].score == pytest.approx(rrf(2, 2))


def test_score_is_sum_of_reciprocal_ranks():
    fusion = ReciprocalRankFusion([StubRanker(["a", "b"]), StubRanker(["b"])])
    scores = {c.chunk_id: c.score for c in fusion.rank("q")}
    assert scores["a"] == pytest.approx(rrf(1))
    assert scores["b"] == pytest.approx(rrf(2, 1))


def test_chunk_missing_from_one_ranker_still_included():
    fusion = ReciprocalRankFusion([StubRanker(["a"]), StubRanker(["b"])])
    assert {c.chunk_id for c in fusion.rank("q")} == {"a", "b"}


def test_weights_shift_the_balance():
    rankers = [StubRanker(["a", "b"]), StubRanker(["b", "a"])]
    assert [c.chunk_id for c in ReciprocalRankFusion(rankers).rank("q")] == ["a", "b"]
    weighted = ReciprocalRankFusion(rankers, weights=[1.0, 5.0])
    assert [c.chunk_id for c in weighted.rank("q")] == ["b", "a"]


def test_ties_are_resolved_deterministically_by_first_appearance():
    fusion = ReciprocalRankFusion([StubRanker(["a", "b"]), StubRanker(["a", "b"])])
    assert [c.chunk_id for c in fusion.rank("q")] == ["a", "b"]


def test_query_chunks_and_subject_are_forwarded_to_every_ranker():
    rankers = [StubRanker(["a"]), StubRanker(["a"])]
    candidates = [make_chunk("a"), make_chunk("b")]
    ReciprocalRankFusion(rankers).rank("q", candidates, subject="physics")
    for ranker in rankers:
        assert [c.chunk_id for c in ranker.received] == ["a", "b"]
        assert ranker.received_subject == "physics"


def test_empty_sources_produce_empty_result():
    assert ReciprocalRankFusion([StubRanker([])]).rank("q") == []


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        ReciprocalRankFusion([])
    with pytest.raises(ValueError):
        ReciprocalRankFusion([StubRanker(["a"])], rrf_k=0)
    with pytest.raises(ValueError):
        ReciprocalRankFusion([StubRanker(["a"])], weights=[1.0, 2.0])
