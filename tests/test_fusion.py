import pytest

from retrieve.rankers.fusion import (
    DEFAULT_RRF_K,
    PrimaryCandidateUnion,
    ReciprocalRankFusion,
)
from retrieve.rankers.rerank import KnowledgeReranker
from schemas.retrieve import RetrievedChunk


def make_chunk(chunk_id: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=chunk_id, score=score, metadata={})


class StubRanker:
    """Всегда отдаёт заранее заданный порядок chunk_id."""

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.received: list[RetrievedChunk] | None = None
        self.received_subject: str | None = None

    def rank(self, query, chunks=None, subject=None, grade=None):
        self.received = None if chunks is None else list(chunks)
        self.received_subject = subject
        return [make_chunk(chunk_id) for chunk_id in self.order]


class ScoredStubRanker:
    def __init__(self, values: list[tuple[str, float]]) -> None:
        self.values = values
        self.calls = 0

    def rank(self, query, chunks=None, subject=None, grade=None):
        self.calls += 1
        return [make_chunk(chunk_id, score) for chunk_id, score in self.values]


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


def test_primary_candidate_union_preserves_lexical_head_and_stable_semantic_ties():
    primary = ScoredStubRanker([("p2", 9.0), ("p1", 8.0), ("p3", 7.0)])
    semantic = ScoredStubRanker(
        [("p3", 99.0), ("s-b", 0.5), ("s-a", 0.5), ("s-c", 0.2)]
    )

    results = PrimaryCandidateUnion(
        primary,
        semantic,
        primary_k=2,
        semantic_k=2,
    ).rank("q")

    assert [chunk.chunk_id for chunk in results] == [
        "p2",
        "p1",
        "s-a",
        "s-b",
        "p3",
    ]
    assert len({chunk.chunk_id for chunk in results}) == len(results)


def test_primary_candidate_fallback_does_not_invoke_semantic_when_primary_is_sufficient():
    primary = ScoredStubRanker([("p1", 2.0), ("p2", 1.0)])
    semantic = ScoredStubRanker([("s1", 1.0)])
    union = PrimaryCandidateUnion(
        primary,
        semantic,
        mode="fallback",
        fallback_min_candidates=2,
    )

    assert [chunk.chunk_id for chunk in union.rank("q")] == ["p1", "p2"]
    assert semantic.calls == 0


def test_primary_candidate_union_rejects_invalid_mode_and_nonfinite_scores():
    with pytest.raises(ValueError, match="mode"):
        PrimaryCandidateUnion(
            ScoredStubRanker([]),
            ScoredStubRanker([]),
            mode="replace",
        )

    union = PrimaryCandidateUnion(
        ScoredStubRanker([]),
        ScoredStubRanker([("bad", float("nan"))]),
    )
    with pytest.raises(ValueError, match="non-finite"):
        union.rank("q")


def test_candidate_union_calibrates_raw_bm25_and_cosine_without_losing_priority():
    union = PrimaryCandidateUnion(
        ScoredStubRanker([("lexical", 100.0)]),
        ScoredStubRanker([("semantic", 0.9)]),
        primary_k=1,
        semantic_k=1,
    )

    candidates = union.rank("semantic")
    scores = {chunk.chunk_id: chunk.score for chunk in candidates}
    assert [chunk.chunk_id for chunk in candidates] == ["lexical", "semantic"]
    assert scores["lexical"] > scores["semantic"]
    assert scores["lexical"] / scores["semantic"] < 2.0
    assert [
        chunk.chunk_id
        for chunk in KnowledgeReranker().rank("semantic", candidates)
    ][0] == "semantic"

    with pytest.raises(ValueError, match="lexical-first"):
        PrimaryCandidateUnion(
            ScoredStubRanker([]),
            ScoredStubRanker([]),
            primary_weight=0.5,
            semantic_weight=1.0,
        )


def test_candidate_union_preserves_original_semantic_rank_after_overlap_dedup():
    overlapping = [(f"shared-{position}", 100.0 - position) for position in range(31)]
    union = PrimaryCandidateUnion(
        ScoredStubRanker(overlapping),
        ScoredStubRanker(overlapping + [("semantic-tail", 0.9)]),
        primary_k=31,
        semantic_k=1,
    )

    candidates = union.rank("q")

    assert candidates[-1].chunk_id == "semantic-tail"
    assert candidates[-1].score == pytest.approx(0.85 / (DEFAULT_RRF_K + 32))
