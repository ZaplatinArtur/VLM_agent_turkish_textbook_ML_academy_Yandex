import math

import pytest

from retrieve.rankers.cross_encoder import CrossEncoderRanker
from schemas.retrieve import RetrievedChunk


def chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=0.0,
        metadata={},
    )


class StubCrossEncoder:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen: list[list[str]] = []

    def predict(self, pairs, batch_size=None):
        self.seen = list(pairs)
        return [self.scores[text] for _query, text in pairs]


class SigmoidActivation:
    pass


class ProbabilityCrossEncoder(StubCrossEncoder):
    default_activation_function = SigmoidActivation()


def test_reorders_candidates_and_accepts_current_grade_api():
    model = StubCrossEncoder({"a": -2.0, "b": 3.0, "c": 0.0})
    results = CrossEncoderRanker(cross_encoder=model).rank(
        "query",
        [chunk("A", "a"), chunk("B", "b"), chunk("C", "c")],
        subject="math",
        grade=9,
    )

    assert [item.chunk_id for item in results] == ["B", "C", "A"]
    assert model.seen == [["query", "a"], ["query", "b"], ["query", "c"]]


def test_only_bounded_head_is_rescored_and_tail_is_untouched():
    model = StubCrossEncoder({"a": -5.0, "b": 5.0})
    results = CrossEncoderRanker(cross_encoder=model, top_n=2).rank(
        "q",
        [chunk("A", "a"), chunk("B", "b"), chunk("C", "c")],
    )

    assert [item.chunk_id for item in results] == ["B", "A", "C"]
    assert len(model.seen) == 2


def test_probability_outputs_are_not_squashed_twice_and_ties_are_stable():
    model = ProbabilityCrossEncoder({"a": 0.2, "b": 0.9, "c": 0.9})
    results = CrossEncoderRanker(cross_encoder=model).rank(
        "q",
        [chunk("A", "a"), chunk("B", "b"), chunk("C", "c")],
    )

    assert [item.chunk_id for item in results] == ["B", "C", "A"]
    assert [item.score for item in results] == [0.9, 0.9, 0.2]


def test_activation_can_be_forced_and_malformed_scores_fail_closed():
    forced = CrossEncoderRanker(
        cross_encoder=ProbabilityCrossEncoder({"a": 0.0}),
        activation="sigmoid",
    )
    assert forced.rank("q", [chunk("A", "a")])[0].score == 0.5

    malformed = CrossEncoderRanker(
        cross_encoder=StubCrossEncoder({"a": math.nan}),
    )
    with pytest.raises(ValueError, match="malformed"):
        malformed.rank("q", [chunk("A", "a")])


def test_empty_candidates_do_not_load_the_model():
    ranker = CrossEncoderRanker(cross_encoder=StubCrossEncoder({}))
    assert ranker.rank("q", []) == []
    assert ranker.rank("q", None) == []
    assert ranker.score_pairs([]) == []


def test_score_pairs_batches_multiple_queries_and_validates_contract():
    model = ProbabilityCrossEncoder({"a": 0.8, "b": 0.3})
    ranker = CrossEncoderRanker(cross_encoder=model)

    assert ranker.score_pairs([["q1", "a"], ["q2", "b"]]) == [0.8, 0.3]
    assert model.seen == [["q1", "a"], ["q2", "b"]]
    with pytest.raises(ValueError, match="two strings"):
        ranker.score_pairs([["q-only"]])


def test_unattested_adapter_is_rejected_explicitly(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(RuntimeError, match="unavailable and unattested"):
        CrossEncoderRanker(adapter_path=adapter)

    monkeypatch.setenv("RETRIEVE_RERANKER_ADAPTER", str(adapter))
    with pytest.raises(RuntimeError, match="unavailable and unattested"):
        CrossEncoderRanker()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_n": 0},
        {"batch_size": 0},
        {"activation": "maybe"},
        {"revision": ""},
        {"model_name": ""},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CrossEncoderRanker(**kwargs)
