from retrieve.rankers.reranker import CrossEncoderRanker
from schemas.retrieve import RetrievedChunk


def chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, score=0.0, metadata={})


class StubCrossEncoder:
    """Возвращает заранее заданный логит по тексту чанка."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen: list[list[str]] = []

    def predict(self, pairs, batch_size=None):
        self.seen = pairs
        return [self.scores[text] for _query, text in pairs]


def test_reranker_reorders_by_cross_encoder_score():
    ce = StubCrossEncoder({"a": -2.0, "b": 3.0, "c": 0.0})
    ranker = CrossEncoderRanker(cross_encoder=ce)
    candidates = [chunk("A", "a"), chunk("B", "b"), chunk("C", "c")]
    results = ranker.rank("q", candidates)
    assert [c.chunk_id for c in results] == ["B", "C", "A"]
    # score приведён к (0,1) сигмоидой и отсортирован по убыванию.
    assert results[0].score > results[1].score > results[2].score
    assert 0.0 < results[0].score < 1.0


def test_only_top_n_are_rescored_tail_kept_after():
    ce = StubCrossEncoder({"a": -5.0, "b": 5.0})
    ranker = CrossEncoderRanker(cross_encoder=ce, top_n=2)
    tail = [chunk("C", "c"), chunk("D", "d")]
    candidates = [chunk("A", "a"), chunk("B", "b"), *tail]
    results = ranker.rank("q", candidates)
    # Переоценены только A и B (B выше), C/D остаются в исходном порядке в хвосте.
    assert [c.chunk_id for c in results] == ["B", "A", "C", "D"]
    assert len(ce.seen) == 2


def test_empty_candidates_return_empty():
    ranker = CrossEncoderRanker(cross_encoder=StubCrossEncoder({}))
    assert ranker.rank("q", []) == []
    assert ranker.rank("q", None) == []
