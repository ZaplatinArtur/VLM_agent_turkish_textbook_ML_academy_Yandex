from concurrent.futures import ThreadPoolExecutor
import threading

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


class SigmoidActivation:
    pass


class ProbabilityCrossEncoder(StubCrossEncoder):
    """Как sentence_transformers.CrossEncoder: сигмоида уже внутри predict."""

    default_activation_function = SigmoidActivation()


def test_scores_from_model_with_sigmoid_are_not_squashed_again():
    # Модель отдаёт готовые вероятности — ранкер обязан оставить их как есть,
    # иначе шкала схлопывается в [0.5, 0.73] и порог по score теряет смысл.
    ce = ProbabilityCrossEncoder({"a": 0.0001, "b": 0.99})
    results = CrossEncoderRanker(cross_encoder=ce).rank("q", [chunk("A", "a"), chunk("B", "b")])
    assert [c.chunk_id for c in results] == ["B", "A"]
    assert results[0].score == 0.99
    assert results[1].score == 0.0001


def test_activation_can_be_forced():
    ce = ProbabilityCrossEncoder({"a": 0.0, "b": 2.0})
    forced = CrossEncoderRanker(cross_encoder=ce, activation="sigmoid")
    assert forced.rank("q", [chunk("A", "a")])[0].score == 0.5
    raw = CrossEncoderRanker(cross_encoder=StubCrossEncoder({"a": 2.0}), activation="none")
    assert raw.rank("q", [chunk("A", "a")])[0].score == 2.0


def test_empty_candidates_return_empty():
    ranker = CrossEncoderRanker(cross_encoder=StubCrossEncoder({}))
    assert ranker.rank("q", []) == []
    assert ranker.rank("q", None) == []


class BlockingCrossEncoder:
    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0
        self.call_count = 0
        self.first_entered = threading.Event()
        self.second_entered = threading.Event()
        self.release_first = threading.Event()

    def predict(self, pairs, batch_size=None):
        with self._state_lock:
            self.call_count += 1
            call_number = self.call_count
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        if call_number == 1:
            self.first_entered.set()
            if not self.release_first.wait(timeout=2):
                raise TimeoutError("test did not release the first predict call")
        else:
            self.second_entered.set()
        with self._state_lock:
            self.active_calls -= 1
        return [0.0 for _pair in pairs]


def test_predict_calls_are_serialized_across_concurrent_requests():
    model = BlockingCrossEncoder()
    ranker = CrossEncoderRanker(cross_encoder=model)
    second_started = threading.Event()

    def second_call():
        second_started.set()
        return ranker.score_pairs([["q2", "b"]])

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(ranker.score_pairs, [["q1", "a"]])
        assert model.first_entered.wait(timeout=1)
        second = executor.submit(second_call)
        assert second_started.wait(timeout=1)
        assert not model.second_entered.wait(timeout=0.1)
        model.release_first.set()
        assert first.result(timeout=1) == [0.5]
        assert second.result(timeout=1) == [0.5]

    assert model.second_entered.is_set()
    assert model.max_active_calls == 1
