from src.retrieve.confidence import (
    DEFAULT_MIN_SCORE,
    Relevance,
    assess_relevance,
)
from src.schemas.retrieve import RetrievedChunk


def chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=chunk_id, score=score, metadata={})


def test_empty_results_are_flagged_empty():
    verdict = assess_relevance([])
    assert verdict.relevance is Relevance.EMPTY
    assert not verdict.is_useful
    assert verdict.top_score is None


def test_low_top_score_is_weak():
    verdict = assess_relevance([chunk("a", DEFAULT_MIN_SCORE - 0.1)])
    assert verdict.relevance is Relevance.WEAK
    assert not verdict.is_useful


def test_high_top_score_is_confident():
    verdict = assess_relevance([chunk("a", 0.72), chunk("b", 0.4)])
    assert verdict.relevance is Relevance.CONFIDENT
    assert verdict.is_useful
    assert verdict.top_score == 0.72


def test_margin_check_flags_flat_distribution_when_enabled():
    results = [chunk("a", 0.70), chunk("b", 0.69)]
    assert assess_relevance(results, min_margin=0.0).is_useful  # off by default
    weak = assess_relevance(results, min_margin=0.05)
    assert weak.relevance is Relevance.WEAK


def test_margin_check_passes_on_clear_winner():
    results = [chunk("a", 0.70), chunk("b", 0.40)]
    assert assess_relevance(results, min_margin=0.05).is_useful
