from retrieve.confidence import (
    CROSS_ENCODER_MIN_SCORE,
    DEFAULT_MIN_SCORE,
    PROFILE_MIN_SCORE,
    Relevance,
    assess_relevance,
    min_score_for,
)
from retrieve.pipelines import PROFILES
from schemas.retrieve import RetrievedChunk


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


def test_profile_sets_its_own_threshold():
    # 0.70 — уверенное попадание для MiniLM и явный мусор для e5-small,
    # у которого даже чужие запросы дают ~0.84.
    results = [chunk("a", 0.70)]
    assert assess_relevance(results, profile="minilm").is_useful
    assert not assess_relevance(results, profile="e5-small").is_useful


def test_explicit_min_score_wins_over_profile():
    results = [chunk("a", 0.70)]
    assert assess_relevance(results, min_score=0.6, profile="e5-small").is_useful


def test_unknown_and_missing_profile_fall_back_to_default():
    assert min_score_for(None) == DEFAULT_MIN_SCORE
    assert min_score_for("m3") == DEFAULT_MIN_SCORE
    assert min_score_for("e5-small") == PROFILE_MIN_SCORE["e5-small"]


def test_cross_encoder_profiles_share_one_threshold():
    # Скор кросс-энкодера — вероятность релевантности пары, а не косинус:
    # шкала не зависит от того, какой ретривер подал кандидатов.
    assert (
        min_score_for("rrf_e5-small_bm25_cross-encoder")
        == min_score_for("rrf_e5-base_m3_bm25_cross-encoder")
        == CROSS_ENCODER_MIN_SCORE
    )


def test_every_cross_encoder_ended_profile_uses_cross_encoder_threshold():
    profiles = [profile for profile in PROFILES if profile.endswith("_cross-encoder")]
    assert profiles
    assert all(min_score_for(profile) == CROSS_ENCODER_MIN_SCORE for profile in profiles)
