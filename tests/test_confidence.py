import pytest

from retrieve.confidence import (
    CROSS_ENCODER_MIN_SCORE,
    DEFAULT_MIN_SCORE,
    PROFILE_MIN_SCORE,
    UNCALIBRATED_PROFILES,
    Relevance,
    UncalibratedProfileError,
    assess_relevance,
    min_score_for,
    validate_profile,
)
from retrieve.pipelines import PROFILES
from retrieve.rankers.fusion import DEFAULT_RRF_K
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


def test_missing_profile_falls_back_to_default():
    # None — вызывающий про профиль не знает; это не то же самое, что профиль
    # без порога, и ломать этот путь нельзя: так зовёт запасная ветка тулы.
    assert min_score_for(None) == DEFAULT_MIN_SCORE
    assert min_score_for("") == DEFAULT_MIN_SCORE
    assert min_score_for("e5-small") == PROFILE_MIN_SCORE["e5-small"]


def test_uncalibrated_profile_is_rejected_not_defaulted():
    with pytest.raises(UncalibratedProfileError, match="m3"):
        min_score_for("m3")
    with pytest.raises(UncalibratedProfileError):
        assess_relevance([chunk("a", 0.9)], profile="rrf_e5-small_bm25")


def test_unknown_profile_is_rejected():
    # Новый профиль не должен молча получить чужой порог.
    with pytest.raises(UncalibratedProfileError):
        min_score_for("visual_colqwen25_cascade")


def test_explicit_min_score_bypasses_calibration_check():
    verdict = assess_relevance([chunk("a", 18.4)], min_score=12.0, profile="m3")
    assert verdict.is_useful


def test_validate_profile_passes_for_calibrated():
    validate_profile("rrf_e5-small_bm25_cross-encoder")
    with pytest.raises(UncalibratedProfileError):
        validate_profile("bm25")


def test_every_profile_is_either_calibrated_or_declared_uncalibrated():
    known = set(PROFILE_MIN_SCORE) | UNCALIBRATED_PROFILES
    unclassified = [profile for profile in PROFILES if profile not in known]
    assert not unclassified, (
        f"профили без записи в реестре порогов: {unclassified}. "
        "Добавьте порог в PROFILE_MIN_SCORE либо честно объявите "
        "профиль в UNCALIBRATED_PROFILES."
    )
    assert not (set(PROFILE_MIN_SCORE) & UNCALIBRATED_PROFILES)


def test_rrf_tail_profiles_stay_uncalibrated():
    # Счёт RRF складывается из рангов и при rrf_k=60 не превышает 2/61:
    # любой порог из шкалы [0, 1] отсекал бы всю выдачу.
    unreachable = 2 / (DEFAULT_RRF_K + 1)
    assert unreachable < CROSS_ENCODER_MIN_SCORE
    rrf_tails = [
        profile
        for profile in PROFILES
        if profile.startswith("rrf_") and profile.removesuffix("_gate").endswith("bm25")
    ]
    assert rrf_tails
    for profile in rrf_tails:
        assert profile in UNCALIBRATED_PROFILES, profile


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
