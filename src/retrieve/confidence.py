"""Детектор бесполезного поиска: top-1 score ниже порога — выдаче не доверяем.

Порог свой у каждого профиля, шкалы у энкодеров разные (PROFILE_MIN_SCORE).
Новый профиль калибруется заново:

    python -m retrieve.compare --systems <профиль> --reference none --calibrate

Профили с хвостом-кросс-энкодером берут общий CROSS_ENCODER_MIN_SCORE.
Альтернатива порогу — retrieve.gate, там решает LLM.

Некалиброванный профиль отвергается, а не откатывается к дефолту: чужой порог
в чужой шкале не выключает гейт, а ломает его молча. У RRF счёт складывается из
рангов и не превышает 2/(rrf_k+1) — дефолтные 0.57 отсекали бы всё подряд;
у MaxSim он суммируется по токенам запроса и уходит далеко за единицу —
те же 0.57 пропустили бы всё подряд.
"""

from dataclasses import dataclass
from enum import Enum

from schemas.retrieve import RetrievedChunk

DEFAULT_MIN_SCORE = 0.57
CROSS_ENCODER_MIN_SCORE = 0.5


class UncalibratedProfileError(RuntimeError):
    """Профиль без снятого порога: в какой шкале судить выдачу — неизвестно."""


# Суффикс _gate добавляет LLM-ступень, которая меняет только порядок выдачи,
# поэтому счёт остаётся от кросс-энкодера и порог у них общий.
PROFILE_MIN_SCORE = {
    "minilm": DEFAULT_MIN_SCORE,
    "e5-small": 0.8643,
    "rrf_e5-small_bm25_cross-encoder": CROSS_ENCODER_MIN_SCORE,
    "rrf_e5-small_bm25_cross-encoder_gate": CROSS_ENCODER_MIN_SCORE,
    "rrf_e5-small_bm25_qwen3-reranker": CROSS_ENCODER_MIN_SCORE,
    "rrf_e5-base_m3_bm25_cross-encoder": CROSS_ENCODER_MIN_SCORE,
    "rrf_e5-small_m3_bm25_cross-encoder": CROSS_ENCODER_MIN_SCORE,
    "rrf_m3_bm25_cross-encoder": CROSS_ENCODER_MIN_SCORE,
    "rrf_m3_bm25_cross-encoder_gate": CROSS_ENCODER_MIN_SCORE,
}

# Порог не снимался. Голые ретриверы — потому что руки не дошли, RRF-хвосты —
# потому что ранговая сумма релевантность и не измеряет.
UNCALIBRATED_PROFILES = frozenset({
    "bm25",
    "e5-base",
    "m3",
    "qwen3-embedding",
    "rrf_e5-small_bm25",
    "rrf_e5-small_bm25_gate",
    "rrf_qwen3-embedding_bm25",
    "rrf_e5-base_bm25",
    "rrf_e5-base_m3_bm25",
    "rrf_e5-small_m3_bm25",
})


def min_score_for(profile: str | None) -> float:
    """Порог профиля. None — вызывающий профиля не знает, берём исторический дефолт."""
    if not profile:
        return DEFAULT_MIN_SCORE
    try:
        return PROFILE_MIN_SCORE[profile]
    except KeyError:
        raise UncalibratedProfileError(
            f"профиль {profile!r} без откалиброванного порога — гейт применить не к чему. "
            "Снимите порог: python -m retrieve.compare "
            f"--systems {profile} --reference none --calibrate, "
            "затем впишите его в PROFILE_MIN_SCORE. "
            "Разово обойти можно явным assess_relevance(min_score=...)."
        ) from None


def validate_profile(profile: str | None) -> None:
    """Падает до прогона, а не на первом запросе посреди него."""
    min_score_for(profile)


class Relevance(str, Enum):
    CONFIDENT = "confident"
    WEAK = "weak"
    EMPTY = "empty"
    ERROR = "error"  # оценить не удалось (гейт недоступен) — выдачу скрываем


@dataclass
class RelevanceVerdict:
    relevance: Relevance
    top_score: float | None
    reason: str

    @property
    def is_useful(self) -> bool:
        return self.relevance is Relevance.CONFIDENT


def assess_relevance(
        results: list[RetrievedChunk],
        min_score: float | None = None,
        profile: str | None = None,
) -> RelevanceVerdict:
    """Оценивает выдачу. min_score важнее profile; без обоих — DEFAULT_MIN_SCORE.

    Некалиброванный profile поднимает UncalibratedProfileError.
    """
    if min_score is None:
        min_score = min_score_for(profile)
    if not results:
        return RelevanceVerdict(Relevance.EMPTY, None, "выдача пуста")
    top_score = results[0].score
    if top_score < min_score:
        return RelevanceVerdict(
            Relevance.WEAK,
            top_score,
            f"top-1 score {top_score:.3f} < порога {min_score:.2f}",
        )
    return RelevanceVerdict(Relevance.CONFIDENT, top_score, "уверенное попадание")
