"""Детектор бесполезного поиска: top-1 score ниже порога — выдаче не доверяем.

Порог свой у каждого профиля, шкалы у энкодеров разные (PROFILE_MIN_SCORE).
Новый профиль калибруется заново:

    python -m retrieve.compare --systems <профиль> --reference none --calibrate

Профили с хвостом-кросс-энкодером берут общий CROSS_ENCODER_MIN_SCORE.
Альтернатива порогу — retrieve.gate, там решает LLM.
"""

from dataclasses import dataclass
from enum import Enum

from schemas.retrieve import RetrievedChunk

DEFAULT_MIN_SCORE = 0.57
CROSS_ENCODER_MIN_SCORE = 0.5

PROFILE_MIN_SCORE = {
    "minilm": DEFAULT_MIN_SCORE,
    "e5-small": 0.8643,
    "rrf_e5-small_bm25_cross-encoder": CROSS_ENCODER_MIN_SCORE,
    "rrf_e5-small_bm25_qwen3-reranker": CROSS_ENCODER_MIN_SCORE,
    "rrf_e5-base_m3_bm25_cross-encoder": CROSS_ENCODER_MIN_SCORE,
    "rrf_e5-small_m3_bm25_cross-encoder": CROSS_ENCODER_MIN_SCORE,
    "rrf_m3_bm25_cross-encoder": CROSS_ENCODER_MIN_SCORE,
}


def min_score_for(profile: str | None) -> float:
    return PROFILE_MIN_SCORE.get(profile or "", DEFAULT_MIN_SCORE)


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
    """Оценивает выдачу. min_score важнее profile; без обоих — DEFAULT_MIN_SCORE."""
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
