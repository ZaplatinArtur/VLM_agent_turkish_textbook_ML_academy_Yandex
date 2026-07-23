"""Детектор бесполезного поиска.

Плотный ретрив ВСЕГДА что-то возвращает — даже на вопрос не по теме учебника
он выдаст k наименее далёких чанков с низким score. Если отдать их агенту как
есть, тот начнёт сочинять ответ по нерелевантному тексту. Детектор смотрит на
score выдачи и решает, стоит ли ей доверять.

Два сигнала (score — косинус в [−1, 1], т.к. векторы нормализованы):
  * абсолют: top-1 score ниже порога → ничего похожего в корпусе нет;
  * маржа:   top-1 почти не оторвался от top-2 → «размазанное» совпадение,
             ретрив не уверен, какой чанк релевантнее (по умолчанию выключено).
"""

from dataclasses import dataclass
from enum import Enum

from schemas.retrieve import RetrievedChunk

DEFAULT_MIN_SCORE = 0.5
DEFAULT_MIN_MARGIN = 0.0


class Relevance(str, Enum):
    CONFIDENT = "confident"
    WEAK = "weak"
    EMPTY = "empty"


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
        min_score: float = DEFAULT_MIN_SCORE,
        min_margin: float = DEFAULT_MIN_MARGIN,
) -> RelevanceVerdict:
    if not results:
        return RelevanceVerdict(Relevance.EMPTY, None, "выдача пуста")
    top_score = results[0].score
    if top_score < min_score:
        return RelevanceVerdict(
            Relevance.WEAK,
            top_score,
            f"top-1 score {top_score:.3f} < порога {min_score:.2f}",
        )
    if min_margin > 0 and len(results) >= 2:
        margin = top_score - results[1].score
        if margin < min_margin:
            return RelevanceVerdict(
                Relevance.WEAK,
                top_score,
                f"маржа top1−top2 {margin:.3f} < {min_margin:.2f}",
            )
    return RelevanceVerdict(Relevance.CONFIDENT, top_score, "уверенное попадание")
