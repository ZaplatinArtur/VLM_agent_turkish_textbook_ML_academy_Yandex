from __future__ import annotations

import os
import threading
from functools import cache
from typing import TYPE_CHECKING

from paths import INDEX_DIR

from schemas.retrieve import RetrievedChunk

from .confidence import RelevanceVerdict, assess_relevance
from .index import Index
from .parsing import get_retrieved_chunks
from .pipeline import RetrievalPipeline

if TYPE_CHECKING:
    from .gate import SemanticGate

_pipeline: RetrievalPipeline | None = None
_pipeline_lock = threading.Lock()


def active_profile() -> str:
    from .pipelines import DEFAULT_PROFILE

    return os.environ.get("RETRIEVE_PROFILE", DEFAULT_PROFILE)


@cache
def get_gate() -> SemanticGate | None:
    """Семантический гейт, если он настроен через RETRIEVE_GATE_URL, иначе None."""
    from .gate import gate_from_env

    return gate_from_env()


def build_pipeline(
        chunks: list[RetrievedChunk] | None = None,
        profile: str | None = None,
) -> RetrievalPipeline:
    """Собирает пайплайн выбранного профиля (см. retrieve.pipelines)."""
    from .pipelines import build_profile

    profile = profile or active_profile()
    raw_corpus = get_retrieved_chunks() if chunks is None else chunks
    corpus = [chunk for chunk in raw_corpus if chunk.text.strip()]
    return build_profile(profile, Index(corpus), index_root=INDEX_DIR)


def get_pipeline() -> RetrievalPipeline:
    """Пайплайн процесса. Не functools.cache: под гонкой он собрал бы второй,
    а сборка — это загрузка моделей и индекса по 42k чанков."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = build_pipeline()
    return _pipeline


def textbook_retrieve(
        query: str,
        k: int = 5,
        subject: str | None = None,
) -> list[RetrievedChunk]:
    return get_pipeline().run(query, k=k, subject=subject)


def textbook_retrieve_checked(
        query: str,
        k: int = 5,
        subject: str | None = None,
) -> tuple[list[RetrievedChunk], RelevanceVerdict]:
    """Ищет и оценивает выдачу: семантическим гейтом, иначе порогом профиля.

    Гейт ещё и прореживает выдачу, поэтому список возвращается уже отфильтрованный.
    """
    results = get_pipeline().run(query, k=k, subject=subject)
    gate = get_gate()
    if gate is not None:
        return gate.judge(query, results)
    return results, assess_relevance(results, profile=active_profile())
