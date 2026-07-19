from ..schemas.retrieve import RetrievedChunk
from .pipeline import RetrievalPipeline

_pipeline: RetrievalPipeline | None = None


def get_pipeline() -> RetrievalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RetrievalPipeline(rankers=[])
    return _pipeline


def textbook_retrieve(
    query: str,
    k: int = 5,
    subject: str | None = None,
) -> list[RetrievedChunk]:
    return get_pipeline().run(query, subject=subject)[:k]
