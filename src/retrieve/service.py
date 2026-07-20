from ..schemas.retrieve import RetrievedChunk
from .embedders import SentenceTransformerEmbedder
from .pipeline import RetrievalPipeline
from .rankers import DenseRanker

_pipeline: RetrievalPipeline | None = None


def build_pipeline() -> RetrievalPipeline:
    embedder = SentenceTransformerEmbedder()
    return RetrievalPipeline(rankers=[DenseRanker(embedder=embedder)])


def get_pipeline() -> RetrievalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def textbook_retrieve(
        query: str,
        k: int = 5,
        subject: str | None = None,
    ) -> list[RetrievedChunk]:
    pipeline = get_pipeline()
    return pipeline.run(query, k, subject=subject)
