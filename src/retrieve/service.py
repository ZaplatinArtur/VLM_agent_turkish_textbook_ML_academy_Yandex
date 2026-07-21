from ..schemas.retrieve import RetrievedChunk

from .embedders import SentenceTransformerEmbedder
from .index import Index
from .parsing import get_retrieved_chunks
from .pipeline import RetrievalPipeline
from .rankers import DenseRanker

_pipeline: RetrievalPipeline | None = None


def build_pipeline(
    chunks: list[RetrievedChunk] | None = None,
) -> RetrievalPipeline:
    corpus = get_retrieved_chunks() if chunks is None else chunks
    index = Index(corpus)
    embedder = SentenceTransformerEmbedder()
    return RetrievalPipeline(
        rankers=[DenseRanker(embedder=embedder, index=index)],
    )


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
    return get_pipeline().run(query, k=k, subject=subject)
