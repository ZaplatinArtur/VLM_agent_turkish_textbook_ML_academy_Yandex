import threading

from paths import INDEX_DIR

from schemas.retrieve import RetrievedChunk

from .confidence import RelevanceVerdict, assess_relevance
from .index import Index
from .parsing import get_retrieved_chunks
from .pipeline import RetrievalPipeline

_pipeline: RetrievalPipeline | None = None
_pipeline_lock = threading.Lock()


def build_pipeline(
        chunks: list[RetrievedChunk] | None = None,
) -> RetrievalPipeline:
    from .embedders import SentenceTransformerEmbedder
    from .rankers import DenseRanker

    raw_corpus = get_retrieved_chunks() if chunks is None else chunks
    corpus = [chunk for chunk in raw_corpus if chunk.text.strip()]
    index = Index(corpus)
    embedder = SentenceTransformerEmbedder()
    return RetrievalPipeline(
        rankers=[DenseRanker(embedder=embedder, index=index, index_dir=INDEX_DIR)],
    )


def get_pipeline() -> RetrievalPipeline:
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
    """Как textbook_retrieve, но с вердиктом детектора бесполезного поиска."""
    results = get_pipeline().run(query, k=k, subject=subject)
    return results, assess_relevance(results)
