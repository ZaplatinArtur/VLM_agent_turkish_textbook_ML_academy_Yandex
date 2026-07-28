import os
import threading
from pathlib import Path

from paths import INDEX_DIR

from schemas.retrieve import RetrievedChunk

from .confidence import RelevanceVerdict, assess_relevance
from .graph import KnowledgeGraph, KnowledgeGraphBuilder
from .index import Index
from .knowledge import KnowledgeBaseBuilder
from .pipeline import RetrievalPipeline

_pipeline: RetrievalPipeline | None = None
_pipeline_lock = threading.Lock()


def build_pipeline(
        chunks: list[RetrievedChunk] | None = None,
) -> RetrievalPipeline:
    from .embedders import SentenceTransformerEmbedder
    from .rankers import (
        BM25Ranker,
        DenseRanker,
        GraphExpansionRanker,
        KnowledgeReranker,
        ReciprocalRankFusion,
    )

    graph: KnowledgeGraph | None = None
    graph_dir_value = os.environ.get("MLA_KNOWLEDGE_GRAPH_DIR", "").strip()
    graph_dir = Path(graph_dir_value).expanduser() if graph_dir_value else None
    if (
        chunks is None
        and graph_dir is not None
        and (graph_dir / "nodes.jsonl").is_file()
        and (graph_dir / "edges.jsonl").is_file()
    ):
        graph = KnowledgeGraph.load(graph_dir)
        corpus = graph.searchable_nodes()
    else:
        if chunks is None:
            from .parsing import get_retrieved_chunks

            raw_corpus = get_retrieved_chunks()
        else:
            raw_corpus = chunks
        corpus = [chunk for chunk in raw_corpus if chunk.text.strip()]
    has_educational_units = any(
        chunk.metadata.get("unit_kind") not in (None, "", "page")
        for chunk in corpus
    )
    has_knowledge_cards = any(
        bool(chunk.metadata.get("knowledge_card")) for chunk in corpus
    )
    use_knowledge_cards = (
        os.environ.get("MLA_KNOWLEDGE_CARDS", "true").casefold()
        not in {"0", "false", "no"}
    )
    use_knowledge_graph = (
        os.environ.get("MLA_KNOWLEDGE_GRAPH", "true").casefold()
        not in {"0", "false", "no"}
    )
    if (
        graph is None
        and has_educational_units
        and not has_knowledge_cards
        and use_knowledge_graph
    ):
        graph = KnowledgeGraphBuilder().build(corpus)
        corpus = graph.searchable_nodes()
    elif (
        graph is None
        and has_educational_units
        and not has_knowledge_cards
        and use_knowledge_cards
    ):
        corpus = KnowledgeBaseBuilder().build(corpus)
    index = Index(corpus)
    embedder = SentenceTransformerEmbedder()
    dense = DenseRanker(
        embedder=embedder,
        index=index,
        index_dir=INDEX_DIR,
        fetch_k=int(os.environ.get("MLA_DENSE_FETCH_K", "80")),
    )
    lexical = BM25Ranker(
        index=index,
        fetch_k=int(os.environ.get("MLA_LEXICAL_FETCH_K", "80")),
    )
    fused = ReciprocalRankFusion(
        [dense, lexical],
        weights=[0.65, 0.35],
    )
    rerank_model = os.environ.get("MLA_RERANK_MODEL", "").strip() or None
    rankers = [
        fused,
        KnowledgeReranker(
            model_name=rerank_model,
            top_n=int(os.environ.get("MLA_RERANK_TOP_N", "40")),
            min_graph_theory_chars=int(
                os.environ.get("MLA_GRAPH_MIN_THEORY_CHARS", "70")
            ),
        ),
    ]
    if graph is not None:
        rankers.append(
            GraphExpansionRanker(
                graph,
                max_exercise_candidates=int(
                    os.environ.get("MLA_GRAPH_MAX_EXERCISES", "20")
                ),
                include_solutions=(
                    os.environ.get(
                        "MLA_GRAPH_INCLUDE_SOLUTIONS",
                        "false",
                    ).casefold()
                    in {"1", "true", "yes", "on"}
                ),
            )
        )
    return RetrievalPipeline(
        rankers=rankers,
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
        grade: int | str | None = None,
) -> list[RetrievedChunk]:
    return get_pipeline().run(query, k=k, subject=subject, grade=grade)


def textbook_retrieve_checked(
        query: str,
        k: int = 5,
        subject: str | None = None,
        grade: int | str | None = None,
) -> tuple[list[RetrievedChunk], RelevanceVerdict]:
    """Как textbook_retrieve, но с вердиктом детектора бесполезного поиска."""
    results = get_pipeline().run(query, k=k, subject=subject, grade=grade)
    return results, assess_relevance(results)
