import os
import threading
from pathlib import Path

from paths import INDEX_DIR

from schemas.retrieve import RetrievedChunk

from .confidence import RelevanceVerdict, assess_relevance
from .config import BGE_M3_REVISION, BgeM3Config
from .graph import KnowledgeGraph, KnowledgeGraphBuilder
from .index import Index
from .knowledge import KnowledgeBaseBuilder
from .pipeline import RetrievalPipeline

_pipeline: RetrievalPipeline | None = None
_pipeline_lock = threading.Lock()


def build_pipeline(
        chunks: list[RetrievedChunk] | None = None,
        *,
        bge_m3_config: BgeM3Config | None = None,
) -> RetrievalPipeline:
    from .embedders import SentenceTransformerEmbedder
    from .rankers import (
        BM25Ranker,
        DenseRanker,
        GraphExpansionRanker,
        KnowledgeReranker,
        PrimaryCandidateUnion,
        ReciprocalRankFusion,
    )

    semantic_config = bge_m3_config or BgeM3Config.from_env()

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
    lexical = BM25Ranker(
        index=index,
        fetch_k=int(os.environ.get("MLA_LEXICAL_FETCH_K", "80")),
    )
    if semantic_config.enabled:
        bge_provenance = semantic_config.embedder_provenance()
        runtime_versions = bge_provenance["runtime_packages"]
        if not isinstance(runtime_versions, dict):
            raise ValueError("BGE-M3 runtime package provenance is malformed")
        bge_embedder = SentenceTransformerEmbedder(
            model_name=semantic_config.model_id,
            revision=semantic_config.revision,
            license_id=semantic_config.license,
            batch_size=semantic_config.batch_size,
            max_length=semantic_config.max_length,
            expected_dimension=semantic_config.embedding_dimension,
            normalize_embeddings=False,
            task_contract=semantic_config.task_contract,
            local_files_only=semantic_config.local_files_only,
            cache_dir=semantic_config.cache_dir,
            device=semantic_config.device,
            runtime_versions=runtime_versions,
            validate_bge_m3_semantics=True,
            faiss_index_kind=semantic_config.faiss_index_kind,
        )
        bge_index_dir = semantic_config.index_dir or (
            INDEX_DIR / f"bge_m3_{BGE_M3_REVISION[:12]}"
        )
        if bge_index_dir.resolve(strict=False) == INDEX_DIR.resolve(strict=False):
            raise ValueError(
                "MLA_BGE_M3_INDEX_DIR must not equal the legacy index directory"
            )
        bge_dense = DenseRanker(
            embedder=bge_embedder,
            index=index,
            index_dir=bge_index_dir,
            fetch_k=semantic_config.semantic_candidate_k,
            strict_provenance=True,
            expected_embedder_provenance=bge_provenance,
            expected_embedding_dimension=semantic_config.embedding_dimension,
            index_kind=semantic_config.faiss_index_kind,
        )
        fused = PrimaryCandidateUnion(
            primary=lexical,
            semantic=bge_dense,
            primary_k=semantic_config.primary_candidate_k,
            semantic_k=semantic_config.semantic_candidate_k,
            mode=semantic_config.candidate_mode,
            fallback_min_candidates=(
                semantic_config.fallback_min_candidates
            ),
        )
    else:
        # Backward-compatible legacy path. Its model, cache keys, weights and
        # index directory intentionally remain unchanged while BGE is disabled.
        embedder = SentenceTransformerEmbedder()
        dense = DenseRanker(
            embedder=embedder,
            index=index,
            index_dir=INDEX_DIR,
            fetch_k=int(os.environ.get("MLA_DENSE_FETCH_K", "80")),
        )
        fused = ReciprocalRankFusion(
            [dense, lexical],
            weights=[0.65, 0.35],
        )
    rerank_model = os.environ.get("MLA_RERANK_MODEL", "").strip() or None
    rerank_top_n_raw = os.environ.get("MLA_RERANK_TOP_N")
    try:
        rerank_top_n = int(rerank_top_n_raw or "40")
    except ValueError as exc:
        raise ValueError("MLA_RERANK_TOP_N must be an integer") from exc
    if rerank_top_n <= 0:
        raise ValueError("MLA_RERANK_TOP_N must be positive")
    if semantic_config.enabled:
        required_rerank_top_n = (
            semantic_config.primary_candidate_k
            + semantic_config.semantic_candidate_k
        )
        if rerank_top_n_raw is not None and rerank_top_n < required_rerank_top_n:
            raise ValueError(
                "MLA_RERANK_TOP_N must be at least the enabled BGE candidate "
                f"window ({required_rerank_top_n})"
            )
        rerank_top_n = max(rerank_top_n, required_rerank_top_n)
    rankers = [
        fused,
        KnowledgeReranker(
            model_name=rerank_model,
            top_n=rerank_top_n,
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
