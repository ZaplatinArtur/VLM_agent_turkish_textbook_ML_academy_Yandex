import os
import threading
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from paths import INDEX_DIR

from schemas.retrieve import RetrievedChunk

from .confidence import RelevanceVerdict, assess_relevance
from .config import BGE_M3_REVISION, BgeM3Config
from .graph import KnowledgeGraph, KnowledgeGraphBuilder
from .index import Index
from .knowledge import KnowledgeBaseBuilder
from .pipeline import RetrievalPipeline

_pipelines: dict[tuple[object, ...], RetrievalPipeline] = {}
_pipeline_lock = threading.Lock()

if TYPE_CHECKING:
    from .gate import SemanticGate

# Значение RETRIEVE_PROFILE, уводящее на графовый пайплайн вместо профиля.
ADVANCED_PIPELINE = "advanced"


def active_profile(profile: str | None = None) -> str | None:
    """Профиль по умолчанию — DEFAULT_PROFILE; None означает графовый пайплайн.

    RETRIEVE_PROFILE=advanced возвращает сборку с графом знаний, BGE-M3 и
    KnowledgeReranker — на ней получены прежние прогоны команды.
    """
    from .pipelines import DEFAULT_PROFILE

    selected = profile if profile is not None else os.environ.get("RETRIEVE_PROFILE", "")
    selected = selected.strip() or DEFAULT_PROFILE
    return None if selected == ADVANCED_PIPELINE else selected


@cache
def get_gate() -> "SemanticGate | None":
    from .gate import gate_from_env

    return gate_from_env()


def build_pipeline(
        chunks: list[RetrievedChunk] | None = None,
        profile: str | None = None,
        *,
        bge_m3_config: BgeM3Config | None = None,
        fetch_k: int | None = None,
        mmr_lambda: float | None = None,
) -> RetrievalPipeline:
    """Build the advanced pipeline or an explicitly selected experiment profile.

    ``RETRIEVE_PROFILE`` and the ``profile`` argument keep the named dense,
    hybrid and reranker profiles available. Without either, the educational
    chunking / knowledge-graph / BGE-M3 pipeline remains the default.

    Passing ``fetch_k`` or ``mmr_lambda`` explicitly selects the frozen dense
    experiment path used by the MMR evaluations.
    """
    if fetch_k is not None and fetch_k <= 0:
        raise ValueError("fetch_k must be positive")
    if mmr_lambda is not None and not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("mmr_lambda must be between 0 and 1")

    selected_profile = active_profile(profile)
    if selected_profile:
        from .parsing import get_retrieved_chunks
        from .pipelines import build_profile

        raw_corpus = get_retrieved_chunks() if chunks is None else chunks
        corpus = [chunk for chunk in raw_corpus if chunk.text.strip()]
        return build_profile(
            selected_profile,
            Index(corpus),
            index_root=INDEX_DIR,
            fetch_k=fetch_k or 200,
        )

    if fetch_k is not None or mmr_lambda is not None:
        from .embedders import SentenceTransformerEmbedder
        from .parsing import get_retrieved_chunks
        from .rankers import DenseRanker, MaximalMarginalRelevanceRanker

        raw_corpus = get_retrieved_chunks() if chunks is None else chunks
        corpus = [chunk for chunk in raw_corpus if chunk.text.strip()]
        index = Index(corpus)
        embedder = SentenceTransformerEmbedder()
        rankers = [
            DenseRanker(
                embedder=embedder,
                index=index,
                fetch_k=fetch_k or 200,
                index_dir=INDEX_DIR,
            )
        ]
        if mmr_lambda is not None:
            rankers.append(
                MaximalMarginalRelevanceRanker(
                    embedder=embedder,
                    lambda_mult=mmr_lambda,
                )
            )
        return RetrievalPipeline(rankers=rankers)

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


def get_pipeline(
        *,
        profile: str | None = None,
        fetch_k: int | None = None,
        mmr_lambda: float | None = None,
) -> RetrievalPipeline:
    selected_profile = active_profile(profile)
    key = (selected_profile, fetch_k, mmr_lambda)
    if key not in _pipelines:
        with _pipeline_lock:
            if key not in _pipelines:
                _pipelines[key] = build_pipeline(
                    profile=selected_profile,
                    fetch_k=fetch_k,
                    mmr_lambda=mmr_lambda,
                )
    return _pipelines[key]


def textbook_retrieve(
        query: str,
        k: int = 5,
        subject: str | None = None,
        grade: int | str | None = None,
        *,
        profile: str | None = None,
        fetch_k: int | None = None,
        mmr_lambda: float | None = None,
) -> list[RetrievedChunk]:
    return get_pipeline(
        profile=profile,
        fetch_k=fetch_k,
        mmr_lambda=mmr_lambda,
    ).run(
        query,
        k=k,
        subject=subject,
        grade=grade,
    )


def textbook_retrieve_checked(
        query: str,
        k: int = 5,
        subject: str | None = None,
        grade: int | str | None = None,
        *,
        profile: str | None = None,
        fetch_k: int | None = None,
        mmr_lambda: float | None = None,
) -> tuple[list[RetrievedChunk], RelevanceVerdict]:
    """Как textbook_retrieve, но с вердиктом детектора бесполезного поиска."""
    results = get_pipeline(
        profile=profile,
        fetch_k=fetch_k,
        mmr_lambda=mmr_lambda,
    ).run(
        query,
        k=k,
        subject=subject,
        grade=grade,
    )
    gate = get_gate()
    if gate is not None:
        return gate.judge(query, results)
    return results, assess_relevance(results, profile=active_profile(profile))
