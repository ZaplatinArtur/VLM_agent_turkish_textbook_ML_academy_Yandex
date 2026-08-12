import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

from retrieve import config as config_module
from retrieve.config import BgeM3Config
from retrieve.rankers import (
    BM25Ranker,
    CrossEncoderRanker,
    DenseRanker,
    KnowledgeReranker,
    PrimaryCandidateUnion,
    ReciprocalRankFusion,
)
from retrieve.service import build_pipeline
from paths import INDEX_DIR
from schemas.retrieve import RetrievedChunk
from retrieve.vector_store import IndexKind


RUNTIME_VERSIONS = {
    "sentence-transformers": "5.3.0",
    "transformers": "4.57.1",
    "torch": "2.8.0",
    "tokenizers": "0.22.1",
    "faiss-cpu": "1.15.0",
    "numpy": "2.3.0",
}


@pytest.fixture
def bge_runtime_versions(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "resolve_bge_runtime_versions",
        lambda: dict(RUNTIME_VERSIONS),
    )


def make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=0.0,
        metadata={"grade": 9, "subject": "math", "textbook": "book-a"},
    )


def test_disabled_bge_keeps_legacy_hybrid_pipeline_shape(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "resolve_bge_runtime_versions",
        lambda: (_ for _ in ()).throw(AssertionError("BGE runtime resolved")),
    )
    pipeline = build_pipeline(
        [make_chunk("a", "triangle area")],
        bge_m3_config=BgeM3Config(enabled=False),
    )

    candidate_ranker = pipeline.rankers[0]
    assert isinstance(candidate_ranker, ReciprocalRankFusion)
    assert isinstance(candidate_ranker.rankers[0], DenseRanker)
    assert isinstance(candidate_ranker.rankers[1], BM25Ranker)
    assert candidate_ranker.weights == [0.65, 0.35]
    assert candidate_ranker.rankers[0].strict_provenance is False
    assert isinstance(pipeline.rankers[1], KnowledgeReranker)


def test_teslov_cross_encoder_is_explicit_lazy_and_pinned(monkeypatch):
    monkeypatch.setenv("MLA_RERANK_BACKEND", "teslov_cross_encoder")
    monkeypatch.delenv("MLA_RERANK_MODEL", raising=False)
    monkeypatch.setenv("MLA_RERANK_DEVICE", "cpu")

    pipeline = build_pipeline(
        [make_chunk("a", "triangle area")],
        bge_m3_config=BgeM3Config(enabled=False),
    )

    reranker = pipeline.rankers[1]
    assert isinstance(reranker, CrossEncoderRanker)
    assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert reranker.revision == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert reranker.local_files_only is True
    assert reranker.device == "cpu"
    assert reranker._model is None, "pipeline construction must remain lazy"


def test_teslov_cross_encoder_rejects_unpinned_model(monkeypatch):
    monkeypatch.setenv("MLA_RERANK_BACKEND", "teslov_cross_encoder")
    monkeypatch.setenv("MLA_RERANK_MODEL", "some/other-model")

    with pytest.raises(ValueError, match="pinned"):
        build_pipeline(
            [make_chunk("a", "triangle area")],
            bge_m3_config=BgeM3Config(enabled=False),
        )


def test_enabled_bge_fails_closed_when_runtime_version_metadata_is_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        config_module,
        "resolve_bge_runtime_versions",
        lambda: (_ for _ in ()).throw(RuntimeError("missing transformers")),
    )
    with pytest.raises(RuntimeError, match="missing transformers"):
        build_pipeline(
            [make_chunk("a", "triangle area")],
            bge_m3_config=BgeM3Config(enabled=True),
        )


def test_enabled_bge_is_pinned_lazy_lexical_first_and_has_full_rerank_window(
    monkeypatch,
    tmp_path,
    bge_runtime_versions,
):
    monkeypatch.delenv("MLA_RERANK_TOP_N", raising=False)
    config = BgeM3Config(
        enabled=True,
        index_dir=tmp_path / "bge-index",
        candidate_mode="fallback",
        fallback_min_candidates=3,
    )
    pipeline = build_pipeline(
        [make_chunk("a", "triangle area")],
        bge_m3_config=config,
    )

    candidate_ranker = pipeline.rankers[0]
    assert isinstance(candidate_ranker, PrimaryCandidateUnion)
    assert isinstance(candidate_ranker.primary, BM25Ranker)
    assert isinstance(candidate_ranker.semantic, DenseRanker)
    assert candidate_ranker.primary_k == 32
    assert candidate_ranker.semantic_k == 32
    assert candidate_ranker.mode == "fallback"
    assert candidate_ranker.fallback_min_candidates == 3
    assert pipeline.rankers[1].top_n == 64

    dense = candidate_ranker.semantic
    assert dense.strict_provenance is True
    assert dense.fetch_k == 32
    assert dense.index_dir == tmp_path / "bge-index"
    assert dense.expected_embedder_provenance == config.embedder_provenance(
        RUNTIME_VERSIONS
    )
    assert dense.expected_embedding_dimension == 1024
    assert dense.embedder.provenance == config.embedder_provenance(
        RUNTIME_VERSIONS
    )
    assert dense.embedder.local_files_only is True
    assert dense.embedder.validate_bge_m3_semantics is True
    assert dense.embedder._model is None, "pipeline construction must not load the model"


def test_enabled_bge_rejects_explicit_rerank_window_smaller_than_union(
    monkeypatch,
    bge_runtime_versions,
):
    monkeypatch.setenv("MLA_RERANK_TOP_N", "63")
    with pytest.raises(ValueError, match="at least the enabled BGE candidate window"):
        build_pipeline(
            [make_chunk("a", "triangle area")],
            bge_m3_config=BgeM3Config(enabled=True),
        )


def test_enabled_bge_rejects_legacy_index_directory_collision(
    bge_runtime_versions,
):
    with pytest.raises(ValueError, match="legacy index directory"):
        build_pipeline(
            [make_chunk("a", "triangle area")],
            bge_m3_config=BgeM3Config(
                enabled=True,
                index_dir=INDEX_DIR,
            ),
        )


def test_enabled_bge_threads_explicit_hnsw_kind_to_dense(
    tmp_path,
    bge_runtime_versions,
):
    config = BgeM3Config(
        enabled=True,
        index_dir=tmp_path / "hnsw-index",
        faiss_index_kind="hnsw",
    )

    pipeline = build_pipeline(
        [make_chunk("a", "triangle area")],
        bge_m3_config=config,
    )
    dense = pipeline.rankers[0].semantic

    assert dense.index_kind is IndexKind.HNSW
    assert dense.expected_embedder_provenance["faiss_index_kind"] == "hnsw"
    assert dense.embedder.provenance["faiss_index_kind"] == "hnsw"
