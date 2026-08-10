import math

import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

from retrieve.embedders.base import SymmetricTextEmbedder
from retrieve.index import Index
from retrieve.persistence import (
    BUILD_LOCK_FILE,
    IndexValidationError,
    MANIFEST_FILE,
)
from retrieve.rankers.dense import DenseRanker
from retrieve.vector_store import CHUNK_IDS_FILE, INDEX_FILE, IndexKind
from schemas.retrieve import RetrievedChunk

VOCAB = ["üçgen", "alan", "hız", "kuvvet", "hücre"]
HUGE_FLOAT32_VECTOR = [3e38] * len(VOCAB)


def assert_finite_unit_vector(vector: list[float]) -> None:
    assert all(math.isfinite(item) for item in vector)
    assert math.isclose(
        math.hypot(*vector),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )

class FakeEmbedder(SymmetricTextEmbedder):
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(text.lower().count(word)) for word in VOCAB] for text in texts]

class DictCache:
    def __init__(self) -> None:
        self.store: dict[str, list[float]] = {}

    def get_embedding(self, chunk_id: str) -> list[float] | None:
        return self.store.get(chunk_id)

    def set_embedding(self, chunk_id: str, vector: list[float]) -> None:
        self.store[chunk_id] = vector

class StubIndex:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def get(self, subject: str | None = None) -> list[RetrievedChunk]:
        if subject is None:
            return self._chunks
        return [c for c in self._chunks if c.metadata.get("subject") == subject]

def make_chunk(chunk_id: str, text: str, subject: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=0.0,
        metadata={"subject": subject, "textbook": "test", "page": 1},
    )

@pytest.fixture
def corpus() -> list[RetrievedChunk]:
    return [
        make_chunk("m1", "Bir üçgenin alanı taban çarpı yükseklik bölü iki.", "math"),
        make_chunk("m2", "Üçgen çeşitleri: eşkenar, ikizkenar, çeşitkenar.", "math"),
        make_chunk("p1", "Hız yol bölü zaman olarak tanımlanır.", "physics"),
        make_chunk("p2", "Kuvvet kütle çarpı ivme, Newton'un ikinci yasası.", "physics"),
        make_chunk("b1", "Hücre canlıların en küçük yapı birimidir.", "biology"),
    ]


def build_ranker(corpus: list[RetrievedChunk], **kwargs) -> DenseRanker:
    return DenseRanker(
        embedder=FakeEmbedder(),
        index=StubIndex(corpus),
        embedding_cache=DictCache(),
        **kwargs,
    )

def test_retrieves_most_relevant_chunk_first(corpus):
    ranker = build_ranker(corpus)
    results = ranker.rank("bir üçgenin alanı nasıl bulunur")
    assert results, "expected at least one hit"
    assert results[0].chunk_id == "m1"
    assert results[0].score > 0

def test_subject_filter_excludes_other_subjects(corpus):
    ranker = build_ranker(corpus)
    results = ranker.rank("kuvvet ve hız", subject="physics")
    assert results
    assert {c.chunk_id for c in results} <= {"p1", "p2"}
    assert all(c.metadata["subject"] == "physics" for c in results)

def test_fetch_k_caps_retriever_candidates(corpus):
    ranker = build_ranker(corpus, fetch_k=2)
    results = ranker.rank("üçgen alan hız kuvvet hücre")
    assert len(results) == 2

def test_chunks_restrict_search_to_subset(corpus):
    ranker = build_ranker(corpus)
    subset = [c for c in corpus if c.chunk_id in {"p1", "p2"}]
    results = ranker.rank("üçgen alanı", chunks=subset)
    assert {c.chunk_id for c in results} <= {"p1", "p2"}
    assert all(c.chunk_id not in {"m1", "m2"} for c in results)

def test_reranker_returns_all_given_chunks(corpus):
    ranker = build_ranker(corpus, fetch_k=1)
    subset = [c for c in corpus if c.chunk_id in {"m1", "m2", "p1"}]
    results = ranker.rank("üçgen alanı", chunks=subset)
    assert {c.chunk_id for c in results} == {"m1", "m2", "p1"}

def test_empty_chunks_searches_whole_index(corpus):
    ranker = build_ranker(corpus)
    results = ranker.rank("bir üçgenin alanı nasıl bulunur", chunks=[])
    assert results[0].chunk_id == "m1"

def test_empty_corpus_returns_empty(corpus):
    ranker = build_ranker([])
    assert ranker.rank("herhangi bir sorgu") == []

def test_embeddings_are_cached_not_recomputed(corpus):
    class CountingEmbedder(FakeEmbedder):
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return super().encode(texts)

    cache = DictCache()

    embedder1 = CountingEmbedder()
    DenseRanker(embedder1, StubIndex(corpus), cache).build()
    assert set(cache.store) == {c.chunk_id for c in corpus}
    assert embedder1.calls == 1

    embedder2 = CountingEmbedder()
    DenseRanker(embedder2, StubIndex(corpus), cache).build()
    assert embedder2.calls == 0


def test_original_corpus_chunk_score_not_mutated(corpus):
    ranker = build_ranker(corpus)
    ranker.rank("üçgen alanı")
    assert all(c.score == 0.0 for c in corpus)


def test_invalidate_forces_rebuild_on_next_rank(corpus):
    ranker = build_ranker(corpus)
    ranker.rank("üçgen")
    assert ranker._built
    ranker.invalidate()
    assert not ranker._built
    # После инвалидации следующий rank пересобирает индекс и снова работает.
    assert ranker.rank("bir üçgenin alanı nasıl bulunur")[0].chunk_id == "m1"
    assert ranker._built


class NamedCountingEmbedder(FakeEmbedder):
    """Считает вызовы encode и имеет стабильное имя модели для манифеста."""

    def __init__(self) -> None:
        self.model_name = "fake-model"
        self.calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return super().encode(texts)


class StrictCountingEmbedder(NamedCountingEmbedder):
    def __init__(
        self,
        revision: str,
        runtime_version: str = "5.3.0",
        index_kind: str = "auto",
    ) -> None:
        super().__init__()
        self.model_name = "BAAI/bge-m3"
        self.revision = revision
        self.runtime_version = runtime_version
        self.index_kind = index_kind

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "backend": "sentence-transformers",
            "embedding_dimension": len(VOCAB),
            "encode_normalize_embeddings": False,
            "faiss_index_kind": self.index_kind,
            "license": "mit",
            "max_length": 1024,
            "model_id": self.model_name,
            "revision": self.revision,
            "runtime_packages": {
                "sentence-transformers": self.runtime_version,
                "torch": "2.8.0",
                "transformers": "4.57.1",
            },
            "task_contract": "symmetric_retrieval_text_v1",
            "trust_remote_code": False,
            "vector_store_normalization": "l2",
        }


def test_persisted_index_is_loaded_without_re_embedding(corpus, tmp_path):
    first = NamedCountingEmbedder()
    DenseRanker(
        embedder=first,
        index=Index(corpus),
        embedding_cache=DictCache(),
        index_dir=tmp_path,
    ).build()  # строит и сохраняет снимок
    assert first.calls == 1

    # Второй ранкер (как после перезапуска): пустой кэш, тот же корпус и снимок.
    second = NamedCountingEmbedder()
    reloaded = DenseRanker(
        embedder=second,
        index=Index(corpus),
        embedding_cache=DictCache(),
        index_dir=tmp_path,
    )
    reloaded.build()
    assert second.calls == 0, "должен был загрузить снимок, а не эмбеддить чанки"
    # Поиск работает; encode здесь вызовется лишь для эмбеддинга самого запроса.
    assert reloaded.rank("bir üçgenin alanı nasıl bulunur")[0].chunk_id == "m1"


def test_strict_cache_namespace_prevents_cross_model_vector_reuse(corpus, tmp_path):
    cache = DictCache()
    first = StrictCountingEmbedder("revision-a")
    DenseRanker(
        first,
        Index(corpus),
        cache,
        index_dir=tmp_path / "first",
        strict_provenance=True,
        expected_embedder_provenance=first.provenance,
        expected_embedding_dimension=len(VOCAB),
    ).build()

    second = StrictCountingEmbedder("revision-b")
    DenseRanker(
        second,
        Index(corpus),
        cache,
        index_dir=tmp_path / "second",
        strict_provenance=True,
        expected_embedder_provenance=second.provenance,
        expected_embedding_dimension=len(VOCAB),
    ).build()

    assert first.calls == 1
    assert second.calls == 1
    assert len(cache.store) == 2 * len(corpus)
    assert all(key.startswith("v2:") for key in cache.store)
    assert not ({chunk.chunk_id for chunk in corpus} & set(cache.store))

    same_revision = StrictCountingEmbedder("revision-b")
    DenseRanker(
        same_revision,
        Index(corpus),
        cache,
        index_dir=tmp_path / "same-revision",
        strict_provenance=True,
        expected_embedder_provenance=same_revision.provenance,
        expected_embedding_dimension=len(VOCAB),
    ).build()
    assert same_revision.calls == 0


def test_strict_cache_namespace_includes_runtime_package_versions(corpus, tmp_path):
    cache = DictCache()
    first = StrictCountingEmbedder("revision-a", runtime_version="5.3.0")
    DenseRanker(
        first,
        Index(corpus),
        cache,
        index_dir=tmp_path / "first-runtime",
        strict_provenance=True,
        expected_embedder_provenance=first.provenance,
        expected_embedding_dimension=len(VOCAB),
    ).build()

    upgraded = StrictCountingEmbedder("revision-a", runtime_version="5.3.1")
    DenseRanker(
        upgraded,
        Index(corpus),
        cache,
        index_dir=tmp_path / "upgraded-runtime",
        strict_provenance=True,
        expected_embedder_provenance=upgraded.provenance,
        expected_embedding_dimension=len(VOCAB),
    ).build()

    assert first.calls == 1
    assert upgraded.calls == 1
    assert len(cache.store) == 2 * len(corpus)


def test_strict_explicit_hnsw_kind_builds_hnsw_despite_global_flat(
    corpus,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MLA_FAISS_INDEX_KIND", "flat")
    embedder = StrictCountingEmbedder("revision-a", index_kind="hnsw")
    ranker = DenseRanker(
        embedder,
        Index(corpus),
        DictCache(),
        index_dir=tmp_path / "hnsw",
        strict_provenance=True,
        expected_embedder_provenance=embedder.provenance,
        expected_embedding_dimension=len(VOCAB),
        index_kind="hnsw",
    )

    ranker.build()

    assert ranker._store is not None
    assert ranker._store.index_kind is IndexKind.HNSW


def test_strict_auto_kind_ignores_process_global_override(
    corpus,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MLA_FAISS_INDEX_KIND", "hnsw")
    embedder = StrictCountingEmbedder("revision-a", index_kind="auto")
    ranker = DenseRanker(
        embedder,
        Index(corpus),
        DictCache(),
        index_dir=tmp_path / "auto",
        strict_provenance=True,
        expected_embedder_provenance=embedder.provenance,
        expected_embedding_dimension=len(VOCAB),
        index_kind="auto",
    )

    ranker.build()

    assert ranker._store is not None
    assert ranker._store.index_kind is IndexKind.FLAT


def test_strict_cache_rejects_wrong_dimension_under_valid_namespace(
    corpus,
    tmp_path,
):
    class PinnedDimensionEmbedder(StrictCountingEmbedder):
        @property
        def provenance(self) -> dict[str, object]:
            return {**super().provenance, "embedding_dimension": 1024}

        def encode(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [[1.0] + [0.0] * 1023 for _ in texts]

    cache = DictCache()
    first = PinnedDimensionEmbedder("revision-a")
    one_chunk = corpus[:1]
    DenseRanker(
        first,
        Index(one_chunk),
        cache,
        index_dir=tmp_path / "valid-cache",
        strict_provenance=True,
        expected_embedder_provenance=first.provenance,
        expected_embedding_dimension=1024,
    ).build()
    cache.store[next(iter(cache.store))] = [0.0] * 768

    second = PinnedDimensionEmbedder("revision-a")
    with pytest.raises(ValueError, match="dimension 768; expected 1024"):
        DenseRanker(
            second,
            Index(one_chunk),
            cache,
            index_dir=tmp_path / "tampered-cache",
            strict_provenance=True,
            expected_embedder_provenance=second.provenance,
            expected_embedding_dimension=1024,
        ).build()
    assert second.calls == 0


def test_strict_cache_rejects_zero_vector_under_valid_namespace(
    corpus,
    tmp_path,
):
    cache = DictCache()
    first = StrictCountingEmbedder("revision-a")
    DenseRanker(
        first,
        Index(corpus),
        cache,
        index_dir=tmp_path / "valid-cache",
        strict_provenance=True,
        expected_embedder_provenance=first.provenance,
        expected_embedding_dimension=len(VOCAB),
    ).build()
    cache.store[next(iter(cache.store))] = [0.0] * len(VOCAB)

    second = StrictCountingEmbedder("revision-a")
    with pytest.raises(ValueError, match="near-zero L2 norm"):
        DenseRanker(
            second,
            Index(corpus),
            cache,
            index_dir=tmp_path / "zero-cache",
            strict_provenance=True,
            expected_embedder_provenance=second.provenance,
            expected_embedding_dimension=len(VOCAB),
        ).build()
    assert second.calls == 0


def test_strict_query_rejects_zero_vector_but_legacy_still_accepts_it(
    corpus,
    tmp_path,
):
    legacy = build_ranker(corpus)
    assert legacy.rank("tokens outside fake vocabulary")

    embedder = StrictCountingEmbedder("revision-a")
    strict = DenseRanker(
        embedder,
        Index(corpus),
        DictCache(),
        index_dir=tmp_path,
        strict_provenance=True,
        expected_embedder_provenance=embedder.provenance,
        expected_embedding_dimension=len(VOCAB),
    )
    with pytest.raises(ValueError, match="near-zero L2 norm"):
        strict.rank("tokens outside fake vocabulary")


def test_strict_cached_huge_float32_vector_is_stably_normalized_before_faiss(
    corpus,
    tmp_path,
):
    one_chunk = corpus[:1]
    cache = DictCache()
    embedder = StrictCountingEmbedder("revision-a")
    ranker = DenseRanker(
        embedder,
        Index(one_chunk),
        cache,
        index_dir=tmp_path,
        strict_provenance=True,
        expected_embedder_provenance=embedder.provenance,
        expected_embedding_dimension=len(VOCAB),
    )
    cache.store[ranker._embedding_cache_key(one_chunk[0])] = list(
        HUGE_FLOAT32_VECTOR
    )

    ranker.build()

    assert embedder.calls == 0
    assert ranker._store is not None
    reconstructed = [float(item) for item in ranker._store._index.reconstruct(0)]
    assert_finite_unit_vector(reconstructed)


def test_strict_generated_huge_float32_vector_is_cached_as_finite_unit_vector(
    corpus,
    tmp_path,
):
    class HugeGeneratedEmbedder(StrictCountingEmbedder):
        def encode(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [list(HUGE_FLOAT32_VECTOR) for _ in texts]

    cache = DictCache()
    embedder = HugeGeneratedEmbedder("revision-a")
    ranker = DenseRanker(
        embedder,
        Index(corpus[:1]),
        cache,
        index_dir=tmp_path,
        strict_provenance=True,
        expected_embedder_provenance=embedder.provenance,
        expected_embedding_dimension=len(VOCAB),
    )

    ranker.build()

    assert embedder.calls == 1
    assert len(cache.store) == 1
    assert_finite_unit_vector(next(iter(cache.store.values())))


def test_strict_huge_float32_query_is_normalized_before_search_and_legacy_unchanged(
    corpus,
    tmp_path,
):
    class HugeQueryEmbedder(StrictCountingEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return list(HUGE_FLOAT32_VECTOR)

    class CapturingStore:
        vector: list[float] | None = None

        def search(self, vector, pool, allowed_ids):
            self.vector = vector
            return []

    embedder = HugeQueryEmbedder("revision-a")
    strict = DenseRanker(
        embedder,
        Index(corpus),
        DictCache(),
        index_dir=tmp_path,
        strict_provenance=True,
        expected_embedder_provenance=embedder.provenance,
        expected_embedding_dimension=len(VOCAB),
    )
    strict.build()
    capture = CapturingStore()
    strict._store = capture

    assert strict.rank("huge query") == []
    assert capture.vector is not None
    assert_finite_unit_vector(capture.vector)

    legacy = build_ranker(corpus)
    assert legacy._validated_vector(
        HUGE_FLOAT32_VECTOR,
        label="legacy huge vector",
    ) == HUGE_FLOAT32_VECTOR


def test_strict_index_refuses_content_mismatch_without_overwrite(
    corpus,
    tmp_path,
):
    cache = DictCache()
    first = StrictCountingEmbedder("revision-a")
    DenseRanker(
        first,
        Index(corpus),
        cache,
        index_dir=tmp_path,
        strict_provenance=True,
        expected_embedder_provenance=first.provenance,
        expected_embedding_dimension=len(VOCAB),
    ).build()
    assert not (tmp_path / BUILD_LOCK_FILE).exists()
    protected = {
        filename: (tmp_path / filename).read_bytes()
        for filename in (MANIFEST_FILE, INDEX_FILE, CHUNK_IDS_FILE)
    }

    changed = list(corpus)
    source = changed[0]
    changed[0] = source.model_copy(update={"text": source.text + " changed"})
    second = StrictCountingEmbedder("revision-a")
    ranker = DenseRanker(
        second,
        Index(changed),
        cache,
        index_dir=tmp_path,
        strict_provenance=True,
        expected_embedder_provenance=second.provenance,
        expected_embedding_dimension=len(VOCAB),
    )

    assert first.calls == 1
    with pytest.raises(IndexValidationError, match="corpus projection"):
        ranker.build()
    assert second.calls == 0
    assert protected == {
        filename: (tmp_path / filename).read_bytes()
        for filename in (MANIFEST_FILE, INDEX_FILE, CHUNK_IDS_FILE)
    }


def test_preexisting_strict_build_lock_blocks_before_embedding(corpus, tmp_path):
    lock_path = tmp_path / BUILD_LOCK_FILE
    lock_path.write_text("stale-writer", encoding="ascii")
    embedder = StrictCountingEmbedder("revision-a")
    ranker = DenseRanker(
        embedder,
        Index(corpus),
        DictCache(),
        index_dir=tmp_path,
        strict_provenance=True,
        expected_embedder_provenance=embedder.provenance,
        expected_embedding_dimension=len(VOCAB),
    )

    with pytest.raises(IndexValidationError, match="build lock"):
        ranker.build()

    assert embedder.calls == 0
    assert lock_path.read_text(encoding="ascii") == "stale-writer"
    assert not (tmp_path / INDEX_FILE).exists()
    assert not (tmp_path / CHUNK_IDS_FILE).exists()
    assert not (tmp_path / MANIFEST_FILE).exists()


def test_strict_persist_failure_retains_build_lock(
    corpus,
    tmp_path,
    monkeypatch,
):
    from retrieve.rankers import dense as dense_module

    def fail_persist(*args, **kwargs):
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(dense_module, "save_index", fail_persist)
    embedder = StrictCountingEmbedder("revision-a")
    ranker = DenseRanker(
        embedder,
        Index(corpus),
        DictCache(),
        index_dir=tmp_path,
        strict_provenance=True,
        expected_embedder_provenance=embedder.provenance,
        expected_embedding_dimension=len(VOCAB),
    )

    with pytest.raises(OSError, match="simulated persistence failure"):
        ranker.build()

    assert embedder.calls == 1
    assert (tmp_path / BUILD_LOCK_FILE).is_file()
    assert not ranker._built


def test_strict_dense_constructor_requires_complete_expected_contract(corpus):
    embedder = StrictCountingEmbedder("revision-a")
    with pytest.raises(ValueError, match="expected embedder provenance"):
        DenseRanker(
            embedder,
            Index(corpus),
            DictCache(),
            strict_provenance=True,
            expected_embedding_dimension=len(VOCAB),
        )
    with pytest.raises(ValueError, match="expected_embedding_dimension"):
        DenseRanker(
            embedder,
            Index(corpus),
            DictCache(),
            strict_provenance=True,
            expected_embedder_provenance=embedder.provenance,
            expected_embedding_dimension=True,
        )

    with pytest.raises(ValueError, match="requires index_dir"):
        DenseRanker(
            embedder,
            Index(corpus),
            DictCache(),
            strict_provenance=True,
            expected_embedder_provenance=embedder.provenance,
            expected_embedding_dimension=len(VOCAB),
        )


def test_strict_cache_key_changes_when_retrieval_content_changes(corpus, tmp_path):
    cache = DictCache()
    first = StrictCountingEmbedder("revision-a")
    DenseRanker(
        first,
        Index(corpus),
        cache,
        index_dir=tmp_path / "original-content",
        strict_provenance=True,
        expected_embedder_provenance=first.provenance,
        expected_embedding_dimension=len(VOCAB),
    ).build()

    changed = list(corpus)
    source = changed[0]
    changed[0] = source.model_copy(update={"text": source.text + " changed"})
    second = StrictCountingEmbedder("revision-a")
    DenseRanker(
        second,
        Index(changed),
        cache,
        index_dir=tmp_path / "changed-content",
        strict_provenance=True,
        expected_embedder_provenance=second.provenance,
        expected_embedding_dimension=len(VOCAB),
    ).build()

    assert second.calls == 1, "changed retrieval text must not reuse a stale vector"
    assert len(cache.store) == len(corpus) + 1


def test_strict_dense_ranker_rejects_configured_provenance_mismatch(
    corpus,
    tmp_path,
):
    embedder = StrictCountingEmbedder("revision-a")
    expected = {**embedder.provenance, "revision": "revision-b"}
    ranker = DenseRanker(
        embedder,
        Index(corpus),
        DictCache(),
        index_dir=tmp_path,
        strict_provenance=True,
        expected_embedder_provenance=expected,
        expected_embedding_dimension=len(VOCAB),
    )

    with pytest.raises(ValueError, match="immutable pin"):
        ranker.build()


def test_dense_score_ties_preserve_legacy_order_but_are_strictly_deterministic(
    tmp_path,
):
    tied = [
        make_chunk("z", "unseen token", "math"),
        make_chunk("a", "another unseen token", "math"),
    ]

    class OrderedTieStore:
        def search(self, vector, pool, allowed_ids):
            return [("z", 0.0), ("a", 0.0)]

    legacy = build_ranker(tied, fetch_k=2)
    legacy._store = OrderedTieStore()
    legacy._chunks_by_id = {chunk.chunk_id: chunk for chunk in tied}
    legacy._built = True
    assert [
        chunk.chunk_id for chunk in legacy.rank("üçgen")
    ] == ["z", "a"]

    embedder = StrictCountingEmbedder("revision-a")
    strict = DenseRanker(
        embedder,
        Index(tied),
        DictCache(),
        fetch_k=2,
        index_dir=tmp_path,
        strict_provenance=True,
        expected_embedder_provenance=embedder.provenance,
        expected_embedding_dimension=len(VOCAB),
    )
    strict._store = OrderedTieStore()
    strict._chunks_by_id = {chunk.chunk_id: chunk for chunk in tied}
    strict._built = True
    assert [
        chunk.chunk_id for chunk in strict.rank("üçgen")
    ] == ["a", "z"]


def test_persisted_index_rebuilt_when_corpus_changes(corpus, tmp_path):
    first = NamedCountingEmbedder()
    DenseRanker(first, Index(corpus), DictCache(), index_dir=tmp_path).build()

    grown = corpus + [make_chunk("x1", "Hücre zarı.", "biology")]
    second = NamedCountingEmbedder()
    DenseRanker(second, Index(grown), DictCache(), index_dir=tmp_path).build()
    assert second.calls == 1, "изменился состав корпуса — снимок невалиден, пересборка"
