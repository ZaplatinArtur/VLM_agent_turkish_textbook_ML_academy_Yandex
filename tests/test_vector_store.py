import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

import faiss
import numpy as np

from retrieve import vector_store as vs
from retrieve.vector_store import (
    FaissVectorStore,
    IndexKind,
    make_index,
    migrate_index,
)


def make_store(n: int, dim: int = 16, seed: int = 0, **kwargs) -> FaissVectorStore:
    rng = np.random.default_rng(seed)
    vectors = rng.random((n, dim)).astype("float32").tolist()
    return FaissVectorStore.from_vectors([f"c{i}" for i in range(n)], vectors, **kwargs)


def test_auto_picks_exact_index_for_small_corpus():
    index = make_index(16, IndexKind.AUTO, vs.FLAT_MAX_VECTORS)
    assert isinstance(index, faiss.IndexFlat)


def test_auto_switches_to_hnsw_for_large_corpus():
    index = make_index(16, IndexKind.AUTO, vs.FLAT_MAX_VECTORS + 1)
    assert isinstance(index, faiss.IndexHNSWFlat)
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT


def test_explicit_kind_overrides_size():
    assert isinstance(make_index(16, IndexKind.HNSW, size_hint=1), faiss.IndexHNSWFlat)
    assert isinstance(make_index(16, IndexKind.FLAT, size_hint=10**9), faiss.IndexFlat)


def test_environment_can_force_exact_store(monkeypatch):
    monkeypatch.setenv("MLA_FAISS_INDEX_KIND", "flat")
    store = make_store(vs.FLAT_MAX_VECTORS + 1, dim=2)
    assert store.is_exact


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        make_index(16, "quantum")  # type: ignore[arg-type]


def test_migrate_index_preserves_vectors_both_ways():
    rng = np.random.default_rng(3)
    vectors = rng.random((40, 8)).astype("float32")
    flat = make_index(8, IndexKind.FLAT)
    flat.add(vectors)

    hnsw = migrate_index(flat, IndexKind.HNSW)
    assert isinstance(hnsw, faiss.IndexHNSWFlat)
    assert hnsw.ntotal == 40
    back = migrate_index(hnsw, IndexKind.FLAT)
    assert isinstance(back, faiss.IndexFlat)
    np.testing.assert_allclose(back.reconstruct_n(0, 40), vectors, rtol=1e-5)


def test_store_migrate_to_keeps_search_working():
    ids = ["a", "b", "c"]
    store = FaissVectorStore.from_vectors(ids, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert store.is_exact
    store.migrate_to(IndexKind.HNSW)
    assert not store.is_exact
    assert store.search([0.0, 5.0], 1)[0][0] == "b"


def test_search_returns_scores_sorted_descending():
    store = make_store(50)
    hits = store.search([1.0] + [0.0] * 15, 10)
    assert len(hits) == 10
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_query_matching_a_stored_vector_ranks_it_first():
    ids = ["a", "b", "c"]
    store = FaissVectorStore.from_vectors(ids, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert store.search([0.0, 5.0], 1)[0][0] == "b"


def test_k_is_clamped_to_index_size():
    assert len(make_store(3, dim=4).search([1.0, 0.0, 0.0, 0.0], 100)) == 3


def test_empty_and_degenerate_queries_return_nothing():
    store = make_store(5, dim=4)
    assert store.search([1.0, 0.0, 0.0, 0.0], 0) == []
    assert FaissVectorStore.from_vectors([], np.zeros((0, 4), dtype="float32")).search(
        [1.0, 0.0, 0.0, 0.0], 5
    ) == []


def test_allowed_ids_restrict_results():
    store = make_store(50)
    allowed = {"c3", "c7", "c11"}
    hits = store.search([1.0] + [0.0] * 15, 10, allowed)
    assert {cid for cid, _ in hits} <= allowed


def test_allowed_ids_fill_k_from_the_subset_not_from_the_global_top():
    # Ключевое отличие пред-фильтра от пост-фильтра: даже если разрешённые
    # чанки лежат далеко от топа, мы всё равно набираем полные k штук.
    store = make_store(200)
    allowed = {f"c{i}" for i in range(190, 200)}
    hits = store.search([1.0] + [0.0] * 15, 5, allowed)
    assert len(hits) == 5
    assert {cid for cid, _ in hits} <= allowed


def test_unknown_allowed_ids_are_ignored():
    store = make_store(10, dim=4)
    hits = store.search([1.0, 0.0, 0.0, 0.0], 5, {"c1", "nope"})
    assert [cid for cid, _ in hits] == ["c1"]
    assert store.search([1.0, 0.0, 0.0, 0.0], 5, {"nope"}) == []


def test_empty_allowed_ids_returns_nothing():
    assert make_store(10, dim=4).search([1.0, 0.0, 0.0, 0.0], 5, set()) == []


def test_ann_index_agrees_with_exact_search_on_the_top_hit():
    rng = np.random.default_rng(1)
    vectors = rng.random((3000, 32)).astype("float32").tolist()
    ids = [f"c{i}" for i in range(3000)]
    query = rng.random(32).astype("float32").tolist()

    exact = FaissVectorStore.from_vectors(ids, vectors)
    ann = FaissVectorStore.from_vectors(ids, vectors, kind=IndexKind.HNSW)
    assert not ann.is_exact

    exact_top = [cid for cid, _ in exact.search(query, 10)]
    ann_top = [cid for cid, _ in ann.search(query, 10)]
    assert ann_top[0] == exact_top[0]
    assert len(set(ann_top) & set(exact_top)) >= 8  # recall@10 >= 0.8


def test_ann_index_filters_exactly_on_a_selective_subset():
    # HNSW с жёстким селектором проваливается, поэтому стор обязан уйти
    # на точный перебор подмножества — проверяем совпадение с эталоном.
    rng = np.random.default_rng(2)
    vectors = rng.random((3000, 32)).astype("float32").tolist()
    ids = [f"c{i}" for i in range(3000)]
    query = rng.random(32).astype("float32").tolist()
    allowed = {f"c{i}" for i in range(0, 3000, 100)}

    exact = FaissVectorStore.from_vectors(ids, vectors)
    ann = FaissVectorStore.from_vectors(ids, vectors, kind=IndexKind.HNSW)
    assert [cid for cid, _ in ann.search(query, 5, allowed)] == [
        cid for cid, _ in exact.search(query, 5, allowed)
    ]
