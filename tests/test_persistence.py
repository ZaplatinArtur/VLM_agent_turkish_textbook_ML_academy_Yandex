import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

from src.retrieve.persistence import (
    corpus_fingerprint,
    load_index,
    load_manifest,
    save_index,
)
from src.retrieve.vector_store import FaissVectorStore


def make_store() -> FaissVectorStore:
    return FaissVectorStore.from_vectors(
        ["bookA:1", "bookA:2", "bookB:1"],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )


def test_fingerprint_is_order_independent():
    assert corpus_fingerprint(["a", "b"], "m") == corpus_fingerprint(["b", "a"], "m")


def test_fingerprint_changes_with_corpus_and_embedder():
    base = corpus_fingerprint(["a", "b"], "m")
    assert base != corpus_fingerprint(["a", "b", "c"], "m")  # new chunk
    assert base != corpus_fingerprint(["a", "b"], "other")   # new embedder


def test_manifest_records_counts(tmp_path):
    manifest = save_index(tmp_path, make_store(), "embedder-x")
    assert manifest["n_vectors"] == 3
    assert manifest["n_chunks"] == 3
    assert manifest["n_books"] == 2  # bookA + bookB
    assert manifest["embedder"] == "embedder-x"
    assert load_manifest(tmp_path)["corpus_hash"] == manifest["corpus_hash"]


def test_roundtrip_preserves_search(tmp_path):
    save_index(tmp_path, make_store(), "m")
    loaded = load_index(tmp_path, ["bookA:1", "bookA:2", "bookB:1"], "m")
    assert loaded is not None
    assert loaded.search([0.0, 5.0], 1)[0][0] == "bookA:2"


def test_load_rejects_changed_corpus(tmp_path):
    save_index(tmp_path, make_store(), "m")
    assert load_index(tmp_path, ["bookA:1", "bookA:2"], "m") is None  # a chunk dropped
    assert load_index(tmp_path, ["bookA:1", "bookA:2", "bookB:1"], "other") is None


def test_load_returns_none_without_snapshot(tmp_path):
    assert load_index(tmp_path, ["x"], "m") is None
