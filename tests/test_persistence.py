import json
import stat
from types import SimpleNamespace

import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

from retrieve.storage import persistence as persistence_module
from retrieve.storage.persistence import (
    BUILD_LOCK_FILE,
    IndexValidationError,
    MANIFEST_FILE,
    corpus_fingerprint,
    load_index,
    load_manifest,
    retrieval_corpus_projection_sha256,
    save_index,
)
from retrieve.storage.vector_store import CHUNK_IDS_FILE, INDEX_FILE, FaissVectorStore
from schemas.retrieve import RetrievedChunk


def make_store() -> FaissVectorStore:
    return FaissVectorStore.from_vectors(
        ["bookA:1", "bookA:2", "bookB:1"],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )


def make_strict_store() -> FaissVectorStore:
    return FaissVectorStore.from_vectors(
        ["bookA:1", "bookA:2", "bookB:1"],
        [
            [1.0] + [0.0] * 1023,
            [0.0, 1.0] + [0.0] * 1022,
            [1.0, 1.0] + [0.0] * 1022,
        ],
    )


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    subject: str = "math",
    grade: int = 9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=0.0,
        metadata={
            "grade": grade,
            "subject": subject,
            "textbook": "book-a",
        },
    )


STRICT_PROVENANCE = {
    "backend": "sentence-transformers",
    "embedding_dimension": 1024,
    "encode_normalize_embeddings": False,
    "faiss_index_kind": "auto",
    "license": "mit",
    "max_length": 1024,
    "model_id": "BAAI/bge-m3",
    "revision": "revision-a",
    "runtime_packages": {
        "sentence-transformers": "5.3.0",
        "torch": "2.8.0",
        "transformers": "4.57.1",
    },
    "task_contract": "symmetric_retrieval_text_v1",
    "trust_remote_code": False,
    "vector_store_normalization": "l2",
}


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


def test_manifest_accepts_explicit_books_for_opaque_graph_node_ids(tmp_path):
    store = FaissVectorStore.from_vectors(
        ["edu_a", "edu_b", "edu_c"],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )

    manifest = save_index(
        tmp_path,
        store,
        "embedder-x",
        book_ids=["textbook-a", "textbook-a", "textbook-b"],
    )

    assert manifest["n_books"] == 2


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


def test_strict_load_allows_a_completely_clean_directory(tmp_path):
    assert load_index(
        tmp_path,
        ["bookA:1"],
        "BAAI/bge-m3",
        embedder_provenance=STRICT_PROVENANCE,
        corpus_projection_sha256="a" * 64,
        require_strict_manifest=True,
    ) is None


def test_strict_load_rejects_nonempty_directory_without_manifest(tmp_path):
    note = tmp_path / "notes.txt"
    note.write_text("operator note", encoding="utf-8")

    with pytest.raises(IndexValidationError, match="unrecognized entries"):
        load_index(
            tmp_path,
            ["bookA:1"],
            "BAAI/bge-m3",
            embedder_provenance=STRICT_PROVENANCE,
            corpus_projection_sha256="a" * 64,
            require_strict_manifest=True,
        )

    assert note.read_text(encoding="utf-8") == "operator note"


def _symlink_or_skip(link, target, *, target_is_directory=False):
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable in this test environment: {exc}")


def test_strict_reparse_guard_is_exercised_without_symlink_privileges(monkeypatch):
    reparse_flag = 0x400
    monkeypatch.setattr(
        persistence_module.stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        reparse_flag,
        raising=False,
    )

    class FakeReparsePath:
        def lstat(self):
            return SimpleNamespace(
                st_mode=stat.S_IFREG,
                st_file_attributes=reparse_flag,
            )

    with pytest.raises(IndexValidationError, match="symlink or reparse"):
        persistence_module._strict_lstat(FakeReparsePath(), "test artifact")


def test_strict_load_rejects_symlinked_index_directory(tmp_path):
    target = tmp_path / "outside-index"
    target.mkdir()
    link = tmp_path / "linked-index"
    _symlink_or_skip(link, target, target_is_directory=True)

    with pytest.raises(IndexValidationError, match="symlink or reparse"):
        load_index(
            link,
            ["bookA:1"],
            "BAAI/bge-m3",
            embedder_provenance=STRICT_PROVENANCE,
            corpus_projection_sha256="a" * 64,
            require_strict_manifest=True,
        )


@pytest.mark.parametrize(
    "filename",
    [MANIFEST_FILE, INDEX_FILE, CHUNK_IDS_FILE],
)
def test_strict_load_rejects_symlinked_known_artifacts(tmp_path, filename):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    target = tmp_path / f"outside-{filename}"
    target.write_bytes(b"outside")
    _symlink_or_skip(index_dir / filename, target)

    with pytest.raises(IndexValidationError, match="symlink or reparse"):
        load_index(
            index_dir,
            ["bookA:1"],
            "BAAI/bge-m3",
            embedder_provenance=STRICT_PROVENANCE,
            corpus_projection_sha256="a" * 64,
            require_strict_manifest=True,
        )


def test_strict_save_rejects_symlink_before_writing_artifacts(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    target = tmp_path / "outside-manifest.json"
    target.write_bytes(b"protected")
    _symlink_or_skip(index_dir / MANIFEST_FILE, target)

    with pytest.raises(IndexValidationError, match="symlink or reparse"):
        save_index(
            index_dir,
            make_strict_store(),
            "BAAI/bge-m3",
            embedder_provenance=STRICT_PROVENANCE,
            corpus_projection_sha256="a" * 64,
        )

    assert target.read_bytes() == b"protected"
    assert not (index_dir / INDEX_FILE).exists()
    assert not (index_dir / CHUNK_IDS_FILE).exists()


def test_content_projection_changes_with_text_and_routing_metadata():
    original = [make_chunk("same-id", "original")]
    changed_text = [make_chunk("same-id", "changed")]
    changed_subject = [make_chunk("same-id", "original", subject="physics")]
    changed_grade = [make_chunk("same-id", "original", grade=10)]

    fingerprint = retrieval_corpus_projection_sha256(original)
    assert fingerprint != retrieval_corpus_projection_sha256(changed_text)
    assert fingerprint != retrieval_corpus_projection_sha256(changed_subject)
    assert fingerprint != retrieval_corpus_projection_sha256(changed_grade)


def test_strict_manifest_roundtrip_binds_provenance_projection_and_artifacts(tmp_path):
    chunks = [
        make_chunk("bookA:1", "one"),
        make_chunk("bookA:2", "two"),
        make_chunk("bookB:1", "three"),
    ]
    projection = retrieval_corpus_projection_sha256(chunks)
    manifest = save_index(
        tmp_path,
        make_strict_store(),
        "BAAI/bge-m3",
        embedder_provenance=STRICT_PROVENANCE,
        corpus_projection_sha256=projection,
    )

    assert manifest["embedder_provenance"] == STRICT_PROVENANCE
    assert manifest["embedding_dimension"] == 1024
    assert manifest["resolved_index_kind"] == "flat"
    assert manifest["corpus_projection_sha256"] == projection
    assert set(manifest["artifacts"]) == {"index.faiss", "chunk_ids.json"}
    assert not (tmp_path / BUILD_LOCK_FILE).exists()
    assert load_index(
        tmp_path,
        [chunk.chunk_id for chunk in chunks],
        "BAAI/bge-m3",
        embedder_provenance=STRICT_PROVENANCE,
        corpus_projection_sha256=projection,
        require_strict_manifest=True,
    ) is not None


def test_strict_save_rejects_actual_index_kind_mismatch(tmp_path):
    hnsw_provenance = {**STRICT_PROVENANCE, "faiss_index_kind": "hnsw"}

    with pytest.raises(ValueError, match="index kind"):
        save_index(
            tmp_path,
            make_strict_store(),
            "BAAI/bge-m3",
            embedder_provenance=hnsw_provenance,
            corpus_projection_sha256="a" * 64,
        )

    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("resolved_index_kind", "hnsw", "resolved FAISS kind"),
        ("index_type", "IndexHNSWFlat", "FAISS type"),
    ],
)
def test_strict_load_rejects_manifest_index_kind_tamper(
    tmp_path,
    field,
    value,
    message,
):
    chunks = [
        make_chunk("bookA:1", "one"),
        make_chunk("bookA:2", "two"),
        make_chunk("bookB:1", "three"),
    ]
    projection = retrieval_corpus_projection_sha256(chunks)
    save_index(
        tmp_path,
        make_strict_store(),
        "BAAI/bge-m3",
        embedder_provenance=STRICT_PROVENANCE,
        corpus_projection_sha256=projection,
    )
    manifest_path = tmp_path / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IndexValidationError, match=message):
        load_index(
            tmp_path,
            [chunk.chunk_id for chunk in chunks],
            "BAAI/bge-m3",
            embedder_provenance=STRICT_PROVENANCE,
            corpus_projection_sha256=projection,
            require_strict_manifest=True,
        )


def test_strict_save_rejects_vector_and_chunk_id_count_mismatch(tmp_path):
    store = make_strict_store()
    store.chunk_ids.append("bookC:1")

    with pytest.raises(ValueError, match="vector count"):
        save_index(
            tmp_path,
            store,
            "BAAI/bge-m3",
            embedder_provenance=STRICT_PROVENANCE,
            corpus_projection_sha256="a" * 64,
        )

    assert not any(tmp_path.iterdir())


def test_strict_load_rejects_faiss_artifact_tamper(tmp_path):
    chunks = [
        make_chunk("bookA:1", "one"),
        make_chunk("bookA:2", "two"),
        make_chunk("bookB:1", "three"),
    ]
    projection = retrieval_corpus_projection_sha256(chunks)
    save_index(
        tmp_path,
        make_strict_store(),
        "BAAI/bge-m3",
        embedder_provenance=STRICT_PROVENANCE,
        corpus_projection_sha256=projection,
    )
    index_path = tmp_path / INDEX_FILE
    index_path.write_bytes(index_path.read_bytes() + b"tamper")

    with pytest.raises(IndexValidationError, match="artifact hash"):
        load_index(
            tmp_path,
            [chunk.chunk_id for chunk in chunks],
            "BAAI/bge-m3",
            embedder_provenance=STRICT_PROVENANCE,
            corpus_projection_sha256=projection,
            require_strict_manifest=True,
        )


def test_strict_load_rejects_manifest_provenance_tamper(tmp_path):
    chunks = [
        make_chunk("bookA:1", "one"),
        make_chunk("bookA:2", "two"),
        make_chunk("bookB:1", "three"),
    ]
    projection = retrieval_corpus_projection_sha256(chunks)
    save_index(
        tmp_path,
        make_strict_store(),
        "BAAI/bge-m3",
        embedder_provenance=STRICT_PROVENANCE,
        corpus_projection_sha256=projection,
    )
    manifest_path = tmp_path / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["embedder_provenance"]["revision"] = "tampered-revision"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IndexValidationError, match="provenance"):
        load_index(
            tmp_path,
            [chunk.chunk_id for chunk in chunks],
            "BAAI/bge-m3",
            embedder_provenance=STRICT_PROVENANCE,
            corpus_projection_sha256=projection,
            require_strict_manifest=True,
        )


def test_strict_load_rejects_provenance_hash_tamper(tmp_path):
    chunks = [
        make_chunk("bookA:1", "one"),
        make_chunk("bookA:2", "two"),
        make_chunk("bookB:1", "three"),
    ]
    projection = retrieval_corpus_projection_sha256(chunks)
    save_index(
        tmp_path,
        make_strict_store(),
        "BAAI/bge-m3",
        embedder_provenance=STRICT_PROVENANCE,
        corpus_projection_sha256=projection,
    )
    manifest_path = tmp_path / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["embedder_provenance_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IndexValidationError, match="provenance hash"):
        load_index(
            tmp_path,
            [chunk.chunk_id for chunk in chunks],
            "BAAI/bge-m3",
            embedder_provenance=STRICT_PROVENANCE,
            corpus_projection_sha256=projection,
            require_strict_manifest=True,
        )


def test_strict_load_rejects_content_projection_mismatch(tmp_path):
    chunks = [
        make_chunk("bookA:1", "one"),
        make_chunk("bookA:2", "two"),
        make_chunk("bookB:1", "three"),
    ]
    projection = retrieval_corpus_projection_sha256(chunks)
    save_index(
        tmp_path,
        make_strict_store(),
        "BAAI/bge-m3",
        embedder_provenance=STRICT_PROVENANCE,
        corpus_projection_sha256=projection,
    )
    changed = list(chunks)
    changed[0] = changed[0].model_copy(update={"text": "changed"})

    with pytest.raises(IndexValidationError, match="corpus projection"):
        load_index(
            tmp_path,
            [chunk.chunk_id for chunk in changed],
            "BAAI/bge-m3",
            embedder_provenance=STRICT_PROVENANCE,
            corpus_projection_sha256=retrieval_corpus_projection_sha256(changed),
            require_strict_manifest=True,
        )


@pytest.mark.parametrize("filename", [MANIFEST_FILE, INDEX_FILE, "chunk_ids.json"])
def test_strict_load_rejects_partial_existing_directory(tmp_path, filename):
    (tmp_path / filename).write_bytes(b"partial")

    with pytest.raises(IndexValidationError, match="partial|malformed"):
        load_index(
            tmp_path,
            ["bookA:1"],
            "BAAI/bge-m3",
            embedder_provenance=STRICT_PROVENANCE,
            corpus_projection_sha256="a" * 64,
            require_strict_manifest=True,
        )
