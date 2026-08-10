import math

import pytest

from retrieve.embedders import sentence_transformer as st_module
from retrieve.embedders.base import SymmetricTextEmbedder
from retrieve.embedders.sentence_transformer import (
    BGE_M3_SEMANTIC_SMOKE_CONTRACT,
    SentenceTransformerEmbedder,
)
from schemas.retrieve import RetrievedChunk


def make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, score=0.0, metadata={})

class RecordingEmbedder(SymmetricTextEmbedder):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text))] for text in texts]


def test_embed_chunks_empty_returns_empty_without_encoding():
    embedder = RecordingEmbedder()
    assert embedder.embed_chunks([]) == []
    assert embedder.calls == []


def test_embed_chunks_encodes_chunk_texts_in_order():
    embedder = RecordingEmbedder()
    chunks = [
        make_chunk("m1", "üçgen"),
        make_chunk("p1", "hız"),
        make_chunk("b1", "hücre"),
    ]
    vectors = embedder.embed_chunks(chunks)
    assert embedder.calls == [["üçgen", "hız", "hücre"]]
    assert vectors == [[len("üçgen")], [len("hız")], [len("hücre")]]


def test_embed_query_returns_single_vector():
    embedder = RecordingEmbedder()
    vector = embedder.embed_query("kuvvet")
    assert embedder.calls == [["kuvvet"]]
    assert vector == [float(len("kuvvet"))]


class FakeEncoded:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def tolist(self) -> list[list[float]]:
        return self.values


def test_sentence_transformer_is_lazy_and_forwards_immutable_loader_contract(
    monkeypatch,
    tmp_path,
):
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeModel:
        def __init__(self, model_name: str, **kwargs) -> None:
            calls.append((model_name, kwargs))
            self.max_seq_length = None

        def encode(self, texts, **kwargs):
            assert kwargs == {
                "batch_size": 2,
                "convert_to_numpy": True,
                "normalize_embeddings": False,
                "show_progress_bar": False,
            }
            return FakeEncoded([[1.0, float(len(text))] for text in texts])

    monkeypatch.setattr(st_module, "_sentence_transformer_class", lambda: FakeModel)
    embedder = SentenceTransformerEmbedder(
        "BAAI/bge-m3",
        batch_size=2,
        revision="abc123",
        license_id="MIT",
        max_length=1024,
        expected_dimension=2,
        task_contract="symmetric_retrieval_text_v1",
        local_files_only=True,
        cache_dir=tmp_path,
        device="cpu",
        runtime_versions={
            "sentence-transformers": "5.3.0",
            "transformers": "4.57.1",
            "torch": "2.8.0",
        },
    )

    assert embedder._model is None
    assert embedder.encode(["a", "abcd"]) == [[1.0, 1.0], [1.0, 4.0]]
    assert calls == [
        (
            "BAAI/bge-m3",
            {
                "cache_folder": str(tmp_path),
                "device": "cpu",
                "local_files_only": True,
                "revision": "abc123",
                "trust_remote_code": False,
            },
        )
    ]
    assert embedder.model.max_seq_length == 1024
    assert embedder.provenance == {
        "backend": "sentence-transformers",
        "batch_size": 2,
        "device": "cpu",
        "embedding_dimension": 2,
        "encode_normalize_embeddings": False,
        "faiss_index_kind": "auto",
        "license": "mit",
        "max_length": 1024,
        "model_id": "BAAI/bge-m3",
        "revision": "abc123",
        "runtime_packages": {
            "sentence-transformers": "5.3.0",
            "torch": "2.8.0",
            "transformers": "4.57.1",
        },
        "task_contract": "symmetric_retrieval_text_v1",
        "trust_remote_code": False,
        "vector_store_normalization": "l2",
    }


def test_sentence_transformer_rejects_remote_code_and_malformed_vectors():
    with pytest.raises(ValueError, match="remote model code"):
        SentenceTransformerEmbedder(trust_remote_code=True)

    class BadModel:
        def encode(self, texts, **kwargs):
            return FakeEncoded([[float("nan")]])

    embedder = SentenceTransformerEmbedder()
    embedder._model = BadModel()
    with pytest.raises(ValueError, match="malformed embeddings"):
        embedder.encode(["query"])

    wrong_dimension = SentenceTransformerEmbedder(expected_dimension=2)
    wrong_dimension._model = type(
        "OneDimensionalModel",
        (),
        {"encode": lambda self, texts, **kwargs: FakeEncoded([[1.0]])},
    )()
    with pytest.raises(ValueError, match="dimension"):
        wrong_dimension.encode(["query"])


def test_bge_semantic_smoke_runs_once_through_sentence_transformer_encode(
    monkeypatch,
):
    calls: list[list[str]] = []

    class SmokeModel:
        def __init__(self, model_name: str, **kwargs) -> None:
            self.max_seq_length = None

        def encode(self, texts, **kwargs):
            calls.append(list(texts))
            assert self.max_seq_length == 8192
            assert kwargs == {
                "batch_size": 4,
                "convert_to_numpy": True,
                "normalize_embeddings": False,
                "show_progress_bar": False,
            }
            right_one_tail = math.sqrt(1.0 - 0.6260**2 - 0.3499**2)
            right_two_tail = math.sqrt(1.0 - 0.3474**2 - 0.6782**2)
            return FakeEncoded(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.6260, 0.3499, right_one_tail],
                    [0.3474, 0.6782, right_two_tail],
                ]
            )

    monkeypatch.setattr(st_module, "_sentence_transformer_class", lambda: SmokeModel)
    embedder = SentenceTransformerEmbedder(
        "BAAI/bge-m3",
        batch_size=4,
        max_length=1024,
        expected_dimension=3,
        validate_bge_m3_semantics=True,
    )

    assert embedder.model is embedder.model
    assert embedder.model.max_seq_length == 1024
    assert calls == [[
        "What is BGE M3?",
        "Defination of BM25",
        "BGE M3 is an embedding model supporting dense retrieval, lexical "
        "matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of "
        "documents based on the query terms appearing in each document",
    ]]
    assert embedder.runtime_validation is not None
    assert embedder.runtime_validation["passed"] is True
    assert embedder.provenance["semantic_smoke_contract"] == (
        BGE_M3_SEMANTIC_SMOKE_CONTRACT
    )


def test_bge_semantic_smoke_failure_does_not_publish_model(monkeypatch):
    class WrongSmokeModel:
        def __init__(self, model_name: str, **kwargs) -> None:
            self.max_seq_length = None

        def encode(self, texts, **kwargs):
            return FakeEncoded(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.9, 0.1],
                    [0.1, 0.9],
                ]
            )

    monkeypatch.setattr(
        st_module,
        "_sentence_transformer_class",
        lambda: WrongSmokeModel,
    )
    embedder = SentenceTransformerEmbedder(
        "BAAI/bge-m3",
        expected_dimension=2,
        validate_bge_m3_semantics=True,
    )

    with pytest.raises(RuntimeError, match="official semantic smoke test"):
        _ = embedder.model
    assert embedder._model is None
    assert embedder.runtime_validation is None
