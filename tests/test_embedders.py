import pytest

pytest.importorskip("numpy")
pytest.importorskip("sentence_transformers")

import numpy as np
from src.retrieve.embedders.base import SymmetricTextEmbedder
from src.retrieve.embedders import sentence_transformer as st_module
from src.retrieve.embedders.sentence_transformer import SentenceTransformerEmbedder
from src.schemas.retrieve import RetrievedChunk


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