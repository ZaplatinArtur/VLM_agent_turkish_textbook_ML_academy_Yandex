import pytest

pytest.importorskip("numpy")
pytest.importorskip("sentence_transformers")

import numpy as np
from retrieve.embedders import sentence_transformer as st_module
from retrieve.embedders.base import AsymmetricTextEmbedder, SymmetricTextEmbedder
from retrieve.embedders.sentence_transformer import (
    E5_SMALL_MODEL,
    QWEN3_EMBEDDING_MODEL,
    E5Embedder,
    PlainEmbedder,
    Qwen3Embedder,
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


class RecordingE5(E5Embedder):
    """E5 без загрузки модели: проверяем только, что уходит в encode."""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0] for _ in texts]


def test_e5_applies_query_and_passage_prefixes():
    # Модели e5 обучены с этими префиксами: без них падает качество,
    # а несовпадение префиксов запроса и документа ломает поиск целиком.
    embedder = RecordingE5(E5_SMALL_MODEL)
    embedder.embed_query("üçgen")
    embedder.embed_chunks([make_chunk("m1", "üçgen alanı")])
    assert embedder.calls == [["query: üçgen"], ["passage: üçgen alanı"]]


class RecordingQwen3(Qwen3Embedder):
    """Qwen3 без загрузки модели: смотрим только на то, что уходит в encode."""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0] for _ in texts]


def test_qwen3_puts_the_instruction_on_the_query_only():
    # У Qwen3-Embedding инструкция задачи приписывается к запросу, а документ
    # кодируется как есть; префикс у документа испортил бы совпадение.
    embedder = RecordingQwen3(QWEN3_EMBEDDING_MODEL)
    embedder.embed_query("üçgen")
    embedder.embed_chunks([make_chunk("m1", "üçgen alanı")])
    query_text = embedder.calls[0][0]
    assert query_text.startswith("Instruct: ")
    assert query_text.endswith("\nQuery:üçgen")
    assert embedder.calls[1] == ["üçgen alanı"]


def test_e5_is_asymmetric_and_plain_embedder_is_not():
    assert isinstance(RecordingE5(E5_SMALL_MODEL), AsymmetricTextEmbedder)
    assert issubclass(PlainEmbedder, SymmetricTextEmbedder)


def test_model_name_is_required_and_kept():
    # Модель — параметр, а не подкласс: имя попадает в namespace кэша и
    # в манифест снапшота, поэтому оно обязано доезжать без изменений.
    assert PlainEmbedder("BAAI/bge-m3").model_name == "BAAI/bge-m3"
    assert E5Embedder(E5_SMALL_MODEL).model_name == E5_SMALL_MODEL
    with pytest.raises(TypeError):
        PlainEmbedder()
