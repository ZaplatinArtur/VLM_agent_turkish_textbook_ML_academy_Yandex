import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

from src.retrieve.embedders.base import SymmetricTextEmbedder
from src.retrieve.rankers.dense import DenseRanker
from src.schemas.retrieve import RetrievedChunk

VOCAB = ["üçgen", "alan", "hız", "kuvvet", "hücre"]

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