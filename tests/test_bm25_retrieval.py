from retrieve.rankers.bm25 import BM25Ranker, tokenize
from schemas.retrieve import RetrievedChunk


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


def corpus() -> list[RetrievedChunk]:
    return [
        make_chunk("d1", "üçgen alan hesabı taban çarpı yükseklik", "math"),
        make_chunk("d2", "üçgen çeşitleri eşkenar ikizkenar çeşitkenar", "math"),
        make_chunk("d3", "hız yol zaman formülü tanımı", "physics"),
        make_chunk("d4", "kuvvet kütle ivme newton yasası", "physics"),
    ]


def build() -> BM25Ranker:
    return BM25Ranker(index=StubIndex(corpus()))


def test_tokenize_drops_single_char_noise_keeps_digits():
    assert tokenize("A üçgen 7 !!") == ["üçgen", "7"]


def test_most_lexically_relevant_chunk_ranks_first():
    results = build().rank("üçgen alan")
    assert results[0].chunk_id == "d1"  # оба термина, тогда как у d2 только "üçgen"
    assert results[0].score > 0


def test_subject_filter_restricts_candidates():
    results = build().rank("kuvvet hız", subject="physics")
    assert {c.chunk_id for c in results} <= {"d3", "d4"}


def test_chunks_subset_restricts_search():
    subset = [c for c in corpus() if c.chunk_id in {"d1", "d3"}]
    results = build().rank("üçgen alan", chunks=subset)
    assert {c.chunk_id for c in results} == {"d1"}


def test_query_without_known_terms_returns_empty():
    assert build().rank("zzz qqq") == []


def test_empty_corpus_returns_empty():
    assert BM25Ranker(index=StubIndex([])).rank("üçgen") == []


def test_original_chunk_scores_not_mutated():
    original = corpus()
    BM25Ranker(index=StubIndex(original)).rank("üçgen alan")
    assert all(c.score == 0.0 for c in original)
