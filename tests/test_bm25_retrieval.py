import pytest

pytest.importorskip("bm25s")
pytest.importorskip("Stemmer")

from retrieve.rankers.bm25 import BM25Ranker, fold_case
from schemas.retrieve import RetrievedChunk


class StubIndex:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def get(
        self,
        subject: str | None = None,
        grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        chunks = self._chunks
        if subject is not None:
            chunks = [
                chunk
                for chunk in chunks
                if chunk.metadata.get("subject") == subject
            ]
        if grade is not None:
            chunks = [
                chunk
                for chunk in chunks
                if str(chunk.metadata.get("grade")) == str(grade)
            ]
        return chunks


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


def test_turkish_case_folding_follows_the_language():
    # casefold() дал бы "işik" и "ki" + комбинирующая точка + "tap":
    # в турецком I → ı, İ → i, иначе заглавные формы не совпадут со строчными.
    assert fold_case("IŞIK") == "ışık"
    assert fold_case("KİTAP") == "kitap"
    assert fold_case("İLK") == "ilk"
    # для английского правило другое: "I" остаётся "i"
    assert fold_case("I AM", language="english") == "i am"


def test_uppercase_query_finds_lowercase_text():
    results = build().rank("ÜÇGEN ALAN")
    assert results and results[0].chunk_id == "d1"


def test_inflected_forms_match_through_the_stemmer():
    # kitaplarımızdan -> kitap: без стемминга это разные термы и совпадения нет.
    books = [make_chunk("b1", "kitaplarımızdan öğrendiklerimiz", "turkce")]
    results = BM25Ranker(index=StubIndex(books)).rank("kitap")
    assert [c.chunk_id for c in results] == ["b1"]


def test_most_lexically_relevant_chunk_ranks_first():
    results = build().rank("üçgen alan")
    assert results[0].chunk_id == "d1"  # оба термина, тогда как у d2 только "üçgen"
    assert results[0].score > 0


def test_subject_filter_restricts_candidates():
    results = build().rank("kuvvet hız", subject="physics")
    assert {c.chunk_id for c in results} <= {"d3", "d4"}


def test_grade_filter_restricts_candidates():
    chunks = [
        make_chunk("m7", "triangle area", "math").model_copy(
            update={"metadata": {"subject": "math", "grade": 7}}
        ),
        make_chunk("m8", "triangle area", "math").model_copy(
            update={"metadata": {"subject": "math", "grade": 8}}
        ),
    ]
    results = BM25Ranker(index=StubIndex(chunks)).rank("triangle", grade=8)
    assert [chunk.chunk_id for chunk in results] == ["m8"]


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
