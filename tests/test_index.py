from src.retrieve.index import Index
from src.schemas.retrieve import RetrievedChunk


def chunk(chunk_id: str, subject: str, textbook: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=chunk_id,
        score=0.0,
        metadata={"subject": subject, "textbook": textbook, "page": 1},
    )


def make_index() -> Index:
    return Index([
        chunk("hist:1", "history", "book-a"),
        chunk("hist:2", "history", "book-a"),
        chunk("phys:1", "physics", "book-b"),
    ])


def test_get_returns_everything_without_filters():
    assert {c.chunk_id for c in make_index().get()} == {"hist:1", "hist:2", "phys:1"}


def test_get_filters_by_subject_and_textbook():
    index = make_index()
    assert {c.chunk_id for c in index.get(subject="history")} == {"hist:1", "hist:2"}
    assert {c.chunk_id for c in index.get(textbook="book-b")} == {"phys:1"}


def test_get_by_id_resolves_chunk_or_none():
    index = make_index()
    assert index.get_by_id("phys:1").metadata["subject"] == "physics"
    assert index.get_by_id("missing") is None
