from retrieve.build_index import corpus_inventory
from schemas.retrieve import RetrievedChunk


def _chunk(chunk_id: str, subject: str, grade: int, text: str = "text"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=0.0,
        metadata={
            "textbook": chunk_id.split(":", 1)[0],
            "subject": subject,
            "grade": grade,
        },
    )


def test_corpus_inventory_reports_data_quality_and_coverage() -> None:
    inventory = corpus_inventory(
        [
            _chunk("math-book:1", "math", 7),
            _chunk("math-book:1", "math", 7),
            _chunk("science-book:1", "science", 8, text=""),
        ]
    )

    assert inventory["chunks"] == 3
    assert inventory["books"] == 2
    assert inventory["duplicate_chunk_ids"] == 1
    assert inventory["empty_texts"] == 1
    assert inventory["subjects"] == {"math": 2, "science": 1}
