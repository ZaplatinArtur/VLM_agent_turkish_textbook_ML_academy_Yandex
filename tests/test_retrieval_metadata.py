import json

from retrieve.index import Index
from retrieve.metadata import (
    canonical_subject,
    enrich_chunk_metadata,
    infer_textbook_metadata,
)
from retrieve.parsing.chunk_store.jsonl import JsonlChunkStore
from schemas.retrieve import RetrievedChunk


def _chunk(textbook: str, **metadata) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{textbook}:0001",
        text="örnek",
        score=0.0,
        metadata={"textbook": textbook, "page": 1, **metadata},
    )


def test_infers_grade_and_subject_from_real_corpus_slug() -> None:
    metadata = infer_textbook_metadata(
        "7-sinif-matematik-ders-kitabi-cevaplari-meb-yayinlari"
    )

    assert metadata == {"grade": 7, "subject": "math"}


def test_normalizes_turkish_and_validation_subject_names() -> None:
    assert canonical_subject("Matematik") == "math"
    assert canonical_subject("ATATÜRKÇÜLÜK") == "history"
    assert canonical_subject("Turkish language and literature") == (
        "turkish language and literature"
    )


def test_normalizes_tumlu_subject_aliases_to_corpus_names() -> None:
    assert canonical_subject("Maths") == "math"
    assert canonical_subject("Native L&L") == "turkish language and literature"
    assert canonical_subject("Religion and Ethics") == (
        "religious culture and ethics"
    )


def test_enriched_chunks_support_english_subject_filter() -> None:
    math = enrich_chunk_metadata(
        _chunk("6-sinif-matematik-ders-kitabi-cevaplari-meb-yayinlari")
    )
    science = enrich_chunk_metadata(
        _chunk("6-sinif-fen-bilimleri-ders-kitabi-cevaplari-meb-yayinlari")
    )

    found = Index([math, science]).get(subject="Mathematics")

    assert [chunk.chunk_id for chunk in found] == [math.chunk_id]
    assert found[0].metadata["grade"] == 6


def test_jsonl_store_enriches_legacy_chunk_metadata(tmp_path) -> None:
    root = tmp_path / "jsonl"
    root.mkdir()
    chunk = _chunk("8-sinif-inkilap-tarihi-ders-kitabi-cevaplari-meb-yayinlari")
    (root / "book.jsonl").write_text(
        json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    loaded = JsonlChunkStore(root).load()

    assert loaded[0].metadata["subject"] == "history"
    assert loaded[0].metadata["grade"] == 8
