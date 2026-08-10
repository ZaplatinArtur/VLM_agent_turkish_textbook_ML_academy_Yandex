from schemas.retrieve import RetrievedChunk

from retrieve.knowledge import KnowledgeBaseBuilder


def _chunk(
    chunk_id: str,
    text: str,
    kind: str,
    index: int,
    *,
    exercise_id: str | None = None,
    page: int = 10,
) -> RetrievedChunk:
    metadata = {
        "textbook": "7-sinif-matematik-test",
        "subject": "math",
        "grade": 7,
        "page": page,
        "source_page": page,
        "parent_chunk_id": f"book:{page:04d}",
        "unit_index": index,
        "unit_kind": kind,
        "section_title": "Dikdörtgenin alanı",
    }
    if exercise_id:
        metadata["exercise_id"] = exercise_id
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=0.0,
        metadata=metadata,
    )


def test_exercise_card_contains_theory_example_and_linked_solution() -> None:
    chunks = [
        _chunk(
            "t1",
            "Dikdörtgenin alanı kısa kenar ile uzun kenarın çarpımıdır.",
            "theory",
            0,
        ),
        _chunk(
            "w1",
            "Örnek: Kenarları 3 ve 4 olan dikdörtgenin alanı 12 olur.",
            "worked_example",
            1,
        ),
        _chunk(
            "e1",
            "Kenarları 5 ve 8 santimetre olan dikdörtgenin alanını bulunuz.",
            "exercise",
            2,
        ),
        _chunk(
            "s1",
            "Çözüm: 5 çarpı 8 eşittir 40 santimetrekare.",
            "solution",
            3,
            exercise_id="e1",
        ),
    ]

    cards = KnowledgeBaseBuilder(min_retrieval_chars=20).build(chunks)
    exercise = next(card for card in cards if card.metadata["knowledge_kind"] == "exercise")

    assert "[THEORY]" in exercise.text
    assert "[WORKED EXAMPLE]" in exercise.text
    assert "[SIMILAR EXERCISE]" in exercise.text
    assert "[SOLUTION]" in exercise.text
    assert exercise.metadata["exercise_chunk_id"] == "e1"
    assert exercise.metadata["solution_chunk_ids"] == ["s1"]
    assert exercise.metadata["retrieval_text"].endswith(chunks[2].text)


def test_short_ocr_fragments_are_not_standalone_cards() -> None:
    chunks = [
        _chunk("tiny", "A A A", "theory", 0),
        _chunk(
            "real",
            "Bir üçgenin iç açılarının toplamı yüz seksen derecedir ve bu temel bir kuraldır.",
            "theory",
            1,
        ),
    ]

    cards = KnowledgeBaseBuilder(min_retrieval_chars=40).build(chunks)

    assert len(cards) == 1
    assert "temel bir kuraldır" in cards[0].text


def test_theory_units_on_same_page_are_merged_into_one_card() -> None:
    chunks = [
        _chunk("t1", "Doğal sayılar sıfırdan başlar.", "theory", 0),
        _chunk("t2", "Toplama işlemi değişme özelliğine sahiptir.", "theory", 1),
    ]

    cards = KnowledgeBaseBuilder(min_retrieval_chars=20).build(chunks)

    assert len(cards) == 1
    assert cards[0].metadata["source_chunk_ids"] == ["t1", "t2"]


def test_links_solution_from_next_page_by_task_number() -> None:
    exercise = _chunk(
        "e7",
        "7. Kenar uzunlukları verilen dikdörtgenin alanını bulunuz.",
        "exercise",
        0,
        page=10,
    )
    exercise.metadata["task_number"] = "7"
    solution = _chunk(
        "s7",
        "7. Çözüm: Kenar uzunlukları çarpılarak sonuç 48 bulunur.",
        "solution",
        0,
        page=11,
    )
    solution.metadata["task_number"] = "7"

    cards = KnowledgeBaseBuilder(min_retrieval_chars=20).build(
        [exercise, solution]
    )
    card = next(card for card in cards if card.metadata["knowledge_kind"] == "exercise")

    assert card.metadata["solution_chunk_ids"] == ["s7"]
    assert card.metadata["solution_link_methods"] == ["task_number"]
