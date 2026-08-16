from schemas.retrieve import RetrievedChunk

from retrieve.ingest.chunking import EducationalChunker, UnitKind


def _page(text: str, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"book:{page:04d}",
        text=text,
        images=[],
        score=0.0,
        metadata={"textbook": "6-sinif-matematik-test", "page": page},
    )


def test_splits_theory_exercise_and_solution() -> None:
    page = _page(
        """4. ÜNİTE
Maddenin Hâl Değişimi

Maddeler ısındıkça ya da soğudukça bir hâlden başka bir hâle geçebilir.

ARAŞTIRALIM
Kutup ayılarının yaşam alanlarının neden daraldığını araştırınız.

Çözüm:
Küresel ısınma nedeniyle buzullar erimektedir."""
    )

    units = EducationalChunker().segment(page)

    assert [unit.kind for unit in units] == [
        UnitKind.THEORY,
        UnitKind.EXERCISE,
        UnitKind.SOLUTION,
    ]
    assert units[2].metadata["exercise_id"] == units[1].unit_id


def test_keeps_answer_options_with_numbered_exercise() -> None:
    page = _page(
        """1. Aşağıdakilerden hangisi doğrudur?
A) Birinci seçenek
B) İkinci seçenek
C) Üçüncü seçenek

2. Sonucu hesaplayınız: 2 + 2 = ?"""
    )

    units = EducationalChunker().segment(page)

    assert [unit.kind for unit in units] == [UnitKind.EXERCISE, UnitKind.EXERCISE]
    assert "A) Birinci seçenek" in units[0].text
    assert units[0].task_number == "1"
    assert units[1].task_number == "2"


def test_theory_page_remains_one_parent_aware_unit() -> None:
    page = _page(
        """Maddenin Hâl Değişimi

Katı bir madde ısı aldığında sıvı hâle geçebilir.

Bu olaya erime adı verilir."""
    )

    units = EducationalChunker().segment(page)
    chunks = EducationalChunker().chunk_page(page)

    assert len(units) == 1
    assert units[0].kind is UnitKind.THEORY
    assert chunks[0].metadata["parent_chunk_id"] == page.chunk_id
    assert chunks[0].metadata["unit_kind"] == "theory"


def test_unit_ids_are_stable() -> None:
    page = _page("Soru: 10 ile 5 sayısının toplamı kaçtır?")
    chunker = EducationalChunker()

    first = chunker.segment(page)
    second = chunker.segment(page)

    assert first[0].unit_id == second[0].unit_id


def test_detects_english_instruction_as_exercise() -> None:
    page = _page(
        "THEME 3\n\nImagine you are a cartoon character. "
        "Write a short paragraph about your daily life."
    )

    units = EducationalChunker().segment(page)

    assert units[-1].kind is UnitKind.EXERCISE


def test_splits_inline_numbered_question_after_theory() -> None:
    page = _page(
        "Healthy nutrition means eating enough food. "
        "1. Aşağıdaki tabloyu inceleyiniz ve soruları cevaplayınız."
    )

    units = EducationalChunker().segment(page)

    assert [unit.kind for unit in units] == [UnitKind.THEORY, UnitKind.EXERCISE]


def test_detects_student_imperative_without_clean_ocr_prefix() -> None:
    page = _page(
        "HGZIHIZ, 2. Yandaki saatin 5'i göstermesi için saatin "
        "akrep ve yelkovanını uygun şekilde çiziniz."
    )

    units = EducationalChunker().segment(page)

    assert units[0].kind is UnitKind.EXERCISE


def test_marks_isolated_page_number_as_other() -> None:
    page = _page("— 239 —")

    units = EducationalChunker().segment(page)

    assert units[0].kind is UnitKind.OTHER


def test_does_not_swallow_long_theory_after_a_question() -> None:
    page = _page(
        "Kur'an-ı Kerim nasıl bir kitaptır?\n\n"
        + (
            "Kur'an-ı Kerim insanların doğruyu öğrenmesi için gönderilen kutsal "
            "kitaptır. Bu bölümde kitabın temel özellikleri açıklanmaktadır. "
        )
        * 4
    )

    units = EducationalChunker().segment(page)

    assert [unit.kind for unit in units] == [UnitKind.EXERCISE, UnitKind.THEORY]
