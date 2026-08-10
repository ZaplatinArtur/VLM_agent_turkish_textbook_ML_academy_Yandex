from visual_retrive.manifest import clean_answer_text
from visual_retrive.query_gen import heuristic_queries, neighbor_hard_negatives


def test_clean_answer_text_strips_boilerplate() -> None:
    text = "Soru 1: Kesir nedir?\nÇözüm: 1/2\nHenüz onaylanmış öğrenci çözümü yok."
    cleaned = clean_answer_text(text)
    assert "Kesir" in cleaned
    assert "öğrenci çözümü" not in cleaned.casefold()


def test_heuristic_queries_are_short_and_turkishish() -> None:
    bundle = {
        "page_id": "5-sinif-matematik-x:0020",
        "book_slug": "5-sinif-matematik-ders-kitabi-cevaplari-meb-yayinlari",
        "page_number": 20,
        "grade": 5,
        "subject": "math",
        "answer_text": (
            "Soru 1:\nKesirlerde payda eşitleme nasıl yapılır?\n"
            "Çözüm: payda eşitleyerek toplama yapılır."
        ),
        "page_image": "books/x/pages/0020.jpg",
        "has_solution": True,
    }
    queries = heuristic_queries(bundle, n_queries=3)
    assert queries
    assert all(len(query.split()) <= 16 for query in queries)
    assert any("matematik" in query.casefold() or "kesir" in query.casefold() for query in queries)


def test_neighbor_hard_negatives_prefers_nearby_pages() -> None:
    bundles = [
        {
            "page_id": f"book:{i:04d}",
            "book_slug": "book",
            "page_number": i,
            "page_image": f"books/book/pages/{i:04d}.jpg",
        }
        for i in (18, 19, 20, 21, 50)
    ]
    by_book = {"book": bundles}
    center = bundles[2]
    negatives = neighbor_hard_negatives(center, by_book, k=3)
    assert negatives == ["book:0019", "book:0021", "book:0018"]
