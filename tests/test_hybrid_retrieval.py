from schemas.retrieve import RetrievedChunk

from retrieve.index import Index
from retrieve.rankers.lexical import BM25Ranker
from retrieve.rankers.rerank import KnowledgeReranker


def _card(
    chunk_id: str,
    retrieval_text: str,
    *,
    subject: str = "math",
    grade: int = 7,
    has_solution: bool = False,
    score: float = 0.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"[CARD]\n{retrieval_text}",
        score=score,
        metadata={
            "retrieval_text": retrieval_text,
            "subject": subject,
            "grade": grade,
            "knowledge_kind": "exercise",
            "has_solution": has_solution,
            "has_theory": True,
        },
    )


def test_bm25_uses_retrieval_text_and_metadata_filters() -> None:
    cards = [
        _card("m7", "dikdörtgen alan kısa kenar uzun kenar", grade=7),
        _card("m6", "dikdörtgen alan kısa kenar uzun kenar", grade=6),
        _card("p7", "kuvvet ivme newton", subject="physics", grade=7),
    ]

    results = BM25Ranker(Index(cards)).rank(
        "dikdörtgen alan",
        subject="math",
        grade=7,
    )

    assert [card.chunk_id for card in results] == ["m7"]


def test_quality_reranker_penalizes_tiny_fragment_and_rewards_complete_card() -> None:
    tiny = _card("tiny", "alan", score=1.0)
    complete = _card(
        "complete",
        "dikdörtgen alan kısa kenar uzun kenar çarpımı",
        has_solution=True,
        score=0.7,
    )

    results = KnowledgeReranker().rank(
        "dikdörtgen alan nasıl bulunur",
        [tiny, complete],
    )

    assert results[0].chunk_id == "complete"


def test_semantic_relevance_beats_graph_connectivity() -> None:
    irrelevant = _card(
        "connected-but-irrelevant",
        "triangle circle rectangle arithmetic unrelated classroom material",
        has_solution=True,
        score=1.0,
    )
    irrelevant.metadata["has_example"] = True
    relevant = RetrievedChunk(
        chunk_id="relevant-theory",
        text="Integral derivative limit theorem " * 4,
        score=0.9,
        metadata={
            "retrieval_text": "Integral derivative limit theorem " * 4,
            "subject": "math",
            "grade": 12,
            "knowledge_kind": "theory",
            "has_solution": False,
            "has_theory": False,
            "has_example": False,
        },
    )

    results = KnowledgeReranker().rank(
        "integral derivative limit theorem",
        [irrelevant, relevant],
    )

    assert results[0].chunk_id == "relevant-theory"


def test_reranker_drops_short_graph_theory_anchor() -> None:
    fragment = RetrievedChunk(
        chunk_id="fragment",
        text="FİTRE\ndir.",
        score=1.0,
        metadata={
            "retrieval_text": "FİTRE\ndir.",
            "unit_kind": "theory",
            "knowledge_graph_node": True,
        },
    )
    complete = RetrievedChunk(
        chunk_id="complete",
        text="Fitre, temel ihtiyaçların dışında yeterli mala sahip olanların verdiği sadakadır.",
        score=0.5,
        metadata={
            "retrieval_text": (
                "Fitre, temel ihtiyaçların dışında yeterli mala sahip olanların "
                "verdiği sadakadır."
            ),
            "unit_kind": "theory",
            "knowledge_graph_node": True,
        },
    )

    results = KnowledgeReranker().rank("fitre nedir", [fragment, complete])

    assert [result.chunk_id for result in results] == ["complete"]


def test_short_graph_theory_filter_can_be_disabled() -> None:
    fragment = RetrievedChunk(
        chunk_id="fragment",
        text="FİTRE",
        score=1.0,
        metadata={
            "retrieval_text": "FİTRE",
            "unit_kind": "theory",
            "knowledge_graph_node": True,
        },
    )

    results = KnowledgeReranker(min_graph_theory_chars=0).rank(
        "fitre",
        [fragment],
    )

    assert [result.chunk_id for result in results] == ["fragment"]


def test_short_graph_theory_filter_keeps_exact_boundary() -> None:
    below = RetrievedChunk(
        chunk_id="below",
        text="x" * 69,
        score=1.0,
        metadata={
            "retrieval_text": "x" * 69,
            "unit_kind": "theory",
            "knowledge_graph_node": True,
        },
    )
    boundary = RetrievedChunk(
        chunk_id="boundary",
        text="x" * 70,
        score=0.9,
        metadata={
            "retrieval_text": "x" * 70,
            "unit_kind": "theory",
            "knowledge_graph_node": True,
        },
    )

    results = KnowledgeReranker(min_graph_theory_chars=70).rank(
        "x",
        [below, boundary],
    )

    assert [result.chunk_id for result in results] == ["boundary"]


def test_short_theory_filter_is_scoped_to_graph_nodes() -> None:
    page_theory = RetrievedChunk(
        chunk_id="page-theory",
        text="Kısa tanım",
        score=1.0,
        metadata={
            "retrieval_text": "Kısa tanım",
            "unit_kind": "theory",
        },
    )
    graph_example = RetrievedChunk(
        chunk_id="graph-example",
        text="Kısa örnek",
        score=0.9,
        metadata={
            "retrieval_text": "Kısa örnek",
            "unit_kind": "worked_example",
            "knowledge_graph_node": True,
        },
    )

    results = KnowledgeReranker(min_graph_theory_chars=70).rank(
        "kısa",
        [page_theory, graph_example],
    )

    assert {result.chunk_id for result in results} == {
        "page-theory",
        "graph-example",
    }
