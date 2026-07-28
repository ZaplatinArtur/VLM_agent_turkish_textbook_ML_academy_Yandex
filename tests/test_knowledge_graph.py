from pathlib import Path

from schemas.retrieve import RetrievedChunk

from retrieve.graph import KnowledgeGraph, KnowledgeGraphBuilder, RelationType
from retrieve.rankers.graph import GraphExpansionRanker


def _node(
    chunk_id: str,
    text: str,
    kind: str,
    index: int,
    *,
    page: int = 10,
    task_number: str | None = None,
    exercise_id: str | None = None,
    score: float = 0.0,
) -> RetrievedChunk:
    metadata = {
        "textbook": "7-sinif-matematik",
        "subject": "math",
        "grade": 7,
        "page": page,
        "source_page": page,
        "unit_index": index,
        "unit_kind": kind,
        "section_title": "Dikdörtgenin alanı",
    }
    if task_number is not None:
        metadata["task_number"] = task_number
    if exercise_id is not None:
        metadata["exercise_id"] = exercise_id
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata=metadata,
    )


def _sample_nodes() -> list[RetrievedChunk]:
    return [
        _node(
            "theory",
            "Dikdörtgenin alanı kısa kenar ile uzun kenarın çarpımıdır. "
            "Sonuç uygun kare birimiyle yazılır.",
            "theory",
            0,
        ),
        _node(
            "example",
            "Örnek: Kenarları 3 ve 4 olan dikdörtgenin alanı 12 olur.",
            "worked_example",
            1,
        ),
        _node(
            "exercise",
            "Kenarları 5 ve 8 olan dikdörtgenin alanını bulunuz.",
            "exercise",
            2,
            task_number="7",
            score=0.9,
        ),
        _node(
            "solution",
            "Çözüm: 5 çarpı 8 eşittir 40 santimetrekare.",
            "solution",
            3,
            exercise_id="exercise",
        ),
    ]


def test_builds_typed_relations_and_expands_exercise() -> None:
    graph = KnowledgeGraphBuilder().build(_sample_nodes())

    assert len(graph.outgoing("exercise", RelationType.THEORY_FOR)) == 1
    assert len(graph.outgoing("exercise", RelationType.WORKED_EXAMPLE_FOR)) == 1
    assert len(graph.outgoing("exercise", RelationType.SOLUTION_OF)) == 1

    bundle = graph.bundle("exercise", score=0.75)

    assert "[THEORY]" in bundle.text
    assert "[WORKED EXAMPLE]" in bundle.text
    assert "[SIMILAR EXERCISE]" in bundle.text
    assert "[SOLUTION]" in bundle.text
    assert bundle.score == 0.75
    assert bundle.metadata["has_theory"] is True
    assert bundle.metadata["has_example"] is True
    assert bundle.metadata["has_solution"] is True
    assert {
        path["relation"] for path in bundle.metadata["graph_paths"]
    } == {
        "theory_for",
        "worked_example_for",
        "solution_of",
    }


def test_links_solution_on_next_page_by_task_number() -> None:
    exercise = _node(
        "exercise",
        "7. Dikdörtgenin alanını bulunuz.",
        "exercise",
        0,
        page=10,
        task_number="7",
    )
    solution = _node(
        "solution",
        "7. Çözüm: Alan 48 birimkaredir.",
        "solution",
        0,
        page=11,
        task_number="7",
    )

    graph = KnowledgeGraphBuilder().build([exercise, solution])
    edge = graph.outgoing("exercise", RelationType.SOLUTION_OF)[0]

    assert edge.target_id == "solution"
    assert edge.method == "task_number"
    assert edge.confidence == 0.9


def test_low_confidence_nearest_solution_stays_in_graph_but_not_agent_bundle() -> None:
    exercise = _node(
        "exercise",
        "Dikdörtgenin alanını bulunuz.",
        "exercise",
        0,
    )
    solution = _node(
        "solution",
        "Çözüm olduğu tahmin edilen komşu metin.",
        "solution",
        1,
    )

    graph = KnowledgeGraphBuilder().build([exercise, solution])
    edge = graph.outgoing("exercise", RelationType.SOLUTION_OF)[0]
    bundle = graph.bundle("exercise")

    assert edge.method == "nearest_exercise"
    assert edge.confidence == 0.58
    assert "[SOLUTION]" not in bundle.text
    assert bundle.metadata["has_solution"] is False


def test_does_not_attach_boilerplate_as_theory() -> None:
    nodes = [
        _node(
            "copyright",
            "ISBN 978-000. Her hakkı saklıdır. Yayın basım dağıtım şirketi.",
            "theory",
            0,
        ),
        _node(
            "exercise",
            "Bir sayının iki katını bulunuz ve sonucu açıklayınız.",
            "exercise",
            1,
        ),
    ]

    graph = KnowledgeGraphBuilder().build(nodes)

    assert graph.outgoing("exercise", RelationType.THEORY_FOR) == []


def test_graph_expansion_preserves_upstream_relevance_order() -> None:
    nodes = _sample_nodes()
    graph = KnowledgeGraphBuilder().build(nodes)
    ranked = [
        nodes[0].model_copy(update={"score": 1.0}),
        nodes[2].model_copy(update={"score": 0.8}),
    ]

    results = GraphExpansionRanker(
        graph,
        include_solutions=True,
    ).rank("dikdörtgen alan", ranked)

    assert results[0].metadata["graph_anchor_id"] == "theory"
    assert results[1].metadata["graph_anchor_id"] == "exercise"
    assert "[SOLUTION]" in results[1].text
    assert results[1].metadata["has_solution"] is True


def test_agent_graph_bundle_keeps_anchor_first_and_excludes_solutions() -> None:
    nodes = _sample_nodes()
    graph = KnowledgeGraphBuilder().build(nodes)
    ranked = [nodes[2].model_copy(update={"score": 0.8})]

    result = GraphExpansionRanker(graph).rank(
        "dikdörtgen alan",
        ranked,
    )[0]

    assert result.text.startswith("[SIMILAR EXERCISE]")
    assert "[SOLUTION]" not in result.text
    assert result.metadata["has_solution"] is False
    assert result.metadata["bundle_anchor_first"] is True
    assert result.metadata["bundle_includes_solutions"] is False


def test_expansions_cannot_consume_the_anchor_budget() -> None:
    theory = _node("long-theory", "T" * 2_000, "theory", 0)
    exercise = _node(
        "long-exercise",
        "A" * 1_800,
        "exercise",
        1,
        score=0.9,
    )
    graph = KnowledgeGraphBuilder().build([theory, exercise])

    bundle = graph.bundle(
        "long-exercise",
        max_chars=1_900,
        include_solutions=False,
    )

    assert bundle.text.startswith("[SIMILAR EXERCISE]\n" + "A" * 1_800)
    anchor_part, theory_part = bundle.text.split("\n\n[THEORY]\n", 1)
    assert anchor_part.split("\n", 1)[1] == "A" * 1_800
    assert theory_part == "T" * 100


def test_graph_round_trip(tmp_path: Path) -> None:
    graph = KnowledgeGraphBuilder().build(_sample_nodes())

    manifest = graph.save(tmp_path)
    loaded = KnowledgeGraph.load(tmp_path)

    assert manifest["nodes"] == 4
    assert manifest["edges"] == 4
    assert loaded.relation_counts() == graph.relation_counts()
    assert loaded.bundle("exercise").metadata["has_solution"] is True
