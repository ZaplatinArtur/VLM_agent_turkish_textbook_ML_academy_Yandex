import pytest

from mla_baseline.tools.hybrid_search import HYBRID_PROFILE, HybridSearchClient
from mla_baseline.tools.textbook_search import TextbookSearchError


def branch(ids, *, useful=True, label="confident"):
    class Client:
        def __init__(self):
            self.calls = []

        def search(self, query, *, top_k=5, subject=None, grade=None, mode="or"):
            self.calls.append(top_k)
            return {
                "relevance": {"label": label, "is_useful": useful,
                              "top_score": 1.0, "reason": ""},
                "hits": [{"chunk_id": i, "page_id": i, "text": f"текст {i}",
                          "score": 1.0} for i in ids],
            }

    return Client()


def broken():
    class Client:
        def search(self, *a, **k):
            raise TextbookSearchError("бэкенд лежит")

    return Client()


def test_both_branches_are_queried_with_depth():
    text, visual = branch(["a"]), branch(["b"])
    HybridSearchClient(text, visual, fetch_multiplier=2).search("üçgen alanı", top_k=5)
    assert text.calls == [10] and visual.calls == [10]


def test_agreed_hit_outranks_singletons():
    # b нашли обе ветки — RRF складывает вклады, и он обязан быть первым,
    # хотя в каждой ветке стоял вторым.
    result = HybridSearchClient(branch(["a", "b"]), branch(["c", "b"])).search(
        "üçgen alanı", top_k=3
    )
    assert result["hits"][0]["chunk_id"] == "b"
    assert result["hits"][0]["found_by"] == ["text", "visual"]


def test_ranks_are_renumbered_after_merge():
    result = HybridSearchClient(branch(["a", "b"]), branch(["c"])).search("üçgen", top_k=3)
    assert [h["rank"] for h in result["hits"]] == [1, 2, 3]


def test_weak_branch_is_excluded_from_merge():
    # Гейт уже сказал, что выдаче не верить — в слияние она не идёт.
    result = HybridSearchClient(
        branch(["a"]), branch(["b"], useful=False, label="weak")
    ).search("üçgen alanı")
    assert [h["chunk_id"] for h in result["hits"]] == ["a"]
    assert result["branches"] == {"text": 1, "visual": 0}


def test_one_dead_branch_does_not_kill_the_request():
    result = HybridSearchClient(branch(["a"]), broken()).search("üçgen alanı")
    assert result["relevance"]["is_useful"] is True
    assert [h["chunk_id"] for h in result["hits"]] == ["a"]


def test_both_dead_branches_report_error():
    result = HybridSearchClient(broken(), broken()).search("üçgen alanı")
    assert result["relevance"]["label"] == "error"
    assert result["hits"] == []


def test_empty_when_both_branches_return_nothing():
    result = HybridSearchClient(branch([]), branch([])).search("üçgen alanı")
    assert result["relevance"]["is_useful"] is False
    assert result["hits"] == []


def test_payload_shape_matches_the_other_backends():
    result = HybridSearchClient(branch(["a"]), branch(["b"])).search("üçgen alanı")
    for key in ("query", "top_k", "context_order", "mode", "filters",
                "latency_ms", "retrieved", "returned", "relevance", "hits"):
        assert key in result, key
    assert result["profile"] == HYBRID_PROFILE


def test_page_image_comes_from_the_visual_branch():
    visual = branch(["b"])
    visual.get_page_image = lambda page_id: b"JPEG" if page_id == "b" else None
    client = HybridSearchClient(branch(["a"]), visual)
    assert client.get_page_image("b") == b"JPEG"
    assert client.get_page_image("zzz") is None


def test_missing_image_support_is_not_an_error():
    client = HybridSearchClient(branch(["a"]), branch(["b"]))
    assert client.get_page_image("b") is None


def test_rejects_nonsense_configuration():
    with pytest.raises(ValueError):
        HybridSearchClient(branch([]), branch([]), rrf_k=0)
    with pytest.raises(ValueError):
        HybridSearchClient(branch([]), branch([]), fetch_multiplier=0)


def test_backend_switch_builds_hybrid(monkeypatch):
    from mla_baseline.config import Settings
    from mla_baseline.solvers.agent_rag import _search_client_from_env

    monkeypatch.setenv("MLA_RETRIEVAL_BACKEND", "hybrid")
    monkeypatch.setenv("MLA_VISUAL_MIN_SCORE", "12.5")
    assert isinstance(_search_client_from_env(Settings()), HybridSearchClient)
