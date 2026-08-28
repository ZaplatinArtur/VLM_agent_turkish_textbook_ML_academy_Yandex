import base64

import pytest

from mla_baseline.tools.textbook_search import TextbookSearchError
from mla_baseline.tools.visual_search import (
    VISUAL_PROFILE,
    VisualSearchClient,
    visual_client_from_env,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"payload"


def page(page_id: str, score: float, *, with_image: bool = False) -> dict:
    row = {
        "page_id": page_id,
        "book_slug": "matematik-6",
        "page_number": 42,
        "grade": 6,
        "subject": "math",
        "page_image": f"books/matematik-6/pages/{page_id}.png",
        "answer_text": f"çözüm {page_id}",
        "has_solution": True,
        "score": score,
    }
    if with_image:
        row["page_image_b64"] = base64.b64encode(PNG).decode("ascii")
    return row


def fake_retriever(rows: list[dict]):
    def retrieve(query, *, k=5, subject=None, grade=None, index_dir=None):
        return {"query": query, "hits": rows[:k], "returned": len(rows[:k])}

    return retrieve


def client(rows: list[dict], *, min_score: float = 10.0) -> VisualSearchClient:
    return VisualSearchClient(min_score=min_score, retriever=fake_retriever(rows))


def test_returns_the_same_payload_shape_as_text_backend():
    result = client([page("p1", 18.4), page("p2", 15.1)]).search("üçgen alanı")
    for key in (
        "query", "top_k", "context_order", "mode", "filters",
        "latency_ms", "retrieved", "returned", "relevance", "hits",
    ):
        assert key in result, key
    assert result["profile"] == VISUAL_PROFILE
    assert result["hits"][0]["chunk_id"] == "p1"
    assert result["hits"][0]["text"] == "çözüm p1"
    assert result["hits"][0]["rank"] == 1


def test_weak_output_is_hidden_from_the_agent():
    # Порог 10, top-1 = 4.2: та же дисциплина, что у текстового клиента —
    # слабую выдачу агент не видит вовсе.
    result = client([page("p1", 4.2)]).search("üçgen alanı")
    assert result["relevance"]["label"] == "weak"
    assert result["relevance"]["is_useful"] is False
    assert result["hits"] == []
    assert result["retrieved"] == 1


def test_confident_output_reaches_the_agent():
    result = client([page("p1", 18.4)]).search("üçgen alanı")
    assert result["relevance"]["label"] == "confident"
    assert result["hits"]


def test_empty_output_is_flagged_empty():
    result = client([]).search("üçgen alanı")
    assert result["relevance"]["label"] == "empty"


def test_maxsim_scale_needs_its_own_threshold():
    # Тот же счёт 18.4 при дефолтном пороге 0.57 был бы «уверенным» всегда:
    # ради этого визуальный клиент и требует калиброванное значение.
    rows = [page("p1", 18.4)]
    assert client(rows, min_score=25.0).search("üçgen")["relevance"]["label"] == "weak"
    assert client(rows, min_score=10.0).search("üçgen")["relevance"]["label"] == "confident"


def test_client_refuses_to_start_without_calibration():
    with pytest.raises(ValueError, match="порог"):
        VisualSearchClient(min_score=None)


def test_unknown_transport_is_rejected():
    with pytest.raises(ValueError, match="inprocess"):
        VisualSearchClient(transport="carrier-pigeon", min_score=1.0)


def test_image_bytes_never_reach_the_prompt():
    # format_search_result_for_model переносит metadata в промпт дословно,
    # поэтому base64 в выдаче не должно быть ни на одном уровне.
    result = client([page("p1", 18.4, with_image=True)]).search("üçgen alanı")
    encoded = base64.b64encode(PNG).decode("ascii")
    assert encoded not in repr(result)
    assert result["hits"][0]["metadata"]["has_page_image"] is True


def test_image_is_available_by_page_id():
    backend = client([page("p1", 18.4, with_image=True)])
    backend.search("üçgen alanı")
    assert backend.get_page_image("p1") == PNG
    assert backend.get_page_image("p404") is None


def test_image_cache_is_bounded():
    rows = [page(f"p{i}", 18.4, with_image=True) for i in range(5)]
    backend = VisualSearchClient(
        min_score=1.0,
        retriever=fake_retriever(rows),
        image_cache_size=2,
    )
    backend.search("üçgen", top_k=5)
    assert backend.get_page_image("p0") is None
    assert backend.get_page_image("p4") == PNG


def test_backend_failure_becomes_a_tool_error():
    def broken(*args, **kwargs):
        raise RuntimeError("index is on fire")

    backend = VisualSearchClient(min_score=1.0, retriever=broken)
    with pytest.raises(TextbookSearchError, match="visual retrieval failed"):
        backend.search("üçgen alanı")


def test_missing_index_names_the_variable_to_set():
    def missing(*args, **kwargs):
        raise FileNotFoundError("no meta.json")

    backend = VisualSearchClient(min_score=1.0, retriever=missing)
    with pytest.raises(TextbookSearchError, match="MLA_VISUAL_INDEX_DIR"):
        backend.search("üçgen alanı")


def test_env_factory_requires_a_calibrated_threshold(monkeypatch):
    monkeypatch.delenv("MLA_VISUAL_MIN_SCORE", raising=False)
    with pytest.raises(ValueError, match="MLA_VISUAL_MIN_SCORE"):
        visual_client_from_env()


def test_env_factory_reads_transport_and_threshold(monkeypatch):
    monkeypatch.setenv("MLA_VISUAL_MIN_SCORE", "12.5")
    monkeypatch.setenv("MLA_VISUAL_TRANSPORT", "http")
    monkeypatch.setenv("MLA_VISUAL_URL", "http://gpu-node:8780/")
    backend = visual_client_from_env()
    assert backend.transport == "http"
    assert backend.min_score == 12.5
    assert backend.url == "http://gpu-node:8780"


def test_backend_switch_defaults_to_text(monkeypatch):
    # Дефолт text — существующие прогоны от появления визуального режима
    # не меняются ни на байт.
    from mla_baseline.config import Settings
    from mla_baseline.solvers.agent_rag import _search_client_from_env
    from mla_baseline.tools import LocalTextbookSearchClient

    monkeypatch.delenv("MLA_RETRIEVAL_BACKEND", raising=False)
    assert isinstance(_search_client_from_env(Settings()), LocalTextbookSearchClient)


def test_backend_switch_selects_visual(monkeypatch):
    from mla_baseline.config import Settings
    from mla_baseline.solvers.agent_rag import _search_client_from_env

    monkeypatch.setenv("MLA_RETRIEVAL_BACKEND", "visual")
    monkeypatch.setenv("MLA_VISUAL_MIN_SCORE", "12.5")
    monkeypatch.delenv("MLA_VISUAL_TRANSPORT", raising=False)
    backend = _search_client_from_env(Settings())
    assert isinstance(backend, VisualSearchClient)
    assert backend.transport == "inprocess"


def test_unknown_backend_is_rejected(monkeypatch):
    from mla_baseline.config import Settings
    from mla_baseline.solvers.agent_rag import _search_client_from_env

    monkeypatch.setenv("MLA_RETRIEVAL_BACKEND", "telepathy")
    with pytest.raises(ValueError, match="text | visual"):
        _search_client_from_env(Settings())
