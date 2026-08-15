import base64
import io

import pytest

from visual_retrive.search_service import VisualSearchApp


class FakeIndex:
    """Индекс без модели и без карты: проверяем обвязку, а не MaxSim."""

    def __init__(self, hits=None):
        self.pages = ["p1", "p2"]
        self.calls = []
        self._hits = hits if hits is not None else [
            {
                "page_id": "p1",
                "book_slug": "matematik-6",
                "page_number": 42,
                "grade": 6,
                "subject": "math",
                "page_image": "books/matematik-6/pages/p1.png",
                "answer_text": "çözüm",
                "has_solution": True,
                "score": 18.4,
            }
        ]

    def search(self, query, *, top_k=5, subject=None, grade=None):
        self.calls.append({"query": query, "top_k": top_k, "subject": subject, "grade": grade})
        return self._hits[:top_k]


def png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2000, 2000), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def corpus(tmp_path):
    page = tmp_path / "books" / "matematik-6" / "pages" / "p1.png"
    page.parent.mkdir(parents=True)
    page.write_bytes(png_bytes())
    return tmp_path


def app(corpus, **kwargs) -> VisualSearchApp:
    kwargs.setdefault("index", FakeIndex())
    return VisualSearchApp(root=corpus, **kwargs)


def test_search_returns_hits_with_rank(corpus):
    result = app(corpus).search({"query": "üçgen alanı", "top_k": 3})
    assert result["returned"] == 1
    assert result["hits"][0]["rank"] == 1
    assert result["hits"][0]["page_id"] == "p1"
    assert "latency_ms" in result


def test_page_image_is_returned_as_base64(corpus):
    result = app(corpus).search({"query": "üçgen alanı"})
    encoded = result["hits"][0]["page_image_b64"]
    raw = base64.b64decode(encoded)
    assert raw[:2] == b"\xff\xd8"  # JPEG, не исходный PNG


def test_image_is_downscaled(corpus):
    from PIL import Image

    result = app(corpus, image_max_side=256).search({"query": "üçgen alanı"})
    raw = base64.b64decode(result["hits"][0]["page_image_b64"])
    with Image.open(io.BytesIO(raw)) as image:
        assert max(image.size) <= 256


def test_images_can_be_switched_off(corpus):
    result = app(corpus, include_images=False).search({"query": "üçgen alanı"})
    assert "page_image_b64" not in result["hits"][0]


def test_missing_page_file_does_not_break_the_request(tmp_path):
    # Битая или отсутствующая страница — не повод ронять запрос: текст всё ещё
    # полезен, а клиент увидит, что картинки нет.
    result = app(tmp_path).search({"query": "üçgen alanı"})
    assert result["returned"] == 1
    assert "page_image_b64" not in result["hits"][0]


def test_path_traversal_is_refused(corpus):
    index = FakeIndex(hits=[{"page_id": "evil", "page_image": "../../etc/passwd", "score": 1.0}])
    result = VisualSearchApp(root=corpus, index=index).search({"query": "üçgen"})
    assert "page_image_b64" not in result["hits"][0]


def test_short_query_is_refused(corpus):
    with pytest.raises(ValueError, match="too short"):
        app(corpus).search({"query": "a"})


def test_top_k_bounds_are_enforced(corpus):
    with pytest.raises(ValueError, match="top_k"):
        app(corpus).search({"query": "üçgen alanı", "top_k": 99})


def test_filters_reach_the_index(corpus):
    index = FakeIndex()
    service = VisualSearchApp(root=corpus, index=index)
    service.search({"query": "üçgen alanı", "subject": "math", "grade": 6})
    assert index.calls[0]["subject"] == "math"
    assert index.calls[0]["grade"] == 6


def test_blank_grade_is_treated_as_absent(corpus):
    index = FakeIndex()
    service = VisualSearchApp(root=corpus, index=index)
    service.search({"query": "üçgen alanı", "grade": ""})
    assert index.calls[0]["grade"] is None


def test_health_reports_loaded_pages(corpus):
    assert app(corpus).health() == {
        "status": "ok",
        "pages": 2,
        "index_dir": "",
        "images": True,
    }


def test_client_and_service_agree_over_http(corpus):
    """Стык клиента и сервиса: контракт по проводу, а не по договорённости."""
    import threading
    from http.server import ThreadingHTTPServer

    from mla_baseline.tools.visual_search import VisualSearchClient
    from visual_retrive.search_service import _handler

    service = VisualSearchApp(root=corpus, index=FakeIndex())
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        client = VisualSearchClient(
            transport="http",
            url=f"http://127.0.0.1:{port}",
            min_score=10.0,
        )
        result = client.search("üçgen alanı", top_k=3)
        assert result["relevance"]["label"] == "confident"
        assert result["hits"][0]["chunk_id"] == "p1"
        assert result["hits"][0]["text"] == "çözüm"
        # Картинка доехала, но в выдачу для промпта не попала.
        assert client.get_page_image("p1") is not None
        assert "page_image_b64" not in result["hits"][0]
        assert result["hits"][0]["metadata"]["has_page_image"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_client_reports_a_dead_service_as_tool_error():
    from mla_baseline.tools.textbook_search import TextbookSearchError
    from mla_baseline.tools.visual_search import VisualSearchClient

    client = VisualSearchClient(
        transport="http",
        url="http://127.0.0.1:1",
        min_score=10.0,
        timeout=(1, 1),
    )
    with pytest.raises(TextbookSearchError, match="visual search service"):
        client.search("üçgen alanı")
