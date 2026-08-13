import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from retrieve.rankers.rerank_api import RerankApiRanker
from schemas.retrieve import RetrievedChunk


def chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, score=0.0, metadata={})


class _Handler(BaseHTTPRequestHandler):
    """Отвечает как /v1/rerank у vLLM: отсортированный список с индексами."""

    scores: dict[str, float] = {}
    seen: dict = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.clear()
        _Handler.seen.update(body)
        _Handler.seen["path"] = self.path
        results = [
            {"index": index, "relevance_score": _Handler.scores[document]}
            for index, document in enumerate(body["documents"])
        ]
        results.sort(key=lambda item: -item["relevance_score"])
        data = json.dumps({"results": results}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def rerank_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_reorders_by_relevance_score(rerank_server):
    _Handler.scores = {"a": 0.1, "b": 0.9, "c": 0.5}
    ranker = RerankApiRanker("Qwen/Qwen3-Reranker-0.6B", base_url=rerank_server)
    results = ranker.rank("q", [chunk("A", "a"), chunk("B", "b"), chunk("C", "c")])
    assert [c.chunk_id for c in results] == ["B", "C", "A"]
    assert [c.score for c in results] == [0.9, 0.5, 0.1]
    assert _Handler.seen["path"] == "/v1/rerank"
    assert _Handler.seen["model"] == "Qwen/Qwen3-Reranker-0.6B"
    assert _Handler.seen["query"] == "q"


def test_only_top_n_go_to_the_server(rerank_server):
    _Handler.scores = {"a": 0.2, "b": 0.8}
    ranker = RerankApiRanker("m", base_url=rerank_server, top_n=2)
    tail = [chunk("C", "c"), chunk("D", "d")]
    results = ranker.rank("q", [chunk("A", "a"), chunk("B", "b"), *tail])
    assert [c.chunk_id for c in results] == ["B", "A", "C", "D"]
    assert _Handler.seen["documents"] == ["a", "b"]


def test_empty_candidates_skip_the_request():
    ranker = RerankApiRanker("m", base_url="http://127.0.0.1:1")
    assert ranker.rank("q", []) == []
    assert ranker.rank("q", None) == []


def test_base_url_comes_from_env_when_not_given(monkeypatch):
    monkeypatch.setenv("RETRIEVE_RERANK_URL", "http://gpu-host:8002/")
    assert RerankApiRanker("m").base_url == "http://gpu-host:8002"
    monkeypatch.delenv("RETRIEVE_RERANK_URL")
    assert RerankApiRanker("m").base_url == "http://127.0.0.1:8002"
