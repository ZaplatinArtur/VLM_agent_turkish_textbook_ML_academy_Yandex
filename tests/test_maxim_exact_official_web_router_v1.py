from __future__ import annotations

import copy
import hashlib
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from scripts import run_maxim_exact_official_web_router_v1 as router


REPO = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO / "configs" / "maxim_exact_official_web_router_v1.json"


def profile_for_test(*, expected_rows: int = 1) -> dict[str, Any]:
    value = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    value["expected_rows"] = expected_rows
    value["authority"]["allowed_schemes"] = ["http"]
    value["authority"]["official_host_suffixes"] = ["127.0.0.1"]
    value["certificate"]["min_full_token_coverage"] = 0.40
    return value


def task(task_id: str = "synthetic_0001") -> dict[str, Any]:
    return {
        "task_id": task_id,
        "subject": "Turkish language and literature",
        "grade": None,
        "question": "(soru görselde)",
        "question_images": [
            {
                "image_id": task_id + "_img1",
                "format": "file_path",
                "data": f"images/{task_id}.png",
                "mime_type": "image/png",
                "caption": None,
            }
        ],
        "answer_type": "choice",
    }


QUESTION_TEXT = (
    "9. Ergenlikten sonra büyümesi duran insanların aksine pek çok balık "
    "yaşamları boyunca gelişimini sürdürür. Aşırı avlanma balıkların uzun "
    "yaşamasını engeller. Bu parçada numaralanmış cümlelerle ilgili olarak "
    "aşağıdakilerden hangisi söylenebilir? A) İnsanlarla balıklar büyüme "
    "özelliği üzerinden karşılaştırılmıştır. B) Yaşam süresi ile boy arasında "
    "kesin ilişki vardır. C) Avlanma yaşam süresini uzatır. D) Neslin tükenme "
    "nedenleri açıklanmıştır. E) Dünya genelinde yasak gereklidir."
)


def ocr_row(text: str = QUESTION_TEXT, task_id: str = "synthetic_0001") -> dict[str, Any]:
    return {
        "schema_version": "maxim-paddleocr-vl16-task-parse-v1",
        "task_id": task_id,
        "parser": {
            "pipeline_version": "synthetic",
            "layout_model": "synthetic",
            "recognition_model": "synthetic",
            "recognition_backend": "fixture",
            "max_new_tokens": 128,
            "gold_access": False,
        },
        "images": [
            {
                "image_index": 0,
                "image_basename": task_id + ".png",
                "image_sha256": "1" * 64,
                "width": 800,
                "height": 1000,
                "input_decode": {"kind": "fixture"},
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": text,
                        "block_bbox": [0, 0, 800, 1000],
                        "block_id": 0,
                        "block_order": 1,
                    }
                ],
            }
        ],
    }


class StaticSearch:
    network_calls = 0

    def __init__(self, urls: list[str]):
        self.urls = urls
        self.calls: list[str] = []

    def search(self, query: str, *, limit: int) -> router.SearchResponse:
        self.calls.append(query)
        hits = tuple(
            router.SearchHit(title=f"fixture-{index}", url=url)
            for index, url in enumerate(self.urls[:limit])
        )
        return router.SearchResponse(
            hits=hits,
            provenance={
                "raw_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "cache_hit": True,
            },
        )


class FixtureHandler(BaseHTTPRequestHandler):
    routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
    hits: dict[str, int] = {}

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = self.path.split("?", 1)[0]
        type(self).hits[path] = type(self).hits.get(path, 0) + 1
        status, headers, body = type(self).routes.get(
            path,
            (404, {"Content-Type": "text/plain; charset=utf-8"}, b"missing"),
        )
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def fixture_server(
    routes: dict[str, tuple[int, dict[str, str], bytes]],
) -> Iterator[tuple[str, type[FixtureHandler]]]:
    handler = type("BoundFixtureHandler", (FixtureHandler,), {})
    handler.routes = routes
    handler.hits = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def minimal_pdf(text: str, *, compressed: bool = False) -> bytes:
    import zlib

    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("cp1254")
    if compressed:
        payload = zlib.compress(stream)
        filter_entry = b" /Filter /FlateDecode"
    else:
        payload = stream
        filter_entry = b""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog >> endobj\n"
        b"2 0 obj << /Length "
        + str(len(payload)).encode("ascii")
        + filter_entry
        + b" >>\nstream\n"
        + payload
        + b"\nendstream\nendobj\n%%EOF\n"
    )


def make_fetcher(
    tmp_path: Path, profile: dict[str, Any], *, network_enabled: bool = True
) -> router.OfficialDocumentFetcher:
    return router.OfficialDocumentFetcher(
        policy=router.OfficialURLPolicy.from_profile(profile),
        cache_dir=tmp_path / "documents",
        timeout_s=3,
        max_bytes=2_000_000,
        network_enabled=network_enabled,
        user_agent="synthetic-test",
    )


def test_recursive_key_and_false_attestation_guards() -> None:
    router.assert_gold_blind_input(
        {"parser": {"gold_access": False}, "task": {"answer_type": "choice"}},
        location="safe",
    )
    with pytest.raises(router.RouterInputError, match="forbidden key"):
        router.assert_gold_blind_input(
            {"safe": [{"nested": {"reference_answer": "SECRET"}}]},
            location="bad",
        )
    with pytest.raises(router.RouterInputError, match="must be false"):
        router.assert_gold_blind_input(
            {"generation_gold_access": True}, location="bad"
        )


def test_forbidden_input_filename_and_public_schema(tmp_path: Path) -> None:
    forbidden = tmp_path / "judge_queue.jsonl"
    forbidden.write_text("{}\n", encoding="utf-8")
    with pytest.raises(router.RouterInputError, match="forbidden input filename"):
        router.guard_input_path(forbidden, role="queue", suffixes={".jsonl"})

    valid = task()
    router.validate_public_queue([valid], expected_rows=1)
    leaked = copy.deepcopy(valid)
    leaked["reference_answer"] = "A"
    with pytest.raises(router.RouterInputError, match="schema mismatch"):
        router.validate_public_queue([leaked], expected_rows=1)


def test_query_plan_is_deterministic_quoted_and_bounded() -> None:
    profile = profile_for_test()
    first = router.build_query_plan(task(), ocr_row(), profile)
    second = router.build_query_plan(task(), ocr_row(), profile)
    assert first == second
    assert first["eligible"] is True
    assert 2 <= len(first["queries"]) <= 4
    assert first["question_number"] == 9
    for item in first["queries"]:
        query = item["query"]
        assert query.count('"') >= 2
        assert query.count('"') % 2 == 0
        assert len(query) <= profile["query"]["max_query_chars"]


def test_empty_ocr_is_task_local_failclosed() -> None:
    profile = profile_for_test()
    plan = router.build_query_plan(task(), ocr_row("<script>hidden()</script>"), profile)
    assert plan["eligible"] is False
    assert plan["ineligibility_reasons"] == ["no_usable_ocr_phrase"]
    assert plan["queries"] == []
    assert plan["ocr"]["visible_characters"] == 0


def test_official_host_allowlist_rejects_lookalikes() -> None:
    production = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    policy = router.OfficialURLPolicy.from_profile(production)
    assert policy.is_allowed_url("https://dokuman.osym.gov.tr/a.pdf")
    assert policy.is_allowed_url("https://ogmmateryal.eba.gov.tr/book/page1.html")
    assert not policy.is_allowed_url("https://osym.gov.tr.evil.example/a.pdf")
    assert not policy.is_allowed_url("https://eba.gov.tr@evil.example/a")
    assert not policy.is_allowed_url("http://dokuman.osym.gov.tr/a.pdf")


def test_html_fetch_cache_sha_and_final_redirect_guard(tmp_path: Path) -> None:
    profile = profile_for_test()
    routes = {
        "/book/page.html": (
            200,
            {"Content-Type": "text/html; charset=utf-8"},
            b"<html><script>bad()</script><body>Gorunen resmi metin</body></html>",
        ),
        "/redirect-bad": (
            302,
            {"Location": "http://example.invalid/not-official"},
            b"",
        ),
    }
    with fixture_server(routes) as (base, handler):
        fetcher = make_fetcher(tmp_path, profile)
        first = fetcher.fetch(base + "/book/page.html")
        second = fetcher.fetch(base + "/book/page.html")
        assert "Gorunen resmi metin" in first.text
        assert "bad()" not in first.text
        assert first.raw_sha256 == second.raw_sha256
        assert second.provenance["cache_hit"] is True
        assert handler.hits["/book/page.html"] == 1
        assert fetcher.network_calls == 1
        with pytest.raises(router.RouterInputError, match="outside official allowlist"):
            fetcher.fetch(base + "/redirect-bad")
        assert "http://example.invalid" not in handler.hits


@pytest.mark.parametrize("compressed", [False, True])
def test_pdf_support_without_optional_pdf_dependency(compressed: bool) -> None:
    raw = minimal_pdf("Cevap Anahtari 9.B", compressed=compressed)
    text, method = router.extract_document_text(
        raw,
        content_type="application/pdf",
        url="https://dokuman.osym.gov.tr/test.pdf",
    )
    assert "Cevap Anahtari 9.B" in text
    assert method in {"stdlib-pdf-stream-text-v1", "pypdf"}


def test_exact_html_question_plus_numbered_key_accepts_certificate(tmp_path: Path) -> None:
    profile = profile_for_test()
    question_html = f"<html><body><article>{QUESTION_TEXT}</article></body></html>".encode(
        "utf-8"
    )
    key_html = b"<html><body><h1>CEVAP ANAHTARI</h1><table><tr><td>9.B</td></tr></table></body></html>"
    routes = {
        "/book/files/basic-html/page1.html": (
            200,
            {"Content-Type": "text/html; charset=utf-8"},
            question_html,
        ),
        "/book/files/basic-html/page99.html": (
            200,
            {"Content-Type": "text/html; charset=utf-8"},
            key_html,
        ),
    }
    with fixture_server(routes) as (base, _handler):
        urls = [
            base + "/book/files/basic-html/page1.html",
            base + "/book/files/basic-html/page99.html",
            "http://evil.invalid/mirror.html",
        ]
        search = StaticSearch(urls)
        fetcher = make_fetcher(tmp_path, profile)
        exact = router.ExactOfficialWebRouter(
            profile=profile,
            profile_sha256="a" * 64,
            queue_sha256="b" * 64,
            ocr_sha256="c" * 64,
            search_client=search,
            fetcher=fetcher,
        )
        decision = exact.route(task(), ocr_row())

    assert decision["accepted"] is True
    assert decision["final_answer"] == "B"
    assert decision["certificate"]["answer"] == "B"
    assert len(decision["certificate"]["certificate_sha256"]) == 64
    assert decision["generation"]["gold_access"] is False
    assert decision["generation"]["default_solver_access"] is False
    assert len(search.calls) == len(decision["query_plan"]["queries"])
    assert all("evil.invalid" not in str(item) for item in decision["document_fetches"])


def test_adjacent_column_conflict_fails_closed(tmp_path: Path) -> None:
    profile = profile_for_test()
    question_html = f"<html><body>{QUESTION_TEXT}</body></html>".encode("utf-8")
    # The nearby C mimics the known adjacent-column contamination failure mode.
    ambiguous_key = b"<html><body>CEVAP ANAHTARI <div>9.B</div><div>9.C</div></body></html>"
    routes = {
        "/book/files/basic-html/page1.html": (
            200,
            {"Content-Type": "text/html; charset=utf-8"},
            question_html,
        ),
        "/book/files/basic-html/page99.html": (
            200,
            {"Content-Type": "text/html; charset=utf-8"},
            ambiguous_key,
        ),
    }
    with fixture_server(routes) as (base, _handler):
        search = StaticSearch(
            [
                base + "/book/files/basic-html/page1.html",
                base + "/book/files/basic-html/page99.html",
            ]
        )
        exact = router.ExactOfficialWebRouter(
            profile=profile,
            profile_sha256="a" * 64,
            queue_sha256="b" * 64,
            ocr_sha256="c" * 64,
            search_client=search,
            fetcher=make_fetcher(tmp_path, profile),
        )
        decision = exact.route(task(), ocr_row())

    assert decision["accepted"] is False
    assert decision["final_answer"] is None
    assert "ambiguous_answer_markers_or_adjacent_columns" in decision["rejection_reasons"]


def test_nonofficial_results_and_network_disabled_cache_miss_fail_closed(tmp_path: Path) -> None:
    profile = profile_for_test()
    exact = router.ExactOfficialWebRouter(
        profile=profile,
        profile_sha256="a" * 64,
        queue_sha256="b" * 64,
        ocr_sha256="c" * 64,
        search_client=StaticSearch(["http://evil.invalid/answer.html"]),
        fetcher=make_fetcher(tmp_path, profile, network_enabled=False),
    )
    decision = exact.route(task(), ocr_row())
    assert decision["accepted"] is False
    assert decision["rejection_reasons"] == ["no_fetchable_official_documents"]
    assert decision["final_answer"] is None


def test_searx_cache_binds_raw_sha_and_replays_without_second_call(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "results": [
                {
                    "title": "official",
                    "url": "https://ogmmateryal.eba.gov.tr/book/page1.html",
                    "content": "snippet",
                    "engine": "fixture",
                }
            ]
        }
    ).encode("utf-8")
    routes = {
        "/search": (200, {"Content-Type": "application/json"}, payload),
    }
    with fixture_server(routes) as (base, handler):
        search = router.SearxngSearchClient(
            base_url=base,
            cache_dir=tmp_path / "search",
            timeout_s=3,
            language="tr",
            network_enabled=True,
            user_agent="synthetic-test",
        )
        first = search.search('"ayırt edici türkçe soru cümlesi"', limit=5)
        second = search.search('"ayırt edici türkçe soru cümlesi"', limit=5)
    assert first.hits == second.hits
    assert first.provenance["raw_sha256"] == hashlib.sha256(payload).hexdigest()
    assert second.provenance["cache_hit"] is True
    assert search.network_calls == 1
    assert handler.hits["/search"] == 1


def test_dry_run_report_has_no_network_and_no_answer_fields() -> None:
    profile = profile_for_test()
    report = router.dry_run_report(
        queue=[task()],
        ocr_index={"synthetic_0001": ocr_row()},
        profile=profile,
        hashes={"queue": "a" * 64, "ocr": "b" * 64, "profile": "c" * 64},
    )
    assert report["status"] == "PASS"
    assert report["tasks"] == 1
    assert report["eligible_tasks"] == 1
    assert report["network_calls"] == 0
    assert report["cache_writes"] == 0
    assert report["generation"]["gold_access"] is False
    assert "final_answer" not in json.dumps(report)
