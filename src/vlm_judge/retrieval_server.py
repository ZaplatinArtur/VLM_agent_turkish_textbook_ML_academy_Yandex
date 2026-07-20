from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .retrieval import get_chunk, index_info, search_bm25


def _handler_factory(index_path: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VLMRetrieval/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _search(self, payload: dict[str, Any]) -> None:
            try:
                query = str(payload.get("query") or payload.get("q") or "").strip()
                if not query:
                    raise ValueError("query is required")
                result = search_bm25(
                    index_path,
                    query,
                    top_k=int(payload.get("top_k") or 10),
                    subject=str(payload["subject"]) if payload.get("subject") not in (None, "") else None,
                    grade=payload.get("grade"),
                    mode=str(payload.get("mode") or "or"),
                    low_information_weight=float(payload.get("low_information_weight", 0.25)),
                )
            except (TypeError, ValueError) as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(result)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/api/health":
                self._send_json({"ok": True, "index": index_info(index_path)})
                return
            if parsed.path == "/api/search":
                self._search({key: values[0] for key, values in query.items()})
                return
            if parsed.path == "/api/chunk":
                chunk_id = query.get("id", [""])[0]
                result = get_chunk(index_path, chunk_id)
                if result is None:
                    self._send_json({"error": "chunk not found"}, HTTPStatus.NOT_FOUND)
                else:
                    self._send_json(result)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urllib.parse.urlparse(self.path).path != "/api/search":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 64_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
            except Exception as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)
                return
            self._search(payload)

    return Handler


def serve_retrieval(
    index_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8770,
) -> None:
    info = index_info(index_path)
    server = ThreadingHTTPServer((host, port), _handler_factory(index_path))
    print(f"VLM retrieval API: http://{host}:{server.server_port}/")
    print(f"Index: {index_path} ({info['indexed_text_chunks']} text chunks)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm-retrieval-api")
    parser.add_argument("--index", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args()
    serve_retrieval(Path(args.index), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
