import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from vlm_judge.retrieval import build_bm25_index
from vlm_judge.retrieval_server import _handler_factory


class RetrievalServerTests(unittest.TestCase):
    def test_health_search_and_chunk_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks.jsonl"
            chunks.write_text(
                json.dumps(
                    {
                        "chunk_id": "c1",
                        "page_id": "p1",
                        "kind": "text",
                        "text": "equal denominator fraction addition",
                        "metadata": {"subject": "math", "grade": 4},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            index = root / "index.sqlite"
            build_bm25_index(chunks, index)
            server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(index))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(base + "/api/health") as response:
                    health = json.load(response)
                self.assertTrue(health["ok"])

                request = urllib.request.Request(
                    base + "/api/search",
                    data=json.dumps({"query": "fraction addition", "top_k": 1}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    result = json.load(response)
                self.assertEqual(result["hits"][0]["chunk_id"], "c1")

                with urllib.request.urlopen(base + "/api/chunk?id=c1") as response:
                    chunk = json.load(response)
                self.assertEqual(chunk["page_id"], "p1")

                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(base + "/api/search?query=")
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
