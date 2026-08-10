import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vlm_judge.backends import OpenAICompatibleBackend
from vlm_judge.prompts import JudgeRequest


class _MockHandler(BaseHTTPRequestHandler):
    payload = None
    authorization = None

    def log_message(self, format, *args):
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        type(self).payload = json.loads(self.rfile.read(length))
        type(self).authorization = self.headers.get("Authorization")
        verdict = {
            "label": "fully_correct", "score": 4, "strict_correct": True,
            "final_answer_correct": True, "reasoning_correct": None,
            "complete": True, "confidence": 0.9, "error_types": [],
            "rationale": "ok", "reference_quality_issue": False,
        }
        response = {
            "id": "mock-1",
            "choices": [{"message": {"content": json.dumps(verdict)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class OpenAICompatibleBackendTests(unittest.TestCase):
    def test_multimodal_chat_payload(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = OpenAICompatibleBackend(
                f"http://127.0.0.1:{server.server_port}/v1",
                "Qwen-VL-mock",
                api_key="test-key",
                enable_thinking=False,
            )
            response = backend.complete(
                JudgeRequest(
                    "system",
                    "user",
                    ("https://example.test/question.png",),
                    ("question image",),
                )
            )
            self.assertIn("fully_correct", response.text)
            self.assertEqual(response.metadata["response_id"], "mock-1")
            payload = _MockHandler.payload
            self.assertEqual(payload["model"], "Qwen-VL-mock")
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(payload["messages"][1]["content"][1]["text"], "question image:")
            self.assertEqual(payload["messages"][1]["content"][2]["type"], "image_url")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_data_url_mode_uses_local_image_cache(self) -> None:
        class FakeCache:
            def get(self, url):
                self.url = url
                return b"png-bytes", "image/png"

        with tempfile.TemporaryDirectory() as directory:
            backend = OpenAICompatibleBackend(
                "http://127.0.0.1:1/v1",
                "Qwen-VL-mock",
                image_mode="data_url",
                image_cache_dir=Path(directory),
            )
            cache = FakeCache()
            backend._image_cache = cache
            value = backend._image_reference("https://yadi.sk/i/example")
            self.assertTrue(value.startswith("data:image/png;base64,"))
            self.assertEqual(cache.url, "https://yadi.sk/i/example")

    def test_openrouter_payload_uses_reasoning_and_bearer_key(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = OpenAICompatibleBackend(
                f"http://127.0.0.1:{server.server_port}/v1",
                "qwen/qwen3.5-9b",
                api_key="test-key",
                enable_thinking=False,
                provider="openrouter",
            )
            backend.complete(JudgeRequest("system", "user", (), ()))

            payload = _MockHandler.payload
            self.assertEqual(payload["reasoning"], {"effort": "none"})
            self.assertNotIn("chat_template_kwargs", payload)
            self.assertEqual(_MockHandler.authorization, "Bearer test-key")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_data_url_mode_reads_local_image_without_remote_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "question.png"
            image_path.write_bytes(b"local-png")
            backend = OpenAICompatibleBackend(
                "http://127.0.0.1:1/v1",
                "Qwen-VL-mock",
                image_mode="data_url",
                image_cache_dir=Path(directory) / "cache",
            )
            value = backend._image_reference(str(image_path))
            self.assertTrue(value.startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
