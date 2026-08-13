from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nonstream_protocol as p
from generic_candidate import build_request, canonical_json_bytes, fixed_smoke_row, sha256_bytes


def metadata(provider: str = "SiliconFlow") -> dict:
    return {
        "requested": "qwen/qwen3.5-9b",
        "strategy": "direct",
        "attempt": 1,
        "pipeline": [],
        "endpoints": {"available": [{"selected": True, "provider": provider, "model": "qwen/qwen3.5-9b-20260310"}]},
        "attempts": [{"provider": provider, "model": "qwen/qwen3.5-9b-20260310", "status": 200}],
    }


def body(**changes) -> bytes:
    value = {
        "id": "gen-test",
        "created": 1,
        "model": "qwen/qwen3.5-9b",
        "provider": "SiliconFlow",
        "choices": [{"index": 0, "finish_reason": "stop", "native_finish_reason": "stop", "message": {"content": '{"answer":"D"}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 0.001},
        "openrouter_metadata": metadata(),
    }
    value.update(changes)
    return json.dumps(value).encode()


class V6Tests(unittest.TestCase):
    def test_request_contract_is_nonstream_private_and_seedless(self):
        request, aliases = build_request(fixed_smoke_row())
        p.validate_request_body(request)
        self.assertIs(request["stream"], False)
        self.assertNotIn("seed", request)
        self.assertEqual(request["reasoning"], {"effort": "medium", "exclude": True})
        self.assertEqual(request["max_tokens"], 32768)
        self.assertEqual(request["provider"], {"only": ["siliconflow"], "allow_fallbacks": False, "require_parameters": True, "quantizations": ["fp8"], "data_collection": "deny", "zdr": True})
        self.assertEqual(set(aliases), set("ABCDE"))

    def test_nonstream_success_closure(self):
        result = p.parse_nonstream_response(body(), response_headers={"x-generation-id": "gen-test", "x-openrouter-cache-status": "miss"})
        self.assertEqual(result["content"], '{"answer":"D"}')
        self.assertTrue(result["routing_validation"]["passed"])
        p.validate_result(result)

    def test_medium_reasoning_usage_count_allowed_but_body_forbidden(self):
        value = json.loads(body())
        value["usage"]["completion_tokens_details"] = {"reasoning_tokens": 17}
        result = p.parse_nonstream_response(json.dumps(value).encode(), response_headers={"x-generation-id": "gen-test"})
        self.assertEqual(result["usage"]["completion_tokens_details"]["reasoning_tokens"], 17)
        value["choices"][0]["message"]["reasoning"] = "must not persist"
        with self.assertRaises(p.StreamFailure):
            p.parse_nonstream_response(json.dumps(value).encode(), response_headers={"x-generation-id": "gen-test"})

    def test_reject_identity_cache_routing_finish_and_reasoning(self):
        cases = [
            (body(provider="Other"), {"x-generation-id": "gen-test"}),
            (body(), {"x-generation-id": "wrong"}),
            (body(), {"x-generation-id": "gen-test", "x-openrouter-cache-status": "hit"}),
            (body(openrouter_metadata=metadata("Other")), {"x-generation-id": "gen-test"}),
        ]
        value = json.loads(body())
        value["choices"][0]["finish_reason"] = "length"
        cases.append((json.dumps(value).encode(), {"x-generation-id": "gen-test"}))
        value = json.loads(body())
        value["choices"][0]["message"]["reasoning"] = "secret"
        cases.append((json.dumps(value).encode(), {"x-generation-id": "gen-test"}))
        for raw, headers in cases:
            with self.subTest(raw=raw[:50]):
                with self.assertRaises(p.StreamFailure):
                    p.parse_nonstream_response(raw, response_headers=headers)

    def test_exact_429_allowlist_and_transport_types(self):
        good = json.dumps({"error": {"code": 429, "metadata": {"provider_name": "SiliconFlow", "error_type": "provider_overloaded"}}})
        self.assertTrue(p._exact_provider_overloaded_429(429, good))
        self.assertFalse(p._exact_provider_overloaded_429(429, good.replace("SiliconFlow", "Other")))
        self.assertFalse(p._exact_provider_overloaded_429(500, good))
        self.assertEqual(p._typed_transport_kind(TimeoutError()), "transport_timeout")
        self.assertIsNone(p._typed_transport_kind(ValueError()))

    def test_http_and_exception_details_cannot_echo_prompt_or_secret(self):
        hostile = "SORU-GİZLİ sk-or-v1-secret-should-never-persist"
        raw = json.dumps({"error": {"code": 400, "message": hostile, "metadata": {"provider_name": "SiliconFlow", "error_type": "bad_request", "raw": hostile}}})
        kind, detail = p._normalized_http_failure(400, raw)
        self.assertEqual(kind, "http_error")
        self.assertNotIn("SORU", detail)
        self.assertNotIn("sk-or", detail)
        self.assertEqual(json.loads(detail), {"http_status": 400, "openrouter_error_code": 400, "provider_name": "SiliconFlow", "error_type": "bad_request"})
        exc_detail = p._normalized_exception_detail("transport_untyped", ValueError(hostile))
        self.assertEqual(exc_detail, "transport_untyped:ValueError")

    def test_orphan_intent_is_not_replayed(self):
        request, _ = build_request(fixed_smoke_row())
        digest = sha256_bytes(canonical_json_bytes(request))
        with tempfile.TemporaryDirectory() as raw:
            cache = Path(raw)
            intent = p._attempt_intent(digest, 1)
            p.exclusive_json(p._intent_path(cache, digest, 1), intent)
            call = p.call_with_retries(request, api_key="sk-or-v1-test-not-real-123456789", cache_dir=cache)
            self.assertFalse(call["terminal_success"])
            self.assertEqual(call["attempts"][-1]["error_kind"], "ambiguous_inflight_after_power_loss")

    def test_queue_and_prompt_identity_to_direct_v3(self):
        here = Path(__file__).resolve().parent
        queue = here / "frozen" / "queue_public_content_only.jsonl"
        v3 = here.parent / "maxim_9b_ykslop_generic_reasoning_sse_alt_v3_dev_20260812" / "frozen" / "queue_public_content_only.jsonl"
        self.assertEqual(queue.read_bytes(), v3.read_bytes())
        rows = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 185)
        prompt_sha = sha256_bytes(canonical_json_bytes([build_request(row)[0]["messages"] for row in rows]))
        self.assertEqual(prompt_sha, "844e11d2007f88ca6732a00b7a87a34da200c730f52327f1f0c7c3819998061f")
        for row in rows:
            request, _ = build_request(row)
            text = "\n".join(message["content"] for message in request["messages"])
            self.assertIn("Türkçe", text)
            self.assertIn("GENEL TEORİ DESTEĞİ", text)
            self.assertNotRegex(text, r"(?:TГ|Д[°±џ]|Р[џЅ])")


if __name__ == "__main__":
    unittest.main()
