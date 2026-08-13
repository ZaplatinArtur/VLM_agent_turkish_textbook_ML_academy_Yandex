from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nonstream_protocol as protocol
import score_private
from generic_candidate import build_request, canonical_json_bytes, fixed_smoke_row, sha256_bytes, validate_model_content


def metadata(provider: str = "SiliconFlow") -> dict:
    return {"requested": "qwen/qwen3.5-9b", "strategy": "direct", "attempt": 1, "pipeline": [], "endpoints": {"available": [{"selected": True, "provider": provider, "model": "qwen/qwen3.5-9b-20260310"}]}, "attempts": [{"provider": provider, "model": "qwen/qwen3.5-9b-20260310", "status": 200}]}


def response(content: str = '{"answer":"4","option_label":"D"}') -> bytes:
    return json.dumps({"id": "gen-test", "created": 1, "model": "qwen/qwen3.5-9b", "provider": "SiliconFlow", "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "completion_tokens_details": {"reasoning_tokens": 3}}, "openrouter_metadata": metadata()}).encode()


class CandidateTests(unittest.TestCase):
    def test_request_exact_provider_privacy_seedless_nonstream(self):
        request, _ = build_request(fixed_smoke_row())
        protocol.validate_request_body(request)
        self.assertFalse(request["stream"])
        self.assertNotIn("seed", request)
        self.assertEqual(request["reasoning"], {"effort": "medium", "exclude": True})
        self.assertEqual(request["max_tokens"], 32768)
        self.assertEqual(request["provider"], {"only": ["siliconflow"], "allow_fallbacks": False, "require_parameters": True, "quantizations": ["fp8"], "data_collection": "deny", "zdr": True})

    def test_strict_nonstream_response_and_answer_types(self):
        result = protocol.parse_nonstream_response(response(), response_headers={"x-generation-id": "gen-test", "x-openrouter-cache-status": "miss"})
        protocol.validate_result(result)
        self.assertEqual(validate_model_content(result["content"], "choice")["option_label"], "D")
        self.assertEqual(validate_model_content('{"answer":"42","option_label":"NA"}', "numeric")["answer"], "42")
        with self.assertRaises(Exception):
            validate_model_content('{"answer":"42","option_label":"D"}', "numeric")

    def test_routing_reasoning_body_and_cache_fail_closed(self):
        value = json.loads(response())
        value["openrouter_metadata"] = metadata("Other")
        with self.assertRaises(protocol.StreamFailure):
            protocol.parse_nonstream_response(json.dumps(value).encode(), response_headers={"x-generation-id": "gen-test"})
        value = json.loads(response())
        value["choices"][0]["message"]["reasoning"] = "hidden"
        with self.assertRaises(protocol.StreamFailure):
            protocol.parse_nonstream_response(json.dumps(value).encode(), response_headers={"x-generation-id": "gen-test"})
        with self.assertRaises(protocol.StreamFailure):
            protocol.parse_nonstream_response(response(), response_headers={"x-generation-id": "gen-test", "x-openrouter-cache-status": "hit"})

    def test_error_redaction_and_exact_retry_allowlist(self):
        hostile = "visible question sk-or-v1-do-not-persist-secret"
        raw = json.dumps({"error": {"code": 400, "message": hostile, "metadata": {"provider_name": "SiliconFlow", "error_type": "bad_request", "raw": hostile}}})
        kind, detail = protocol._normalized_http_failure(400, raw)
        self.assertEqual(kind, "http_error")
        self.assertNotIn("question", detail)
        self.assertNotIn("sk-or", detail)
        overloaded = json.dumps({"error": {"code": 429, "metadata": {"provider_name": "SiliconFlow", "error_type": "provider_overloaded"}}})
        self.assertTrue(protocol._exact_provider_overloaded_429(429, overloaded))
        self.assertFalse(protocol._exact_provider_overloaded_429(429, overloaded.replace("SiliconFlow", "Other")))

    def test_orphan_intent_never_replayed(self):
        request, _ = build_request(fixed_smoke_row())
        digest = sha256_bytes(canonical_json_bytes(request))
        with tempfile.TemporaryDirectory() as raw:
            cache = Path(raw)
            protocol.exclusive_json(protocol._intent_path(cache, digest, 1), protocol._attempt_intent(digest, 1))
            result = protocol.call_with_retries(request, api_key="sk-or-v1-test-not-real-123456789", cache_dir=cache)
            self.assertEqual(result["attempts"][-1]["error_kind"], "ambiguous_inflight_after_power_loss")

    def test_frozen_image_capability_is_ocr_only(self):
        evidence = json.loads((Path(__file__).parent / "frozen" / "IMAGE_CAPABILITY_EVIDENCE.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence["decision"], "ocr_only_fail_closed")
        self.assertFalse(evidence["exact_endpoint"]["record_has_endpoint_specific_input_modalities"])
        self.assertFalse(evidence["runtime_contract"]["image_bytes_sent"])
        self.assertEqual(evidence["source"]["response_sha256"], "cb5746185b6c1b1d34ca4ac2a50a88e61d99033aa140cef7e2740d3262266744")

    def test_all_hybrid_abstentions_project_to_id_hash_free_ocr_wire(self):
        here = Path(__file__).resolve().parent
        source = here.parent / "maxim_9b_strict_noid_db_generic_hybrid_v3_1_20260812" / "runs" / "maxim274" / "generic_queue.jsonl"
        self.assertEqual(protocol.sha256_file(source), "b222a2fbc17afd33141802b727e302052b352743753221e44202a2bb5e156820")
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 256)
        hashes = []
        degraded = 0
        for row in rows:
            projected = {"schema_version": "maxim256-idfree-ocr-row-v1", "subject": row["subject"], "answer_type": row["answer_type"], "ocr_text": row["ocr_text"], "source_input_mode": "text_only" if row["input_mode"] == "text_only" else "multimodal_degraded_to_ocr_only"}
            degraded += int(projected["source_input_mode"] != "text_only")
            request, _ = build_request(projected)
            protocol.validate_request_body(request)
            wire = json.dumps(request, ensure_ascii=False, sort_keys=True)
            self.assertNotIn("data:image", wire)
            self.assertNotIn(row["controller_id"], wire)
            hashes.append(sha256_bytes(canonical_json_bytes(request)))
        self.assertEqual(degraded, 186)
        self.assertEqual(len(set(hashes)), 255)  # one exact visible-content duplicate group

    def test_private_score_rejects_unrelated_valid_score_provenance(self):
        freeze = {"artifacts": {"benchmark": {"sha256": "b" * 64}, "baseline_judge": {"sha256": "j" * 64}, "standard_scorer": {"sha256": "s" * 64}}}
        compose = {"composed_solver_sha256": "c" * 64}
        score = {
            "schema_version": "maxim-full274-score-v1",
            "overall": {"new_correct": 240, "n": 274},
            "provenance": {"benchmark": {"sha256": "b" * 64}, "frozen_page_rag_judge": {"sha256": "j" * 64}, "scorer": {"sha256": "s" * 64}, "solver_results": {"sha256": "x" * 64}, "image_judge": {"sha256": "i" * 64}},
            "guardrails": {"benchmark_rows_verified": 274, "solver_rows_verified": 274, "baseline_rows_verified": 274, "task_id_sets_match": True, "duplicate_task_ids": 0, "forbidden_gold_fields_in_solver": 0, "explicit_nonfalse_generation_gold_access": 0, "frozen_sha_pins_checked": True},
        }
        with self.assertRaises(score_private.ScoreError):
            score_private.validate_standard_score_provenance(score, freeze, compose, "i" * 64)
        score["provenance"]["solver_results"]["sha256"] = compose["composed_solver_sha256"]
        with self.assertRaises(score_private.ScoreError):
            score_private.validate_standard_score_provenance(score, freeze, compose, "z" * 64)
        self.assertEqual(
            score_private.validate_standard_score_provenance(score, freeze, compose, "i" * 64),
            (240, 274),
        )


if __name__ == "__main__":
    unittest.main()
