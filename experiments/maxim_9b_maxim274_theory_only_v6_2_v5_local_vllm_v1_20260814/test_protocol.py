"""Hostile offline tests for the Maxim-274 theory-only namespace."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import prepare_freeze
import protocol


class TheoryOnlyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source, cls.corpus = prepare_freeze._source_inputs()
        cls.public = [protocol.public_content_projection(row) for row in cls.source]

    def test_exact_public_and_corpus_pins(self) -> None:
        self.assertEqual(len(self.source), 274)
        self.assertEqual(len(self.corpus), 75)
        self.assertEqual(protocol.sha256_file(protocol.SOURCE_QUEUE), protocol.SOURCE_QUEUE_SHA256)
        self.assertEqual(protocol.sha256_file(protocol.SOURCE_THEORY), protocol.SOURCE_THEORY_SHA256)

    def test_queue_has_no_gold_or_outcomes(self) -> None:
        for row in self.source:
            protocol._walk_forbidden_keys(row, label="hostile test")
        poisoned = dict(self.source[0])
        poisoned["reference_answer"] = "A"
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_source_public_rows([poisoned, *self.source[1:]])

    def test_task_identity_never_enters_request(self) -> None:
        row = self.public[0]
        retrieval = protocol.retrieve_theory(row, self.corpus)
        request = protocol.build_primary_request(row, retrieval)
        wire = protocol.canonical_json_bytes(request)
        self.assertNotIn(b"task_id", wire)
        self.assertNotIn(b"controller_id", wire)
        self.assertNotIn(self.source[0]["controller_id"].encode("ascii"), wire)
        poisoned = json.loads(json.dumps(request))
        poisoned["task_id"] = self.source[0]["controller_id"]
        with self.assertRaises(protocol.ProtocolError):
            protocol._assert_wire_blind(poisoned)

    def test_content_seed_ignores_outer_alignment(self) -> None:
        row = self.public[0]
        first = protocol.seed_for(row, "v6.2-primary")
        source = dict(self.source[0])
        source["controller_id"] = "val_9999"
        self.assertEqual(first, protocol.seed_for(protocol.public_content_projection(source), "v6.2-primary"))

    def test_retrieval_is_subject_only_and_deterministic(self) -> None:
        for row in self.public:
            first = protocol.retrieve_theory(row, self.corpus)
            second = protocol.retrieve_theory(row, list(reversed(self.corpus)))
            self.assertEqual(first, second)
            desired = protocol.subject_key(row["subject"])
            allowed = {item["chunk_id"] for item in self.corpus if item["subject"] == desired}
            self.assertTrue(all(item["chunk_id"] in allowed for item in first))

    def test_strict_corpus_rejects_task_like_schema_poison(self) -> None:
        poisoned = [dict(row) for row in self.corpus]
        poisoned[0]["contains_exercise_condition_solution_example"] = True
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_theory_rows(poisoned)
        poisoned = [dict(row) for row in self.corpus]
        poisoned[0]["solution"] = "do not admit"
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_theory_rows(poisoned)

    def test_all_answer_contracts_fail_closed(self) -> None:
        self.assertEqual(protocol.parse_answer_content('{"final_answer":"A"}', "choice")["final_answer"], "A")
        for malformed in ('{"final_answer":"Z"}', '{"answer":"A"}', 'A', ''):
            with self.assertRaises(protocol.ProtocolError):
                protocol.parse_answer_content(malformed, "choice")
        self.assertEqual(protocol.parse_answer_content('{"final_answer":"42"}', "numeric")["final_answer"], "42")

    def test_v5_fallback_is_generic_theory_only(self) -> None:
        row = self.public[0]
        retrieval = protocol.retrieve_theory(row, self.corpus)
        for variant in protocol.FALLBACK_VARIANTS:
            request = protocol.build_fallback_request(row, retrieval, variant)
            wire = protocol.canonical_json_bytes(request)
            self.assertNotIn(b"task_id", wire)
            self.assertIn(b"enable_thinking", wire)
        answer, mode = protocol.choose_fallback(
            [
                {"final_answer": "A", "evidence": "x"},
                {"final_answer": "A", "evidence": "y"},
                {"final_answer": "B", "evidence": "z"},
            ],
            "choice",
        )
        self.assertEqual((answer, mode), ("A", "v5_valid_consensus"))

    def test_coverage_is_aggregate_only(self) -> None:
        coverage = protocol.coverage_aggregate(self.public, self.corpus)
        self.assertEqual(coverage["benchmark_rows"], 274)
        self.assertFalse(coverage["per_task_identifiers_present"])
        raw = protocol.canonical_json_bytes(coverage)
        self.assertNotIn(b"val_", raw)

    def test_exclusive_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "append-only.json"
            protocol.exclusive_json(path, {"first": True})
            with self.assertRaises(FileExistsError):
                protocol.exclusive_json(path, {"second": True})
            self.assertEqual(protocol.read_json(path), {"first": True})


if __name__ == "__main__":
    unittest.main()
