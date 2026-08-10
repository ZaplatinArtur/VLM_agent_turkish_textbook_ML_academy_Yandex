#!/usr/bin/env python3
"""CPU-only tests for the preregistered paired RAG/no-RAG branch."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import compose_maxim_paired_rag_norag_semantic_support_v1 as composer
    import prepare_maxim_paired_rag_norag_semantic_support_v1 as preparation
    import run_maxim_paired_rag_norag_semantic_support_v1 as runner
except ModuleNotFoundError:  # Imported from repository root.
    from scripts import compose_maxim_paired_rag_norag_semantic_support_v1 as composer
    from scripts import prepare_maxim_paired_rag_norag_semantic_support_v1 as preparation
    from scripts import run_maxim_paired_rag_norag_semantic_support_v1 as runner


def task(task_id: str, answer_type: str = "choice") -> dict:
    return {
        "task_id": task_id,
        "subject": "Math",
        "grade": 8,
        "question": "Visible question",
        "question_images": [{"data": f"{task_id}.png"}],
        "answer_type": answer_type,
        "reference_answer": "must never enter queue",
    }


def solver(task_id: str, answer: str) -> dict:
    return {
        "task_id": task_id,
        "condition": "hidden-source-condition",
        "model": "hidden-model",
        "final_answer": answer,
        "reasoning": f"reasoning for {answer}",
        "solution_steps": f"steps for {answer}",
        "generation": {"gold_access": False},
        "error": None,
    }


def context(task_id: str, *, safe: bool = True, text: str = "FACT alpha beta") -> dict:
    return {
        "task_id": task_id,
        "route": "structural_evidence" if safe else "router_page_rag_fallback",
        "safety": {"safe": safe},
        "evidence": [
            {
                "chunk_id": f"chunk-{task_id}",
                "document_id": "book",
                "page_number": 7,
                "primary_type": "theory",
                "relation": "direct",
                "text": text,
            }
        ],
    }


def positive_verdict(task_id: str, request_sha256: str) -> dict:
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "task_id": task_id,
        "queue_index": 0,
        "request_sha256": request_sha256,
        "condition": runner.CONDITION,
        "prompt_version": runner.PROMPT_VERSION,
        "parsed": {
            "question_reconstruction": "question",
            "rag_answer_supported": True,
            "confidence": 0.91,
            "contradiction_found": False,
            "unsupported_decisive_steps": [],
            "citations": [
                {
                    "chunk_id": f"chunk-{task_id}",
                    "exact_quote": "alpha beta",
                    "supports": "decisive relation",
                    "decisive": True,
                }
            ],
            "answer_format_verified": True,
            "audit_summary": "supported",
        },
        "call": {"attempt": 1},
        "error": None,
        "gold_access": False,
    }


PROFILE = {
    "selection_gate": {
        "min_confidence": 0.85,
        "min_decisive_valid_citations": 1,
        "all_returned_citations_must_validate": True,
    }
}


class PreparationTests(unittest.TestCase):
    def test_gold_keys_are_removed_and_only_disagreement_with_context_is_queued(self) -> None:
        tasks = [task("a"), task("b"), task("c")]
        page = [solver("a", "A"), solver("b", "B"), solver("c", "C")]
        no_rag = [solver("a", "A."), solver("b", "C"), solver("c", "D")]
        contexts = [context("a"), context("b"), context("c", safe=False)]
        with mock.patch.object(preparation, "EXPECTED_ROWS", 3):
            queue, stats = preparation.build_queue(tasks, page, no_rag, contexts)
        self.assertEqual([row["task_id"] for row in queue], ["b"])
        self.assertEqual(stats["route_counts"]["same_answer_default_no_rag"], 1)
        self.assertEqual(
            stats["route_counts"]["unsafe_or_missing_context_default_no_rag"], 1
        )
        serialized = preparation.stable_json(queue)
        self.assertNotIn("reference_answer", serialized)
        self.assertNotIn("hidden-source-condition", serialized)
        self.assertNotIn("hidden-model", serialized)

    def test_context_packing_is_deterministic_and_capped(self) -> None:
        long = "x" * (preparation.MAX_CONTEXT_CHARS + 50)
        packed = preparation.pack_safe_context(context("a", text=long))
        self.assertEqual(len(packed), 1)
        self.assertEqual(len(packed[0]["text"]), preparation.MAX_CONTEXT_CHARS)
        self.assertTrue(packed[0]["text_cut_by_queue"])

    def test_forbidden_key_audit_is_recursive(self) -> None:
        with self.assertRaises(preparation.PreparationError):
            preparation.audit_gold_free({"nested": [{"judge": "hidden"}]})


class GateTests(unittest.TestCase):
    def queue_row(self) -> dict:
        payload = {
            "schema_version": preparation.QUEUE_SCHEMA_VERSION,
            "queue_index": 0,
            "task_id": "b",
            "subject": "Math",
            "grade": 8,
            "question": "Visible",
            "question_images": [{"data": "b.png"}],
            "answer_type": "choice",
            "rag_candidate": preparation.candidate_payload(solver("b", "B"), "page"),
            "no_rag_candidate": preparation.candidate_payload(solver("b", "C"), "no"),
            "contexts": preparation.pack_safe_context(context("b")),
        }
        payload["request_sha256"] = preparation.stable_sha256(payload)
        return payload

    def test_positive_gate_requires_exact_quote(self) -> None:
        queue = self.queue_row()
        verdict = positive_verdict("b", queue["request_sha256"])
        passed, reasons, audit = composer.gate_verdict(verdict, queue, PROFILE)
        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        self.assertEqual(audit["decisive_valid_count"], 1)

        bad = copy.deepcopy(verdict)
        bad["parsed"]["citations"][0]["exact_quote"] = "Alpha beta"
        passed, reasons, audit = composer.gate_verdict(bad, queue, PROFILE)
        self.assertFalse(passed)
        self.assertIn("invalid_returned_citation", reasons)
        self.assertEqual(audit["decisive_valid_count"], 0)

    def test_every_failclosed_condition_defaults(self) -> None:
        queue = self.queue_row()
        base = positive_verdict("b", queue["request_sha256"])
        mutations = [
            ("confidence", 0.84, "confidence_below_threshold"),
            ("contradiction_found", True, "contradiction_found"),
            ("answer_format_verified", False, "answer_format_not_verified"),
            ("rag_answer_supported", False, "rag_not_supported"),
        ]
        for key, value, expected_reason in mutations:
            with self.subTest(key=key):
                verdict = copy.deepcopy(base)
                verdict["parsed"][key] = value
                passed, reasons, _ = composer.gate_verdict(verdict, queue, PROFILE)
                self.assertFalse(passed)
                self.assertIn(expected_reason, reasons)
        unsupported = copy.deepcopy(base)
        unsupported["parsed"]["unsupported_decisive_steps"] = ["missing derivation"]
        passed, reasons, _ = composer.gate_verdict(unsupported, queue, PROFILE)
        self.assertFalse(passed)
        self.assertIn("unsupported_decisive_steps", reasons)


class CompositionTests(unittest.TestCase):
    def test_full_composition_selects_rag_only_for_passing_gate(self) -> None:
        tasks = [task("a"), task("b"), task("c")]
        page = [solver("a", "A"), solver("b", "B"), solver("c", "C")]
        no_rag = [solver("a", "A."), solver("b", "C"), solver("c", "D")]
        contexts = [context("a"), context("b"), context("c", safe=False)]
        with mock.patch.object(preparation, "EXPECTED_ROWS", 3):
            queue, _ = preparation.build_queue(tasks, page, no_rag, contexts)
            verdict = positive_verdict("b", queue[0]["request_sha256"])
            output, stats = composer.compose(
                tasks, page, no_rag, contexts, queue, [verdict], PROFILE
            )
        self.assertEqual([row["final_answer"] for row in output], ["A.", "B", "D"])
        self.assertEqual(stats["rag_selected_rows"], 1)
        self.assertEqual(stats["no_rag_selected_rows"], 2)
        self.assertTrue(all(row["generation"]["gold_access"] is False for row in output))


class RunnerTests(unittest.TestCase):
    def test_runner_requests_exactly_one_internal_error_retry(self) -> None:
        queue = GateTests().queue_row()

        class FakePool:
            def __init__(self) -> None:
                self.retries = None

            def complete(self, **kwargs):
                self.retries = kwargs["retries"]
                parsed = positive_verdict("b", queue["request_sha256"])["parsed"]
                return {
                    "parsed": parsed,
                    "endpoint": "fake",
                    "finish_reason": "stop",
                    "attempt": 1,
                    "latency_s": 0.1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "recovered_partial": False,
                    "parse_error": None,
                }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "b.png").write_bytes(b"image")
            pool = FakePool()
            result = runner.run_one(
                queue,
                pool=pool,  # type: ignore[arg-type]
                image_root=root,
                image_url_root="file:///images",
                max_tokens=1800,
                base_seed=20260803,
            )
        self.assertEqual(pool.retries, 1)
        self.assertIsNone(result["error"])
        self.assertFalse(result["gold_access"])


if __name__ == "__main__":
    unittest.main()
