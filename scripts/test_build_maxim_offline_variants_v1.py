from __future__ import annotations

import unittest

from scripts import build_maxim_offline_variants_v1 as composer


def _benchmark(task_id: str, subject: str, answer_type: str) -> dict:
    # Gold-like fields are included to prove that composition projects only the
    # explicitly visible benchmark fields and never copies references out.
    return {
        "task_id": task_id,
        "subject": subject,
        "answer_type": answer_type,
        "question": "visible question",
        "reference_answer": "SECRET",
        "reference_solution": "SECRET SOLUTION",
    }


def _solver(task_id: str, answer: str, condition: str) -> dict:
    return {
        "task_id": task_id,
        "condition": condition,
        "prompt_version": "source-v1",
        "model": "test-model",
        "final_answer": answer,
        "solution_steps": f"steps for {answer}",
        "reasoning": f"reasoning for {answer}",
        "raw_response": answer,
        "tool_calls": [],
        "usage": {"input_tokens": 10, "output_tokens": 2, "latency_s": 0.5},
        "error": None,
    }


def _parallel_row(task_id: str, selected: str, answers: list[str]) -> dict:
    row = _solver(task_id, selected, "parallel-source")
    row["generation"] = {
        "gold_access": False,
        "selected_index": 8,
        "call_count": 9,
        "candidate_traces": [
            {
                "index": index,
                "route": f"route-{index}",
                "final_answer": answer,
                "reasoning": f"candidate reasoning {index}",
            }
            for index, answer in enumerate(answers, start=1)
        ],
    }
    return row


def _search(chunk_ids: list[str], error: str | None = None) -> dict:
    return {
        "tool": "search_textbooks",
        "args": {"query": "visible query"},
        "returned_chunk_ids": chunk_ids,
        "error": error,
    }


class SubjectRouterTests(unittest.TestCase):
    def test_math_uses_no_tools_and_non_math_uses_page_rag(self) -> None:
        benchmark = [
            _benchmark("m", "Math", "numeric"),
            _benchmark("h", "History", "choice"),
        ]
        page = [_solver("m", "page-m", "page"), _solver("h", "page-h", "page")]
        no_tools = [
            _solver("m", "plain-m", "no-tools"),
            _solver("h", "plain-h", "no-tools"),
        ]

        rows, stats = composer.compose_subject_router(benchmark, page, no_tools)

        self.assertEqual(["plain-m", "page-h"], [row["final_answer"] for row in rows])
        self.assertEqual({"no_tools": 1, "page_rag": 1}, stats["chosen_source_counts"])
        self.assertEqual(
            ["task_id", "subject"], rows[0]["offline_provenance"]["benchmark_fields_used"]
        )
        self.assertEqual("no_tools", rows[0]["offline_provenance"]["decision"]["chosen_source"])
        self.assertFalse(rows[0]["generation"]["gold_access"])
        self.assertNotIn("reference_answer", rows[0])
        self.assertNotIn("reference_solution", rows[0])


class ParallelConsensusTests(unittest.TestCase):
    def test_unique_modal_cluster_and_tie_fallback(self) -> None:
        benchmark = [
            _benchmark("modal", "Math", "choice"),
            _benchmark("tie", "Math", "choice"),
        ]
        parallel = [
            _parallel_row(
                "modal",
                "E",
                ["(a)", "A", "cevap: a", "B", "B", "C", "D", "E"],
            ),
            _parallel_row("tie", "E", ["A", "A", "B", "B", "C", "C", "D", "D"]),
        ]

        rows, stats = composer.compose_parallel_consensus(benchmark, parallel)

        self.assertEqual("A", rows[0]["final_answer"])
        self.assertEqual("candidate reasoning 1", rows[0]["reasoning"])
        self.assertEqual("candidate reasoning 1", rows[0]["solution_steps"])
        self.assertEqual(
            "unique_modal_cluster", rows[0]["offline_provenance"]["decision"]["kind"]
        )
        self.assertEqual("E", rows[1]["final_answer"])
        self.assertEqual(
            "tie_fallback_saved_selector",
            rows[1]["offline_provenance"]["decision"]["kind"],
        )
        self.assertEqual(1, stats["final_answer_changed_rows"])
        self.assertEqual(
            {"tie_fallback_saved_selector": 1, "unique_modal_cluster": 1},
            stats["decision_counts"],
        )

    def test_parallel_requires_exactly_eight_valid_traces(self) -> None:
        benchmark = [_benchmark("x", "Math", "choice")]
        parallel = [_parallel_row("x", "A", ["A"] * 7)]
        with self.assertRaisesRegex(composer.CompositionError, "expected 8"):
            composer.compose_parallel_consensus(benchmark, parallel)

    def test_empty_candidate_falls_back_for_entire_row(self) -> None:
        benchmark = [_benchmark("x", "Math", "choice")]
        parallel = [_parallel_row("x", "E", ["A", "A", "A", "B", "C", "D", "E", ""])]

        rows, stats = composer.compose_parallel_consensus(benchmark, parallel)

        self.assertEqual("E", rows[0]["final_answer"])
        decision = rows[0]["offline_provenance"]["decision"]
        self.assertEqual("invalid_candidate_fallback_saved_selector", decision["kind"])
        self.assertEqual([8], decision["invalid_candidate_indices"])
        self.assertEqual(
            {"invalid_candidate_fallback_saved_selector": 1}, stats["decision_counts"]
        )


class CanonicalizationTests(unittest.TestCase):
    def test_only_safe_answer_type_formatting_changes(self) -> None:
        benchmark = [
            _benchmark("choice", "History", "choice"),
            _benchmark("numeric_unit", "Math", "numeric"),
            _benchmark("two_numbers", "Math", "numeric"),
            _benchmark("wrapped", "History", "choice"),
            _benchmark("text", "English", "short_text"),
        ]
        sources = [
            _solver("choice", " (c) ", "source"),
            _solver("numeric_unit", "-18 \u00b0C", "source"),
            _solver("two_numbers", "12 naneli ve 8 limonlu", "source"),
            _solver("wrapped", '{"final_answer": "b"}', "source"),
            _solver("text", "  x  +  y  ", "source"),
        ]

        rows, stats = composer.compose_answer_canonicalization(benchmark, sources)

        self.assertEqual(
            ["C", "-18", "12 naneli ve 8 limonlu", "B", "x  +  y"],
            [row["final_answer"] for row in rows],
        )
        self.assertEqual(4, stats["changed_rows"])
        self.assertEqual(
            {"choice": 2, "numeric": 1, "short_text": 1},
            stats["changed_rows_by_answer_type"],
        )
        for row in rows:
            self.assertNotIn("reference_answer", row)
            self.assertFalse(row["offline_provenance"]["gold_access"])


class SelectiveRagMetadataTests(unittest.TestCase):
    def test_repeated_chunk_without_errors_uses_rag(self) -> None:
        page = _solver("x", "rag-answer", "page")
        page["tool_calls"] = [_search(["a", "b"]), _search(["b", "c"])]
        gate = composer.retrieval_metadata_gate(page)
        self.assertTrue(gate["use_rag"])
        self.assertEqual(1, gate["repeated_chunk_count"])

    def test_error_or_no_cross_call_agreement_falls_back(self) -> None:
        error_row = _solver("x", "rag-answer", "page")
        error_row["tool_calls"] = [
            _search(["a", "b"]),
            _search(["b", "c"]),
            _search([], error="duplicate query"),
        ]
        no_repeat_row = _solver("y", "rag-answer", "page")
        no_repeat_row["tool_calls"] = [_search(["a"]), _search(["b"])]
        self.assertFalse(composer.retrieval_metadata_gate(error_row)["use_rag"])
        self.assertFalse(composer.retrieval_metadata_gate(no_repeat_row)["use_rag"])

    def test_composition_records_both_branch_costs(self) -> None:
        benchmark = [
            _benchmark("rag", "History", "choice"),
            _benchmark("plain", "Math", "numeric"),
        ]
        page_rag = [
            _solver("rag", "rag-answer", "page"),
            _solver("plain", "wrong-rag-answer", "page"),
        ]
        page_rag[0]["tool_calls"] = [_search(["a", "b"]), _search(["b", "c"])]
        page_rag[1]["tool_calls"] = [_search(["a"]), _search(["b"])]
        no_tools = [
            _solver("rag", "plain-answer", "plain"),
            _solver("plain", "plain-answer", "plain"),
        ]

        rows, stats = composer.compose_selective_rag_metadata(
            benchmark, page_rag, no_tools
        )

        self.assertEqual(["rag-answer", "plain-answer"], [row["final_answer"] for row in rows])
        self.assertEqual({"no_tools": 1, "page_rag": 1}, stats["chosen_source_counts"])
        self.assertEqual(20, rows[0]["usage"]["input_tokens"])
        self.assertEqual(4, rows[0]["usage"]["output_tokens"])
        self.assertEqual(1.0, rows[0]["usage"]["latency_s"])
        self.assertEqual(2, rows[0]["generation"]["call_count"])
        self.assertEqual(2, len(rows[0]["offline_provenance"]["sources"]))


class FailClosedTests(unittest.TestCase):
    def test_duplicate_task_id_is_rejected(self) -> None:
        benchmark = [
            _benchmark("dup", "Math", "numeric"),
            _benchmark("dup", "Math", "numeric"),
        ]
        with self.assertRaisesRegex(composer.CompositionError, "duplicate task_id"):
            composer.compose_subject_router(
                benchmark,
                [_solver("dup", "1", "page")],
                [_solver("dup", "2", "plain")],
            )

    def test_gold_field_in_solver_source_is_rejected(self) -> None:
        benchmark = [_benchmark("x", "Math", "numeric")]
        bad_page = _solver("x", "1", "page")
        bad_page["reference_answer"] = "1"
        with self.assertRaisesRegex(composer.CompositionError, "forbidden gold fields"):
            composer.compose_subject_router(
                benchmark,
                [bad_page],
                [_solver("x", "2", "plain")],
            )


if __name__ == "__main__":
    unittest.main()
