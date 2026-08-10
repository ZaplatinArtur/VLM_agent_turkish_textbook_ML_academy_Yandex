import json
import tempfile
import threading
import unittest
from pathlib import Path

from vlm_judge.backends import BackendResponse, ReplayBackend
from vlm_judge.runner import evaluate_items
from vlm_judge.schema import EvaluationItem


VALID_RESPONSE = json.dumps(
    {
        "label": "fully_correct",
        "score": 4,
        "strict_correct": True,
        "final_answer_correct": True,
        "reasoning_correct": True,
        "complete": True,
        "confidence": 0.9,
        "error_types": [],
        "rationale": "Matches the reference.",
        "reference_quality_issue": False,
    }
)


class RunnerTests(unittest.TestCase):
    def test_retry_blinding_and_cache(self) -> None:
        item = EvaluationItem(
            task_id="x1",
            candidate_answer="C",
            subject="Math",
            answer_type="multiple_choice",
            setup="textbook_retrieval",
            question_image_url="https://example.test/q.png",
            reference_answer="C",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = ReplayBackend(["not json", VALID_RESPONSE])
            output = root / "results.jsonl"
            count = evaluate_items(
                [item], backend, output, cache_dir=root / "cache", max_attempts=2
            )
            self.assertEqual(count, 1)
            self.assertEqual(len(backend.requests), 2)
            self.assertNotIn("textbook_retrieval", backend.requests[0].user_prompt)
            first = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(first["verdict"]["strict_correct"])
            self.assertEqual(first["judge"]["attempts"], 2)

            cached_backend = ReplayBackend([])
            evaluate_items([item], cached_backend, output, cache_dir=root / "cache")
            second = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(second["judge"]["cache_hit"])
            self.assertEqual(cached_backend.requests, [])

    def test_parallel_workers_preserve_input_order(self) -> None:
        class ConcurrentBackend:
            name = "concurrent-test"
            model = "test-model"

            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.calls = 0

            def complete(self, request):
                with self.lock:
                    self.calls += 1
                return BackendResponse(VALID_RESPONSE, self.model)

        items = [
            EvaluationItem(
                task_id=f"p{index}",
                candidate_answer="C",
                subject="Math",
                answer_type="multiple_choice",
                setup="no_tools",
                question_text=f"question {index}",
                reference_answer="C",
            )
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "parallel.jsonl"
            backend = ConcurrentBackend()
            count = evaluate_items(items, backend, output, workers=3)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(count, 6)
            self.assertEqual(backend.calls, 6)
            self.assertEqual([record["task_id"] for record in records], [f"p{index}" for index in range(6)])

    def test_cache_is_scoped_to_model_and_decoding_configuration(self) -> None:
        item = EvaluationItem(
            task_id="cache-scope",
            candidate_answer="C",
            question_text="Choose one.",
            reference_answer="C",
            answer_type="multiple_choice",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluate_items(
                [item],
                ReplayBackend([VALID_RESPONSE], model="judge-a"),
                root / "first.jsonl",
                cache_dir=root / "cache",
            )
            other_model = ReplayBackend([VALID_RESPONSE], model="judge-b")
            evaluate_items(
                [item],
                other_model,
                root / "second.jsonl",
                cache_dir=root / "cache",
            )
            second = json.loads((root / "second.jsonl").read_text(encoding="utf-8"))
            self.assertFalse(second["judge"]["cache_hit"])
            self.assertEqual(len(other_model.requests), 1)

    def test_cache_is_scoped_to_provider(self) -> None:
        class ProviderReplayBackend(ReplayBackend):
            def __init__(self, responses, *, provider: str) -> None:
                super().__init__(responses, model="judge")
                self.provider = provider

        item = EvaluationItem(
            task_id="provider-scope",
            candidate_answer="C",
            question_text="Choose one.",
            reference_answer="C",
            answer_type="multiple_choice",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluate_items(
                [item],
                ProviderReplayBackend([VALID_RESPONSE], provider="vllm"),
                root / "first.jsonl",
                cache_dir=root / "cache",
            )
            openrouter = ProviderReplayBackend([VALID_RESPONSE], provider="openrouter")
            evaluate_items(
                [item],
                openrouter,
                root / "second.jsonl",
                cache_dir=root / "cache",
            )
            second = json.loads((root / "second.jsonl").read_text(encoding="utf-8"))
            self.assertFalse(second["judge"]["cache_hit"])
            self.assertEqual(len(openrouter.requests), 1)


if __name__ == "__main__":
    unittest.main()
