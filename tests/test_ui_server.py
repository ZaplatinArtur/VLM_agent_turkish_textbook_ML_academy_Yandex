import json
import tempfile
import unittest
from pathlib import Path

from vlm_judge.ui_server import AnnotationStore, GoldStore, ImageCache, build_binary_judge_context


class AnnotationStoreTests(unittest.TestCase):
    def test_binary_judge_context_keeps_zero_one_results_separate(self) -> None:
        tasks = [
            {"task_id": "m1", "setup": "no_tools", "subject": "math"},
            {"task_id": "m2", "setup": "no_tools", "subject": "physics"},
            {"task_id": "m3", "setup": "no_tools", "subject": "chemistry"},
        ]
        results = [
            {
                "task_id": "m1",
                "setup": "no_tools",
                "prompt_version": "text-binary-v4",
                "verdict": {"score": 1, "rationale": "Correct."},
                "judge": {"model": "Qwen", "metadata": {}},
            },
            {
                "task_id": "m2",
                "setup": "no_tools",
                "prompt_version": "text-binary-v4",
                "verdict": {"score": 0, "rationale": "Wrong choice."},
                "judge": {
                    "model": "Qwen",
                    "metadata": {"deterministic_choice_mismatch": True},
                },
            },
        ]

        context = build_binary_judge_context(tasks, results)

        self.assertTrue(context["enabled"])
        self.assertEqual(context["summary"]["valid"], 2)
        self.assertEqual(context["summary"]["failed"], 1)
        self.assertEqual(context["summary"]["score_0"], 1)
        self.assertEqual(context["summary"]["score_1"], 1)
        self.assertEqual(context["summary"]["guarded"], 1)
        self.assertEqual(context["items"][0]["_binary_judge"]["verdict"]["score"], 1)
        self.assertIsNone(context["items"][2]["_binary_judge"])

    def test_binary_judge_context_ignores_regular_rubric_results(self) -> None:
        context = build_binary_judge_context(
            [{"task_id": "m1", "setup": "no_tools"}],
            [
                {
                    "task_id": "m1",
                    "setup": "no_tools",
                    "prompt_version": "judge-v1",
                    "verdict": {
                        "score": 1,
                        "label": "partially_correct",
                        "rationale": "Incomplete.",
                    },
                }
            ],
        )
        self.assertFalse(context["enabled"])
        self.assertEqual(context["summary"]["valid"], 0)

    def test_upsert_reload_and_csv_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.jsonl"
            store = AnnotationStore(path)
            store.upsert(
                {
                    "task_id": "m1",
                    "status": "complete",
                    "score": 4,
                    "error_types": ["unit_or_format"],
                }
            )
            reloaded = AnnotationStore(path)
            records = reloaded.list()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["score"], 4)
            csv_text = reloaded.export_csv().decode("utf-8-sig")
            self.assertIn("unit_or_format", csv_text)

    def test_annotation_store_rejects_invalid_or_inconsistent_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AnnotationStore(Path(directory) / "annotations.jsonl")
            with self.assertRaises(ValueError):
                store.upsert(
                    {
                        "task_id": "m1",
                        "mode": "pointwise",
                        "status": "complete",
                        "score": "4",
                    }
                )
            with self.assertRaises(ValueError):
                store.upsert(
                    {
                        "task_id": "m1",
                        "mode": "pointwise",
                        "status": "complete",
                        "score": 1,
                        "label": "incorrect",
                    }
                )
            saved = store.upsert(
                {
                    "task_id": "m1",
                    "mode": "pointwise",
                    "status": "complete",
                    "score": 1,
                    "label": "partially_correct",
                    "strict_correct": False,
                    "confidence": 0.8,
                    "error_types": [],
                }
            )
            self.assertEqual(saved["label"], "partially_correct")

    def test_image_allowlist(self) -> None:
        cache = ImageCache(Path(tempfile.gettempdir()) / "vlm-judge-test-images")
        cache._validate("https://yadi.sk/i/example")
        with self.assertRaises(ValueError):
            cache._validate("http://127.0.0.1/private")

    def test_gold_store_is_task_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.jsonl"
            store = GoldStore(path)
            store.upsert(
                {
                    "task_id": "m1",
                    "status": "verified",
                    "quality": "clear",
                    "transcription": "42",
                    "acceptable_answers": ["42", "42.0"],
                    "subanswers": [],
                }
            )
            self.assertEqual(GoldStore(path).list()[0]["transcription"], "42")
            self.assertIn("acceptable_answers", store.export_csv().decode("utf-8-sig"))

            with self.assertRaisesRegex(ValueError, "explicit quality"):
                store.upsert(
                    {
                        "task_id": "m2",
                        "status": "verified",
                        "quality": "unknown",
                        "transcription": "7",
                    }
                )
            with self.assertRaisesRegex(ValueError, "transcription or subanswers"):
                store.upsert(
                    {
                        "task_id": "m3",
                        "status": "verified",
                        "quality": "clear",
                    }
                )


if __name__ == "__main__":
    unittest.main()
