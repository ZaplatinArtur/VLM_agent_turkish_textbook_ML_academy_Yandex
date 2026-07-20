from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vlm_judge.adjudication import AdjudicationStore, build_adjudication_context


def _task(task_id: str, candidate: str = "answer") -> dict:
    return {
        "task_id": task_id,
        "setup": "no_tools",
        "subject": "math",
        "answer_type": "short_text",
        "question_text": "q",
        "reference_answer": "gold",
        "candidate_answer": candidate,
    }


def _judge(task_id: str, score: int, confidence: float = 0.9) -> dict:
    return {
        "task_id": task_id,
        "setup": "no_tools",
        "verdict": {
            "score": score,
            "label": "fully_correct" if score == 4 else "incorrect",
            "strict_correct": score == 4,
            "final_answer_correct": score == 4,
            "reasoning_correct": None,
            "complete": True,
            "confidence": confidence,
            "error_types": [],
            "reference_quality_issue": False,
            "rationale": "judge",
        },
        "judge": {"backend": "mock", "model": "qwen", "error": None},
    }


def _human(task_id: str, score: int) -> dict:
    return {
        "annotation_id": f"{task_id}::no_tools",
        "task_id": task_id,
        "setup": "no_tools",
        "status": "complete",
        "score": score,
        "strict_correct": score == 4,
        "confidence": 0.9,
        "rationale": "human",
    }


class AdjudicationTests(unittest.TestCase):
    def test_build_context_prioritizes_disagreement_and_samples_agreement(self) -> None:
        context = build_adjudication_context(
            [_task("a"), _task("b")],
            [_judge("a", 0), _judge("b", 4)],
            [_human("a", 4), _human("b", 4)],
            agreement_sample_rate=1.0,
        )

        self.assertTrue(context["enabled"])
        self.assertEqual(context["stats"]["eligible"], 2)
        self.assertEqual(context["stats"]["strict_disagreements"], 1)
        self.assertEqual(context["stats"]["agreement_controls"], 1)
        self.assertEqual([item["task_id"] for item in context["items"]], ["a", "b"])
        self.assertEqual(
            context["items"][0]["_adjudication"]["reasons"],
            ["score_disagreement", "strict_disagreement"],
        )
        self.assertEqual(
            context["items"][1]["_adjudication"]["reasons"],
            ["agreement_control"],
        )

    def test_context_requires_completed_human_annotation(self) -> None:
        human = _human("a", 4)
        human["status"] = "draft"
        context = build_adjudication_context([_task("a")], [_judge("a", 0)], [human])
        self.assertEqual(context["stats"]["eligible"], 0)
        self.assertEqual(context["items"], [])

    def test_store_validates_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AdjudicationStore(Path(directory) / "adjudications.jsonl")
            saved = store.upsert(
                {
                    "adjudication_id": "adj::a::no_tools",
                    "task_id": "a",
                    "setup": "no_tools",
                    "status": "resolved",
                    "decision": "custom",
                    "final_score": 3,
                    "issue_source": "judge",
                    "rationale": "checked",
                }
            )
            self.assertEqual(saved["final_score"], 3)
            self.assertEqual(saved["final_label"], "mostly_correct")
            self.assertEqual(AdjudicationStore(store.path).list()[0]["decision"], "custom")

            with self.assertRaisesRegex(ValueError, "requires final_score"):
                store.upsert(
                    {
                        "adjudication_id": "adj::b::no_tools",
                        "task_id": "b",
                        "status": "resolved",
                        "decision": "human",
                    }
                )


if __name__ == "__main__":
    unittest.main()
