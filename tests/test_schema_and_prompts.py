import unittest
import json

from vlm_judge.parsing import parse_judge_verdict
from vlm_judge.pipeline import request_id
from vlm_judge.prompts import build_judge_request
from vlm_judge.schema import EvaluationItem


class SchemaAndPromptTests(unittest.TestCase):
    def test_setup_is_blinded(self) -> None:
        item = EvaluationItem(
            task_id="m0101",
            candidate_answer="2, 4, 6",
            subject="math",
            setup="textbook_retrieval",
            question_image_url="https://example.test/question.png",
            reference_image_url="https://example.test/reference.png",
            metadata={
                "run_id": "experiment-textbook_retrieval",
                "retrieved_chunk_ids": ["secret-retrieval-trace"],
                "expected_labels": ["fully_correct"],
                "required_subanswers": ["a", "b"],
            },
        )
        request = build_judge_request(item)
        self.assertNotIn("textbook_retrieval", request.user_prompt)
        self.assertNotIn("secret-retrieval-trace", request.user_prompt)
        self.assertNotIn("expected_labels", request.user_prompt)
        self.assertIn("required_subanswers", request.user_prompt)
        self.assertEqual(len(request.image_urls), 2)

        equivalent = EvaluationItem(
            task_id="m0101",
            candidate_answer="2, 4, 6",
            subject="math",
            setup="web_search",
            question_image_url="https://example.test/question.png",
            reference_image_url="https://example.test/reference.png",
            metadata={"run_id": "different-web-run", "required_subanswers": ["a", "b"]},
        )
        self.assertEqual(request_id(item), request_id(equivalent))

    def test_valid_verdict(self) -> None:
        raw = """{
          "label": "fully_correct",
          "score": 4,
          "strict_correct": true,
          "final_answer_correct": true,
          "reasoning_correct": true,
          "complete": true,
          "confidence": 0.95,
          "error_types": [],
          "rationale": "All required answers match.",
          "reference_quality_issue": false
        }"""
        verdict = parse_judge_verdict(raw)
        self.assertTrue(verdict.strict_correct)

    def test_task_identifier_cannot_leak_synthetic_expected_label(self) -> None:
        item = EvaluationItem(
            task_id="m1__wrong_concise__textbook_retrieval",
            candidate_answer="Final answer: B",
            setup="textbook_retrieval",
            question_text="Choose the correct answer.",
            reference_answer="A",
            answer_type="multiple_choice",
        )
        request = build_judge_request(item)
        self.assertNotIn("m1__wrong_concise", request.user_prompt)
        self.assertNotIn("textbook_retrieval", request.user_prompt)

    def test_verdict_schema_rejects_coercion_extra_keys_and_label_score_mismatch(self) -> None:
        base = {
            "label": "fully_correct",
            "score": 4,
            "strict_correct": True,
            "final_answer_correct": True,
            "reasoning_correct": True,
            "complete": True,
            "confidence": 0.9,
            "error_types": [],
            "rationale": "ok",
            "reference_quality_issue": False,
        }
        for invalid in (
            {**base, "strict_correct": "true"},
            {**base, "extra": "not allowed"},
            {**base, "label": "incorrect", "strict_correct": False},
            {**base, "final_answer_correct": False},
            {
                **base,
                "label": "unjudgeable",
                "score": 0,
                "strict_correct": False,
                "final_answer_correct": None,
                "reasoning_correct": True,
                "complete": None,
            },
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_judge_verdict(json.dumps(invalid))

    def test_single_reference_image_is_labeled_unambiguously(self) -> None:
        item = EvaluationItem(
            task_id="single-reference",
            candidate_answer="42",
            question_text="What is the answer?",
            reference_image_url="https://example.test/reference.png",
        )
        request = build_judge_request(item)
        self.assertEqual(request.image_labels, ("annotated reference-answer image",))
        self.assertIn("image 1: annotated reference-answer image", request.user_prompt)


if __name__ == "__main__":
    unittest.main()
