import unittest

from vlm_judge.calibration_analysis import analyze_arena_annotations, analyze_calibration


def verdict(score: int, confidence: float) -> dict:
    return {
        "score": score,
        "label": "fully_correct" if score == 4 else "incorrect",
        "strict_correct": score == 4,
        "final_answer_correct": score == 4,
        "reasoning_correct": None,
        "complete": True,
        "confidence": confidence,
        "error_types": [],
        "rationale": "test",
        "reference_quality_issue": False,
    }


class CalibrationAnalysisTests(unittest.TestCase):
    def test_perfect_judge_agreement(self) -> None:
        humans = [
            {"task_id": "q1", "setup": "no_tools", "status": "complete", "score": 4, "subject": "Math"},
            {"task_id": "q2", "setup": "no_tools", "status": "complete", "score": 0, "subject": "Math"},
        ]
        judges = [
            {"task_id": "q1", "setup": "no_tools", "verdict": verdict(4, 0.9)},
            {"task_id": "q2", "setup": "no_tools", "verdict": verdict(0, 0.8)},
        ]
        report = analyze_calibration(humans, judges)
        self.assertEqual(report["overall"]["exact_score_agreement"], 1.0)
        self.assertEqual(report["overall"]["quadratic_weighted_kappa"], 1.0)
        self.assertEqual(report["overall"]["macro_f1_5_score"], 1.0)
        self.assertEqual(report["by_setup"]["no_tools"]["comparisons"], 2)
        self.assertEqual(
            report["selective_agreement_by_minimum_confidence"]["0.80"]["exact_score_agreement"],
            1.0,
        )

    def test_mirrored_arena_normalizes_sides(self) -> None:
        records = [
            {
                "task_id": "q1", "status": "complete", "winner": "A",
                "candidate_a_setup": "no_tools", "candidate_b_setup": "web_search",
                "side_swapped": False,
            },
            {
                "task_id": "q1", "status": "complete", "winner": "B",
                "candidate_a_setup": "web_search", "candidate_b_setup": "no_tools",
                "side_swapped": True,
            },
        ]
        report = analyze_arena_annotations(records)
        self.assertEqual(report["underlying_decision_consistency"], 1.0)
        self.assertEqual(report["same_side_selection_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
