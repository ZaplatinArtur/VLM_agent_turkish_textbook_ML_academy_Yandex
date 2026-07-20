import unittest
from collections import Counter

from vlm_judge.calibration import (
    DEFAULT_QUOTAS,
    build_synthetic_mc_stress,
    select_calibration_tasks,
)


class CalibrationTests(unittest.TestCase):
    def test_calibration_quotas(self) -> None:
        tasks = []
        for answer_type, quota in DEFAULT_QUOTAS.items():
            for index in range(quota + 3):
                tasks.append(
                    {
                        "task_id": f"{answer_type}-{index}",
                        "answer_type": answer_type,
                        "grade": (index % 12) + 1,
                        "metadata": {
                            "easy": index % 3 == 0,
                            "medium": index % 3 == 1,
                            "hard": index % 3 == 2,
                        },
                    }
                )
        selected = select_calibration_tasks(tasks)
        self.assertEqual(len(selected), sum(DEFAULT_QUOTAS.values()))
        self.assertEqual(
            Counter(task["answer_type"] for task in selected),
            Counter(DEFAULT_QUOTAS),
        )
        self.assertEqual(len({task["task_id"] for task in selected}), len(selected))

    def test_synthetic_mc_stress_has_three_cases(self) -> None:
        task = {
            "task_id": "m1",
            "answer_type": "multiple_choice",
            "reference_answer": "C",
            "metadata": {},
        }
        records = build_synthetic_mc_stress([task])
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["candidate_answer"], "Final answer: C")
        self.assertEqual(records[1]["candidate_answer"], "Final answer: D")


if __name__ == "__main__":
    unittest.main()
