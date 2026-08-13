from __future__ import annotations

import unittest

from failover_rule import apply_failover


def fallback(index: int, answer: str = "B") -> dict[str, object]:
    return {
        "schema_version": "generic-v5-theory-content-fallback-v1",
        "content_sha256": f"{index:064x}",
        "prediction": answer,
        "source_arm": "local_textbook_theory_bm25",
        "gold_access": False,
        "final_access": False,
        "opaque_identifier_retained": False,
    }


def valid_v6(index: int, answer: str = "A") -> dict[str, object]:
    return {
        "schema_version": "generic-medium-nonstream-content-prediction-v6",
        "content_sha256": f"{index:064x}",
        "request_sha256": "f" * 64,
        "prediction": answer,
        "terminal_success": True,
        "attempt_count": 1,
        "terminal_error_kind": None,
        "model_contract_error": None,
        "gold_access": False,
        "final_access": False,
        "opaque_identifier_access": False,
    }


class FailoverRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fallbacks = [fallback(index) for index in range(185)]

    def test_exact_valid_success_selects_v6(self) -> None:
        result = apply_failover([valid_v6(0, "D")], self.fallbacks)
        self.assertEqual(result[0]["prediction"], "D")
        self.assertEqual(result[0]["selected_source"], "v6_2_strict_success")
        self.assertIsNone(result[0]["fallback_reason"])

    def test_missing_and_error_fall_back(self) -> None:
        row = valid_v6(1, "D")
        row["terminal_success"] = False
        row["terminal_error_kind"] = "transport_timeout"
        result = apply_failover([row], self.fallbacks)
        self.assertEqual(result[0]["fallback_reason"], "v6_missing")
        self.assertEqual(result[1]["prediction"], "B")
        self.assertEqual(result[1]["fallback_reason"], "v6_invalid_schema_or_error")

    def test_schema_extension_is_rejected(self) -> None:
        row = valid_v6(2, "E")
        row["confidence"] = 1.0
        result = apply_failover([row], self.fallbacks)
        self.assertEqual(result[2]["prediction"], "B")
        self.assertEqual(result[2]["selected_source"], "v5_theory_fallback")

    def test_duplicate_content_falls_back(self) -> None:
        row = valid_v6(3, "E")
        result = apply_failover([row, dict(row)], self.fallbacks)
        self.assertEqual(result[3]["fallback_reason"], "v6_duplicate_content")

    def test_output_has_no_identity_or_outcome_fields(self) -> None:
        result = apply_failover([valid_v6(0)], self.fallbacks)
        forbidden = {"benchmark_id", "task_id", "gold", "correct", "confidence"}
        self.assertFalse(forbidden.intersection(result[0]))
        self.assertEqual(len(result), 185)


if __name__ == "__main__":
    unittest.main()
