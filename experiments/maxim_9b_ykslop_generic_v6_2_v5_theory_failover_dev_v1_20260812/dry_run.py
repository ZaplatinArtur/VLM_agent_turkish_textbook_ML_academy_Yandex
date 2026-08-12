"""Synthetic-only dry run; it never opens benchmark or model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from failover_rule import apply_failover, selection_counts
from test_failover_rule import fallback, valid_v6


HERE = Path(__file__).resolve().parent


def main() -> None:
    fallbacks = [fallback(index, "B") for index in range(185)]
    error = valid_v6(1, "D")
    error["terminal_success"] = False
    error["terminal_error_kind"] = "transport_timeout"
    malformed = valid_v6(2, "E")
    malformed["prediction"] = "Z"
    outputs = apply_failover([valid_v6(0, "C"), error, malformed], fallbacks)
    value = {
        "schema_version": "generic-failover-synthetic-dry-run-v1",
        "synthetic_only": True,
        "rows": len(outputs),
        "selection_counts": selection_counts(outputs),
        "first_four_predictions": [row["prediction"] for row in outputs[:4]],
        "expected_first_four_predictions": ["C", "B", "B", "B"],
        "all_answers_valid": all(row["prediction"] in tuple("ABCDE") for row in outputs),
        "benchmark_opened": False,
        "gold_opened": False,
        "v6_outputs_opened": False,
    }
    if value["first_four_predictions"] != value["expected_first_four_predictions"]:
        raise RuntimeError("synthetic failover mismatch")
    destination = HERE / "DRY_RUN.json"
    if destination.exists():
        raise RuntimeError("dry-run artifact already exists")
    destination.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
