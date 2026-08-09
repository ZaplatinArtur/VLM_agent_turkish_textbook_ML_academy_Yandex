from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import verify_result  # noqa: E402


def test_hash_bound_result_and_arithmetic() -> None:
    report = verify_result.verify()
    assert report == {
        "status": "PASS",
        "arms": {
            "v1_1_primary": 239,
            "v1_1_secondary": 237,
            "v1_2_exploratory": 239,
            "v1_2_primary": 240,
        },
        "best": 240,
        "fixed_vs_source": ["val_0089", "val_0251"],
        "regressed_vs_source": [],
    }


def test_reported_metric_tamper_fails_closed(tmp_path: Path) -> None:
    result = json.loads(verify_result.RESULT_PATH.read_text(encoding="utf-8"))
    result["wave"]["arms"]["v1_2_primary"]["correct"] = 239
    tampered = tmp_path / "RESULT.json"
    tampered.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(verify_result.ResultError, match="reported total mismatch"):
        verify_result.verify(tampered)
