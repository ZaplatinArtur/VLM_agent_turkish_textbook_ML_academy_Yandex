from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from scripts.analyze_maxim_v7_source_first_speedup_v1 import (
    _answer_fingerprint,
    _stable_trace,
    _validate_certificate,
    analyze,
)


ROOT = Path(__file__).resolve().parents[1]


def _certificate_fixture() -> tuple[dict, dict]:
    candidate = {"final_answer": "A"}
    trace = {"accepted": True, "checks": {"source": True, "binding": True}}
    certificate = {
        "status": "pass",
        "strength": "strong",
        "input_bound": True,
        "answer_bound": True,
        "input_fingerprint": "a" * 64,
        "answer_fingerprint": _answer_fingerprint("A"),
        "claim_coverage": 1.0,
        "contradiction_count": 0,
        "deterministic_checks": [True, True],
        "verifier": "fixture-verifier",
        "trace": trace,
        "trace_fingerprint": hashlib.sha256(_stable_trace(trace)).hexdigest(),
    }
    return certificate, candidate


def test_frozen_v7_source_first_replay_is_answer_equivalent(tmp_path: Path) -> None:
    result = analyze(
        ROOT / "configs/maxim_composite_source_pipeline_v7.json",
        tmp_path / "analysis.json",
        tmp_path / "REPORT.md",
        ROOT,
    )

    assert result["rows"] == 274
    assert result["source_shortcuts"] == 131
    assert result["anchor_fallbacks"] == 143
    assert result["answer_equivalent_rows"] == 131
    assert result["conflicting_source_rows"] == []
    assert result["recorded_anchor_usage"]["avoidable_latency_fraction"] > 0.44
    assert result["claims"] == {
        "artifact_answer_equivalence_measured": True,
        "certificate_artifacts_replayed": True,
        "online_wall_clock_speedup_measured": False,
        "source_lookup_cost_included": False,
        "accuracy_or_gold_read": False,
        "task_id_policy_feature": False,
        "task_id_alignment_only": True,
    }
    assert result["inputs"]["main_certificates"]["rows"] == 130
    assert result["inputs"]["history_certificates"]["rows"] == 1
    assert (tmp_path / "analysis.json").is_file()
    assert "не новый score" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")


def test_certificate_replay_rejects_tampered_proof_fields() -> None:
    certificate, candidate = _certificate_fixture()
    _validate_certificate("ok", certificate, candidate)

    mutations = (
        ("status", "fail"),
        ("strength", "weak"),
        ("answer_fingerprint", "b" * 64),
        ("trace_fingerprint", "c" * 64),
    )
    for field, value in mutations:
        tampered = deepcopy(certificate)
        tampered[field] = value
        with pytest.raises(ValueError, match="invalid source certificate"):
            _validate_certificate(field, tampered, candidate)

    tampered = deepcopy(certificate)
    tampered["deterministic_checks"][0] = False
    with pytest.raises(ValueError, match="invalid source certificate"):
        _validate_certificate("deterministic", tampered, candidate)

    tampered = deepcopy(certificate)
    tampered["trace"]["checks"]["binding"] = False
    tampered["trace_fingerprint"] = hashlib.sha256(
        _stable_trace(tampered["trace"])
    ).hexdigest()
    with pytest.raises(ValueError, match="invalid source certificate"):
        _validate_certificate("trace-check", tampered, candidate)
