#!/usr/bin/env python3
"""Freeze the post-map adapter code and its already-produced private result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_os.math12_stress_binding_compat_eval import (  # noqa: E402
    ADAPTER_STATUS,
    DEFAULT_PINS,
    EVALUATION_SCHEMA,
    PREDICTION_STATUS,
    PREREGISTRATION_STATUS,
    SCOPE,
)
from src.evidence_os.official_ogm import (  # noqa: E402
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
)


FILES = (
    "src/evidence_os/math12_stress_binding_compat_eval.py",
    "scripts/evaluate_math12_stress_binding_compat_v1.py",
    "scripts/freeze_math12_stress_binding_compat_v1.py",
    "tests/test_math12_stress_binding_compat_eval.py",
    "reports/maxim_math12_stress_binding_compat_v1_20260808/README.md",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def _entry(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit-result", default="9 passed")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("freeze output already exists")
    evaluation = _load(args.evaluation)
    metrics = evaluation.get("metrics")
    if (
        evaluation.get("schema_version") != EVALUATION_SCHEMA
        or evaluation.get("prediction_status") != PREDICTION_STATUS
        or evaluation.get("adapter_status") != ADAPTER_STATUS
        or evaluation.get("preregistration_status") != PREREGISTRATION_STATUS
        or evaluation.get("scope") != SCOPE
        or evaluation.get("official_default_pins_used") is not True
        or evaluation.get("pins_projection_sha256") != DEFAULT_PINS.projection_sha256
        or not isinstance(metrics, dict)
        or metrics.get("total") != 20
        or metrics.get("correct") != 20
        or metrics.get("source_binding_accuracy") != 1.0
    ):
        raise ValueError("evaluation is not the exact completed default-pins result")
    projected = dict(evaluation)
    declared_projection = projected.pop("evaluation_projection_sha256", None)
    if canonical_json_sha256(projected) != declared_projection:
        raise ValueError("evaluation projection mismatch")

    manifest: dict[str, Any] = {
        "schema_version": "math12-stress-source-binding-compat-freeze-v1",
        "created_date": "2026-08-08",
        "freeze_timing": "after_private_map_read",
        "prediction_status": PREDICTION_STATUS,
        "adapter_status": ADAPTER_STATUS,
        "preregistration_status": PREREGISTRATION_STATUS,
        "scope": SCOPE,
        "timeline_disclosure": (
            "predictions/output seal were fixed before private-map read; adapter, "
            "evaluation, report and this freeze were created after private-map read"
        ),
        "claim_limit": "source-binding synthetic robustness only; not QA/reasoning accuracy",
        "original_artifacts_mutated": False,
        "exact_pins": {**DEFAULT_PINS.__dict__},
        "pins_projection_sha256": DEFAULT_PINS.projection_sha256,
        "result": metrics,
        "evaluation_artifact": {
            "sha256": sha256_file(args.evaluation),
            "size_bytes": args.evaluation.stat().st_size,
            "evaluation_projection_sha256": declared_projection,
            "rows_projection_sha256": evaluation["rows_projection_sha256"],
            "storage": "private_holdout_workspace_not_committed",
        },
        "files": [_entry(ROOT / name) for name in FILES],
        "tests": {
            "command": "python -m pytest -q tests/test_math12_stress_binding_compat_eval.py",
            "observed_result": args.unit_result,
        },
    }
    manifest["manifest_projection_sha256"] = canonical_json_sha256(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(manifest) + b"\n")
    freeze_sha = sha256_file(args.output)
    (args.output.parent / "FREEZE_SHA256.txt").write_text(
        f"{freeze_sha}  {args.output.name}\n", encoding="ascii", newline="\n"
    )
    print(json.dumps({"freeze_sha256": freeze_sha, **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
