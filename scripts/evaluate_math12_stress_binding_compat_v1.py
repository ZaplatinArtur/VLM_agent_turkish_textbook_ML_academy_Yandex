#!/usr/bin/env python3
"""Run the explicitly post-map Math12 stress compatibility evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_os.math12_stress_binding_compat_eval import (  # noqa: E402
    ADAPTER_STATUS,
    PREDICTION_STATUS,
    PREREGISTRATION_STATUS,
    evaluate_math12_stress_bindings_compat,
    write_compatibility_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stress-dir", type=Path, required=True)
    parser.add_argument("--clean-input-seal", type=Path, required=True)
    parser.add_argument("--clean-input-jsonl", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--output-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = evaluate_math12_stress_bindings_compat(
        run_dir=args.run_dir,
        stress_dir=args.stress_dir,
        clean_input_seal_path=args.clean_input_seal,
        clean_input_jsonl_path=args.clean_input_jsonl,
        private_map_path=args.private_map,
        output_seal_path=args.output_seal,
    )
    write_compatibility_evaluation(args.output, value)
    print(
        json.dumps(
            {
                "prediction_status": PREDICTION_STATUS,
                "adapter_status": ADAPTER_STATUS,
                "preregistration_status": PREREGISTRATION_STATUS,
                "scope": value["scope"],
                **value["metrics"],
                "evaluation_projection_sha256": value["evaluation_projection_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
