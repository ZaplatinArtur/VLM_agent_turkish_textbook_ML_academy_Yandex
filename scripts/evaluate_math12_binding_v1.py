#!/usr/bin/env python3
"""Evaluate a frozen Math12 opaque run against its sealed source-address map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_os.math12_binding_eval import (  # noqa: E402
    evaluate_math12_bindings,
    write_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-seal", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = evaluate_math12_bindings(
        run_dir=args.run_dir,
        input_seal_path=args.input_seal,
        private_map_path=args.private_map,
    )
    write_evaluation(args.output, evaluation)
    print(json.dumps({key: evaluation[key] for key in (
        "scope", "total", "accepted", "abstained", "correct", "incorrect",
        "coverage", "source_binding_accuracy", "conditional_precision",
    )}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

