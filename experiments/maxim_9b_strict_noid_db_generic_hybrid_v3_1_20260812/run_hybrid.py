from __future__ import annotations

import argparse
import json
from pathlib import Path

from strict_hybrid import compose, route


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    routing = sub.add_parser("route")
    routing.add_argument("--eval-set", choices=("maxim274", "ykslop_dev185"), required=True)
    routing.add_argument("--public", type=Path, required=True)
    routing.add_argument("--output-dir", type=Path, required=True)
    routing.add_argument("--expected-freeze-sha256", required=True)
    composition = sub.add_parser("compose")
    composition.add_argument("--decisions", type=Path, required=True)
    composition.add_argument("--generic-predictions", type=Path, required=True)
    composition.add_argument("--output", type=Path, required=True)
    composition.add_argument("--generic-mode", choices=("qwen35_9b_frozen_candidate", "maxim_base240_control"), required=True)
    composition.add_argument("--generic-candidate-freeze", type=Path)
    composition.add_argument("--expected-generic-candidate-freeze-sha256")
    composition.add_argument("--generic-candidate-independent-audit", type=Path)
    composition.add_argument("--expected-generic-candidate-independent-audit-sha256")
    composition.add_argument("--expected-freeze-sha256", required=True)
    args = parser.parse_args()
    if args.command == "route":
        result = route(args.eval_set, args.public, args.output_dir, args.expected_freeze_sha256)
    else:
        result = compose(
            args.decisions, args.generic_predictions, args.output, args.expected_freeze_sha256,
            args.generic_mode, args.generic_candidate_freeze,
            args.expected_generic_candidate_freeze_sha256,
            args.generic_candidate_independent_audit,
            args.expected_generic_candidate_independent_audit_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
