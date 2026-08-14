"""Offline zero-call dry run for the frozen theory-only experiment."""

from __future__ import annotations

import argparse
import json

import run_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--base-url", action="append", default=[])
    args = parser.parse_args()
    value = run_candidate.dry_run(
        args.expected_freeze_sha256,
        args.base_url or ["http://127.0.0.1:8000/v1"],
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
