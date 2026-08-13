"""Post-freeze outcome delta reporter; it has no default gold/reference path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_content import canonical_json_bytes, read_jsonl


def outcome_map(path: Path) -> dict[str, bool]:
    output: dict[str, bool] = {}
    for row in read_jsonl(path):
        identifier = row.get("task_id") or row.get("runtime_alignment_id")
        correct = row.get("correct")
        if type(identifier) is not str or type(correct) is not bool or identifier in output:
            raise RuntimeError("outcome schema/uniqueness mismatch")
        output[identifier] = correct
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-outcomes", type=Path, required=True)
    parser.add_argument("--hybrid-outcomes", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, hybrid = outcome_map(args.baseline_outcomes), outcome_map(args.hybrid_outcomes)
    if set(baseline) != set(hybrid):
        raise RuntimeError("outcome ID sets differ")
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    fixes = sorted(key for key in baseline if not baseline[key] and hybrid[key])
    regressions = sorted(key for key in baseline if baseline[key] and not hybrid[key])
    result = {
        "schema_version": "content-only-hybrid-private-delta-v1",
        "rows": len(hybrid),
        "baseline_correct": sum(baseline.values()),
        "hybrid_correct": sum(hybrid.values()),
        "score": sum(hybrid.values()) / len(hybrid),
        "fixes": fixes,
        "regressions": regressions,
        "coverage": coverage["db_coverage"],
        "abstention": coverage["db_abstention_rate"],
    }
    args.output.write_bytes(canonical_json_bytes(result))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
