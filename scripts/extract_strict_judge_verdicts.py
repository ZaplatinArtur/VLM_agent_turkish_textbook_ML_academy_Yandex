from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a canonical judge run and emit minimal strict verdicts."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=97)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if len(rows) != args.expected:
        raise ValueError(f"expected {args.expected} judge rows, found {len(rows)}")
    seen: set[str] = set()
    minimal: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in seen:
            raise ValueError(f"missing or duplicate task_id at row {position}: {task_id}")
        seen.add(task_id)
        error = (row.get("judge") or {}).get("error")
        verdict = row.get("verdict")
        if error:
            raise ValueError(f"judge error for {task_id}: {error}")
        if not isinstance(verdict, dict) or not isinstance(
            verdict.get("strict_correct"), bool
        ):
            raise ValueError(f"judge verdict is missing strict_correct for {task_id}")
        minimal.append(
            {
                "task_id": task_id,
                "strict_correct": verdict["strict_correct"],
                "error": None,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as destination:
        for row in minimal:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "rows": len(minimal),
                "strict_correct": sum(row["strict_correct"] for row in minimal),
                "output": str(args.output),
                "output_sha256": sha256(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
