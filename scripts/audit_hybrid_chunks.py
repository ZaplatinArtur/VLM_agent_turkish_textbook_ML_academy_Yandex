from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve.chunking import QwenEducationalRefiner
from schemas.retrieve import RetrievedChunk


def _candidate_pages(input_dir: Path) -> list[tuple[str, list[RetrievedChunk]]]:
    candidates: list[tuple[str, list[RetrievedChunk]]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk = RetrievedChunk.model_validate_json(line)
                parent_id = str(chunk.metadata.get("parent_chunk_id") or "")
                grouped[parent_id].append(chunk)
        for parent_id, units in grouped.items():
            if any(unit.metadata.get("low_confidence") for unit in units):
                candidates.append((parent_id, units))
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Use Qwen to audit selectively sampled ambiguous hybrid chunks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hybrid_chunks",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--pages", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "hybrid_qwen_audit.json",
    )
    args = parser.parse_args(argv)

    candidates = _candidate_pages(args.input_dir)
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.pages]
    refiner = QwenEducationalRefiner(
        base_url=args.base_url,
        model=args.model,
    )
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def run(item: tuple[str, list[RetrievedChunk]]) -> dict[str, Any]:
        page_id, units = item
        refined = refiner.refine(page_id, units)
        rows = []
        for unit, decision in zip(units, refined.decisions):
            rule_kind = str(unit.metadata.get("unit_kind"))
            rows.append(
                {
                    "index": decision.index,
                    "rule_kind": rule_kind,
                    "qwen_kind": decision.kind.value,
                    "agrees": rule_kind == decision.kind.value,
                    "qwen_confidence": decision.confidence,
                    "reason": decision.reason,
                    "text": " ".join(unit.text.split())[:500],
                }
            )
        return {"page_id": page_id, "units": rows}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, item): item[0] for item in selected}
        completed = 0
        for future in as_completed(futures):
            page_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"page_id": page_id, "error": str(exc)})
            completed += 1
            print(
                f"[{completed}/{len(selected)}] audited, failures={len(failures)}",
                flush=True,
            )

    confusion: Counter[str] = Counter()
    total = agreements = high_conf_disagreements = 0
    disagreements: list[dict[str, Any]] = []
    for page in results:
        for unit in page["units"]:
            total += 1
            agreements += int(unit["agrees"])
            confusion[f"{unit['rule_kind']} -> {unit['qwen_kind']}"] += 1
            if not unit["agrees"]:
                row = {"page_id": page["page_id"], **unit}
                disagreements.append(row)
                high_conf_disagreements += int(unit["qwen_confidence"] >= 0.85)

    report = {
        "candidate_pages": len(candidates),
        "sampled_pages": len(selected),
        "successful_pages": len(results),
        "failures": failures,
        "audited_units": total,
        "agreement_rate": round(agreements / total, 4) if total else 0.0,
        "disagreements": len(disagreements),
        "high_confidence_disagreements": high_conf_disagreements,
        "confusion": dict(confusion.most_common()),
        "disagreement_samples": disagreements[:100],
        "pages": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "pages"}, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
