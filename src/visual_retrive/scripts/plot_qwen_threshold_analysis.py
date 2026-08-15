"""Plot Qwen acceptance as a function of E5 similarity."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bin-width", type=float, default=0.002)
    args = ap.parse_args()

    rows = []
    for path in args.judgments:
        rows.extend(json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
    if not rows:
        raise SystemExit("No Qwen judgments yet")
    lo = np.floor(min(float(r["semantic"]) for r in rows) / args.bin_width) * args.bin_width
    hi = np.ceil(max(float(r["semantic"]) for r in rows) / args.bin_width) * args.bin_width + args.bin_width
    edges = np.arange(lo, hi + args.bin_width / 2, args.bin_width)
    stats = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = [r for r in rows if left <= float(r["semantic"]) < right]
        yes = sum(bool(r["same_intent"]) for r in selected)
        stats.append({"left": left, "right": right, "total": len(selected), "yes": yes,
                      "no": len(selected) - yes, "acceptance": yes / len(selected) if selected else None})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys()); writer.writeheader(); writer.writerows(stats)
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    centers = np.array([(s["left"] + s["right"]) / 2 for s in stats])
    yes = np.array([s["yes"] for s in stats]); no = np.array([s["no"] for s in stats])
    rate = np.array([np.nan if s["acceptance"] is None else 100 * s["acceptance"] for s in stats])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    ax1.bar(centers, yes, width=args.bin_width * .9, label="Qwen: same intent", color="#2ca02c")
    ax1.bar(centers, no, width=args.bin_width * .9, bottom=yes, label="Qwen: different", color="#d62728")
    ax1.set_ylabel("Candidate pairs"); ax1.legend(); ax1.grid(axis="y", alpha=.25)
    ax2.plot(centers, rate, marker="o", linewidth=2, color="#1f77b4")
    ax2.set_ylabel("Qwen acceptance, %"); ax2.set_xlabel("multilingual-e5-base cosine similarity")
    ax2.set_ylim(0, 105); ax2.grid(alpha=.25)
    fig.suptitle(f"Qwen relevance decisions by E5 threshold (n={len(rows):,})")
    fig.savefig(args.output, dpi=160)
    print(json.dumps({"judgments": len(rows), "png": str(args.output), "csv": str(csv_path)}))


if __name__ == "__main__":
    main()
