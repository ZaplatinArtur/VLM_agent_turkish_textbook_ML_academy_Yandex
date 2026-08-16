from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .scoring import DEFAULT_CUTOFFS, evaluate_dense_mmr, prepare_qrels_from_agent_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Dense and Dense+MMR retrieval")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare-qrels",
        help="freeze agent search queries and create a manual relevance template",
    )
    prepare.add_argument("--tasks", required=True, type=Path)
    prepare.add_argument("--run", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)

    evaluate = commands.add_parser(
        "run",
        help="run current Dense and MMR stages and score annotated qrels",
    )
    evaluate.add_argument("--qrels", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--k", action="append", type=int)
    evaluate.add_argument("--fetch-k", type=int)
    evaluate.add_argument("--mmr-lambda", type=float, default=0.5)
    evaluate.add_argument("--no-subject-filter", action="store_true")
    evaluate.add_argument("--candidate-text-chars", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare-qrels":
        summary = prepare_qrels_from_agent_run(args.tasks, args.run, args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    cutoffs = tuple(args.k or DEFAULT_CUTOFFS)
    fetch_k = args.fetch_k or max(cutoffs)
    report = evaluate_dense_mmr(
        args.qrels,
        cutoffs=cutoffs,
        fetch_k=fetch_k,
        mmr_lambda=args.mmr_lambda,
        use_subject_filter=not args.no_subject_filter,
        candidate_text_chars=args.candidate_text_chars,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "coverage": report["coverage"],
        "dense": report["variants"]["dense"]["metrics"],
        "dense_mmr": report["variants"]["dense_mmr"]["metrics"],
        "output": str(args.output),
    }
    if report["coverage"]["scored_qrels"] == 0:
        summary["warning"] = (
            "no annotated corpus-covered qrels; rankings were saved but "
            "metrics are unavailable"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
