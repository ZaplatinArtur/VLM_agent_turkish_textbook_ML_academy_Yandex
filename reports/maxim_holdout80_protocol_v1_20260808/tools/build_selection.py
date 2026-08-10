#!/usr/bin/env python3
"""Portable launcher for the hash-frozen gold-blind selection implementation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import build_selection_frozen as frozen


PROJECT = Path(__file__).resolve().parents[3]


def first_existing(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise RuntimeError(f"Cannot locate {label}; tried: {candidates}")


def configure() -> None:
    configured_report = os.environ.get("VLM_HOLDOUT_REPORT_DIR")
    frozen.REPORT = (
        Path(configured_report).resolve()
        if configured_report
        else (PROJECT / "reports" / "maxim_holdout80_protocol_v1_rebuild").resolve()
    )
    configured_workspace = os.environ.get("VLM_HOLDOUT_WORKSPACE")
    frozen.WORKSPACE = Path(configured_workspace).resolve() if configured_workspace else PROJECT.resolve()

    configured_corpus = os.environ.get("VLM_HOLDOUT_TEXTBOOK_ROOT")
    corpus_candidates = ([Path(configured_corpus)] if configured_corpus else []) + [
        PROJECT / "artifacts" / "textbooks" / "full_2026_07",
        PROJECT.parent / "VLM" / "artifacts" / "textbooks" / "full_2026_07",
    ]
    frozen.TEXTBOOK_ROOT = first_existing(corpus_candidates, "full_2026_07 textbook corpus")

    configured_basic = os.environ.get("VLM_HOLDOUT_BASIC_RAG_ROOT")
    basic_candidates = ([Path(configured_basic)] if configured_basic else []) + [
        PROJECT,
        PROJECT.parent / "VLM_agent_turkish_textbook_basic_rag",
    ]
    frozen.BASIC_RAG = first_existing(
        [candidate for candidate in basic_candidates if (candidate / "tmp" / "remaining_official_source_audit" / "pdfs").exists()],
        "basic-RAG root with Math12 official PDF",
    )
    frozen.BENCHMARK_OCR = (
        frozen.BASIC_RAG / "reports" / "maxim_document_parser_v1_20260803"
        / "parser_augmented_solver_v1" / "parser_artifacts" / "parser_results_274.jsonl"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "verify"])
    args = parser.parse_args()
    configure()
    if args.command == "build":
        frozen.build()
    else:
        frozen.verify()


if __name__ == "__main__":
    main()
