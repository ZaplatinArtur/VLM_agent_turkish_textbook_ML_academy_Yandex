from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve.ingest.chunking import EducationalChunker, UnitKind
from schemas.retrieve import RetrievedChunk


def _read_jsonl(path: Path) -> list[RetrievedChunk]:
    rows: list[RetrievedChunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(RetrievedChunk.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def _select_books(
    input_dir: Path,
    *,
    names: list[str],
    subjects: list[str],
    max_books: int | None,
) -> list[Path]:
    paths = sorted(input_dir.glob("*.jsonl"))
    if names:
        requested = {name.removesuffix(".jsonl") for name in names}
        paths = [path for path in paths if path.stem in requested]
    if subjects:
        lowered = [subject.casefold() for subject in subjects]
        paths = [
            path
            for path in paths
            if any(subject in path.stem.casefold() for subject in lowered)
        ]
    if max_books is not None:
        paths = paths[:max_books]
    return paths


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = path.with_suffix(".md")
    kind_counts = report["unit_kinds"]
    lines = [
        "# Hybrid educational chunking pilot",
        "",
        f"- Books: {report['books']}",
        f"- Pages: {report['pages']}",
        f"- Units: {report['units']}",
        f"- Pages with exercises: {report['pages_with_exercises']}",
        f"- Exercise units: {kind_counts.get('exercise', 0)}",
        f"- Low-confidence units: {report['low_confidence_units']}",
        f"- Runtime: {report['runtime_seconds']} s",
        "",
        "## Unit kinds",
        "",
        "| kind | count |",
        "|---|---:|",
    ]
    lines.extend(f"| {kind} | {count} |" for kind, count in kind_counts.items())
    lines.extend(["", "## Samples", ""])
    for sample in report["samples"]:
        lines.extend(
            [
                f"### {sample['kind']} — {sample['book']} p.{sample['page']}",
                "",
                "```text",
                sample["text"],
                "```",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split page-level Turkish textbook OCR into educational units."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "chunks" / "jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hybrid_chunks",
    )
    parser.add_argument("--book", action="append", default=[])
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--max-books", type=int)
    parser.add_argument("--max-pages-per-book", type=int)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "hybrid_chunking_pilot.json",
    )
    args = parser.parse_args(argv)

    books = _select_books(
        args.input_dir,
        names=args.book,
        subjects=args.subject,
        max_books=args.max_books,
    )
    if not books:
        raise SystemExit(f"No JSONL books selected in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunker = EducationalChunker()
    started = time.perf_counter()
    kind_counts: Counter[str] = Counter()
    book_reports: list[dict[str, Any]] = []
    unit_lengths: list[int] = []
    samples: list[dict[str, Any]] = []
    pages = units_total = pages_with_exercises = low_confidence = oversized = 0

    for book_index, input_path in enumerate(books, 1):
        page_chunks = _read_jsonl(input_path)
        if args.max_pages_per_book is not None:
            page_chunks = page_chunks[: args.max_pages_per_book]
        output_path = args.output_dir / input_path.name
        book_kinds: Counter[str] = Counter()
        book_units = book_exercise_pages = 0

        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for page in page_chunks:
                chunks = chunker.chunk_page(page)
                page_kinds = {str(chunk.metadata["unit_kind"]) for chunk in chunks}
                book_exercise_pages += int(UnitKind.EXERCISE.value in page_kinds)
                pages_with_exercises += int(UnitKind.EXERCISE.value in page_kinds)
                for chunk in chunks:
                    output.write(chunk.model_dump_json() + "\n")
                    kind = str(chunk.metadata["unit_kind"])
                    kind_counts[kind] += 1
                    book_kinds[kind] += 1
                    units_total += 1
                    book_units += 1
                    unit_lengths.append(len(chunk.text))
                    low_confidence += int(bool(chunk.metadata.get("low_confidence")))
                    oversized += int(bool(chunk.metadata.get("oversized")))
                    if (
                        len(samples) < args.sample_count
                        and kind
                        in {
                            UnitKind.EXERCISE.value,
                            UnitKind.WORKED_EXAMPLE.value,
                            UnitKind.SOLUTION.value,
                        }
                        and len(chunk.text) >= 40
                    ):
                        samples.append(
                            {
                                "book": input_path.stem,
                                "page": chunk.metadata.get("page"),
                                "kind": kind,
                                "confidence": chunk.metadata.get(
                                    "segmentation_confidence"
                                ),
                                "text": " ".join(chunk.text.split())[:700],
                            }
                        )
        pages += len(page_chunks)
        book_reports.append(
            {
                "book": input_path.stem,
                "pages": len(page_chunks),
                "units": book_units,
                "pages_with_exercises": book_exercise_pages,
                "unit_kinds": dict(sorted(book_kinds.items())),
            }
        )
        print(
            f"[{book_index}/{len(books)}] {input_path.stem}: "
            f"{len(page_chunks)} pages -> {book_units} units, "
            f"{book_exercise_pages} exercise pages",
            flush=True,
        )

    runtime = round(time.perf_counter() - started, 3)
    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "books": len(books),
        "pages": pages,
        "units": units_total,
        "pages_with_exercises": pages_with_exercises,
        "unit_kinds": dict(sorted(kind_counts.items())),
        "units_per_page": round(units_total / pages, 3) if pages else 0.0,
        "low_confidence_units": low_confidence,
        "oversized_units": oversized,
        "unit_length_chars": {
            "mean": round(statistics.mean(unit_lengths), 1) if unit_lengths else 0.0,
            "median": statistics.median(unit_lengths) if unit_lengths else 0,
            "p90": _percentile(unit_lengths, 0.9),
            "max": max(unit_lengths, default=0),
        },
        "runtime_seconds": runtime,
        "books_detail": book_reports,
        "samples": samples,
    }
    _write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
