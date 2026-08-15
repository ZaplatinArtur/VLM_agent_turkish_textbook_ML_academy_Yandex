from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def usable_rows(rows: Iterable[dict[str, Any]], data_root: Path) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        query = str(row.get("query") or "").strip()
        rel = str(row.get("positive_image") or "")
        if len(query) < 3 or not rel or not (data_root / rel).is_file():
            continue
        item = dict(row)
        item["query"] = query
        item["positive_image"] = rel
        result.append(item)
    return result


def split_by_book(
    rows: list[dict[str, Any]], *, seed: int = 17, val_ratio: float = 0.08,
    test_ratio: float = 0.02,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic group split; a book can occur in exactly one partition."""
    books = sorted({str(r.get("book_slug") or str(r["positive_page_id"]).split(":")[0]) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(books)
    n_val = max(1, round(len(books) * val_ratio))
    n_test = max(1, round(len(books) * test_ratio))
    val_books, test_books = set(books[:n_val]), set(books[n_val:n_val + n_test])
    out = {"train": [], "val": [], "test": []}
    for row in rows:
        book = str(row.get("book_slug") or str(row["positive_page_id"]).split(":")[0])
        key = "val" if book in val_books else "test" if book in test_books else "train"
        out[key].append(row)
    memberships: dict[str, str] = {}
    for split, items in out.items():
        for item in items:
            page = str(item["positive_page_id"])
            if page in memberships and memberships[page] != split:
                raise AssertionError(f"page leakage: {page}")
            memberships[page] = split
    return out


def split_by_subject_pages(
    rows: list[dict[str, Any]], *, seed: int = 17,
    val_ratio_per_subject: float = 0.08, test_ratio: float = 0.02,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    """Hold out unique pages independently per subject, without page leakage.

    Eight percent by default is selected independently within every subject.
    """
    by_subject: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        subject = str(row.get("subject") or "unknown")
        page = str(row["positive_page_id"])
        by_subject.setdefault(subject, {}).setdefault(page, []).append(row)
    assignment: dict[str, str] = {}
    stats: dict[str, dict[str, int]] = {}
    for subject, page_rows in sorted(by_subject.items()):
        pages = sorted(page_rows)
        random.Random(f"{seed}:{subject}").shuffle(pages)
        val_count = min(max(1, round(len(pages) * val_ratio_per_subject)), max(1, len(pages) - 1))
        remaining = len(pages) - val_count
        test_count = min(max(1, round(len(pages) * test_ratio)), max(0, remaining - 1)) if remaining > 1 else 0
        for page in pages[:val_count]: assignment[page] = "val"
        for page in pages[val_count:val_count + test_count]: assignment[page] = "test"
        for page in pages[val_count + test_count:]: assignment[page] = "train"
        stats[subject] = {
            "total_pages": len(pages), "train_pages": len(pages) - val_count - test_count,
            "val_pages": val_count, "test_pages": test_count,
            "val_ratio_percent": round(100 * val_count / max(1, len(pages)), 4),
        }
    out = {"train": [], "val": [], "test": []}
    for row in rows: out[assignment[str(row["positive_page_id"])]].append(row)
    memberships: dict[str, str] = {}
    for split, items in out.items():
        for item in items:
            page = str(item["positive_page_id"])
            if page in memberships and memberships[page] != split:
                raise AssertionError(f"page leakage: {page}")
            memberships[page] = split
    return out, stats


def dataset_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(str(row.get("positive_page_id", "")).encode())
        h.update(b"\0")
        h.update(str(row.get("query", "")).encode())
        h.update(b"\n")
    return h.hexdigest()


def page_id_to_image(page_id: str) -> str:
    book, page = page_id.rsplit(":", 1)
    return f"books/{book}/pages/{int(page):04d}.jpg"
