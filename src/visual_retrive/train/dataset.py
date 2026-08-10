"""Load and split query→page training pairs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_by_page(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 13,
) -> dict[str, list[dict[str, Any]]]:
    """Keep all queries for a page in the same split to reduce leakage."""

    by_page: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_page.setdefault(str(row["positive_page_id"]), []).append(row)

    page_ids = sorted(by_page)
    rng = random.Random(seed)
    rng.shuffle(page_ids)

    n = len(page_ids)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_ids = set(page_ids[:n_test])
    val_ids = set(page_ids[n_test : n_test + n_val])
    train_ids = set(page_ids[n_test + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for page_id, page_rows in by_page.items():
        if page_id in test_ids:
            splits["test"].extend(page_rows)
        elif page_id in val_ids:
            splits["val"].extend(page_rows)
        else:
            splits["train"].extend(page_rows)
    # silence unused
    _ = train_ids
    return splits


def resolve_image_path(row: dict[str, Any], data_root: Path) -> Path | None:
    rel = row.get("positive_image")
    if not rel:
        return None
    path = Path(str(rel))
    if path.is_absolute() and path.is_file():
        return path
    candidate = data_root / path
    return candidate if candidate.is_file() else None
