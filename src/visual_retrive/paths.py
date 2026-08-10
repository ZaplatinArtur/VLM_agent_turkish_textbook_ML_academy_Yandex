"""Paths for the visual_retrive corpus under data/visual_retrive."""

from __future__ import annotations

from pathlib import Path

from paths import DATA_DIR, PROJECT_ROOT, VISUAL_RETRIVE_DIR

CATALOG_DIR = VISUAL_RETRIVE_DIR / "catalog"
BOOKS_DIR = VISUAL_RETRIVE_DIR / "books"
LEGACY_BOOKS_DIR = DATA_DIR / "books"


def ensure_visual_retrive_dirs() -> None:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)


def book_dir(book_slug: str) -> Path:
    return BOOKS_DIR / book_slug


def pages_dir(book_slug: str) -> Path:
    return book_dir(book_slug) / "pages"


def answers_dir(book_slug: str) -> Path:
    return book_dir(book_slug) / "answers"


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "VISUAL_RETRIVE_DIR",
    "CATALOG_DIR",
    "BOOKS_DIR",
    "LEGACY_BOOKS_DIR",
    "ensure_visual_retrive_dirs",
    "book_dir",
    "pages_dir",
    "answers_dir",
]
