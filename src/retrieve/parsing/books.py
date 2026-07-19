from pathlib import Path

from ...schemas.retrieve import RetrievedChunk
from ..cache import BOOKS_RAW_DIR, ParsingCache
from .base import Parser


def iter_books(raw_dir: Path = BOOKS_RAW_DIR) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(path for path in raw_dir.iterdir() if path.is_dir())


def parse_all_books(
    parser: Parser,
    cache: ParsingCache | None = None,
    raw_dir: Path = BOOKS_RAW_DIR,
) -> list[RetrievedChunk]:
    cache = cache or ParsingCache()
    chunks: list[RetrievedChunk] = []
    for book_dir in iter_books(raw_dir):
        book_chunks = parser.parse_book(book_dir, book_slug=book_dir.name)
        cache.set_chunks(book_chunks)
        chunks.extend(book_chunks)
    return chunks
