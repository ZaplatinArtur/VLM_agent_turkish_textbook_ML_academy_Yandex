from pathlib import Path
from typing import Protocol

from ...schemas.retrieve import RetrievedChunk


class Parser(Protocol):
    def parse_book(
        self,
        book_dir: Path,
        book_slug: str,
    ) -> list[RetrievedChunk]: ...
