from abc import ABC, abstractmethod
from pathlib import Path

from schemas.retrieve import RetrievedChunk

from ..chunk_store import ChunkStore


class Parser(ABC):
    def __init__(self, chunk_store: ChunkStore | None = None) -> None:
        self.chunk_store = chunk_store

    @abstractmethod
    def parse_book(
            self,
            book_dir: Path,
            book_slug: str,
    ) -> list[RetrievedChunk]:
        ...

    def parse_books(self, book_dirs: list[Path]) -> list[RetrievedChunk]:
        all_chunks: list[RetrievedChunk] = []
        total = len(book_dirs)

        for index, book_dir in enumerate(book_dirs, 1):
            book_slug = book_dir.name
            print(f"[{index}/{total}] Parsing {book_slug}...", flush=True)

            chunks = self.parse_book(book_dir, book_slug=book_slug)
            all_chunks.extend(chunks)

            if self.chunk_store is not None:
                self.chunk_store.save_book(book_slug, chunks)
                print(
                    f"[{index}/{total}] Saved {len(chunks)} chunks ({book_slug})",
                    flush=True,
                )
            else:
                print(
                    f"[{index}/{total}] Done {book_slug}: {len(chunks)} chunks",
                    flush=True,
                )

        print(f"Finished: {total} books, {len(all_chunks)} chunks", flush=True)
        return all_chunks
