from pathlib import Path

from paths import CHUNKS_JSONL_DIR, ensure_data_dirs
from schemas.retrieve import RetrievedChunk

from ...metadata import enrich_chunk_metadata
from .base import ChunkStore


class JsonlChunkStore(ChunkStore):
    def __init__(self, root: Path | str = CHUNKS_JSONL_DIR) -> None:
        ensure_data_dirs()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _book_path(self, book_slug: str) -> Path:
        return self.root / f"{book_slug}.jsonl"

    def save_book(self, book_slug: str, chunks: list[RetrievedChunk]) -> None:
        path = self._book_path(book_slug)
        with path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")

    def load_book(self, book_slug: str) -> list[RetrievedChunk]:
        path = self._book_path(book_slug)
        if not path.exists():
            return []
        return self._read_jsonl(path)

    def load(self) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for path in sorted(self.root.glob("*.jsonl")):
            chunks.extend(self._read_jsonl(path))
        return chunks

    @staticmethod
    def _read_jsonl(path: Path) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk = RetrievedChunk.model_validate_json(line)
                    chunks.append(enrich_chunk_metadata(chunk))
        return chunks
