import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytesseract
from PIL import Image
from tqdm import tqdm

from paths import to_data_relpath
from schemas.media import ImageRef
from schemas.retrieve import RetrievedChunk

from ..chunk_store import ChunkStore
from .base import Parser
from .tesseract_env import configure_tesseract

_PAGE_RE = re.compile(r"^(\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)


class OcrParser(Parser):
    def __init__(
            self,
            lang: str = "tur",
            max_workers: int = 16,
            chunk_store: ChunkStore | None = None,
    ) -> None:

        super().__init__(chunk_store=chunk_store)
        configure_tesseract()
        self.lang = lang
        self.max_workers = max_workers

    def parse_book(self, book_dir: Path, book_slug: str) -> list[RetrievedChunk]:
        pages: list[tuple[int, Path]] = []
        for path in sorted(book_dir.iterdir()):
            if not path.is_file():
                continue
            match = _PAGE_RE.match(path.name)
            if match is not None:
                pages.append((int(match.group(1)), path))

        if not pages:
            return []

        def _parse_page(item: tuple[int, Path]) -> RetrievedChunk:
            page, path = item
            text = pytesseract.image_to_string(Image.open(path), lang=self.lang)
            chunk_id = f"{book_slug}:{page:04d}"
            return RetrievedChunk(
                chunk_id=chunk_id,
                text=text.strip(),
                images=[
                    ImageRef(
                        image_id=chunk_id,
                        format="file_path",
                        data=to_data_relpath(path),
                        mime_type="image/jpeg",
                    )
                ],
                score=0.0,
                metadata={"textbook": book_slug, "page": page},
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(
                tqdm(
                    executor.map(_parse_page, pages),
                    total=len(pages),
                    desc=book_slug,
                )
            )
