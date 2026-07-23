"""Ручная проверка dense-ретрива на одной книге из data/chunks/jsonl.

Запуск из корня проекта:
    python scripts/try_retrieve.py
    python scripts/try_retrieve.py --query "Atatürk ilkeleri nelerdir" --k 5
    python scripts/try_retrieve.py --book 8-sinif-inkilap-tarihi-ders-kitabi-cevaplari-meb-yayinlari

Первый запуск скачает модель sentence-transformers (~120 МБ) и посчитает
эмбеддинги всех чанков книги. Векторы кэшируются на диск (EmbeddingCache по
chunk_id), поэтому повторные запуски — быстрые.
"""

import argparse
import sys
from pathlib import Path

# Повторяем конфигурацию pytest (pythonpath=["src"]): на пути и корень проекта
# (для импортов с префиксом src. и относительных ...schemas), и сам src
# (чтобы резолвился `from paths import ...` внутри retrieve/cache).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from paths import CHUNKS_JSONL_DIR
from retrieve.embedders import SentenceTransformerEmbedder
from retrieve.pipeline import RetrievalPipeline
from retrieve.rankers import DenseRanker
from schemas.retrieve import RetrievedChunk

DEFAULT_BOOK = "8-sinif-inkilap-tarihi-ders-kitabi-cevaplari-meb-yayinlari"
DEFAULT_QUERY = "Mustafa Kemal Atatürk'ün ilkeleri nelerdir"


class InMemoryIndex:
    """Мини-индекс поверх готового списка чанков — без ParsingCache/диска.

    Реализует единственный метод, который нужен DenseRanker: `get(subject)`.
    """

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def get(self, subject: str | None = None) -> list[RetrievedChunk]:
        if subject is None:
            return self._chunks
        return [c for c in self._chunks if c.metadata.get("subject") == subject]


def load_chunks(book: str) -> list[RetrievedChunk]:
    """Читает <book>.jsonl: каждая строка — сериализованный RetrievedChunk."""
    path = CHUNKS_JSONL_DIR / f"{book}.jsonl"
    if not path.exists():
        raise SystemExit(f"Нет файла: {path}")
    chunks: list[RetrievedChunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                chunks.append(RetrievedChunk.model_validate_json(line))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ручная проверка ретрива на одной книге")
    parser.add_argument("--book", default=DEFAULT_BOOK, help="имя файла без .jsonl")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="поисковый запрос")
    parser.add_argument("--k", type=int, default=5, help="сколько результатов вывести")
    args = parser.parse_args()

    chunks = load_chunks(args.book)
    print(f"Книга:  {args.book}")
    print(f"Чанков: {len(chunks)}")
    print(f"Запрос: {args.query!r}\n")

    # Реальный эмбеддер + DenseRanker поверх in-memory индекса, обёрнутые в пайплайн.
    embedder = SentenceTransformerEmbedder()
    ranker = DenseRanker(embedder=embedder, index=InMemoryIndex(chunks))
    pipeline = RetrievalPipeline(rankers=[ranker])

    results = pipeline.run(args.query, k=args.k)

    if not results:
        print("Ничего не найдено.")
        return

    for position, chunk in enumerate(results, start=1):
        page = chunk.metadata.get("page")
        snippet = " ".join(chunk.text.split())[:200]
        print(f"#{position}  score={chunk.score:.3f}  page={page}  id={chunk.chunk_id}")
        print(f"     {snippet}\n")


if __name__ == "__main__":
    main()
