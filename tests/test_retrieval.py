import json
import tempfile
import unittest
from pathlib import Path

from vlm_judge.retrieval import build_bm25_index, build_match_query, get_chunk, search_bm25


class RetrievalTests(unittest.TestCase):
    def test_build_search_and_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks.jsonl"
            records = [
                {
                    "chunk_id": "math-1", "page_id": "p1", "kind": "text",
                    "text": "Kesirlerde toplama ve payda eşitleme örnekleri",
                    "metadata": {"subject": "matematik", "grade": 5, "book_id": 1},
                },
                {
                    "chunk_id": "science-1", "page_id": "p2", "kind": "text",
                    "text": "Fotosentez sırasında bitkiler ışık enerjisini kullanır",
                    "metadata": {"subject": "fen bilimleri", "grade": 5, "book_id": 2},
                },
                {
                    "chunk_id": "image-1", "page_id": "p1", "kind": "image",
                    "image_url": "https://example.test/1.webp", "metadata": {},
                },
            ]
            chunks.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            index = root / "bm25.sqlite"
            report = build_bm25_index(chunks, index, batch_size=1)
            self.assertEqual(report["indexed_text_chunks"], 2)
            result = search_bm25(index, "kesir payda", subject="matematik")
            self.assertEqual(result["hits"][0]["chunk_id"], "math-1")
            self.assertEqual(get_chunk(index, "science-1")["subject"], "fen bilimleri")

    def test_query_is_safely_tokenized(self) -> None:
        match, tokens = build_match_query('"OR" + matematik?', mode="and")
        self.assertIn("matematik", tokens)
        self.assertNotIn("+", match)

    def test_low_information_chunks_can_be_downweighted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks.jsonl"
            values = [
                {
                    "chunk_id": "boilerplate", "page_id": "p1", "kind": "text",
                    "text": "fraction fraction fraction addition addition",
                    "metadata": {"index_policy": "downweight"},
                },
                {
                    "chunk_id": "normal", "page_id": "p2", "kind": "text",
                    "text": "fraction addition worked example",
                    "metadata": {"index_policy": "normal"},
                },
            ]
            chunks.write_text(
                "".join(json.dumps(value) + "\n" for value in values),
                encoding="utf-8",
            )
            index = root / "index.sqlite"
            build_bm25_index(chunks, index)
            result = search_bm25(index, "fraction addition", top_k=2, low_information_weight=0.0)
            self.assertEqual(result["hits"][0]["chunk_id"], "normal")
            self.assertEqual(result["hits"][1]["index_policy"], "downweight")


if __name__ == "__main__":
    unittest.main()
