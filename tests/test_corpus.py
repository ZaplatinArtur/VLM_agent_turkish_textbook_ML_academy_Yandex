import json
import tempfile
import unittest
from pathlib import Path

from vlm_judge.corpus import prepare_corpus, split_text


class CorpusPreparationTests(unittest.TestCase):
    def test_chunk_overlap_and_conflict_quarantine(self) -> None:
        self.assertGreater(len(split_text("A " * 500, max_chars=300, overlap_chars=40)), 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            records = [
                {"id": "p1", "content": "Short answer", "metadata": {"image_urls": ["https://example/image"]}},
                {"id": "p1", "content": "A much more useful solution " * 20, "metadata": {"image_urls": ["https://example/image"]}},
                {"id": "p2", "content": "Bu sayfada henüz çözüm bulunmamaktadır.", "metadata": {"image_urls": ["https://example/2"]}},
            ]
            source.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
            report = prepare_corpus(source, root / "prepared", max_chars=300, overlap_chars=40)
            self.assertEqual(report["canonical_pages"], 2)
            self.assertEqual(report["conflicting_ids"], 1)
            self.assertEqual(report["boilerplate_pages_text_suppressed"], 1)
            self.assertEqual(report["image_chunks"], 2)
            self.assertGreater(report["text_chunks"], 1)


if __name__ == "__main__":
    unittest.main()
