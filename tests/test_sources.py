import tempfile
import unittest
from pathlib import Path

from vlm_judge.sources import inspect_odevjet


class SourceInventoryTests(unittest.TestCase):
    def test_duplicate_and_low_information_detection(self) -> None:
        rows = [
            '{"id":"x","content":"Bu sayfada henüz çözüm bulunmamaktadır.","metadata":{"kitap_id":1,"kitap_title":"Book","sinif":5,"ders":"matematik","sayfa_no":1,"url":"u1","image_urls":["i1"]}}',
            '{"id":"x","content":"A different and sufficiently informative solution that changes the duplicate payload substantially.","metadata":{"kitap_id":1,"kitap_title":"Book","sinif":5,"ders":"matematik","sayfa_no":1,"url":"u1","image_urls":["i1"]}}',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            result = inspect_odevjet(path)
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["duplicate_rows"], 1)
        self.assertEqual(result["conflicting_duplicate_ids"], 1)
        self.assertEqual(result["low_information_records"], 2)


if __name__ == "__main__":
    unittest.main()
