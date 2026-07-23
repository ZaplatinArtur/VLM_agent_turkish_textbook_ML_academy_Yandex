import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vlm_judge.ingest import read_records
from vlm_judge.validation_archive import (
    _parse_extraction,
    ValidationExtractor,
    build_image_only_validation_tasks,
    build_validation_manifest,
    build_validation_tasks,
)


try:
    from openpyxl import Workbook
except ImportError:  # pragma: no cover - optional sources dependency
    Workbook = None


class ValidationArchiveTests(unittest.TestCase):
    @unittest.skipIf(Workbook is None, "openpyxl is not installed")
    def test_hidden_mapping_columns_become_local_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            for name in ("q1.png", "q2.png", "a2.png"):
                Image.new("RGB", (8, 8), "white").save(images / name)

            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Sheet1"
            sheet.append(
                [
                    "Type",
                    "Subject",
                    "Class",
                    "Source",
                    "Visual",
                    "Correct answer",
                    "Input format",
                    "Question format",
                    "Question type",
                    "handwriten answer samples",
                    "",
                    "",
                    "",
                ]
            )
            sheet.append(
                ["Test", "Math", 8, "source", "images/q1.png", "D", "screenshot", "text", "single-choice question"]
            )
            sheet.append(
                [
                    "Test",
                    "Physics",
                    9,
                    "source",
                    "q5001",
                    "a5001",
                    "screenshot",
                    "text + visual",
                    "open question (precise answer)",
                    None,
                    None,
                    "images/q2.png",
                    "images/a2.png",
                ]
            )
            sheet.append(["Test", "History", 7, "source", "unresolved", "A"])
            workbook_path = root / "validation.xlsx"
            workbook.save(workbook_path)

            manifest_path = root / "manifest.jsonl"
            report = build_validation_manifest(workbook_path, root, manifest_path)
            records = read_records(manifest_path)

            self.assertEqual(report["records"], 2)
            self.assertEqual(report["skipped_without_local_question"], 1)
            self.assertEqual(report["reference_kinds"], {"text": 1, "image": 1})
            self.assertEqual(records[0]["reference_answer"], "D")
            self.assertEqual(records[1]["question_image_path"], "images/q2.png")
            self.assertEqual(records[1]["reference_image_path"], "images/a2.png")

            extractions_path = root / "extractions.jsonl"
            extractions = [
                {
                    "task_id": records[0]["task_id"],
                    "question_text": "Question one?",
                    "reference_answer": "D",
                    "reference_solution": "",
                    "error": None,
                },
                {
                    "task_id": records[1]["task_id"],
                    "question_text": "Question two?",
                    "reference_answer": "42",
                    "reference_solution": "Worked solution",
                    "error": None,
                },
            ]
            extractions_path.write_text(
                "".join(json.dumps(record) + "\n" for record in extractions), encoding="utf-8"
            )
            tasks_path = root / "tasks.jsonl"
            task_report = build_validation_tasks(
                manifest_path, extractions_path, tasks_path, require_all=True
            )
            tasks = read_records(tasks_path)

            self.assertEqual(task_report["written"], 2)
            self.assertEqual(tasks[0]["question_images"][0]["format"], "file_path")
            self.assertEqual(tasks[1]["reference_solution"], "Worked solution")

    def test_extraction_parser_requires_exact_fields(self) -> None:
        parsed = _parse_extraction(
            '```json\n{"question_text":"Q", "reference_answer":"A", "reference_solution":"S"}\n```'
        )
        self.assertEqual(parsed["reference_answer"], "A")
        with self.assertRaises(ValueError):
            _parse_extraction('{"question_text":"Q", "reference_answer":"A"}')

    def test_merged_manifest_aliases_support_question_and_reference_images(self) -> None:
        class FakeResponse:
            text = json.dumps(
                {
                    "question_text": "Merged question?",
                    "reference_answer": "42",
                    "reference_solution": "Visible worked solution",
                }
            )
            model = "Qwen/Qwen3.5-9B"
            metadata = {}

        class FakeBackend:
            model = "Qwen/Qwen3.5-9B"

            def __init__(self) -> None:
                self.requests = []

            def complete(self, request):
                self.requests.append(request)
                return FakeResponse()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            for name in ("question.png", "answer.png"):
                Image.new("RGB", (8, 8), "white").save(images / name)

            record = {
                "task_id": "merged_001",
                "subject": "Math",
                "grade": 8,
                "question_image": "images/question.png",
                "reference_answer": None,
                "reference_answer_image": "images/answer.png",
                "answer_type": "free_form",
            }
            backend = FakeBackend()
            extracted = ValidationExtractor(backend, root).extract(record)

            self.assertEqual(extracted["reference_answer"], "42")
            self.assertEqual(len(backend.requests), 1)
            self.assertEqual(len(backend.requests[0].image_urls), 2)

            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            extractions_path = root / "extractions.jsonl"
            extractions_path.write_text(
                json.dumps(extracted, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tasks_path = root / "tasks.jsonl"
            report = build_validation_tasks(
                manifest_path,
                extractions_path,
                tasks_path,
                require_all=True,
            )
            task = read_records(tasks_path)[0]

            self.assertEqual(report["written"], 1)
            self.assertEqual(
                task["question_images"][0]["data"],
                "images/question.png",
            )
            self.assertEqual(task["reference_answer"], "42")

            image_tasks_path = root / "image_tasks.jsonl"
            image_report = build_image_only_validation_tasks(
                manifest_path,
                root,
                image_tasks_path,
            )
            image_task = read_records(image_tasks_path)[0]

            self.assertEqual(image_report["written"], 1)
            self.assertFalse(image_report["uses_question_transcriptions"])
            self.assertEqual(image_report["reference_kinds"], {"image": 1})
            self.assertEqual(image_task["question"], "(soru görselde)")
            self.assertEqual(image_task["reference_answer"], "[REFERENCE_IMAGE_ONLY]")
            self.assertEqual(
                image_task["question_images"][0]["data"],
                "images/question.png",
            )


if __name__ == "__main__":
    unittest.main()
