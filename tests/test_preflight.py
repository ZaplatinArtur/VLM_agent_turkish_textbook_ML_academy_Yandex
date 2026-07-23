from pathlib import Path

from mla_baseline.preflight import inspect_tasks, main


def test_preflight_distinguishes_text_and_missing_local_images(tmp_path: Path) -> None:
    tasks = [
        {
            "task_id": "image",
            "question": "(soru görselde)",
            "reference_answer": "A",
            "question_images": [
                {
                    "format": "file_path",
                    "data": "images/missing.png",
                }
            ],
        },
        {
            "task_id": "text",
            "question": "2 + 2 kaçtır?",
            "reference_answer": "4",
            "question_images": [],
        },
    ]

    report = inspect_tasks(tasks, tmp_path)

    assert report["tasks"] == 2
    assert report["text_questions"] == 1
    assert report["placeholder_questions"] == 1
    assert report["missing_local_images"] == 1


def test_text_only_preflight_allows_stale_image_ref_when_text_exists(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        '{"task_id":"a","question":"2 + 2 kaçtır?",'
        '"reference_answer":"4","question_images":['
        '{"format":"file_path","data":"missing.png"}]}\n',
        encoding="utf-8",
    )

    assert main(["--tasks", str(tasks), "--data-root", str(tmp_path), "--text-only"]) == 0
