from mla_baseline.prepare_text_only import prepare_text_only


def _record(question: str, *, images=None) -> dict:
    return {
        "task_id": question[:10],
        "subject": "math",
        "grade": 7,
        "question": question,
        "question_images": images or [],
        "reference_answer": "4",
        "answer_type": "numeric",
        "reference_solution": None,
    }


def test_prepare_text_only_strips_stale_images_and_rejects_image_dependent_text() -> None:
    tasks, report = prepare_text_only(
        [
            _record("2 + 2 kaçtır?", images=[{"unvalidated": True}]),
            _record("(soru görselde)"),
            _record("Grafik: [image omitted in text-only smoke test]"),
        ]
    )

    assert len(tasks) == 1
    assert tasks[0].question_images == []
    assert report["rejected"] == {
        "image_dependent_question": 1,
        "placeholder_question": 1,
    }
