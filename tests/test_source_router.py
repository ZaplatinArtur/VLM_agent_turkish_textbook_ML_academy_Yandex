import json
from pathlib import Path

import pytest

from source_router import ObservableError, load_source_db, route, route_observable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Отказы замороженного прогона Максима: 256 задач, на которых роутер промолчал.
FROZEN_ABSTAINS = (
    PROJECT_ROOT
    / "experiments/maxim_9b_strict_noid_db_generic_hybrid_v3_1_20260812"
    / "runs/maxim274/generic_queue.jsonl"
)


def _records(family: str) -> list[dict]:
    return [r for r in load_source_db()["records"] if r["family"] == family]


def test_source_db_cardinality_is_frozen():
    db = load_source_db()
    assert db["record_count"] == 17
    assert db["answer_binding_count"] == 16
    assert {r["family"] for r in db["records"]} == {"meb7", "math12", "english10"}


@pytest.mark.parametrize("family", ["math12", "english10"])
def test_page_records_recognize_their_own_text(family):
    """Записи, принимаемые по скору и отрыву, должны узнавать сами себя."""
    for record in _records(family):
        found = route(record["retrieval_text"])
        assert found is not None, f"{record['record_id']} не опознан"
        assert found.record_id == record["record_id"]
        assert found.answer == record["answer"]


def test_meb_records_need_question_marker():
    """meb7 принимается только с номером вопроса из OCR: текста страницы мало."""
    for record in _records("meb7"):
        assert route(record["retrieval_text"]) is None


def test_unrelated_query_abstains():
    assert route("Osmanlı Devletinin kuruluş tarihi nedir") is None
    assert route("") is None


def test_observable_rejects_non_observable_fields():
    """Эталонный ответ и идентификаторы строки бенчмарка в роутер не попадают."""
    with pytest.raises(ObservableError):
        route_observable({"ocr_text": "x", "reference_answer": "42"})
    with pytest.raises(ObservableError):
        route_observable({"ocr_text": "x", "task_id": "val_0048"})


def test_allowed_observable_fields_pass():
    assert route_observable(
        {"ocr_text": "hiçbir şey", "answer_type": "numeric", "input_mode": "text"}
    ) is None


@pytest.mark.skipif(not FROZEN_ABSTAINS.is_file(), reason="нет замороженной очереди")
def test_replays_frozen_abstains_without_false_positives():
    """Ни одного ложного срабатывания на 256 задачах, где оригинал промолчал."""
    rows = [json.loads(line) for line in FROZEN_ABSTAINS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 256
    fired = [row for row in rows if route(row["ocr_text"]) is not None]
    assert fired == []
