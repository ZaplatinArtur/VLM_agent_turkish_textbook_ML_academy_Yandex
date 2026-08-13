from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("content_source_router_noid_v1.py")
SPEC = importlib.util.spec_from_file_location("content_source_router_noid_v1", MODULE_PATH)
assert SPEC and SPEC.loader
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)


def test_temperature_parser_is_generic_and_content_only() -> None:
    action = router.temperature_route(
        "Erzurum'da sabah hava sıcaklığı -15°C'dir. Sıcaklık öğle saatinde "
        "7°C ortyor. Akşam ise 10°C azalıyor. Buna göre son sıcaklık kaç °C olur?"
    )
    assert action is not None
    assert action["answer"] == "-18"


def test_integer_lcm_parser_returns_option_from_observable_choices() -> None:
    text = (
        "Aynı büyüklükteki 5 adet küp blok yan yana. Bu kutularından 4 tanesinin "
        "uzunluğu, 6 adet özdeş silginin toplam uzunluğuna eşittir. En küçük değer? "
        "A) 120 B) 90 C) 60 D) 20"
    )
    action = router.lcm_route(text)
    assert action is not None
    assert action["numeric_result"] == 60
    assert action["answer"] == "C"


def test_router_rejects_identity_bearing_fields() -> None:
    source_db = {"records": []}
    try:
        router.route_observable({"ocr_text": "x", "task_id": "opaque"}, source_db)
    except router.BuildError:
        pass
    else:
        raise AssertionError("identity-bearing field was accepted")


def test_counterfactual_reidentification_cannot_change_policy() -> None:
    source_db = router.build_source_database()
    observable = {
        "ocr_text": (
            "5. Read the agenda below and complete the table. TRIP TO WARSAW. "
            "Arrive at Warsaw Chopin Airport at 9 a.m. Take a taxi to the hotel. "
            "Visit Chopin Museum and City Art Gallery. Take a tour along the river."
        ),
        "answer_type": "free_form",
        "input_mode": "multimodal",
    }
    first = router.route_observable(observable, source_db)
    second = router.route_observable(dict(observable), source_db)
    assert first == second
    assert first["family"] == "english10"


def test_public_queue_routes_exact_frozen_candidate_census_without_ids() -> None:
    source_db = router.build_source_database()
    rows = router.read_jsonl(router.QUEUE)
    decisions = []
    for row in rows:
        observable = {
            "ocr_text": row["ocr_text"],
            "answer_type": row["answer_type"],
            "input_mode": row["input_mode"],
        }
        decisions.append(router.route_observable(observable, source_db))
    selected = [item for item in decisions if item["kind"] != "abstain"]
    assert len(selected) == 18
    counts = {}
    for item in selected:
        counts[item["family"]] = counts.get(item["family"], 0) + 1
    assert counts == {
        "meb7": 6,
        "math12": 5,
        "english10": 5,
        "signed_temperature_change": 1,
        "integer_block_lcm": 1,
    }
