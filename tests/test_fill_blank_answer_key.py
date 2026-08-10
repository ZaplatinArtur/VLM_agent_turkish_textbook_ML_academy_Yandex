from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from evidence_os.fill_blank_answer_key import (
    FillBlankAnswerKeyError,
    attest_fill_blank_answer_key,
    parse_fill_blank_canonical_answer,
    verify_fill_blank_answer_key,
)


PDF_SHA256 = "a" * 64
ANSWER = "1=Alpha; 2=Beta/Gamma; 3=Delta; 4=Epsilon"
TITLE = "Boşluk Doldurma"
INSTRUCTION = "Choose the words from the bank."


def _word(text: str, x0: float, top: float, width: float | None = None) -> dict:
    width = width if width is not None else max(6.0, len(text) * 4.0)
    return {
        "text": text,
        "x0": x0,
        "top": top,
        "x1": x0 + width,
        "bottom": top + 6.0,
    }


class FakePage:
    def __init__(
        self,
        words: list[dict],
        *,
        width: float = 200.0,
        height: float = 180.0,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.words = words
        self.width = width
        self.height = height
        self.bbox = bbox

    def extract_words(self, **_kwargs: object) -> list[dict]:
        if self.bbox is None:
            return deepcopy(self.words)
        x0, top, x1, bottom = self.bbox
        return deepcopy(
            [
                word
                for word in self.words
                if x0 <= (word["x0"] + word["x1"]) / 2 <= x1
                and top <= (word["top"] + word["bottom"]) / 2 <= bottom
            ]
        )

    def crop(self, bbox: tuple[float, float, float, float]) -> "FakePage":
        return FakePage(self.words, width=self.width, height=self.height, bbox=bbox)

    def extract_text(self) -> str:
        return " ".join(
            word["text"]
            for word in sorted(
                self.extract_words(), key=lambda item: (item["top"], item["x0"])
            )
        )


def _content_page() -> FakePage:
    words = [
        _word("Boşluk", 50, 5),
        _word("Doldurma", 82, 5),
    ]
    x = 5.0
    for token in INSTRUCTION.split():
        words.append(_word(token, x, 20))
        x += max(6.0, len(token) * 4.0) + 3.0
    words.extend(
        [
            _word("Alpha", 5, 42),
            _word("Beta", 40, 42),
            _word("Gamma", 75, 42),
            _word("Delta", 115, 42),
            _word("Epsilon", 150, 42),
        ]
    )
    for number, top in enumerate((75, 95, 115, 135), start=1):
        words.extend(
            [
                _word(f"{number}.", 5, top, 8),
                _word(f"statement-{number}", 20, top),
            ]
        )
    return FakePage(words)


def _key_page() -> FakePage:
    return FakePage(
        [
            _word("Boşluk", 55, 5),
            _word("Doldurma", 87, 5),
            _word("1.", 10, 30, 8),
            _word("Alpha", 25, 30),
            _word("2.", 10, 55, 8),
            _word("Beta/Gamma", 25, 55),
            _word("3.", 105, 30, 8),
            _word("Delta", 120, 30),
            _word("4.", 105, 55, 8),
            _word("Epsilon", 120, 55),
        ]
    )


def _attest(**overrides: object):
    kwargs = {
        "pdf_sha256": PDF_SHA256,
        "content_page_number": 2,
        "key_page_number": 9,
        "content_bbox": [0, 0, 200, 170],
        "word_bank_bbox": [0, 35, 200, 55],
        "key_bbox": [0, 0, 200, 90],
        "activity_title": TITLE,
        "instruction_text": INSTRUCTION,
        "expected_item_count": 4,
        "expected_column_count": 2,
        "expected_answer": ANSWER,
    }
    kwargs.update(overrides)
    return attest_fill_blank_answer_key(_content_page(), _key_page(), **kwargs)


def test_fill_blank_attestation_derives_every_numbered_component() -> None:
    result = _attest()
    assert result.derived_answer == {
        1: "Alpha",
        2: "Beta/Gamma",
        3: "Delta",
        4: "Epsilon",
    }
    assert all(result.component_matches.values())
    assert len(result.projection_sha256) == 64
    assert len(result.content_projection_sha256) == 64
    assert len(result.key_projection_sha256) == 64


def test_fill_blank_attestation_rejects_one_changed_source_answer() -> None:
    with pytest.raises(FillBlankAnswerKeyError, match="word bank|disagree"):
        _attest(expected_answer="1=Alpha; 2=Beta/Gamma; 3=Wrong; 4=Epsilon")


def test_fill_blank_verifier_rejects_semantically_equal_coordinate_drift() -> None:
    result = _attest()
    changed_key = _key_page()
    changed_key.words[-1]["x0"] += 0.5
    changed_key.words[-1]["x1"] += 0.5
    with pytest.raises(FillBlankAnswerKeyError, match="projection differs"):
        verify_fill_blank_answer_key(
            _content_page(),
            changed_key,
            pdf_sha256=PDF_SHA256,
            content_page_number=2,
            key_page_number=9,
            content_bbox=[0, 0, 200, 170],
            word_bank_bbox=[0, 35, 200, 55],
            key_bbox=[0, 0, 200, 90],
            activity_title=TITLE,
            instruction_text=INSTRUCTION,
            expected_item_count=4,
            expected_column_count=2,
            expected_answer=ANSWER,
            expected_projection_sha256=result.projection_sha256,
            expected_content_projection_sha256=result.content_projection_sha256,
            expected_key_projection_sha256=result.key_projection_sha256,
        )


def test_fill_blank_parser_rejects_nonconsecutive_labels() -> None:
    with pytest.raises(FillBlankAnswerKeyError, match="consecutive"):
        parse_fill_blank_canonical_answer(
            "1=Alpha; 3=Gamma", expected_item_count=2
        )


def test_real_ck_source_replays_frozen_projection_when_pdf_is_available() -> None:
    root = Path(__file__).resolve().parents[1]
    pdf_path = (
        root
        / "tmp/remaining_official_source_audit/pdfs"
        / "MEB-CK-12tcinkiliaptarihiveataturkculukf01.pdf"
    )
    if not pdf_path.exists():
        pytest.skip("external official CK PDF is not available")
    import json

    pdfplumber = pytest.importorskip("pdfplumber")

    fragment_path = (
        root
        / "reports/maxim_official_exact_source_v2_20260805/frozen"
        / "public_workbook_source_fragment_meb_ck_history_fill_blank_candidate_v1.json"
    )
    payload = json.loads(fragment_path.read_text(encoding="utf-8"))
    document = payload["documents"][0]
    activity = document["activities"][0]
    with pdfplumber.open(pdf_path) as pdf:
        result = verify_fill_blank_answer_key(
            pdf.pages[activity["content_page_number"] - 1],
            pdf.pages[activity["key_page_number"] - 1],
            pdf_sha256=document["pdf_sha256"],
            content_page_number=activity["content_page_number"],
            key_page_number=activity["key_page_number"],
            content_bbox=activity["content_bbox"],
            word_bank_bbox=activity["word_bank_bbox"],
            key_bbox=activity["key_bbox"],
            activity_title=activity["activity_title"],
            instruction_text=activity["instruction_text"],
            expected_item_count=activity["expected_item_count"],
            expected_column_count=activity["expected_column_count"],
            expected_answer=activity["answer"],
            expected_projection_sha256=activity["binding_projection_sha256"],
            expected_content_projection_sha256=activity[
                "content_projection_sha256"
            ],
            expected_key_projection_sha256=activity["key_projection_sha256"],
        )
    assert result.derived_answer[1] == "Ali Rıza Paşa"
    assert result.derived_answer[20] == "Gümrü Antlaşması"
