from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.build_maxim_fill_blank_workbook_fragment_v1 import build as build_fill_blank
from scripts.build_maxim_inline_solutions_workbook_fragment_v1 import (
    BuildError,
    _records_for_page,
    build as build_inline,
)
from evidence_os.inline_solution import (
    InlineSolutionError,
    verify_inline_solution_content,
)
from evidence_os.official_workbook import (
    OfficialSourceError,
    WorkbookQuestion,
    WorkbookThresholds,
    observed_inline_question_binding,
    parse_workbook_index,
    verify_workbook_index_pdf,
)
from evidence_os.official_ogm import OcrObservation


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "reports/maxim_official_exact_source_v2_20260805/frozen"


def _word(text: str, x0: float, top: float) -> dict[str, float | str]:
    width = max(5.0, len(text) * 4.0)
    return {
        "text": text,
        "x0": x0,
        "top": top,
        "x1": x0 + width,
        "bottom": top + 7.0,
    }


class FakeInlinePage:
    width = 200.0
    height = 160.0

    def __init__(
        self,
        words: list[dict[str, float | str]],
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.words = words
        self.bbox = bbox

    def extract_words(self) -> list[dict[str, float | str]]:
        return deepcopy(self.words)

    def crop(self, bbox: tuple[float, float, float, float]) -> "FakeInlinePage":
        return FakeInlinePage(self.words, bbox)

    def extract_text(self) -> str:
        words = self.words
        if self.bbox is not None:
            x0, top, x1, bottom = self.bbox
            words = [
                word
                for word in words
                if x0 <= (float(word["x0"]) + float(word["x1"])) / 2 <= x1
                and top <= (float(word["top"]) + float(word["bottom"])) / 2 <= bottom
            ]
        return " ".join(str(word["text"]) for word in words)


def test_inline_builder_ignores_explanatory_lowercase_cevap_token() -> None:
    page = FakeInlinePage(
        [
            _word("1.", 10, 50),
            _word("Prompt", 25, 50),
            _word("doğru", 25, 65),
            _word("cevap", 52, 65),
            _word("A", 80, 65),
            _word("değildir", 88, 65),
            _word("Cevap:", 25, 85),
            _word("B", 58, 85),
        ]
    )
    records = _records_for_page(
        page,
        document_id="source_book_aaaaaaaaaaaa",
        pdf_sha256="a" * 64,
        page_number=3,
    )
    assert len(records) == 1
    assert records[0]["answer"] == "B"
    assert records[0]["key_binding_kind"] == "inline_solution_projected"
    assert records[0]["record_id"] == "source_book_aaaaaaaaaaaa:p3:q1"


def test_inline_builder_requires_exact_cevap_colon_marker() -> None:
    page = FakeInlinePage(
        [
            _word("1.", 10, 50),
            _word("Prompt", 25, 50),
            _word("Cevap", 25, 85),
            _word("B", 58, 85),
        ]
    )
    assert (
        _records_for_page(
            page,
            document_id="source_book_aaaaaaaaaaaa",
            pdf_sha256="a" * 64,
            page_number=3,
        )
        == []
    )


def test_inline_projection_rejects_question_text_tamper() -> None:
    page = FakeInlinePage(
        [
            _word("1.", 10, 50),
            _word("Prompt", 25, 50),
            _word("Cevap:", 25, 85),
            _word("B", 58, 85),
        ]
    )
    record = _records_for_page(
        page,
        document_id="source_book_aaaaaaaaaaaa",
        pdf_sha256="a" * 64,
        page_number=3,
    )[0]
    verify_inline_solution_content(
        page,
        pdf_sha256="a" * 64,
        physical_page=3,
        bbox=record["content_bbox"],
        question_number=1,
        question_text=record["question_text"],
        expected_projection_sha256=record["content_projection_sha256"],
    )
    with pytest.raises(InlineSolutionError):
        verify_inline_solution_content(
            page,
            pdf_sha256="a" * 64,
            physical_page=3,
            bbox=record["content_bbox"],
            question_number=1,
            question_text=record["question_text"] + " tampered",
            expected_projection_sha256=record["content_projection_sha256"],
        )


def _inline_question(record_id: str, number: int, text: str) -> WorkbookQuestion:
    return WorkbookQuestion(
        record_id=record_id,
        content_page_number=3,
        question_number=number,
        question_marker_kind="numbered_item",
        question_text=text,
        answer="A",
        answer_format="choice",
        key_crop_text="",
        key_projection_sha256="",
        content_projection_sha256="a" * 64,
        binding_projection_sha256="",
        key_binding_kind="inline_solution_projected",
        section="",
        test_variant="",
        key_page_number=3,
        key_context_page_number=3,
        key_bbox=(1.0, 1.0, 2.0, 2.0),
        content_bbox=(0.0, 0.0, 100.0, 100.0),
        visually_checked=True,
    )


def test_projected_inline_gate_is_independent_of_disabled_numberless_fallback() -> None:
    selected = _inline_question(
        "source_book_aaaaaaaaaaaa:p3:q1",
        1,
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
    )
    competitor = _inline_question(
        "source_book_aaaaaaaaaaaa:p3:q2",
        2,
        "red orange yellow green blue indigo violet black white gray",
    )
    observation = OcrObservation(
        task_id="alignment-only",
        statement="alpha beta gamma delta epsilon zeta eta theta iota kappa",
        image_sha256="b" * 64,
        width=100,
        height=100,
        question_number=1,
        parser_identity="synthetic",
    )
    thresholds = WorkbookThresholds(
        min_numberless_question_coverage=1.0,
        min_numberless_question_matched_tokens=999,
        min_numberless_question_margin=1.0,
        min_inline_question_coverage=0.85,
        min_inline_question_matched_tokens=8,
        min_inline_question_margin=0.25,
    )
    assert observed_inline_question_binding(
        observation,
        selected,
        (selected, competitor),
        thresholds,
    )["passed"] is True
    strict_inline = WorkbookThresholds(
        min_numberless_question_coverage=1.0,
        min_numberless_question_matched_tokens=999,
        min_numberless_question_margin=1.0,
        min_inline_question_coverage=1.0,
        min_inline_question_matched_tokens=999,
        min_inline_question_margin=1.0,
    )
    assert observed_inline_question_binding(
        observation,
        selected,
        (selected, competitor),
        strict_inline,
    )["passed"] is False


@pytest.mark.parametrize("extra_field", ["task_id", "notes"])
def test_inline_build_rejects_non_allowlisted_spec_metadata(
    tmp_path: Path,
    extra_field: str,
) -> None:
    pdf_path = (
        ROOT
        / "tmp/remaining_official_source_audit/pdfs"
        / "MEB-DD-TYT-felsefe.pdf"
    )
    if not pdf_path.exists():
        pytest.skip("external official philosophy PDF is not available")
    pytest.importorskip("pdfplumber")
    payload = json.loads(
        (ROOT / "configs/maxim_meb_dd_felsefe_inline_candidate_v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload["documents"][0][extra_field] = (
        "val_0001" if extra_field == "task_id" else "unreviewed metadata"
    )
    spec = tmp_path / "tampered-spec.json"
    spec.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises((BuildError, OfficialSourceError)):
        build_inline(
            spec,
            {"meb_dd_tyt_felsefe_1e66a07b236a": pdf_path},
            tmp_path / "fragment.json",
        )


def test_real_philosophy_fragment_rebuild_is_byte_identical_when_pdf_is_available(
    tmp_path: Path,
) -> None:
    pdf_path = (
        ROOT
        / "tmp/remaining_official_source_audit/pdfs"
        / "MEB-DD-TYT-felsefe.pdf"
    )
    if not pdf_path.exists():
        pytest.skip("external official philosophy PDF is not available")
    pytest.importorskip("pdfplumber")
    output = tmp_path / "philosophy.json"
    build_inline(
        ROOT / "configs/maxim_meb_dd_felsefe_inline_candidate_v1.json",
        {"meb_dd_tyt_felsefe_1e66a07b236a": pdf_path},
        output,
    )
    frozen = FROZEN / "public_workbook_source_fragment_meb_dd_felsefe_inline_candidate_v1.json"
    assert output.read_bytes() == frozen.read_bytes()
    payload = json.loads(output.read_text(encoding="utf-8"))
    questions = payload["documents"][0]["questions"]
    assert {(row["content_page_number"], row["question_number"]) for row in questions} == {
        (12, 3),
        (12, 4),
        (28, 3),
        (28, 4),
    }
    assert all(
        row["key_binding_kind"] == "inline_solution_projected"
        and len(row["content_projection_sha256"]) == 64
        for row in questions
    )


def test_real_philosophy_pdf_replay_rejects_question_text_tamper() -> None:
    pdf_path = (
        ROOT
        / "tmp/remaining_official_source_audit/pdfs"
        / "MEB-DD-TYT-felsefe.pdf"
    )
    if not pdf_path.exists():
        pytest.skip("external official philosophy PDF is not available")
    pytest.importorskip("pdfplumber")
    payload = json.loads(
        (
            FROZEN
            / "public_workbook_source_fragment_meb_dd_felsefe_inline_candidate_v1.json"
        ).read_text(encoding="utf-8")
    )
    payload["documents"][0]["questions"][0]["question_text"] += " tampered"
    document = parse_workbook_index(payload).documents[0]
    with pytest.raises(OfficialSourceError):
        verify_workbook_index_pdf(pdf_path, document)


def test_real_history_fragment_rebuild_is_byte_identical_when_pdf_is_available(
    tmp_path: Path,
) -> None:
    pdf_path = (
        ROOT
        / "tmp/remaining_official_source_audit/pdfs"
        / "MEB-CK-12tcinkiliaptarihiveataturkculukf01.pdf"
    )
    if not pdf_path.exists():
        pytest.skip("external official CK PDF is not available")
    pytest.importorskip("pdfplumber")
    output = tmp_path / "history.json"
    build_fill_blank(
        ROOT / "configs/maxim_meb_ck_history_fill_blank_candidate_v1.json",
        {"meb_ck_12_inkilap_tarihi_f7c805d55591": pdf_path},
        output,
    )
    frozen = FROZEN / "public_workbook_source_fragment_meb_ck_history_fill_blank_candidate_v1.json"
    assert output.read_bytes() == frozen.read_bytes()


@pytest.mark.parametrize(
    "name",
    [
        "public_workbook_source_fragment_meb_dd_felsefe_inline_candidate_v1.json",
        "public_workbook_source_fragment_meb_ck_history_fill_blank_candidate_v1.json",
    ],
)
def test_source_fragments_contain_no_benchmark_identifiers(name: str) -> None:
    payload = json.loads((FROZEN / name).read_text(encoding="utf-8"))

    def keys(value: object) -> list[str]:
        if isinstance(value, dict):
            return [str(key) for key in value] + [
                nested for child in value.values() for nested in keys(child)
            ]
        if isinstance(value, list):
            return [nested for child in value for nested in keys(child)]
        return []

    forbidden = {
        "task_id",
        "gold",
        "reference",
        "score",
        "judge",
        "outcome",
        "solver",
        "correctness",
    }
    assert not (set(keys(payload)) & forbidden)
    serialized = json.dumps(payload, ensure_ascii=False).casefold()
    assert "val_" not in serialized
