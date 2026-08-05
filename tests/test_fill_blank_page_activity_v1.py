from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from evidence_os.fill_blank_page_activity import (
    FillBlankPageActivity,
    FillBlankPageDocument,
    FillBlankPageThresholds,
    parse_fill_blank_page_index,
    resolve_fill_blank_page_activity,
    verify_fill_blank_page_index_pdf,
)
from evidence_os.official_ogm import (
    OcrObservation,
    OfficialSourceError,
    PageMatcher,
    canonical_json_bytes,
    sha256_file,
)
from evidence_os.official_workbook import YandexPublicIdentity
from scripts.compose_maxim_fill_blank_page_activity_v1 import (
    CompositionError,
    compose,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "reports/maxim_official_exact_source_v2_20260805/frozen"
HISTORY_FRAGMENT = (
    FROZEN
    / "public_workbook_source_fragment_meb_ck_history_fill_blank_candidate_v1.json"
)
HISTORY_PDF = (
    ROOT
    / "tmp/remaining_official_source_audit/pdfs"
    / "MEB-CK-12tcinkiliaptarihiveataturkculukf01.pdf"
)
SOURCE_URL = (
    "https://docs.yandex.ru/docs/view?"
    "url=ya-disk-public%3A%2F%2Fsynthetic-fill-blank-key"
    "&name=synthetic-fill-blank.pdf&nosw=17"
)


def _synthetic_case() -> tuple[
    OcrObservation,
    FillBlankPageDocument,
    PageMatcher,
    tuple[str, ...],
    dict[str, dict[str, bool]],
]:
    statement = (
        "Full page activity complete each sentence using the source bank.\n"
        "1. alpha appears in the first statement.\n"
        "2. beta appears in the second statement.\n"
        "3. gamma appears in the third statement."
    )
    activity = FillBlankPageActivity(
        record_id="synthetic_fill_aaaaaaaaaaaa:p1:fill_blank",
        content_page_number=1,
        key_page_number=3,
        key_context_page_number=3,
        question_text=statement,
        answer="1=alpha;2=beta;3=gamma",
        activity_title="Full page activity",
        instruction_text="complete each sentence using the source bank",
        expected_item_count=3,
        expected_column_count=1,
        key_crop_text="source-native answer key",
        key_projection_sha256="1" * 64,
        content_projection_sha256="2" * 64,
        binding_projection_sha256="3" * 64,
        content_bbox=(10.0, 10.0, 190.0, 290.0),
        word_bank_bbox=(20.0, 30.0, 180.0, 70.0),
        key_bbox=(10.0, 10.0, 190.0, 150.0),
    )
    document = FillBlankPageDocument(
        document_id="synthetic_fill_aaaaaaaaaaaa",
        identity=YandexPublicIdentity(
            public_locator="ya-disk-public://synthetic-fill-blank-key",
            name="synthetic-fill-blank.pdf",
        ),
        pdf_sha256="a" * 64,
        page_count=3,
        content_page_ranges=((1, 2),),
        activities=(activity,),
    )
    page_texts = (
        statement,
        "unrelated chronology archive treaty geography migration monument",
        "source-native answer key",
    )
    observation = OcrObservation(
        task_id="opaque-alignment-a",
        statement=statement,
        image_sha256="b" * 64,
        width=1200,
        height=1600,
        question_number=None,
        parser_identity="synthetic-source-only-parser",
        text_blocks=tuple(statement.splitlines()[1:]),
    )
    attestation = {
        activity.record_id: {
            "pdf_binding": True,
            "activity_title_once": True,
            "activity_instruction_once": True,
            "complete_item_inventory": True,
            "word_bank_multiset": True,
            "answer_key_components": True,
        }
    }
    return observation, document, PageMatcher(page_texts), page_texts, attestation


def _resolve(observation: OcrObservation):
    observation0, document, matcher, page_texts, attestation = _synthetic_case()
    del observation0
    return resolve_fill_blank_page_activity(
        observation,
        SOURCE_URL,
        document,
        matcher,
        page_texts,
        FillBlankPageThresholds(
            min_page_coverage=0.70,
            min_page_matched_tokens=5,
            min_page_margin=0.20,
            min_activity_coverage=0.85,
            min_activity_matched_tokens=5,
        ),
        verified_record_attestations=attestation,
    )


def test_task_id_rename_is_not_a_fill_blank_policy_feature() -> None:
    observation, _document, _matcher, _page_texts, _attestation = _synthetic_case()
    first = _resolve(observation)
    renamed = _resolve(replace(observation, task_id="opaque-alignment-renamed"))
    assert first.accepted is True
    assert renamed.accepted is True
    assert first.answer == renamed.answer
    assert first.checks == renamed.checks
    assert first.trace == renamed.trace


def test_missing_source_visible_item_forces_abstention() -> None:
    observation, _document, _matcher, _page_texts, _attestation = _synthetic_case()
    malformed = replace(observation, text_blocks=observation.text_blocks[:-1])
    result = _resolve(malformed)
    assert result.accepted is False
    assert dict(result.checks)["complete_numbered_item_inventory"] is False
    assert result.answer is None


def test_duplicate_source_visible_item_forces_abstention() -> None:
    observation, _document, _matcher, _page_texts, _attestation = _synthetic_case()
    malformed = replace(
        observation,
        text_blocks=observation.text_blocks
        + ("2. duplicated marker must not collapse into a set.",),
    )
    result = _resolve(malformed)
    assert result.accepted is False
    assert dict(result.checks)["complete_numbered_item_inventory"] is False
    assert result.trace["match"]["observed_item_inventory"] == [1, 2, 2, 3]
    assert result.answer is None


def test_history_source_index_rejects_benchmark_metadata() -> None:
    payload = json.loads(HISTORY_FRAGMENT.read_text(encoding="utf-8"))
    payload["documents"][0]["activities"][0]["task_id"] = "val_0196"
    with pytest.raises(OfficialSourceError):
        parse_fill_blank_page_index(payload)


@pytest.mark.parametrize("tamper", ["key_page", "content_bbox"])
def test_real_history_pdf_replay_rejects_address_or_bbox_drift(
    tamper: str,
) -> None:
    if not HISTORY_PDF.exists():
        pytest.skip("external official history PDF is not available")
    pytest.importorskip("pdfplumber")
    payload = json.loads(HISTORY_FRAGMENT.read_text(encoding="utf-8"))
    activity = payload["documents"][0]["activities"][0]
    if tamper == "key_page":
        activity["key_page_number"] += 1
        activity["key_context_page_number"] += 1
    else:
        activity["content_bbox"][0] += 0.25
    document = parse_fill_blank_page_index(payload).documents[0]
    with pytest.raises(OfficialSourceError):
        verify_fill_blank_page_index_pdf(HISTORY_PDF, document)


def test_composer_rejects_certified_candidate_answer_tamper(tmp_path: Path) -> None:
    if not HISTORY_PDF.exists():
        pytest.skip("external official history PDF is not available")
    pytest.importorskip("pdfplumber")
    pytest.importorskip("pypdf")
    canonical = (
        ROOT
        / "reports/maxim_official_exact_source_v2_20260805"
        / "fill_blank_page_activity_history_v1_resolver"
    )
    if not (canonical / "manifest.json").exists():
        pytest.skip("canonical source-only resolver artifact is not available")
    copied = tmp_path / "resolver"
    copied.mkdir()
    for name in ("candidate.jsonl", "certificates.jsonl", "audit.jsonl"):
        shutil.copyfile(canonical / name, copied / name)
    candidate_path = copied / "candidate.jsonl"
    rows = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    certified = next(row for row in rows if row.get("abstain") is False)
    certified["final_answer"] = str(certified["final_answer"]) + " tampered"
    candidate_path.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )
    manifest = json.loads((canonical / "manifest.json").read_text(encoding="utf-8"))
    for name, filename in (
        ("candidate", "candidate.jsonl"),
        ("certificates", "certificates.jsonl"),
        ("audit", "audit.jsonl"),
    ):
        artifact_path = copied / filename
        manifest["artifacts"][name] = {
            "path": str(artifact_path),
            "sha256": sha256_file(artifact_path),
        }
    manifest_path = copied / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(CompositionError):
        compose(
            ROOT / "configs/maxim_fill_blank_page_activity_history_v1.json",
            manifest_path,
            tmp_path / "composed",
        )
