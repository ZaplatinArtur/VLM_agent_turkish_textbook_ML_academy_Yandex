from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from evidence_os.adapters import certificate_from_record
from evidence_os.contracts import CandidateEnvelope, CertificateKind
from evidence_os.official_ogm import (
    VERIFIER,
    MatchThresholds,
    OcrObservation,
    OfficialSourceError,
    PageMatcher,
    build_safe_snapshot,
    canonical_json_bytes,
    parser_observation,
    resolve_exact_question,
    safe_project_book,
    strict_book_id,
)


BOOK_ID = "a" * 24
TEST_A_ID = "b" * 24
TEST_B_ID = "c" * 24
QUESTION_A_ID = "d" * 24
QUESTION_B_ID = "e" * 24
SOURCE_URL = f"https://ogmmateryal.eba.gov.tr/ogm-test/book/{BOOK_ID}"
TARGET_TEXT = (
    "algebra triangle orchard compass lantern isotope fraction theorem polygon "
    "velocity cylinder matrix radius quotient symmetry tangent integer prism "
    "coordinate diagonal equation"
)
DISTRACTOR_TEXT = (
    "history empire treaty archive dynasty republic parliament chronology culture "
    "museum language geography migration monument citizenship reform document"
)


def _test_summary(test_id: str, *, start_page: int, first_number: int = 1) -> dict[str, Any]:
    return {
        "id": test_id,
        "bookId": BOOK_ID,
        "testTitle": f"Synthetic test {test_id[0]}",
        "startPage": start_page,
        "pageCount": 1,
        "questionCount": 1,
        "firstQuestionNumber": first_number,
    }


def _book_payload(*, pdf_url: str | None = None, image_root: str | None = None) -> dict[str, Any]:
    summaries = [
        _test_summary(TEST_A_ID, start_page=1),
        _test_summary(TEST_B_ID, start_page=2),
    ]
    return {
        "book": {
            "id": BOOK_ID,
            "bookTitle": "Synthetic OGM book",
            "pdfPublicUrl": pdf_url
            or "https://ogmmateryal.eba.gov.tr/content/synthetic-book.pdf",
            "pageCount": 4,
            "publicImageFolderRootUrl": image_root
            or "https://ogmmateryal.eba.gov.tr/content/synthetic-pages",
            "imageExtension": "JPG",
            "originalImageWidth": 1000,
            "originalImageHeight": 1000,
            "testCount": 2,
            "stats": {"attempts": 987654, "successRate": 0.99},
            "outcomeIds": ["book-outcome-must-not-survive"],
        },
        "tests": [
            {
                **summary,
                "stats": {"correct": 123},
                "outcomeIds": [f"{summary['id']}-outcome-must-not-survive"],
            }
            for summary in summaries
        ],
        "stats": {"global": True},
        "outcomeIds": ["root-outcome-must-not-survive"],
    }


def _test_payload(
    test_id: str,
    question_id: str,
    *,
    page_number: int,
    correct_choice_index: int,
) -> dict[str, Any]:
    summary = _test_summary(test_id, start_page=page_number)
    return {
        "test": {
            **summary,
            "stats": {"correct": 999},
            "outcomeIds": ["test-outcome-must-not-survive"],
        },
        "questions": [
            {
                "id": question_id,
                "testId": test_id,
                "bookId": BOOK_ID,
                "questionNumber": 1,
                "pageNumber": page_number,
                "left": 10.0,
                "top": 20.0,
                "width": 20.0,
                "height": 10.0,
                "choiceCount": 5,
                "correctChoiceIndex": correct_choice_index,
                "visuallyChecked": True,
                "stats": {"answerDistribution": [1, 2, 3, 4, 5]},
                "outcomeIds": ["question-outcome-must-not-survive"],
                "choices": [
                    {
                        "index": index,
                        "left": 10.0 + index * 10.0,
                        "top": 80.0,
                        "size": 2.0,
                        "stats": {"selected": index * 10},
                        "outcomeIds": [f"choice-{index}-outcome-must-not-survive"],
                    }
                    for index in range(5)
                ],
            }
        ],
        "stats": {"endpoint": True},
        "outcomeIds": ["response-outcome-must-not-survive"],
    }


def _snapshot() -> dict[str, Any]:
    return build_safe_snapshot(
        _book_payload(),
        [
            _test_payload(
                TEST_A_ID,
                QUESTION_A_ID,
                page_number=1,
                correct_choice_index=1,
            ),
            _test_payload(
                TEST_B_ID,
                QUESTION_B_ID,
                page_number=2,
                correct_choice_index=3,
            ),
        ],
    )


def _observation() -> OcrObservation:
    return OcrObservation(
        task_id="opaque-alignment-only",
        statement=f"1. {TARGET_TEXT}",
        image_sha256="f" * 64,
        width=800,
        height=400,
        question_number=1,
        parser_identity="pipeline-v2/layout-v2/recognizer-v2",
    )


def _parser_row() -> dict[str, Any]:
    return {
        "task_id": "opaque-alignment-only",
        "parser": {
            "gold_access": False,
            "pipeline_version": "pipeline-v2",
            "layout_model": "layout-v2",
            "recognition_model": "recognizer-v2",
        },
        "images": [
            {
                "image_sha256": "f" * 64,
                "width": 800,
                "height": 400,
                "parsing_res_list": [
                    {"block_label": "text", "block_content": f"1. {TARGET_TEXT}"}
                ],
            }
        ],
    }


def _load_resolver_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_maxim_official_ogm_exact_source_v2.py"
    )
    spec = importlib.util.spec_from_file_location("official_ogm_resolver_script_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_book_url_allowlist_accepts_only_exact_public_shape() -> None:
    assert strict_book_id(SOURCE_URL) == BOOK_ID


@pytest.mark.parametrize(
    "source_url",
    [
        f"http://ogmmateryal.eba.gov.tr/ogm-test/book/{BOOK_ID}",
        f"https://evil.example/ogm-test/book/{BOOK_ID}",
        f"https://ogmmateryal.eba.gov.tr.evil.example/ogm-test/book/{BOOK_ID}",
        f"https://user@ogmmateryal.eba.gov.tr/ogm-test/book/{BOOK_ID}",
        f"https://ogmmateryal.eba.gov.tr:443/ogm-test/book/{BOOK_ID}",
        f"https://ogmmateryal.eba.gov.tr/ogm-test/book/{BOOK_ID}/",
        f"https://ogmmateryal.eba.gov.tr/ogm-test/book/{BOOK_ID}?view=1",
        f"https://ogmmateryal.eba.gov.tr/ogm-test/book/{BOOK_ID}#fragment",
        "https://ogmmateryal.eba.gov.tr/ogm-test/book/not-a-book-id",
    ],
)
def test_strict_book_url_allowlist_rejects_near_misses(source_url: str) -> None:
    with pytest.raises(OfficialSourceError, match="allowlist|exact OGM|book ID"):
        strict_book_id(source_url)


@pytest.mark.parametrize(
    ("field", "bad_url"),
    [
        (
            "pdfPublicUrl",
            "https://user@ogmmateryal.eba.gov.tr/content/synthetic-book.pdf",
        ),
        (
            "pdfPublicUrl",
            "https://ogmmateryal.eba.gov.tr:8443/content/synthetic-book.pdf",
        ),
        (
            "publicImageFolderRootUrl",
            "https://user@ogmmateryal.eba.gov.tr/content/synthetic-pages",
        ),
        (
            "publicImageFolderRootUrl",
            "https://ogmmateryal.eba.gov.tr:8443/content/synthetic-pages",
        ),
    ],
)
def test_official_asset_url_allowlist_rejects_userinfo_and_nonstandard_ports(
    field: str,
    bad_url: str,
) -> None:
    payload = _book_payload()
    payload["book"][field] = bad_url

    with pytest.raises(OfficialSourceError, match="allowlist"):
        safe_project_book(payload)


@pytest.mark.parametrize(
    "leak_path",
    [
        ("root", "judgeScore"),
        ("parser", "referenceAnswer"),
        ("image", "outcomeIds"),
        ("block", "evaluationVerdict"),
    ],
)
def test_parser_observation_rejects_forbidden_evaluator_keys_recursively(
    leak_path: tuple[str, str],
) -> None:
    row = _parser_row()
    location, key = leak_path
    targets = {
        "root": row,
        "parser": row["parser"],
        "image": row["images"][0],
        "block": row["images"][0]["parsing_res_list"][0],
    }
    targets[location][key] = "must-not-be-observable"

    with pytest.raises(OfficialSourceError, match="forbidden"):
        parser_observation(row)


def test_parser_gold_access_attestation_is_only_allowed_at_provenance_path() -> None:
    row = _parser_row()
    row["images"][0]["diagnostics"] = {"gold_access": False}

    with pytest.raises(OfficialSourceError, match="gold_access|forbidden"):
        parser_observation(row)


def test_safe_snapshot_projection_drops_stats_and_outcome_ids_recursively() -> None:
    projected = _snapshot()
    serialized = json.dumps(projected, ensure_ascii=False, sort_keys=True)

    assert "stats" not in serialized
    assert "outcomeIds" not in serialized
    assert "outcome-must-not-survive" not in serialized
    assert set(projected) == {"book", "tests"}
    assert set(projected["book"]) == {
        "id",
        "bookTitle",
        "pdfPublicUrl",
        "pageCount",
        "publicImageFolderRootUrl",
        "imageExtension",
        "originalImageWidth",
        "originalImageHeight",
        "testCount",
    }


def test_unique_page_and_candidate_match_issues_an_exact_answer() -> None:
    matcher = PageMatcher(["cover", TARGET_TEXT, DISTRACTOR_TEXT, "appendix"])
    result = resolve_exact_question(
        _observation(),
        SOURCE_URL,
        _snapshot(),
        matcher,
        MatchThresholds(pdf_page_index_offset=0),
    )

    assert result.accepted is True
    assert result.answer == "B"
    assert all(passed for _, passed in result.checks)
    assert result.trace["source"]["question_id"] == QUESTION_A_ID
    assert result.trace["source"]["pdf_page_index"] == 1
    assert result.trace["observation"]["image_sha256"] == "f" * 64
    assert "task_id" not in json.dumps(result.trace, sort_keys=True)


def test_equal_page_and_candidate_scores_fail_closed() -> None:
    matcher = PageMatcher(["cover", TARGET_TEXT, TARGET_TEXT, "appendix"])
    result = resolve_exact_question(
        _observation(),
        SOURCE_URL,
        _snapshot(),
        matcher,
        MatchThresholds(pdf_page_index_offset=0),
    )
    checks = dict(result.checks)

    assert result.accepted is False
    assert result.answer is None
    assert checks["page_margin"] is False
    assert checks["candidate_margin"] is False
    assert result.trace["match"]["page_margin"] == pytest.approx(0.0)
    assert result.trace["match"]["candidate_margin"] == pytest.approx(0.0)


def test_certificate_binds_inline_trace_observation_source_and_answer() -> None:
    matcher = PageMatcher(["cover", TARGET_TEXT, DISTRACTOR_TEXT, "appendix"])
    result = resolve_exact_question(
        _observation(),
        SOURCE_URL,
        _snapshot(),
        matcher,
        MatchThresholds(pdf_page_index_offset=0),
    )
    assert result.accepted and result.answer == "B"
    result.trace["provenance"] = {
        "profile_sha256": "1" * 64,
        "parser_observations_sha256": "2" * 64,
        "source_locators_sha256": "3" * 64,
        "official_safe_snapshot_sha256": "4" * 64,
        "official_pdf_sha256": "5" * 64,
    }
    resolver_script = _load_resolver_script()
    record = resolver_script._certificate_record(
        "opaque-alignment-only", result, result.answer
    )
    candidate = CandidateEnvelope(source=VERIFIER, final_answer=result.answer)

    certificate = certificate_from_record(
        result.problem,
        candidate,
        record,
        allowed_verifiers=frozenset({VERIFIER}),
        allowed_kinds=frozenset({CertificateKind.SOURCE_ENTAILMENT}),
        require_inline_trace=True,
    )
    expected_trace_fingerprint = hashlib.sha256(
        canonical_json_bytes(record["trace"])
    ).hexdigest()
    assert certificate.trace_fingerprint == expected_trace_fingerprint
    assert record["trace_fingerprint"] == expected_trace_fingerprint

    tampered_trace = deepcopy(record)
    tampered_trace["trace"]["observation"]["image_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="trace fingerprint"):
        certificate_from_record(
            result.problem,
            candidate,
            tampered_trace,
            allowed_verifiers=frozenset({VERIFIER}),
            allowed_kinds=frozenset({CertificateKind.SOURCE_ENTAILMENT}),
        )
    with pytest.raises(ValueError, match="different answer"):
        certificate_from_record(
            result.problem,
            CandidateEnvelope(source=VERIFIER, final_answer="C"),
            record,
            allowed_verifiers=frozenset({VERIFIER}),
            allowed_kinds=frozenset({CertificateKind.SOURCE_ENTAILMENT}),
        )
