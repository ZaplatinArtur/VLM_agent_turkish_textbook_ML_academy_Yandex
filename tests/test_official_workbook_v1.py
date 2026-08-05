from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from evidence_os.official_ogm import OcrObservation, OfficialSourceError, PageMatcher
from evidence_os.official_workbook import (
    WorkbookThresholds,
    _choice_binding_matches,
    document_for_source,
    parse_workbook_index,
    resolve_workbook_question,
    strict_direct_https_identity,
    strict_yandex_public_identity,
    validate_fail_closed_workbook_policy,
)
from scripts.merge_maxim_public_workbook_index_v1 import merge


SOURCE_URL = (
    "https://docs.yandex.ru/docs/view?"
    "url=ya-disk-public%3A%2F%2Fpublic-key&name=book.pdf&nosw=17"
)
TARGET = (
    "algebra triangle orchard compass lantern isotope fraction theorem polygon "
    "velocity cylinder matrix radius quotient symmetry tangent integer prism equation"
)
DISTRACTOR = (
    "history empire treaty archive dynasty republic parliament chronology museum "
    "geography migration monument citizenship reform document"
)
DOCUMENT_ID = "synthetic-book-aaaaaaaaaaaa"
DIRECT_URL = "https://official.example.gov.tr/books/lgs1.pdf"


def _fail_closed_policy() -> dict[str, object]:
    return {
        "require_observed_question_number": True,
        "allow_numberless_question_binding": False,
        "require_unique_printed_number_on_page": True,
        "require_pdf_bound_key_context": True,
        "question_number_projection": "primary_layout_then_unique_v1",
        "example_label_projection": "primary_paragraph_title_order_one_v1",
        "require_observed_source_question_marker": True,
    }


def test_shared_fail_closed_policy_rejects_every_missing_gate() -> None:
    policy = _fail_closed_policy()
    assert validate_fail_closed_workbook_policy(policy) == (
        "primary_layout_then_unique_v1",
        True,
    )
    for key in (
        "require_observed_question_number",
        "allow_numberless_question_binding",
        "require_unique_printed_number_on_page",
        "require_pdf_bound_key_context",
        "require_observed_source_question_marker",
    ):
        mutated = dict(policy)
        mutated.pop(key)
        with pytest.raises(OfficialSourceError):
            validate_fail_closed_workbook_policy(mutated)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "public-workbook-source-index-v1",
        "documents": [
            {
                "document_id": DOCUMENT_ID,
                "locator": {
                    "kind": "yandex_public",
                    "public_locator": "ya-disk-public://public-key",
                    "name": "book.pdf",
                },
                "pdf_sha256": "a" * 64,
                "page_count": 4,
                "content_page_ranges": [[2, 3]],
                "questions": [
                    {
                        "record_id": f"{DOCUMENT_ID}:p2:q7",
                        "content_page_number": 2,
                        "question_number": 7,
                        "question_text": f"7. {TARGET}",
                        "answer": "C",
                        "key_binding_kind": "answer_key_table",
                        "section": "Synthetic section",
                        "test_variant": "Test 1",
                        "key_page_number": 4,
                        "key_bbox": [100, 200, 140, 215],
                        "content_bbox": [20, 30, 500, 400],
                        "visually_checked": True,
                    }
                ],
            }
        ],
    }


def _observation(question_number: int | None = 7) -> OcrObservation:
    prefix = f"{question_number}. " if question_number is not None else ""
    return OcrObservation(
        task_id="alignment-only",
        statement=prefix + TARGET,
        image_sha256="f" * 64,
        width=900,
        height=600,
        question_number=question_number,
        parser_identity="pipeline/layout/recognition",
        text_blocks=(prefix + TARGET,),
    )


def _pages(*, duplicate_target: bool = False) -> list[str]:
    return [
        "cover",
        f"7. {TARGET}",
        f"7. {TARGET}" if duplicate_target else f"8. {DISTRACTOR}",
        "answer key 7 C",
    ]


def _resolve(*, observation: OcrObservation | None = None, pages: list[str] | None = None):
    index = parse_workbook_index(_payload())
    document = document_for_source(index, SOURCE_URL)
    assert document is not None
    page_texts = pages or _pages()
    return resolve_workbook_question(
        observation or _observation(),
        SOURCE_URL,
        document,
        PageMatcher(page_texts),
        page_texts,
        WorkbookThresholds(),
    )


def test_printed_number_page_and_reviewed_key_are_accepted() -> None:
    result = _resolve()

    assert result.accepted is True
    assert result.answer == "C"
    assert all(passed for _, passed in result.checks)
    assert result.trace["match"]["question_binding_method"] == "printed_number"
    assert result.trace["source"]["record_id"] == f"{DOCUMENT_ID}:p2:q7"


def test_numberless_crop_abstains_without_complete_pdf_native_index() -> None:
    result = _resolve(observation=_observation(question_number=None))

    assert result.accepted is False
    assert result.answer is None
    assert result.trace["match"]["question_binding_method"] == "missing_printed_number_abstain"


def test_exact_short_text_source_answer_is_supported() -> None:
    payload = _payload()
    record = payload["documents"][0]["questions"][0]  # type: ignore[index]
    record["answer_format"] = "short_text"
    record["answer"] = "98,8 kg"
    record["key_crop_text"] = "98,8 kg"
    record["key_binding_kind"] = "exact_key_text"
    record.pop("section")
    record.pop("test_variant")
    index = parse_workbook_index(payload)
    document = document_for_source(index, SOURCE_URL)
    assert document is not None
    pages = _pages()

    result = resolve_workbook_question(
        _observation(),
        SOURCE_URL,
        document,
        PageMatcher(pages),
        pages,
        WorkbookThresholds(),
    )

    assert result.accepted is True
    assert result.answer == "98,8 kg"
    assert result.problem.answer_format == "short_text"
    assert result.trace["source"]["answer_format"] == "short_text"


def test_short_text_requires_exact_key_crop_evidence() -> None:
    payload = _payload()
    record = payload["documents"][0]["questions"][0]  # type: ignore[index]
    record["answer_format"] = "short_text"
    record["answer"] = "98,8 kg"
    record["key_binding_kind"] = "exact_key_text"
    record.pop("section")
    record.pop("test_variant")

    with pytest.raises(OfficialSourceError, match="short-text answer requires"):
        parse_workbook_index(payload)


def test_coordinate_short_text_source_binding_is_schema_valid() -> None:
    payload = _payload()
    record = payload["documents"][0]["questions"][0]  # type: ignore[index]
    record["answer_format"] = "short_text"
    record["answer"] = "a=2/5; b=-1/15"
    record["key_crop_text"] = "1. a. 2/5 b. -1/15"
    record["key_binding_kind"] = "coordinate_answer_key"
    record["key_projection_sha256"] = "b" * 64
    record.pop("section")
    record.pop("test_variant")

    index = parse_workbook_index(payload)

    question = index.documents[0].questions[0]
    assert question.key_binding_kind == "coordinate_answer_key"
    assert question.key_projection_sha256 == "b" * 64


def test_coordinate_short_text_binding_requires_projection_and_content_box() -> None:
    payload = _payload()
    record = payload["documents"][0]["questions"][0]  # type: ignore[index]
    record["answer_format"] = "short_text"
    record["answer"] = "a=2/5"
    record["key_crop_text"] = "1. a. 2/5"
    record["key_binding_kind"] = "coordinate_answer_key"
    record.pop("section")
    record.pop("test_variant")
    record.pop("content_bbox")

    with pytest.raises(OfficialSourceError, match="projection pin"):
        parse_workbook_index(payload)


def _coordinate_table_payload() -> dict[str, object]:
    payload = _payload()
    record = payload["documents"][0]["questions"][0]  # type: ignore[index]
    record.update(
        {
            "question_marker_kind": "example_label",
            "answer_format": "short_text",
            "answer": "7",
            "key_crop_text": "7 7",
            "key_binding_kind": "coordinate_table_answer_key",
            "key_projection_sha256": "b" * 64,
            "content_projection_sha256": "c" * 64,
            "key_context_page_number": 4,
            "section": "ÜNİTE 1 - SYNTHETIC ÖRNEKLER",
            "test_variant": "ÖRNEKLER",
        }
    )
    return payload


def test_coordinate_table_source_binding_requires_complete_projection_metadata() -> None:
    payload = _coordinate_table_payload()
    question = parse_workbook_index(payload).documents[0].questions[0]

    assert question.question_marker_kind == "example_label"
    assert question.content_projection_sha256 == "c" * 64
    assert question.key_context_page_number == 4

    for field in (
        "content_projection_sha256",
        "question_marker_kind",
        "key_context_page_number",
    ):
        malformed = _coordinate_table_payload()
        malformed["documents"][0]["questions"][0].pop(field)  # type: ignore[index]
        with pytest.raises(OfficialSourceError, match="coordinate-table"):
            parse_workbook_index(malformed)


def test_exact_primary_example_label_binds_only_with_profile_opt_in() -> None:
    index = parse_workbook_index(_coordinate_table_payload())
    document = document_for_source(index, SOURCE_URL)
    assert document is not None
    observation = replace(
        _observation(question_number=None),
        statement=f"## Örnek 7\n{TARGET}",
        text_blocks=("## Örnek 7", TARGET),
        primary_example_label_number=7,
    )
    pages = _pages()
    marker_counts = {f"{DOCUMENT_ID}:p2:q7": 1}

    disabled = resolve_workbook_question(
        observation,
        SOURCE_URL,
        document,
        PageMatcher(pages),
        pages,
        WorkbookThresholds(),
        verified_content_marker_counts=marker_counts,
    )
    enabled = resolve_workbook_question(
        observation,
        SOURCE_URL,
        document,
        PageMatcher(pages),
        pages,
        WorkbookThresholds(),
        allow_example_label_marker=True,
        verified_content_marker_counts=marker_counts,
    )

    assert disabled.accepted is False
    assert enabled.accepted is True
    assert enabled.answer == "7"
    assert enabled.trace["match"]["question_binding_method"] == (
        "source_visible_example_label"
    )
    assert enabled.trace["observation"]["observed_source_marker_kind"] == (
        "example_label"
    )


def test_wrong_or_conflicting_example_marker_abstains() -> None:
    index = parse_workbook_index(_coordinate_table_payload())
    document = document_for_source(index, SOURCE_URL)
    assert document is not None
    pages = _pages()
    marker_counts = {f"{DOCUMENT_ID}:p2:q7": 1}
    wrong = replace(
        _observation(question_number=None),
        primary_example_label_number=8,
    )
    conflict = replace(
        _observation(question_number=7),
        primary_example_label_number=7,
    )

    for observation in (wrong, conflict):
        result = resolve_workbook_question(
            observation,
            SOURCE_URL,
            document,
            PageMatcher(pages),
            pages,
            WorkbookThresholds(),
            allow_example_label_marker=True,
            verified_content_marker_counts=marker_counts,
        )
        assert result.accepted is False
        assert result.answer is None


def test_equal_page_scores_fail_closed() -> None:
    result = _resolve(pages=_pages(duplicate_target=True))

    assert result.accepted is False
    assert result.answer is None
    assert dict(result.checks)["unique_content_page"] is False


def test_pdf_verified_content_bbox_can_isolate_a_duplicate_page_marker() -> None:
    index = parse_workbook_index(_payload())
    document = document_for_source(index, SOURCE_URL)
    assert document is not None
    pages = _pages()
    pages[1] = f"7. {TARGET} explanatory duplicate 7."

    without_pdf_crop_proof = resolve_workbook_question(
        _observation(),
        SOURCE_URL,
        document,
        PageMatcher(pages),
        pages,
        WorkbookThresholds(),
    )
    with_pdf_crop_proof = resolve_workbook_question(
        _observation(),
        SOURCE_URL,
        document,
        PageMatcher(pages),
        pages,
        WorkbookThresholds(),
        verified_content_marker_counts={f"{DOCUMENT_ID}:p2:q7": 1},
    )

    assert without_pdf_crop_proof.accepted is False
    assert dict(without_pdf_crop_proof.checks)["printed_number_visible_on_page"] is False
    assert with_pdf_crop_proof.accepted is True
    assert with_pdf_crop_proof.answer == "C"


def test_yandex_identity_discards_only_numeric_viewer_flag() -> None:
    first = strict_yandex_public_identity(SOURCE_URL)
    second = strict_yandex_public_identity(SOURCE_URL.replace("nosw=17", "nosw=999"))
    without_nosw = strict_yandex_public_identity(
        SOURCE_URL.replace("&nosw=17", ""),
        allow_missing_nosw=True,
    )

    assert first == second == without_nosw
    with pytest.raises(OfficialSourceError):
        strict_yandex_public_identity(SOURCE_URL.replace("&nosw=17", ""))
    with pytest.raises(OfficialSourceError):
        strict_yandex_public_identity(SOURCE_URL + "&answer=C")
    with pytest.raises(OfficialSourceError):
        strict_yandex_public_identity(SOURCE_URL.replace("nosw=17", "nosw=route-C"))
    with pytest.raises(OfficialSourceError):
        strict_yandex_public_identity(SOURCE_URL.replace("nosw=17", "nosw="))
    with pytest.raises(OfficialSourceError):
        strict_yandex_public_identity(SOURCE_URL + "&name=duplicate.pdf")
    with pytest.raises(OfficialSourceError):
        strict_yandex_public_identity(SOURCE_URL.replace("&name=book.pdf", ""))


def test_direct_https_identity_is_exact_and_query_free() -> None:
    identity = strict_direct_https_identity(DIRECT_URL)

    assert identity.kind == "direct_https"
    assert identity.public_locator == DIRECT_URL
    assert identity.name == "lgs1.pdf"
    for malformed in (
        " " + DIRECT_URL,
        DIRECT_URL + "?answer=A",
        DIRECT_URL + "#fragment",
        DIRECT_URL.replace("official.example.gov.tr", "OFFICIAL.example.gov.tr"),
        DIRECT_URL.replace("/books/", "/books/../books/"),
        DIRECT_URL.replace("lgs1.pdf", "lgs%31.pdf"),
        DIRECT_URL.replace("https://", "http://"),
        DIRECT_URL.replace("official.example.gov.tr", "user@official.example.gov.tr"),
        DIRECT_URL.replace("official.example.gov.tr", "official.example.gov.tr:443"),
    ):
        with pytest.raises(OfficialSourceError):
            strict_direct_https_identity(malformed)


def test_direct_https_document_identity_and_answer_list_are_schema_valid() -> None:
    payload = _payload()
    document = payload["documents"][0]  # type: ignore[index]
    document["locator"] = {
        "kind": "direct_https",
        "public_locator": DIRECT_URL,
        "name": "lgs1.pdf",
    }
    question = document["questions"][0]
    question["key_binding_kind"] = "answer_key_list"
    question["section"] = "MATEMATIK"
    question["test_variant"] = "Official LGS 1"

    index = parse_workbook_index(payload)
    selected = document_for_source(index, DIRECT_URL)

    assert selected is not None
    assert selected.identity.kind == "direct_https"
    assert selected.questions[0].key_binding_kind == "answer_key_list"
    assert document_for_source(index, DIRECT_URL.replace("lgs1.pdf", "other.pdf")) is None


class _KeyCrop:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _KeyPage:
    width = 600.0

    def __init__(self, words: list[dict[str, object]], crop_text: str) -> None:
        self._words = words
        self._crop_text = crop_text

    def extract_words(self) -> list[dict[str, object]]:
        return self._words

    def crop(self, _bbox: tuple[float, float, float, float]) -> _KeyCrop:
        return _KeyCrop(self._crop_text)


def _word(text: str, x0: float, x1: float, top: float) -> dict[str, object]:
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": top + 9.0}


def test_exact_colon_answer_list_requires_book_and_subject_context() -> None:
    payload = _payload()
    record = payload["documents"][0]["questions"][0]  # type: ignore[index]
    record.update(
        {
            "key_binding_kind": "answer_key_list",
            "section": "MATEMATIK",
            "test_variant": "Official LGS 1",
            "key_bbox": [100.0, 200.0, 114.0, 209.0],
        }
    )
    question = parse_workbook_index(payload).documents[0].questions[0]
    page = _KeyPage(
        [
            _word("Official", 10.0, 45.0, 10.0),
            _word("LGS", 48.0, 65.0, 10.0),
            _word("1", 68.0, 73.0, 10.0),
            _word("MATEMATIK", 10.0, 65.0, 100.0),
            _word("7:C", 100.0, 114.0, 200.0),
        ],
        "7:C",
    )

    assert _choice_binding_matches(page, question) is True
    page_without_subject = _KeyPage(
        [word for word in page.extract_words() if word["text"] != "MATEMATIK"],
        "7:C",
    )
    assert _choice_binding_matches(page_without_subject, question) is False

    page_with_nearer_wrong_subject = _KeyPage(
        [
            _word("Official", 10.0, 45.0, 10.0),
            _word("LGS", 48.0, 65.0, 10.0),
            _word("1", 68.0, 73.0, 10.0),
            _word("MATEMATIK", 10.0, 65.0, 100.0),
            _word("FEN", 10.0, 30.0, 170.0),
            _word("BILIMLERI", 33.0, 80.0, 170.0),
            _word("7:C", 100.0, 114.0, 200.0),
        ],
        "7:C",
    )
    assert _choice_binding_matches(page_with_nearer_wrong_subject, question) is False


def test_exact_hyphen_table_cell_is_bound_to_adim_and_section() -> None:
    question = parse_workbook_index(_payload()).documents[0].questions[0]
    question = replace(
        question,
        key_bbox=(112.0, 200.0, 118.0, 209.0),
        section="Sozcukte Anlam",
        test_variant="1. ADIM",
    )
    page = _KeyPage(
        [
            _word("Sozcukte", 10.0, 50.0, 150.0),
            _word("Anlam", 53.0, 78.0, 150.0),
            _word("1.", 10.0, 18.0, 204.0),
            _word("ADIM", 21.0, 45.0, 204.0),
            _word("7-C", 100.0, 118.0, 200.0),
        ],
        "C",
    )

    assert _choice_binding_matches(page, question) is True

    page_with_nearer_wrong_section = _KeyPage(
        [
            _word("Sozcukte", 10.0, 50.0, 100.0),
            _word("Anlam", 53.0, 78.0, 100.0),
            _word("Cumlede", 10.0, 50.0, 150.0),
            _word("Anlam", 53.0, 78.0, 150.0),
            _word("1.", 10.0, 18.0, 204.0),
            _word("ADIM", 21.0, 45.0, 204.0),
            _word("7-C", 100.0, 118.0, 200.0),
        ],
        "C",
    )
    assert _choice_binding_matches(page_with_nearer_wrong_section, question) is False

    page_with_wrong_same_row_adim = _KeyPage(
        [
            _word("Sozcukte", 10.0, 50.0, 150.0),
            _word("Anlam", 53.0, 78.0, 150.0),
            _word("2.", 10.0, 18.0, 204.0),
            _word("ADIM", 21.0, 45.0, 204.0),
            _word("7-C", 100.0, 118.0, 200.0),
        ],
        "C",
    )
    assert _choice_binding_matches(page_with_wrong_same_row_adim, question) is False


def test_optional_nosw_policy_is_enforced_through_document_and_resolver() -> None:
    index = parse_workbook_index(_payload())
    source_without_nosw = SOURCE_URL.replace("&nosw=17", "")
    with pytest.raises(OfficialSourceError):
        document_for_source(index, source_without_nosw)

    document = document_for_source(
        index,
        source_without_nosw,
        allow_missing_nosw=True,
    )
    assert document is not None
    pages = _pages()
    with pytest.raises(OfficialSourceError):
        resolve_workbook_question(
            _observation(),
            source_without_nosw,
            document,
            PageMatcher(pages),
            pages,
            WorkbookThresholds(),
        )

    result = resolve_workbook_question(
        _observation(),
        source_without_nosw,
        document,
        PageMatcher(pages),
        pages,
        WorkbookThresholds(),
        allow_missing_nosw=True,
    )
    assert result.accepted is True
    assert result.answer == "C"


@pytest.mark.parametrize(
    "malformed_url",
    [
        " " + SOURCE_URL,
        SOURCE_URL + "#",
        SOURCE_URL.replace("https://docs.yandex.ru/", "https://docs.yandex.ru:/"),
        SOURCE_URL.replace("docs.yandex.ru", "docs.yandex.\nru"),
        SOURCE_URL.replace("url=", "%75rl="),
        SOURCE_URL.replace("book.pdf", "%20book.pdf"),
        SOURCE_URL.replace("book.pdf", "book%ZZ.pdf"),
        SOURCE_URL.replace("book.pdf", "book%FF.pdf"),
        SOURCE_URL.replace("book.pdf", "book%7F.pdf"),
        SOURCE_URL.replace("book.pdf", "book%C2%80.pdf"),
        SOURCE_URL.replace("public-key", "public%7F-key"),
    ],
)
def test_yandex_identity_rejects_noncanonical_raw_url_syntax(
    malformed_url: str,
) -> None:
    with pytest.raises(OfficialSourceError):
        strict_yandex_public_identity(malformed_url, allow_missing_nosw=True)


def test_source_index_forbids_benchmark_task_mapping() -> None:
    payload = _payload()
    payload["documents"][0]["questions"][0]["task_id"] = "val_0001"  # type: ignore[index]

    with pytest.raises(OfficialSourceError, match="forbidden"):
        parse_workbook_index(payload)


def test_source_record_id_must_be_page_and_question_address() -> None:
    payload = _payload()
    payload["documents"][0]["questions"][0]["record_id"] = "opaque-row"  # type: ignore[index]

    with pytest.raises(OfficialSourceError, match="source-addressed"):
        parse_workbook_index(payload)


def test_unknown_nested_source_index_fields_fail_closed() -> None:
    payload = _payload()
    payload["documents"][0]["metadata"] = {"producer": "unreviewed"}  # type: ignore[index]
    with pytest.raises(OfficialSourceError, match="document fields"):
        parse_workbook_index(payload)

    payload = _payload()
    payload["documents"][0]["locator"]["route"] = "A"  # type: ignore[index]
    with pytest.raises(OfficialSourceError, match="locator fields"):
        parse_workbook_index(payload)

    payload = _payload()
    payload["documents"][0]["questions"][0]["qid"] = "val_0001"  # type: ignore[index]
    with pytest.raises(OfficialSourceError, match="question fields"):
        parse_workbook_index(payload)


def test_page_threshold_boundaries_are_inclusive_and_fail_above_observation() -> None:
    index = parse_workbook_index(_payload())
    document = document_for_source(index, SOURCE_URL)
    assert document is not None
    observation = replace(
        _observation(),
        statement=f"7. {TARGET} unmatched zephyr",
        text_blocks=(f"7. {TARGET} unmatched zephyr",),
    )
    pages = [
        "cover",
        f"7. {TARGET}",
        "8. algebra triangle unrelated archive",
        "answer key 7 C",
    ]
    matcher = PageMatcher(pages)
    probe = resolve_workbook_question(
        observation,
        SOURCE_URL,
        document,
        matcher,
        pages,
        WorkbookThresholds(),
    )
    coverage = probe.trace["match"]["page_idf_coverage"]
    matched_tokens = probe.trace["match"]["page_matched_tokens"]
    margin = probe.trace["match"]["page_margin"]
    assert 0.0 < coverage < 1.0
    assert 0.0 < margin < 1.0

    exact = resolve_workbook_question(
        observation,
        SOURCE_URL,
        document,
        matcher,
        pages,
        WorkbookThresholds(
            min_page_coverage=coverage,
            min_page_matched_tokens=matched_tokens,
            min_page_margin=margin,
        ),
    )
    assert exact.accepted is True

    stricter = (
        WorkbookThresholds(
            min_page_coverage=coverage + 1e-9,
            min_page_matched_tokens=matched_tokens,
            min_page_margin=margin,
        ),
        WorkbookThresholds(
            min_page_coverage=coverage,
            min_page_matched_tokens=matched_tokens + 1,
            min_page_margin=margin,
        ),
        WorkbookThresholds(
            min_page_coverage=coverage,
            min_page_matched_tokens=matched_tokens,
            min_page_margin=margin + 1e-9,
        ),
    )
    for thresholds in stricter:
        result = resolve_workbook_question(
            observation,
            SOURCE_URL,
            document,
            matcher,
            pages,
            thresholds,
        )
        assert result.accepted is False
        assert dict(result.checks)["unique_content_page"] is False


def test_alignment_key_renaming_does_not_change_source_decision() -> None:
    first = _resolve(observation=_observation())
    second = _resolve(
        observation=replace(_observation(), task_id="completely-renamed-alignment-key")
    )

    assert first.accepted == second.accepted
    assert first.answer == second.answer
    assert first.checks == second.checks
    assert first.trace["source"] == second.trace["source"]
    assert first.trace["match"] == second.trace["match"]


def test_source_index_root_fields_are_strictly_allowlisted() -> None:
    payload = _payload()
    payload["metadata"] = {"qid": "val_0001", "correct_answer": "A"}

    with pytest.raises(OfficialSourceError, match="root fields"):
        parse_workbook_index(payload)


def test_task_like_document_id_is_rejected_even_with_pdf_hash_suffix() -> None:
    payload = _payload()
    document = payload["documents"][0]  # type: ignore[index]
    old_id = document["document_id"]
    task_like_id = "val_0001-aaaaaaaaaaaa"
    document["document_id"] = task_like_id
    for question in document["questions"]:
        question["record_id"] = question["record_id"].replace(old_id, task_like_id, 1)

    with pytest.raises(OfficialSourceError, match="source-derived"):
        parse_workbook_index(payload)


def test_merge_output_sha_is_invariant_to_fragment_and_question_order(tmp_path) -> None:
    payload = _payload()
    document = payload["documents"][0]  # type: ignore[index]
    first_question = document["questions"][0]
    second_question = deepcopy(first_question)
    second_question.update(
        {
            "record_id": f"{DOCUMENT_ID}:p3:q8",
            "content_page_number": 3,
            "question_number": 8,
            "answer": "D",
            "key_bbox": [150, 200, 190, 215],
        }
    )
    static_document = {key: value for key, value in document.items() if key != "questions"}
    fragments = []
    for name, question in (("first", first_question), ("second", second_question)):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "public-workbook-source-index-v1",
                    "documents": [{**deepcopy(static_document), "questions": [question]}],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        fragments.append(path)

    output_ab = tmp_path / "merged_ab.json"
    output_ba = tmp_path / "merged_ba.json"
    merge(fragments, output_ab, tmp_path / "manifest_ab.json")
    merge(list(reversed(fragments)), output_ba, tmp_path / "manifest_ba.json")

    sha_ab = hashlib.sha256(output_ab.read_bytes()).hexdigest()
    sha_ba = hashlib.sha256(output_ba.read_bytes()).hexdigest()
    assert sha_ab == sha_ba
    assert output_ab.read_bytes() == output_ba.read_bytes()
