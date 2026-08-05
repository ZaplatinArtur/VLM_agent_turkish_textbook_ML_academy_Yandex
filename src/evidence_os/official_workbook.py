"""Fail-closed bindings from OCR crops to pinned public workbook answer keys.

The benchmark alignment key is deliberately absent from the source index.
Every admitted answer is addressed by document identity, physical PDF page,
printed question number (or a stricter question-text match), and a reviewed
answer-key cell from the same immutable PDF.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any
import unicodedata
from urllib.parse import parse_qsl, urlsplit

from .coordinate_answer_key import (
    CoordinateAnswerKeyError,
    verify_coordinate_answer_key,
)
from .coordinate_table_answer_key import (
    CoordinateTableAnswerKeyError,
    verify_content_question_marker,
    verify_coordinate_table_answer_key,
)
from .official_ogm import (
    MatchResult,
    OcrObservation,
    OfficialSourceError,
    PageMatcher,
    normalize_tokens,
    observed_source_question_marker,
    problem_for,
)


VERIFIER = "public-workbook-ocr-page-key-binding-v1"
INDEX_SCHEMA = "public-workbook-source-index-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NOSW = re.compile(r"^[0-9]+$")
_BAD_PERCENT_ESCAPE = re.compile(r"%(?![0-9a-fA-F]{2})")
_TASK_LIKE_IDENTIFIER = re.compile(
    r"(?:^|[._-])(?:val|test|train|task|qid)[._-]?\d+(?:[._-]|$)",
    re.IGNORECASE,
)
_FORBIDDEN_INDEX_KEYS = frozenset(
    {
        "accuracy",
        "benchmarkanswer",
        "candidate",
        "gold",
        "groundtruth",
        "judge",
        "metric",
        "oracle",
        "outcome",
        "reference",
        "reward",
        "score",
        "solver",
        "taskid",
        "verdict",
    }
)
_KEY_COMPONENT = re.compile(r"[^a-z0-9]+")
_CHOICE_TOKEN = re.compile(r"(?<![A-Z])([A-E])(?![A-Z])")
_DOCUMENT_KEYS = frozenset(
    {"document_id", "locator", "pdf_sha256", "page_count", "content_page_ranges", "questions"}
)
_ROOT_KEYS = frozenset({"schema_version", "documents"})
_LOCATOR_KEYS = frozenset({"kind", "public_locator", "name"})
_QUESTION_KEYS = frozenset(
    {
        "record_id",
        "content_page_number",
        "question_number",
        "question_marker_kind",
        "question_text",
        "answer",
        "answer_format",
        "key_crop_text",
        "key_projection_sha256",
        "content_projection_sha256",
        "key_page_number",
        "key_context_page_number",
        "key_bbox",
        "content_bbox",
        "key_binding_kind",
        "section",
        "test_variant",
        "visually_checked",
    }
)


@dataclass(frozen=True, slots=True)
class WorkbookThresholds:
    min_page_coverage: float = 0.60
    min_page_matched_tokens: int = 10
    min_page_margin: float = 0.08
    min_numberless_question_coverage: float = 0.85
    min_numberless_question_matched_tokens: int = 8
    min_numberless_question_margin: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "min_page_coverage",
            "min_page_margin",
            "min_numberless_question_coverage",
            "min_numberless_question_margin",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.min_page_matched_tokens < 1:
            raise ValueError("min_page_matched_tokens must be positive")
        if self.min_numberless_question_matched_tokens < 1:
            raise ValueError("min_numberless_question_matched_tokens must be positive")


def validate_fail_closed_workbook_policy(
    policy: Mapping[str, Any],
) -> tuple[str, bool]:
    """Validate the source-marker policy shared by every runtime entry point."""

    if (
        policy.get("require_observed_question_number") is not True
        or policy.get("allow_numberless_question_binding") is not False
        or policy.get("require_unique_printed_number_on_page") is not True
        or policy.get("require_pdf_bound_key_context") is not True
    ):
        raise OfficialSourceError(
            "public-workbook profile does not enable all fail-closed bindings"
        )
    number_projection = str(
        policy.get("question_number_projection") or "unique_block_markers_v1"
    )
    if number_projection not in {
        "unique_block_markers_v1",
        "primary_layout_then_unique_v1",
    }:
        raise OfficialSourceError("unsupported workbook question-number projection")
    example_projection = str(policy.get("example_label_projection") or "disabled")
    if example_projection not in {
        "disabled",
        "primary_paragraph_title_order_one_v1",
    }:
        raise OfficialSourceError("unsupported workbook example-label projection")
    allow_example_label_marker = (
        example_projection == "primary_paragraph_title_order_one_v1"
    )
    if allow_example_label_marker and number_projection != "primary_layout_then_unique_v1":
        raise OfficialSourceError(
            "workbook example-label projection requires primary layout"
        )
    if (
        allow_example_label_marker
        and policy.get("require_observed_source_question_marker") is not True
    ):
        raise OfficialSourceError(
            "workbook example-label projection requires an observed source marker"
        )
    return number_projection, allow_example_label_marker


@dataclass(frozen=True, slots=True)
class YandexPublicIdentity:
    public_locator: str
    name: str
    kind: str = "yandex_public"


@dataclass(frozen=True, slots=True)
class WorkbookQuestion:
    record_id: str
    content_page_number: int
    question_number: int
    question_marker_kind: str
    question_text: str
    answer: str
    answer_format: str
    key_crop_text: str
    key_projection_sha256: str
    content_projection_sha256: str
    key_binding_kind: str
    section: str
    test_variant: str
    key_page_number: int
    key_context_page_number: int
    key_bbox: tuple[float, float, float, float]
    content_bbox: tuple[float, float, float, float] | None
    visually_checked: bool


@dataclass(frozen=True, slots=True)
class WorkbookDocument:
    document_id: str
    identity: YandexPublicIdentity
    pdf_sha256: str
    page_count: int
    content_page_ranges: tuple[tuple[int, int], ...]
    questions: tuple[WorkbookQuestion, ...]

    def content_page_indexes(self) -> tuple[int, ...]:
        return tuple(
            page - 1
            for start, end in self.content_page_ranges
            for page in range(start, end + 1)
        )


@dataclass(frozen=True, slots=True)
class WorkbookIndex:
    documents: tuple[WorkbookDocument, ...]


def _compact_key(value: str) -> str:
    return "".join(part for part in _KEY_COMPONENT.split(value.casefold()) if part)


def reject_benchmark_metadata(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject benchmark-derived fields before they enter a source-native index."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise OfficialSourceError(f"non-string source-index key at {'.'.join(path)}")
            compact = _compact_key(raw_key)
            if compact in _FORBIDDEN_INDEX_KEYS:
                raise OfficialSourceError(
                    f"benchmark-derived key is forbidden in source index: {'.'.join(path + (raw_key,))}"
                )
            reject_benchmark_metadata(child, path + (raw_key,))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            reject_benchmark_metadata(child, path + (f"[{index}]",))


def strict_yandex_public_identity(
    source_url: str,
    *,
    allow_missing_nosw: bool = False,
) -> YandexPublicIdentity:
    """Project a Yandex viewer URL to its immutable public-resource identity.

    ``nosw`` is optional.  When present, it is accepted only as an inert
    numeric viewer flag and is discarded.  It must never become a routing or
    answer feature.
    """

    if (
        source_url != source_url.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in source_url)
        or "#" in source_url
        or _BAD_PERCENT_ESCAPE.search(source_url) is not None
        or not source_url.startswith("https://docs.yandex.ru/docs/view?")
    ):
        raise OfficialSourceError("source URL has non-canonical Yandex Docs syntax")
    try:
        parsed = urlsplit(source_url)
    except ValueError as exc:
        raise OfficialSourceError("source URL cannot be parsed safely") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != "docs.yandex.ru"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/docs/view"
        or parsed.fragment
    ):
        raise OfficialSourceError("source URL is outside the strict Yandex Docs allowlist")
    raw_pairs = parsed.query.split("&")
    if any(part.count("=") != 1 for part in raw_pairs):
        raise OfficialSourceError("Yandex Docs query fields must use one literal equals sign")
    raw_keys = [part.split("=", 1)[0] for part in raw_pairs]
    if any(key not in {"url", "name", "nosw"} for key in raw_keys):
        raise OfficialSourceError("Yandex Docs query keys must be literal and allowlisted")
    try:
        pairs = parse_qsl(
            parsed.query,
            keep_blank_values=True,
            strict_parsing=True,
            encoding="utf-8",
            errors="strict",
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise OfficialSourceError("Yandex Docs query encoding is malformed") from exc
    keys = [key for key, _ in pairs]
    key_set = set(keys)
    allowed_key_sets = (
        ({"url", "name"}, {"url", "name", "nosw"})
        if allow_missing_nosw
        else ({"url", "name", "nosw"},)
    )
    if (
        len(keys) not in (2, 3)
        or len(key_set) != len(keys)
        or key_set not in allowed_key_sets
    ):
        raise OfficialSourceError(
            "Yandex Docs locator must contain url and name once, plus optional nosw once"
        )
    values = {key: value for key, value in pairs}
    public_locator = values["url"]
    name = values["name"]
    if (
        not public_locator.startswith("ya-disk-public://")
        or len(public_locator) <= len("ya-disk-public://")
        or any(unicodedata.category(character) == "Cc" for character in public_locator)
        or any(character.isspace() for character in public_locator)
    ):
        raise OfficialSourceError("Yandex public-resource locator is malformed")
    if (
        not name
        or name != name.strip()
        or not name.casefold().endswith(".pdf")
        or "/" in name
        or "\\" in name
        or any(unicodedata.category(character) == "Cc" for character in name)
    ):
        raise OfficialSourceError("Yandex public-resource filename is malformed")
    if "nosw" in values and not _NOSW.fullmatch(values["nosw"]):
        raise OfficialSourceError("Yandex viewer flag must be numeric and inert")
    return YandexPublicIdentity(public_locator=public_locator, name=name)


def strict_direct_https_identity(source_url: str) -> YandexPublicIdentity:
    """Project one canonical, query-free HTTPS PDF URL to its exact identity.

    Unlike a mutable web-page URL, this identity is used only together with a
    profile-pinned PDF SHA-256.  No normalization is performed: alternate
    hosts, ports, paths, escapes, queries, fragments, credentials, or spelling
    are different identities and therefore fail closed.
    """

    if (
        source_url != source_url.strip()
        or not source_url.startswith("https://")
        or any(ord(character) < 32 or ord(character) == 127 for character in source_url)
        or any(character.isspace() for character in source_url)
        or any(character in source_url for character in ("%", "\\", "?", "#"))
    ):
        raise OfficialSourceError("direct PDF URL has non-canonical HTTPS syntax")
    try:
        parsed = urlsplit(source_url)
    except ValueError as exc:
        raise OfficialSourceError("direct PDF URL cannot be parsed safely") from exc
    hostname = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.netloc != hostname
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"[a-z0-9.-]+", hostname)
        or hostname.startswith(".")
        or hostname.endswith(".")
        or ".." in hostname
    ):
        raise OfficialSourceError("direct PDF URL is outside the strict HTTPS identity syntax")
    path_parts = parsed.path.split("/")
    if (
        not parsed.path.startswith("/")
        or "//" in parsed.path
        or any(part in {"", ".", ".."} for part in path_parts[1:])
        or not re.fullmatch(r"/[A-Za-z0-9._~/-]+\.pdf", parsed.path)
    ):
        raise OfficialSourceError("direct PDF URL path is not canonical")
    name = path_parts[-1]
    return YandexPublicIdentity(
        public_locator=source_url,
        name=name,
        kind="direct_https",
    )


def strict_public_document_identity(
    source_url: str,
    *,
    allow_missing_nosw: bool = False,
) -> YandexPublicIdentity:
    """Dispatch only between the two frozen public-PDF identity syntaxes."""

    if source_url.startswith("https://docs.yandex.ru/docs/view?"):
        return strict_yandex_public_identity(
            source_url,
            allow_missing_nosw=allow_missing_nosw,
        )
    return strict_direct_https_identity(source_url)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise OfficialSourceError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise OfficialSourceError(f"{label} must be a positive integer") from exc
    if parsed < 1 or parsed != value:
        raise OfficialSourceError(f"{label} must be a positive integer")
    return parsed


def _bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise OfficialSourceError(f"{label} must contain four coordinates")
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in value
    ):
        raise OfficialSourceError(f"{label} must contain numeric coordinates")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise OfficialSourceError(f"{label} must contain numeric coordinates") from exc
    if not all(math.isfinite(item) for item in result):
        raise OfficialSourceError(f"{label} coordinates must be finite")
    if not (0.0 <= result[0] < result[2] and 0.0 <= result[1] < result[3]):
        raise OfficialSourceError(f"{label} coordinates are not ordered")
    return result  # type: ignore[return-value]


def parse_workbook_index(payload: Mapping[str, Any]) -> WorkbookIndex:
    """Validate and materialize a task-ID-free source-native index."""

    reject_benchmark_metadata(payload)
    if set(payload) != _ROOT_KEYS:
        raise OfficialSourceError("workbook source-index root fields are not on the strict allowlist")
    if payload.get("schema_version") != INDEX_SCHEMA:
        raise OfficialSourceError("unsupported workbook source-index schema")
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes)):
        raise OfficialSourceError("workbook source index has no documents")
    documents: list[WorkbookDocument] = []
    document_ids: set[str] = set()
    identities: set[tuple[str, str, str]] = set()
    for raw_document in raw_documents:
        if not isinstance(raw_document, Mapping):
            raise OfficialSourceError("workbook document entry is malformed")
        if set(raw_document) != _DOCUMENT_KEYS:
            raise OfficialSourceError("workbook document fields are not on the strict allowlist")
        document_id = str(raw_document.get("document_id") or "").strip()
        locator = raw_document.get("locator")
        pdf_sha256 = str(raw_document.get("pdf_sha256") or "")
        page_count = _positive_integer(raw_document.get("page_count"), "page_count")
        if (
            not document_id
            or document_id in document_ids
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,80}", document_id)
            or _TASK_LIKE_IDENTIFIER.search(document_id)
            or not document_id.endswith(pdf_sha256[:12])
        ):
            raise OfficialSourceError(
                "workbook document_id must be unique, source-derived, and end in the PDF hash"
            )
        if not _HEX64.fullmatch(pdf_sha256):
            raise OfficialSourceError("workbook PDF is not pinned by SHA-256")
        if not isinstance(locator, Mapping) or set(locator) != _LOCATOR_KEYS:
            raise OfficialSourceError("workbook locator fields are malformed")
        locator_kind = locator.get("kind")
        locator_value = str(locator.get("public_locator") or "")
        locator_name = str(locator.get("name") or "").strip()
        if locator_kind == "yandex_public":
            identity = YandexPublicIdentity(
                public_locator=locator_value,
                name=locator_name,
            )
            if (
                not identity.public_locator.startswith("ya-disk-public://")
                or not identity.name.casefold().endswith(".pdf")
                or "/" in identity.name
                or "\\" in identity.name
            ):
                raise OfficialSourceError("indexed Yandex public identity is malformed")
        elif locator_kind == "direct_https":
            identity = strict_direct_https_identity(locator_value)
            if locator_name != identity.name:
                raise OfficialSourceError("indexed direct PDF filename does not match its URL")
        else:
            raise OfficialSourceError("workbook locator kind is not allowlisted")
        identity_key = (identity.kind, identity.public_locator, identity.name)
        if identity_key in identities:
            raise OfficialSourceError("workbook public identity is duplicated")

        raw_ranges = raw_document.get("content_page_ranges")
        if not isinstance(raw_ranges, Sequence) or isinstance(raw_ranges, (str, bytes)):
            raise OfficialSourceError("workbook content page ranges are missing")
        ranges: list[tuple[int, int]] = []
        occupied_pages: set[int] = set()
        for raw_range in raw_ranges:
            if (
                not isinstance(raw_range, Sequence)
                or isinstance(raw_range, (str, bytes))
                or len(raw_range) != 2
            ):
                raise OfficialSourceError("workbook content page range is malformed")
            start = _positive_integer(raw_range[0], "content range start")
            end = _positive_integer(raw_range[1], "content range end")
            if start > end or end > page_count:
                raise OfficialSourceError("workbook content page range is outside the PDF")
            pages = set(range(start, end + 1))
            if pages & occupied_pages:
                raise OfficialSourceError("workbook content page ranges overlap")
            occupied_pages.update(pages)
            ranges.append((start, end))
        if len(occupied_pages) < 2:
            raise OfficialSourceError("workbook index exposes too few content pages")

        raw_questions = raw_document.get("questions")
        if not isinstance(raw_questions, Sequence) or isinstance(raw_questions, (str, bytes)):
            raise OfficialSourceError("workbook document has no indexed questions")
        questions: list[WorkbookQuestion] = []
        record_ids: set[str] = set()
        page_numbers: set[tuple[int, int]] = set()
        for raw_question in raw_questions:
            if not isinstance(raw_question, Mapping):
                raise OfficialSourceError("workbook question record is malformed")
            if not set(raw_question) <= _QUESTION_KEYS:
                raise OfficialSourceError("workbook question fields are not on the strict allowlist")
            for field in (
                "record_id",
                "question_marker_kind",
                "question_text",
                "answer",
                "answer_format",
                "key_crop_text",
                "key_projection_sha256",
                "content_projection_sha256",
                "key_binding_kind",
                "section",
                "test_variant",
            ):
                if field in raw_question and not isinstance(raw_question[field], str):
                    raise OfficialSourceError(
                        f"workbook question field {field} must be a literal string"
                    )
            content_page = _positive_integer(
                raw_question.get("content_page_number"), "content_page_number"
            )
            question_number = _positive_integer(
                raw_question.get("question_number"), "question_number"
            )
            key_page = _positive_integer(
                raw_question.get("key_page_number"), "key_page_number"
            )
            key_context_page = _positive_integer(
                raw_question.get("key_context_page_number", key_page),
                "key_context_page_number",
            )
            record_id = str(raw_question.get("record_id") or "")
            expected_record_id = f"{document_id}:p{content_page}:q{question_number}"
            if record_id != expected_record_id or record_id in record_ids:
                raise OfficialSourceError("workbook record_id is not source-addressed or is duplicated")
            if (
                content_page not in occupied_pages
                or key_page > page_count
                or key_context_page > page_count
            ):
                raise OfficialSourceError("workbook question/key page is outside its pinned ranges")
            if (content_page, question_number) in page_numbers:
                raise OfficialSourceError("workbook page/question address is ambiguous")
            raw_answer_format = raw_question.get("answer_format", "choice")
            raw_answer = raw_question.get("answer")
            if not isinstance(raw_answer_format, str) or not isinstance(raw_answer, str):
                raise OfficialSourceError(
                    "workbook answer and answer_format must be literal strings"
                )
            answer_format = raw_answer_format.strip()
            answer = raw_answer.strip()
            if answer_format not in {"choice", "short_text"}:
                raise OfficialSourceError("workbook answer_format must be choice or short_text")
            if answer_format == "choice":
                answer = answer.upper()
                if answer not in frozenset("ABCDE"):
                    raise OfficialSourceError("workbook choice key is not A-E")
            elif (
                not answer
                or len(answer) > 512
                or any(ord(character) < 32 and character not in "\t\n" for character in answer)
            ):
                raise OfficialSourceError("workbook short-text answer is empty or malformed")
            key_crop_text = str(raw_question.get("key_crop_text") or "").strip()
            key_projection_sha256 = str(
                raw_question.get("key_projection_sha256") or ""
            ).strip()
            content_projection_sha256 = str(
                raw_question.get("content_projection_sha256") or ""
            ).strip()
            key_binding_kind = str(raw_question.get("key_binding_kind") or "").strip()
            question_marker_kind = str(
                raw_question.get("question_marker_kind") or "numbered_item"
            ).strip()
            section = str(raw_question.get("section") or "").strip()
            test_variant = str(raw_question.get("test_variant") or "").strip()
            question_text = str(raw_question.get("question_text") or "").strip()
            content_bbox_raw = raw_question.get("content_bbox")
            content_bbox = (
                _bbox(content_bbox_raw, "content_bbox")
                if content_bbox_raw is not None
                else None
            )
            if answer_format == "short_text":
                if key_binding_kind == "exact_key_text":
                    if not key_crop_text or key_projection_sha256:
                        raise OfficialSourceError(
                            "workbook exact short-text answer requires an exact "
                            "extracted key binding"
                        )
                elif key_binding_kind == "coordinate_answer_key":
                    if (
                        not key_crop_text
                        or not question_text
                        or content_bbox is None
                        or _HEX64.fullmatch(key_projection_sha256) is None
                    ):
                        raise OfficialSourceError(
                            "workbook coordinate short-text answer requires "
                            "text, boxes, and a projection pin"
                        )
                elif key_binding_kind == "coordinate_table_answer_key":
                    if (
                        not key_crop_text
                        or not question_text
                        or content_bbox is None
                        or not section
                        or not test_variant
                        or "question_marker_kind" not in raw_question
                        or "content_projection_sha256" not in raw_question
                        or "key_context_page_number" not in raw_question
                        or question_marker_kind
                        not in {"numbered_item", "example_label"}
                        or _HEX64.fullmatch(key_projection_sha256) is None
                        or _HEX64.fullmatch(content_projection_sha256) is None
                        or key_context_page not in {key_page, key_page - 1}
                    ):
                        raise OfficialSourceError(
                            "workbook coordinate-table short-text answer requires "
                            "complete adjacent-page key and content projection pins"
                        )
                else:
                    raise OfficialSourceError(
                        "workbook short-text answer has no supported source binding"
                    )
            if answer_format == "choice":
                if key_crop_text or key_projection_sha256 or key_binding_kind not in {
                    "inline_solution",
                    "answer_key_table",
                    "answer_key_list",
                }:
                    raise OfficialSourceError("choice record has no supported source binding")
                if key_binding_kind == "inline_solution" and (
                    key_page != content_page or section or test_variant
                ):
                    raise OfficialSourceError("inline solution binding is inconsistent")
                if key_binding_kind in {"answer_key_table", "answer_key_list"} and not (
                    section and test_variant
                ):
                    raise OfficialSourceError("answer-key binding lacks source context")
            if key_binding_kind != "coordinate_table_answer_key" and (
                question_marker_kind != "numbered_item"
                or content_projection_sha256
                or key_context_page != key_page
            ):
                raise OfficialSourceError(
                    "legacy workbook binding cannot use coordinate-table marker metadata"
                )
            visually_checked = raw_question.get("visually_checked") is True
            if not visually_checked:
                raise OfficialSourceError("workbook source key lacks visual review")
            questions.append(
                WorkbookQuestion(
                    record_id=record_id,
                    content_page_number=content_page,
                    question_number=question_number,
                    question_marker_kind=question_marker_kind,
                    question_text=question_text,
                    answer=answer,
                    answer_format=answer_format,
                    key_crop_text=key_crop_text,
                    key_projection_sha256=key_projection_sha256,
                    content_projection_sha256=content_projection_sha256,
                    key_binding_kind=key_binding_kind,
                    section=section,
                    test_variant=test_variant,
                    key_page_number=key_page,
                    key_context_page_number=key_context_page,
                    key_bbox=_bbox(raw_question.get("key_bbox"), "key_bbox"),
                    content_bbox=content_bbox,
                    visually_checked=True,
                )
            )
            record_ids.add(record_id)
            page_numbers.add((content_page, question_number))
        if not questions:
            raise OfficialSourceError("workbook document contains no reviewed source questions")
        document_ids.add(document_id)
        identities.add(identity_key)
        documents.append(
            WorkbookDocument(
                document_id=document_id,
                identity=identity,
                pdf_sha256=pdf_sha256,
                page_count=page_count,
                content_page_ranges=tuple(sorted(ranges)),
                questions=tuple(
                    sorted(
                        questions,
                        key=lambda question: (
                            question.content_page_number,
                            question.question_number,
                        ),
                    )
                ),
            )
        )
    if not documents:
        raise OfficialSourceError("workbook source index is empty")
    return WorkbookIndex(documents=tuple(sorted(documents, key=lambda item: item.document_id)))


def document_for_source(
    index: WorkbookIndex,
    source_url: str,
    *,
    allow_missing_nosw: bool = False,
) -> WorkbookDocument | None:
    identity = strict_public_document_identity(
        source_url,
        allow_missing_nosw=allow_missing_nosw,
    )
    matches = [document for document in index.documents if document.identity == identity]
    if len(matches) > 1:
        raise OfficialSourceError("public PDF identity is ambiguous in source index")
    return matches[0] if matches else None


def verify_workbook_index_pdf(
    pdf_path: Path,
    document: WorkbookDocument,
) -> dict[str, Any]:
    """Re-read every indexed answer from its pinned PDF cell.

    The index supplies source coordinates, not authority: a certificate run
    still fails unless the current PDF bytes expose exactly the indexed source
    answer inside every key cell and the printed number on its content page.
    """

    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise OfficialSourceError("workbook source verification requires pdfplumber") from exc
    verified = 0
    content_marker_counts: dict[str, int] = {}
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != document.page_count:
            raise OfficialSourceError("workbook PDF page count differs from source index")
        for question in document.questions:
            key_page = pdf.pages[question.key_page_number - 1]
            key_bbox = question.key_bbox
            if not (
                0.0 <= key_bbox[0] < key_bbox[2] <= float(key_page.width)
                and 0.0 <= key_bbox[1] < key_bbox[3] <= float(key_page.height)
            ):
                raise OfficialSourceError(f"key bbox is outside PDF page for {question.record_id}")
            key_text = key_page.crop(key_bbox).extract_text() or ""
            if question.answer_format == "choice":
                choices = _CHOICE_TOKEN.findall(key_text.upper())
                key_matches = choices == [question.answer] and _choice_binding_matches(
                    key_page, question
                )
            elif question.key_binding_kind == "exact_key_text":
                normalized_key = " ".join(key_text.split()).casefold()
                normalized_crop = " ".join(question.key_crop_text.split()).casefold()
                key_matches = (
                    normalized_key == normalized_crop
                    and normalized_crop == " ".join(question.answer.split()).casefold()
                )
            elif question.key_binding_kind == "coordinate_answer_key":
                try:
                    verification = verify_coordinate_answer_key(
                        key_page,
                        pdf_sha256=document.pdf_sha256,
                        physical_page=question.key_page_number,
                        bbox=question.key_bbox,
                        question_number=question.question_number,
                        expected_answer=question.answer,
                        expected_projection_sha256=question.key_projection_sha256,
                    )
                except CoordinateAnswerKeyError as exc:
                    raise OfficialSourceError(
                        f"coordinate answer key is not verified for {question.record_id}: {exc}"
                    ) from exc
                key_matches = (
                    verification.projection_sha256 == question.key_projection_sha256
                    and all(verification.component_matches.values())
                )
            else:
                context_page = pdf.pages[question.key_context_page_number - 1]
                try:
                    verification = verify_coordinate_table_answer_key(
                        key_page,
                        pdf_sha256=document.pdf_sha256,
                        physical_page=question.key_page_number,
                        bbox=question.key_bbox,
                        question_number=question.question_number,
                        expected_answer=question.answer,
                        expected_section=question.section,
                        expected_test_variant=question.test_variant,
                        expected_projection_sha256=question.key_projection_sha256,
                        section_page=context_page,
                        section_physical_page=question.key_context_page_number,
                    )
                except CoordinateTableAnswerKeyError as exc:
                    raise OfficialSourceError(
                        "coordinate table answer key is not verified for "
                        f"{question.record_id}: {exc}"
                    ) from exc
                key_matches = (
                    verification.projection_sha256 == question.key_projection_sha256
                    and all(verification.component_matches.values())
                )
            if not key_matches:
                raise OfficialSourceError(
                    f"key cell does not expose the indexed answer for {question.record_id}"
                )
            content_page = pdf.pages[question.content_page_number - 1]
            if question.content_bbox is not None:
                content_bbox = question.content_bbox
                if not (
                    0.0 <= content_bbox[0] < content_bbox[2] <= float(content_page.width)
                    and 0.0 <= content_bbox[1] < content_bbox[3] <= float(content_page.height)
                ):
                    raise OfficialSourceError(
                        f"content bbox is outside PDF page for {question.record_id}"
                    )
                content_text = content_page.crop(content_bbox).extract_text() or ""
            else:
                content_text = content_page.extract_text() or ""
            if question.key_binding_kind == "coordinate_table_answer_key":
                assert question.content_bbox is not None
                try:
                    content_verification = verify_content_question_marker(
                        content_page,
                        pdf_sha256=document.pdf_sha256,
                        physical_page=question.content_page_number,
                        bbox=question.content_bbox,
                        question_number=question.question_number,
                        marker_kind=question.question_marker_kind,
                        question_text=question.question_text,
                        expected_projection_sha256=(
                            question.content_projection_sha256
                        ),
                    )
                except CoordinateTableAnswerKeyError as exc:
                    raise OfficialSourceError(
                        "coordinate table content marker is not verified for "
                        f"{question.record_id}: {exc}"
                    ) from exc
                content_marker_count = content_verification.marker_count
            else:
                content_marker_count = _question_marker_count(
                    content_text,
                    question.question_number,
                )
            content_marker_counts[question.record_id] = content_marker_count
            if content_marker_count != 1:
                raise OfficialSourceError(
                    f"printed number is absent or ambiguous for {question.record_id}"
                )
            if question.key_binding_kind == "coordinate_answer_key" and not _contains_contiguous(
                normalize_tokens(content_text), normalize_tokens(question.question_text)
            ):
                raise OfficialSourceError(
                    "coordinate answer key is not bound to its source question "
                    f"for {question.record_id}"
                )
            verified += 1
    return {
        "records": len(document.questions),
        "verified_records": verified,
        "content_marker_counts": dict(sorted(content_marker_counts.items())),
    }


def _word_center(word: Mapping[str, Any]) -> tuple[float, float]:
    return (
        (float(word["x0"]) + float(word["x1"])) / 2.0,
        (float(word["top"]) + float(word["bottom"])) / 2.0,
    )


def _words_in_bbox(
    words: Sequence[Mapping[str, Any]], bbox: tuple[float, float, float, float]
) -> list[Mapping[str, Any]]:
    return [
        word
        for word in words
        if bbox[0] <= _word_center(word)[0] <= bbox[2]
        and bbox[1] <= _word_center(word)[1] <= bbox[3]
    ]


def _normalized_phrase(value: str) -> tuple[str, ...]:
    return normalize_tokens(value)


def _contains_contiguous(values: Sequence[str], target: Sequence[str]) -> bool:
    if not target or len(target) > len(values):
        return False
    return any(tuple(values[index : index + len(target)]) == tuple(target) for index in range(len(values) - len(target) + 1))


def _word_lines(
    words: Sequence[Mapping[str, Any]],
    *,
    y_tolerance: float = 6.0,
) -> list[list[Mapping[str, Any]]]:
    """Group PDF words into source-visible rows without trusting text order."""

    lines: list[list[Mapping[str, Any]]] = []
    line_centers: list[float] = []
    for word in sorted(words, key=lambda item: (_word_center(item)[1], float(item["x0"]))):
        center_y = _word_center(word)[1]
        if lines and abs(center_y - line_centers[-1]) <= y_tolerance:
            lines[-1].append(word)
            line_centers[-1] = sum(_word_center(item)[1] for item in lines[-1]) / len(
                lines[-1]
            )
        else:
            lines.append([word])
            line_centers.append(center_y)
    return [sorted(line, key=lambda item: float(item["x0"])) for line in lines]


def _line_center_y(line: Sequence[Mapping[str, Any]]) -> float:
    return sum(_word_center(word)[1] for word in line) / len(line)


def _line_tokens(line: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return normalize_tokens(" ".join(str(word.get("text") or "") for word in line))


def _line_has_choice_cell(line: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        re.fullmatch(
            r"\d{1,3}\s*[.:-]\s*[A-E]",
            str(word.get("text") or "").strip(),
            re.IGNORECASE,
        )
        is not None
        for word in line
    )


def _line_containing_word(
    lines: Sequence[Sequence[Mapping[str, Any]]],
    target: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]] | None:
    for line in lines:
        if any(word is target for word in line):
            return line
    return None


def _strict_hyphen_table_context(
    words: Sequence[Mapping[str, Any]],
    combined_cell: Mapping[str, Any],
    question: WorkbookQuestion,
) -> bool:
    """Bind an ``N-A`` cell to its same-row ADIM and nearest section row."""

    lines = _word_lines(words)
    answer_line = _line_containing_word(lines, combined_cell)
    if answer_line is None or not _contains_contiguous(
        _line_tokens(answer_line), _normalized_phrase(question.test_variant)
    ):
        return False
    answer_line_y = _line_center_y(answer_line)
    prior_source_rows = [
        line
        for line in lines
        if _line_center_y(line) < answer_line_y and not _line_has_choice_cell(line)
    ]
    if not prior_source_rows:
        return False
    nearest_section_row = max(prior_source_rows, key=_line_center_y)
    return _line_tokens(nearest_section_row) == _normalized_phrase(question.section)


def _strict_answer_list_context(
    page: Any,
    words: Sequence[Mapping[str, Any]],
    combined_cell: Mapping[str, Any],
    question: WorkbookQuestion,
) -> bool:
    """Bind a list cell to the nearest subject in its column and book header."""

    lines = _word_lines(words)
    answer_line = _line_containing_word(lines, combined_cell)
    if answer_line is None:
        return False
    answer_x, answer_y = _word_center(combined_cell)
    midpoint = float(page.width) / 2.0
    answer_is_left = answer_x < midpoint

    def same_column(line: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [
            word
            for word in line
            if (_word_center(word)[0] < midpoint) == answer_is_left
        ]

    prior_source_rows: list[Sequence[Mapping[str, Any]]] = []
    for line in lines:
        if _line_center_y(line) >= answer_y:
            continue
        column_words = same_column(line)
        if column_words and not _line_has_choice_cell(column_words):
            prior_source_rows.append(column_words)
    if not prior_source_rows:
        return False
    nearest_subject_row = max(prior_source_rows, key=_line_center_y)
    if _line_tokens(nearest_subject_row) != _normalized_phrase(question.section):
        return False

    subject_y = _line_center_y(nearest_subject_row)
    header_rows = [
        line
        for line in lines
        if _line_center_y(line) < subject_y
        and _line_tokens(line) == _normalized_phrase(question.test_variant)
    ]
    return len(header_rows) == 1


def _choice_binding_matches(page: Any, question: WorkbookQuestion) -> bool:
    """Prove that an A-E glyph belongs to the indexed source question."""

    words = page.extract_words() or []
    key_text = page.crop(question.key_bbox).extract_text() or ""
    if _CHOICE_TOKEN.findall(key_text.upper()) != [question.answer]:
        return False
    answer_words = [
        word
        for word in _words_in_bbox(words, question.key_bbox)
        if str(word.get("text") or "").strip().upper() == question.answer
    ]
    answer_word = answer_words[0] if len(answer_words) == 1 else None
    answer_x = (question.key_bbox[0] + question.key_bbox[2]) / 2.0
    answer_y = (question.key_bbox[1] + question.key_bbox[3]) / 2.0
    if question.key_binding_kind == "inline_solution":
        if answer_word is None:
            return False
        midpoint = float(page.width) / 2.0
        column = 0 if answer_x < midpoint else 1
        labels = [
            word
            for word in words
            if str(word.get("text") or "").strip().casefold().startswith("cevap")
            and abs(_word_center(word)[1] - answer_y) <= 3.0
            and 0.0 <= float(answer_word["x0"]) - float(word["x1"]) <= 8.0
        ]
        if len(labels) != 1:
            return False
        markers: list[tuple[int, Mapping[str, Any]]] = []
        for word in words:
            match = re.fullmatch(r"(\d{1,3})[.)]", str(word.get("text") or "").strip())
            if match is None or float(word["top"]) >= float(answer_word["top"]):
                continue
            x0 = float(word["x0"])
            word_column = 0 if x0 < midpoint else 1
            in_margin = x0 <= 80.0 or midpoint <= x0 <= midpoint + 80.0
            if word_column == column and in_margin:
                markers.append((int(match.group(1)), word))
        if not markers:
            return False
        primary_x = min(float(item[1]["x0"]) for item in markers)
        markers = [
            item for item in markers if float(item[1]["x0"]) <= primary_x + 10.0
        ]
        nearest_number, _ = max(markers, key=lambda item: float(item[1]["top"]))
        return nearest_number == question.question_number

    if question.key_binding_kind == "answer_key_list":
        combined_cell = [
            word
            for word in words
            if re.fullmatch(
                rf"{question.question_number}\s*[.:-]\s*{question.answer}",
                str(word.get("text") or "").strip(),
                re.IGNORECASE,
            )
            and abs(_word_center(word)[1] - answer_y) <= 3.0
            and float(word["x0"]) <= question.key_bbox[0] + 1.0
            and float(word["x1"]) >= question.key_bbox[2] - 1.0
        ]
        if len(combined_cell) != 1:
            return False
        return _strict_answer_list_context(
            page,
            words,
            combined_cell[0],
            question,
        )

    if question.key_binding_kind != "answer_key_table":
        return False
    context_words = sorted(
        (
            word
            for word in words
            if question.key_bbox[1] - 150.0 <= float(word["top"])
            < question.key_bbox[1]
        ),
        key=lambda word: (round(float(word["top"]), 1), float(word["x0"])),
    )
    nearby_context_tokens = normalize_tokens(
        " ".join(str(word.get("text") or "") for word in context_words)
    )
    prior_context_tokens = normalize_tokens(
        " ".join(
            str(word.get("text") or "")
            for word in sorted(
                (word for word in words if float(word["top"]) < question.key_bbox[1]),
                key=lambda word: (round(float(word["top"]), 1), float(word["x0"])),
            )
        )
    )
    context_matches = _contains_contiguous(
        nearby_context_tokens, _normalized_phrase(question.test_variant)
    ) and _contains_contiguous(
        prior_context_tokens, _normalized_phrase(question.section)
    )
    combined_cell = [
        word
        for word in words
        if re.fullmatch(
            rf"{question.question_number}\s*[.:-]\s*{question.answer}",
            str(word.get("text") or "").strip(),
            re.IGNORECASE,
        )
        and abs(_word_center(word)[1] - answer_y) <= 3.0
        and float(word["x0"]) <= question.key_bbox[0] + 1.0
        and float(word["x1"]) >= question.key_bbox[2] - 1.0
    ]
    if len(combined_cell) == 1:
        if re.fullmatch(
            rf"{question.question_number}\s*-\s*{question.answer}",
            str(combined_cell[0].get("text") or "").strip(),
            re.IGNORECASE,
        ):
            return _strict_hyphen_table_context(words, combined_cell[0], question)
        return context_matches
    if answer_word is None:
        return False
    direct_number = [
        word
        for word in words
        if str(word.get("text") or "").strip() == f"{question.question_number}."
        and abs(_word_center(word)[1] - answer_y) <= 3.0
        and 0.0 <= float(answer_word["x0"]) - float(word["x1"]) <= 16.0
    ]
    if len(direct_number) == 1:
        return context_matches
    header_candidates = [
        word
        for word in words
        if str(word.get("text") or "").strip() == str(question.question_number)
        and float(word["bottom"]) < float(answer_word["top"])
        and abs(_word_center(word)[0] - answer_x) <= 4.0
        and float(answer_word["top"]) - float(word["bottom"]) <= 180.0
    ]
    if not header_candidates:
        return False
    header = max(header_candidates, key=lambda word: float(word["top"]))
    header_top = float(header["top"])
    header_words = [
        word
        for word in words
        if re.fullmatch(r"\d{1,3}", str(word.get("text") or "").strip())
        and abs(float(word["top"]) - header_top) <= 2.5
    ]
    if len(header_words) < 2:
        return False
    first_header_x = min(float(word["x0"]) for word in header_words)
    row_words = sorted(
        (
            word
            for word in words
            if float(word["x1"]) < first_header_x - 2.0
            and not (
                float(word["bottom"]) < float(answer_word["top"]) - 2.5
                or float(word["top"]) > float(answer_word["bottom"]) + 2.5
            )
        ),
        key=lambda word: float(word["x0"]),
    )
    row_text = " ".join(str(word.get("text") or "") for word in row_words)
    if _normalized_phrase(row_text) != _normalized_phrase(question.test_variant):
        return False
    heading_words = sorted(
        (
            word
            for word in words
            if header_top - 55.0 <= float(word["top"]) < header_top
        ),
        key=lambda word: (round(float(word["top"]), 1), float(word["x0"])),
    )
    heading_tokens = normalize_tokens(
        " ".join(str(word.get("text") or "") for word in heading_words)
    )
    return _contains_contiguous(heading_tokens, _normalized_phrase(question.section))


def _question_marker_present(page_text: str, number: int) -> bool:
    return _question_marker_count(page_text, number) >= 1


def _question_marker_count(page_text: str, number: int) -> int:
    return len(re.findall(rf"(?<!\d){number}\s*[.)](?=\s|$)", page_text))


def _numberless_question_match(
    observation: OcrObservation,
    questions: Sequence[WorkbookQuestion],
    thresholds: WorkbookThresholds,
) -> tuple[WorkbookQuestion | None, dict[str, Any]]:
    candidates = [question for question in questions if question.question_text]
    if not candidates:
        return None, {
            "coverage": 0.0,
            "matched_tokens": 0,
            "query_tokens": 0,
            "margin": 0.0,
        }
    matcher = PageMatcher([question.question_text for question in candidates])
    scored = [
        (matcher.score(observation.statement, index), question)
        for index, question in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (-item[0][0], item[1].record_id))
    (coverage, matched, total), selected = scored[0]
    runner = scored[1][0][0] if len(scored) > 1 else 0.0
    margin = coverage - runner
    accepted = (
        coverage >= thresholds.min_numberless_question_coverage
        and matched >= thresholds.min_numberless_question_matched_tokens
        and margin >= thresholds.min_numberless_question_margin
    )
    return (selected if accepted else None), {
        "coverage": coverage,
        "matched_tokens": matched,
        "query_tokens": total,
        "margin": margin,
    }


def resolve_workbook_question(
    observation: OcrObservation,
    source_url: str,
    document: WorkbookDocument,
    matcher: PageMatcher,
    page_texts: Sequence[str],
    thresholds: WorkbookThresholds,
    *,
    allow_missing_nosw: bool = False,
    allow_example_label_marker: bool = False,
    verified_content_marker_counts: Mapping[str, int] | None = None,
) -> MatchResult:
    """Bind one parser crop to one reviewed workbook key, or abstain."""

    observed_identity = strict_public_document_identity(
        source_url,
        allow_missing_nosw=allow_missing_nosw,
    )
    if observed_identity != document.identity:
        raise OfficialSourceError("source URL does not match the pinned workbook identity")
    if matcher.page_count != len(page_texts) or len(page_texts) != document.page_count:
        raise OfficialSourceError("workbook matcher, index, and PDF page counts do not align")
    content_pages = document.content_page_indexes()
    scores = {page: matcher.score(observation.statement, page) for page in content_pages}
    order = sorted(content_pages, key=lambda page: (-scores[page][0], page))
    best_page, runner_page = order[:2]
    page_coverage, page_matched, page_total = scores[best_page]
    page_margin = page_coverage - scores[runner_page][0]
    page_is_unique = (
        page_coverage >= thresholds.min_page_coverage
        and page_matched >= thresholds.min_page_matched_tokens
        and page_margin >= thresholds.min_page_margin
    )
    page_number = best_page + 1
    page_questions = [
        question
        for question in document.questions
        if question.content_page_number == page_number
    ]
    numberless_match = {
        "coverage": 0.0,
        "matched_tokens": 0,
        "query_tokens": 0,
        "margin": 0.0,
    }
    observed_marker_kind, observed_marker_number = observed_source_question_marker(
        observation
    )
    if observation.question_number is not None:
        matches = [
            question
            for question in page_questions
            if question.question_marker_kind == "numbered_item"
            and question.question_number == observation.question_number
        ]
        selected = matches[0] if len(matches) == 1 else None
        binding_method = "printed_number"
        question_binding = selected is not None
    elif (
        allow_example_label_marker
        and observed_marker_kind == "example_label"
        and observed_marker_number is not None
    ):
        matches = [
            question
            for question in page_questions
            if question.question_marker_kind == "example_label"
            and question.question_number == observed_marker_number
        ]
        selected = matches[0] if len(matches) == 1 else None
        binding_method = "source_visible_example_label"
        question_binding = selected is not None
    else:
        # Sparse source indexes cannot prove global question-text uniqueness.
        # Numberless admission remains fail-closed until a complete PDF-native
        # question inventory is available for the document.
        selected = None
        binding_method = "missing_printed_number_abstain"
        question_binding = False
    if (
        selected is not None
        and selected.content_bbox is not None
        and verified_content_marker_counts is not None
    ):
        visible_marker_count = verified_content_marker_counts.get(selected.record_id, 0)
    else:
        visible_marker_count = (
            _question_marker_count(page_texts[best_page], selected.question_number)
            if selected is not None
            else 0
        )
    visible_number = selected is not None and visible_marker_count == 1
    checks = (
        ("strict_public_document_identity", observed_identity == document.identity),
        ("unique_content_page", page_is_unique),
        ("unique_source_question_record", selected is not None),
        ("question_binding", question_binding),
        ("printed_number_visible_on_page", visible_number),
        ("reviewed_embedded_key", selected is not None and selected.visually_checked),
        (
            "valid_source_answer",
            selected is not None
            and (
                (selected.answer_format == "choice" and selected.answer in frozenset("ABCDE"))
                or (selected.answer_format == "short_text" and bool(selected.answer.strip()))
            ),
        ),
        ("source_address_not_task_id", selected is not None and selected.record_id.startswith(f"{document.document_id}:p")),
    )
    accepted = all(passed for _, passed in checks)
    answer = selected.answer if accepted and selected is not None else None
    problem = problem_for(
        observation,
        source_url,
        answer_format=selected.answer_format if selected else "source_answer",
    )
    trace = {
        "schema_version": "public-workbook-source-trace-v1",
        "verifier": VERIFIER,
        "source": {
            "document_id": document.document_id,
            "public_locator": document.identity.public_locator,
            "name": document.identity.name,
            "pdf_sha256": document.pdf_sha256,
            "matched_page_number": page_number,
            "runner_up_page_number": runner_page + 1,
            "record_id": selected.record_id if selected else None,
            "question_number": selected.question_number if selected else observation.question_number,
            "answer_format": selected.answer_format if selected else None,
            "question_marker_kind": (
                selected.question_marker_kind if selected else None
            ),
            "key_binding_kind": selected.key_binding_kind if selected else None,
            "key_projection_sha256": selected.key_projection_sha256 if selected else None,
            "content_projection_sha256": (
                selected.content_projection_sha256 if selected else None
            ),
            "key_page_number": selected.key_page_number if selected else None,
            "key_context_page_number": (
                selected.key_context_page_number if selected else None
            ),
            "key_bbox": list(selected.key_bbox) if selected else None,
            "content_bbox": list(selected.content_bbox) if selected and selected.content_bbox else None,
        },
        "observation": {
            "image_sha256": observation.image_sha256,
            "image_size": [observation.width, observation.height],
            "parser_identity": observation.parser_identity,
            "observed_question_number": observation.question_number,
            "observed_source_marker_kind": observed_marker_kind,
            "observed_source_marker_number": observed_marker_number,
        },
        "match": {
            "page_idf_coverage": page_coverage,
            "page_matched_tokens": page_matched,
            "page_query_tokens": page_total,
            "page_margin": page_margin,
            "question_binding_method": binding_method,
            "numberless_question": numberless_match,
        },
        "thresholds": {
            "min_page_coverage": thresholds.min_page_coverage,
            "min_page_matched_tokens": thresholds.min_page_matched_tokens,
            "min_page_margin": thresholds.min_page_margin,
            "min_numberless_question_coverage": thresholds.min_numberless_question_coverage,
            "min_numberless_question_matched_tokens": thresholds.min_numberless_question_matched_tokens,
            "min_numberless_question_margin": thresholds.min_numberless_question_margin,
        },
        "checks": {name: passed for name, passed in checks},
        "accepted": accepted,
    }
    return MatchResult(
        accepted=accepted,
        answer=answer,
        problem=problem,
        checks=checks,
        trace=trace,
    )
