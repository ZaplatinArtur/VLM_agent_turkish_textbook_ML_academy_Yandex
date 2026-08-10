"""Fail-closed binding for a numberless full-page fill-blank activity.

This is intentionally separate from :mod:`official_workbook`.  Ordinary
workbook records require one source-visible question number (or a numbered
activity label).  A full-page ``Boşluk Doldurma`` sheet instead exposes a
title, instruction, word bank, and a complete numbered item inventory.  Those
source-native signals are verified here without weakening the numbered-record
policy used by every other workbook source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from .fill_blank_answer_key import (
    FillBlankAnswerKeyError,
    parse_fill_blank_canonical_answer,
    verify_fill_blank_answer_key,
)
from .official_ogm import (
    MatchResult,
    OcrObservation,
    OfficialSourceError,
    PageMatcher,
    canonical_json_sha256,
    normalize_tokens,
    observed_source_question_marker,
    problem_for,
    sha256_file,
)
from .official_workbook import (
    YandexPublicIdentity,
    reject_benchmark_metadata,
    strict_public_document_identity,
)


INDEX_SCHEMA = "public-workbook-fill-blank-source-index-v1"
TRACE_SCHEMA = "fill-blank-page-activity-source-trace-v1"
VERIFIER = "fill-blank-page-activity-pdf-binding-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")
_TASK_LIKE = re.compile(
    r"(?:^|[._-])(?:val|test|train|task|qid)[._-]?\d+(?:[._-]|$)",
    re.IGNORECASE,
)
_ROOT_KEYS = frozenset({"schema_version", "documents"})
_DOCUMENT_KEYS = frozenset(
    {
        "document_id",
        "locator",
        "pdf_sha256",
        "page_count",
        "content_page_ranges",
        "activities",
    }
)
_LOCATOR_KEYS = frozenset({"kind", "public_locator", "name"})
_ACTIVITY_KEYS = frozenset(
    {
        "record_id",
        "content_page_number",
        "key_page_number",
        "key_context_page_number",
        "question_marker_kind",
        "question_text",
        "answer",
        "answer_format",
        "source_answer_format",
        "key_binding_kind",
        "activity_title",
        "instruction_text",
        "expected_item_count",
        "expected_column_count",
        "key_crop_text",
        "key_projection_sha256",
        "content_projection_sha256",
        "binding_projection_sha256",
        "content_bbox",
        "word_bank_bbox",
        "key_bbox",
        "visually_checked",
    }
)


@dataclass(frozen=True, slots=True)
class FillBlankPageThresholds:
    min_page_coverage: float = 0.65
    min_page_matched_tokens: int = 20
    min_page_margin: float = 0.12
    min_activity_coverage: float = 0.85
    min_activity_matched_tokens: int = 40

    def __post_init__(self) -> None:
        for name in ("min_page_coverage", "min_page_margin", "min_activity_coverage"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.min_page_matched_tokens < 1 or self.min_activity_matched_tokens < 1:
            raise ValueError("fill-blank matched-token thresholds must be positive")


@dataclass(frozen=True, slots=True)
class FillBlankPageActivity:
    record_id: str
    content_page_number: int
    key_page_number: int
    key_context_page_number: int
    question_text: str
    answer: str
    activity_title: str
    instruction_text: str
    expected_item_count: int
    expected_column_count: int
    key_crop_text: str
    key_projection_sha256: str
    content_projection_sha256: str
    binding_projection_sha256: str
    content_bbox: tuple[float, float, float, float]
    word_bank_bbox: tuple[float, float, float, float]
    key_bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FillBlankPageDocument:
    document_id: str
    identity: YandexPublicIdentity
    pdf_sha256: str
    page_count: int
    content_page_ranges: tuple[tuple[int, int], ...]
    activities: tuple[FillBlankPageActivity, ...]

    def content_page_indexes(self) -> tuple[int, ...]:
        return tuple(
            page - 1
            for start, end in self.content_page_ranges
            for page in range(start, end + 1)
        )


@dataclass(frozen=True, slots=True)
class FillBlankPageIndex:
    documents: tuple[FillBlankPageDocument, ...]


def _positive(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise OfficialSourceError(f"{label} must be a positive integer")
    return value


def _literal(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise OfficialSourceError(f"{label} must be literal text")
    result = value.strip()
    if not result or any(ord(character) < 32 and character not in "\t\n" for character in result):
        raise OfficialSourceError(f"{label} is empty or malformed")
    return result


def _bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise OfficialSourceError(f"{label} is malformed")
    result = tuple(float(item) for item in value)
    if not (0.0 <= result[0] < result[2] and 0.0 <= result[1] < result[3]):
        raise OfficialSourceError(f"{label} is empty or unordered")
    return result  # type: ignore[return-value]


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def parse_fill_blank_page_index(payload: Mapping[str, Any]) -> FillBlankPageIndex:
    """Validate a task-ID-free fill-blank source index."""

    reject_benchmark_metadata(payload, ("fill_blank_page_source_index",))
    if set(payload) != _ROOT_KEYS or payload.get("schema_version") != INDEX_SCHEMA:
        raise OfficialSourceError("unsupported fill-blank page source-index schema")
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise OfficialSourceError("fill-blank page source index has no documents")
    documents: list[FillBlankPageDocument] = []
    document_ids: set[str] = set()
    identities: set[tuple[str, str, str]] = set()
    for raw_document in raw_documents:
        if not isinstance(raw_document, Mapping) or set(raw_document) != _DOCUMENT_KEYS:
            raise OfficialSourceError("fill-blank document fields are not allowlisted")
        document_id = str(raw_document.get("document_id") or "")
        pdf_sha256 = str(raw_document.get("pdf_sha256") or "")
        if (
            _DOCUMENT_ID.fullmatch(document_id) is None
            or document_id in document_ids
            or _TASK_LIKE.search(document_id) is not None
            or _HEX64.fullmatch(pdf_sha256) is None
            or not document_id.endswith(pdf_sha256[:12])
        ):
            raise OfficialSourceError("fill-blank document identity is malformed")
        raw_locator = raw_document.get("locator")
        if not isinstance(raw_locator, Mapping) or set(raw_locator) != _LOCATOR_KEYS:
            raise OfficialSourceError("fill-blank public locator is malformed")
        kind = raw_locator.get("kind")
        public_locator = str(raw_locator.get("public_locator") or "")
        name = str(raw_locator.get("name") or "")
        if (
            kind != "yandex_public"
            or not public_locator.startswith("ya-disk-public://")
            or len(public_locator) <= len("ya-disk-public://")
            or any(character.isspace() for character in public_locator)
            or not name
            or name != name.strip()
            or not name.casefold().endswith(".pdf")
            or "/" in name
            or "\\" in name
        ):
            raise OfficialSourceError("fill-blank Yandex identity is malformed")
        identity = YandexPublicIdentity(public_locator=public_locator, name=name)
        identity_key = (identity.kind, identity.public_locator, identity.name)
        if identity_key in identities:
            raise OfficialSourceError("fill-blank public identity is duplicated")
        page_count = _positive(raw_document.get("page_count"), "page_count")
        raw_ranges = raw_document.get("content_page_ranges")
        if not isinstance(raw_ranges, list) or not raw_ranges:
            raise OfficialSourceError("fill-blank content ranges are missing")
        occupied: set[int] = set()
        ranges: list[tuple[int, int]] = []
        for raw_range in raw_ranges:
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raise OfficialSourceError("fill-blank content range is malformed")
            start = _positive(raw_range[0], "content range start")
            end = _positive(raw_range[1], "content range end")
            pages = set(range(start, end + 1))
            if start > end or end > page_count or pages & occupied:
                raise OfficialSourceError("fill-blank content ranges overlap or leave PDF")
            occupied.update(pages)
            ranges.append((start, end))
        if len(occupied) < 2:
            raise OfficialSourceError("fill-blank index exposes too few content pages")
        raw_activities = raw_document.get("activities")
        if not isinstance(raw_activities, list) or not raw_activities:
            raise OfficialSourceError("fill-blank document has no activities")
        activities: list[FillBlankPageActivity] = []
        record_ids: set[str] = set()
        content_addresses: set[int] = set()
        for raw in raw_activities:
            if not isinstance(raw, Mapping) or set(raw) != _ACTIVITY_KEYS:
                raise OfficialSourceError("fill-blank activity fields are not allowlisted")
            content_page = _positive(raw.get("content_page_number"), "content_page_number")
            key_page = _positive(raw.get("key_page_number"), "key_page_number")
            key_context_page = _positive(
                raw.get("key_context_page_number"), "key_context_page_number"
            )
            record_id = str(raw.get("record_id") or "")
            if (
                record_id != f"{document_id}:p{content_page}:fill_blank"
                or record_id in record_ids
                or _TASK_LIKE.search(record_id) is not None
                or content_page in content_addresses
                or content_page not in occupied
                or key_page > page_count
                or key_page == content_page
                or key_context_page != key_page
            ):
                raise OfficialSourceError("fill-blank activity address is malformed")
            if (
                raw.get("question_marker_kind") != "numberless_page_activity"
                or raw.get("answer_format") != "short_text"
                or raw.get("source_answer_format") != "numbered_short_text"
                or raw.get("key_binding_kind") != "fill_blank_answer_key"
                or raw.get("visually_checked") is not True
            ):
                raise OfficialSourceError("fill-blank activity contract is malformed")
            item_count = _positive(raw.get("expected_item_count"), "expected_item_count")
            column_count = _positive(
                raw.get("expected_column_count"), "expected_column_count"
            )
            answer = _literal(raw.get("answer"), "fill-blank answer")
            if len(answer) > 512:
                raise OfficialSourceError("fill-blank answer is too long")
            try:
                parse_fill_blank_canonical_answer(
                    answer, expected_item_count=item_count
                )
            except FillBlankAnswerKeyError as exc:
                raise OfficialSourceError("fill-blank answer syntax is malformed") from exc
            hashes = tuple(
                str(raw.get(field) or "")
                for field in (
                    "key_projection_sha256",
                    "content_projection_sha256",
                    "binding_projection_sha256",
                )
            )
            if any(_HEX64.fullmatch(value) is None for value in hashes):
                raise OfficialSourceError("fill-blank projection pin is malformed")
            content_bbox = _bbox(raw.get("content_bbox"), "content_bbox")
            word_bank_bbox = _bbox(raw.get("word_bank_bbox"), "word_bank_bbox")
            key_bbox = _bbox(raw.get("key_bbox"), "key_bbox")
            if not (
                content_bbox[0] <= word_bank_bbox[0] < word_bank_bbox[2] <= content_bbox[2]
                and content_bbox[1] <= word_bank_bbox[1] < word_bank_bbox[3] <= content_bbox[3]
            ):
                raise OfficialSourceError("fill-blank word bank is outside content bbox")
            question_text = _literal(raw.get("question_text"), "question_text")
            key_crop_text = _literal(raw.get("key_crop_text"), "key_crop_text")
            activity_title = _literal(raw.get("activity_title"), "activity_title")
            instruction_text = _literal(raw.get("instruction_text"), "instruction_text")
            if len(normalize_tokens(question_text)) < max(20, item_count):
                raise OfficialSourceError("fill-blank source content is unexpectedly short")
            activities.append(
                FillBlankPageActivity(
                    record_id=record_id,
                    content_page_number=content_page,
                    key_page_number=key_page,
                    key_context_page_number=key_context_page,
                    question_text=question_text,
                    answer=answer,
                    activity_title=activity_title,
                    instruction_text=instruction_text,
                    expected_item_count=item_count,
                    expected_column_count=column_count,
                    key_crop_text=key_crop_text,
                    key_projection_sha256=hashes[0],
                    content_projection_sha256=hashes[1],
                    binding_projection_sha256=hashes[2],
                    content_bbox=content_bbox,
                    word_bank_bbox=word_bank_bbox,
                    key_bbox=key_bbox,
                )
            )
            record_ids.add(record_id)
            content_addresses.add(content_page)
        documents.append(
            FillBlankPageDocument(
                document_id=document_id,
                identity=identity,
                pdf_sha256=pdf_sha256,
                page_count=page_count,
                content_page_ranges=tuple(sorted(ranges)),
                activities=tuple(
                    sorted(activities, key=lambda item: item.content_page_number)
                ),
            )
        )
        document_ids.add(document_id)
        identities.add(identity_key)
    return FillBlankPageIndex(documents=tuple(sorted(documents, key=lambda item: item.document_id)))


def document_for_source(
    index: FillBlankPageIndex,
    source_url: str,
    *,
    allow_missing_nosw: bool = False,
) -> FillBlankPageDocument | None:
    identity = strict_public_document_identity(
        source_url, allow_missing_nosw=allow_missing_nosw
    )
    matches = [document for document in index.documents if document.identity == identity]
    if len(matches) > 1:
        raise OfficialSourceError("fill-blank public identity is ambiguous")
    return matches[0] if matches else None


def verify_fill_blank_page_index_pdf(
    pdf_path: Path,
    document: FillBlankPageDocument,
) -> dict[str, Any]:
    """Replay every content/key projection against the pinned PDF bytes."""

    if sha256_file(pdf_path) != document.pdf_sha256:
        raise OfficialSourceError("fill-blank PDF hash differs from source index")
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise OfficialSourceError("fill-blank PDF replay requires pdfplumber") from exc
    verified = 0
    record_attestations: dict[str, dict[str, bool]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != document.page_count:
            raise OfficialSourceError("fill-blank PDF page count changed")
        for activity in document.activities:
            try:
                result = verify_fill_blank_answer_key(
                    pdf.pages[activity.content_page_number - 1],
                    pdf.pages[activity.key_page_number - 1],
                    pdf_sha256=document.pdf_sha256,
                    content_page_number=activity.content_page_number,
                    key_page_number=activity.key_page_number,
                    content_bbox=activity.content_bbox,
                    word_bank_bbox=activity.word_bank_bbox,
                    key_bbox=activity.key_bbox,
                    activity_title=activity.activity_title,
                    instruction_text=activity.instruction_text,
                    expected_item_count=activity.expected_item_count,
                    expected_column_count=activity.expected_column_count,
                    expected_answer=activity.answer,
                    expected_projection_sha256=activity.binding_projection_sha256,
                    expected_content_projection_sha256=(
                        activity.content_projection_sha256
                    ),
                    expected_key_projection_sha256=activity.key_projection_sha256,
                )
            except FillBlankAnswerKeyError as exc:
                raise OfficialSourceError(
                    f"fill-blank PDF proof failed for {activity.record_id}: {exc}"
                ) from exc
            if (
                _normalized_text(result.content_text)
                != _normalized_text(activity.question_text)
                or _normalized_text(result.key_text)
                != _normalized_text(activity.key_crop_text)
                or not all(result.component_matches.values())
            ):
                raise OfficialSourceError(
                    f"fill-blank canonical text differs for {activity.record_id}"
                )
            # ``verify_fill_blank_answer_key`` is deliberately stricter than
            # the benchmark observation: it re-opens the immutable PDF and
            # proves title/instruction uniqueness, the complete marker column,
            # the exact word-bank multiset, and every answer-key component.
            # Keep those PDF facts structured so the resolver never pretends
            # that title/bank text stripped by the OCR projection was observed.
            record_attestations[activity.record_id] = {
                "pdf_binding": True,
                "activity_title_once": True,
                "activity_instruction_once": True,
                "complete_item_inventory": True,
                "word_bank_multiset": True,
                "answer_key_components": all(result.component_matches.values()),
            }
            verified += 1
    return {
        "records": len(document.activities),
        "verified_records": verified,
        "record_attestations": dict(sorted(record_attestations.items())),
    }


def _line_marker_inventory(observation: OcrObservation) -> tuple[int, ...]:
    texts = observation.text_blocks or (observation.statement,)
    markers: list[int] = []
    for text in texts:
        for match in re.finditer(r"(?m)^\s*([1-9]\d*)\s*[.)](?=\s)", text):
            markers.append(int(match.group(1)))
    return tuple(sorted(markers))


def resolve_fill_blank_page_activity(
    observation: OcrObservation,
    source_url: str,
    document: FillBlankPageDocument,
    matcher: PageMatcher,
    page_texts: Sequence[str],
    thresholds: FillBlankPageThresholds,
    *,
    verified_record_attestations: Mapping[str, Mapping[str, bool]],
    allow_missing_nosw: bool = False,
) -> MatchResult:
    """Bind one full-page observation to an attested numberless activity."""

    observed_identity = strict_public_document_identity(
        source_url, allow_missing_nosw=allow_missing_nosw
    )
    if observed_identity != document.identity:
        raise OfficialSourceError("source URL does not match fill-blank document")
    if matcher.page_count != len(page_texts) or len(page_texts) != document.page_count:
        raise OfficialSourceError("fill-blank matcher/PDF/index page counts differ")
    content_pages = document.content_page_indexes()
    scores = {page: matcher.score(observation.statement, page) for page in content_pages}
    order = sorted(content_pages, key=lambda page: (-scores[page][0], page))
    if len(order) < 2:
        raise OfficialSourceError("fill-blank source has too few candidate pages")
    best_page, runner_page = order[:2]
    page_coverage, page_matched, page_total = scores[best_page]
    page_margin = page_coverage - scores[runner_page][0]
    page_unique = (
        page_coverage >= thresholds.min_page_coverage
        and page_matched >= thresholds.min_page_matched_tokens
        and page_margin >= thresholds.min_page_margin
    )
    page_number = best_page + 1
    page_records = [
        activity
        for activity in document.activities
        if activity.content_page_number == page_number
    ]
    selected = page_records[0] if len(page_records) == 1 else None
    marker_kind, marker_number = observed_source_question_marker(observation)
    no_single_marker = marker_kind is None and marker_number is None
    observed_tokens = tuple(normalize_tokens(observation.statement))
    if selected is not None:
        source_matcher = PageMatcher([selected.question_text])
        activity_coverage, activity_matched, activity_total = source_matcher.score(
            observation.statement, 0
        )
        observed_inventory = _line_marker_inventory(observation)
        expected_inventory = tuple(range(1, selected.expected_item_count + 1))
        attestation = verified_record_attestations.get(selected.record_id, {})
        pdf_bound = attestation.get("pdf_binding") is True
        pdf_title_once = attestation.get("activity_title_once") is True
        pdf_instruction_once = (
            attestation.get("activity_instruction_once") is True
        )
        pdf_inventory_complete = (
            attestation.get("complete_item_inventory") is True
        )
        pdf_bank_exact = attestation.get("word_bank_multiset") is True
        pdf_key_components = (
            attestation.get("answer_key_components") is True
        )
    else:
        activity_coverage = 0.0
        activity_matched = 0
        activity_total = 0
        observed_inventory = ()
        expected_inventory = ()
        pdf_bound = False
        pdf_title_once = False
        pdf_instruction_once = False
        pdf_inventory_complete = False
        pdf_bank_exact = False
        pdf_key_components = False
    activity_text_match = (
        activity_coverage >= thresholds.min_activity_coverage
        and activity_matched >= thresholds.min_activity_matched_tokens
    )
    checks = (
        ("strict_public_document_identity", observed_identity == document.identity),
        ("unique_content_page", page_unique),
        ("no_single_source_question_marker", no_single_marker),
        ("unique_page_activity_record", selected is not None),
        ("source_activity_text_match", activity_text_match),
        ("complete_numbered_item_inventory", observed_inventory == expected_inventory),
        ("pdf_activity_title_attested", pdf_title_once),
        ("pdf_activity_instruction_attested", pdf_instruction_once),
        ("pdf_complete_item_inventory_attested", pdf_inventory_complete),
        ("pdf_word_bank_multiset_attested", pdf_bank_exact),
        ("pdf_answer_key_components_attested", pdf_key_components),
        ("pdf_attestation_replayed", pdf_bound),
        ("reviewed_embedded_key", selected is not None),
        ("valid_source_answer", selected is not None and bool(selected.answer)),
        (
            "source_address_not_task_id",
            selected is not None
            and selected.record_id
            == f"{document.document_id}:p{selected.content_page_number}:fill_blank",
        ),
    )
    accepted = all(passed for _name, passed in checks)
    answer = selected.answer if accepted and selected is not None else None
    trace = {
        "schema_version": TRACE_SCHEMA,
        "verifier": VERIFIER,
        "source": {
            "document_id": document.document_id,
            "public_locator": document.identity.public_locator,
            "name": document.identity.name,
            "pdf_sha256": document.pdf_sha256,
            "matched_page_number": page_number,
            "runner_up_page_number": runner_page + 1,
            "record_id": selected.record_id if selected else None,
            "answer_format": "short_text" if selected else None,
            "question_marker_kind": (
                "numberless_page_activity" if selected else None
            ),
            "key_binding_kind": "fill_blank_answer_key" if selected else None,
            "key_page_number": selected.key_page_number if selected else None,
            "key_context_page_number": (
                selected.key_context_page_number if selected else None
            ),
            "content_bbox": list(selected.content_bbox) if selected else None,
            "word_bank_bbox": list(selected.word_bank_bbox) if selected else None,
            "key_bbox": list(selected.key_bbox) if selected else None,
            "expected_item_count": (
                selected.expected_item_count if selected else None
            ),
            "expected_column_count": (
                selected.expected_column_count if selected else None
            ),
            "key_projection_sha256": (
                selected.key_projection_sha256 if selected else None
            ),
            "content_projection_sha256": (
                selected.content_projection_sha256 if selected else None
            ),
            "binding_projection_sha256": (
                selected.binding_projection_sha256 if selected else None
            ),
        },
        "observation": {
            "image_sha256": observation.image_sha256,
            "image_size": [observation.width, observation.height],
            "parser_identity": observation.parser_identity,
            "observed_question_number": observation.question_number,
            "observed_source_marker_kind": marker_kind,
            "observed_source_marker_number": marker_number,
        },
        "match": {
            "page_idf_coverage": page_coverage,
            "page_matched_tokens": page_matched,
            "page_query_tokens": page_total,
            "page_margin": page_margin,
            "question_binding_method": "numberless_full_page_activity_v1",
            "activity_idf_coverage": activity_coverage,
            "activity_matched_tokens": activity_matched,
            "activity_query_tokens": activity_total,
            "observed_item_inventory": list(observed_inventory),
            "expected_item_inventory": list(expected_inventory),
            "pdf_activity_title_attested": pdf_title_once,
            "pdf_activity_instruction_attested": pdf_instruction_once,
            "pdf_complete_item_inventory_attested": pdf_inventory_complete,
            "pdf_word_bank_multiset_attested": pdf_bank_exact,
            "pdf_answer_key_components_attested": pdf_key_components,
            "statement_tokens_sha256": canonical_json_sha256(
                {"tokens": list(observed_tokens)}
            ),
        },
        "thresholds": {
            "min_page_coverage": thresholds.min_page_coverage,
            "min_page_matched_tokens": thresholds.min_page_matched_tokens,
            "min_page_margin": thresholds.min_page_margin,
            "min_activity_coverage": thresholds.min_activity_coverage,
            "min_activity_matched_tokens": thresholds.min_activity_matched_tokens,
        },
        "checks": {name: passed for name, passed in checks},
        "accepted": accepted,
    }
    return MatchResult(
        accepted=accepted,
        answer=answer,
        problem=problem_for(
            observation,
            source_url,
            answer_format="short_text",
        ),
        checks=checks,
        trace=trace,
    )
