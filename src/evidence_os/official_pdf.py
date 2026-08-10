"""Fail-closed lookup in pinned official PDFs with embedded answer keys."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from .official_ogm import (
    MatchResult,
    OcrObservation,
    OfficialSourceError,
    PageMatcher,
    normalize_tokens,
    problem_for,
)


VERIFIER = "official-pdf-ocr-page-key-binding-v2"
_KEY_PAIR = re.compile(r"(?<!\d)(\d{1,2})\s*[.\-:]?\s*([A-E])(?![A-Z])", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DirectPdfThresholds:
    min_page_coverage: float = 0.75
    min_page_matched_tokens: int = 15
    min_page_margin: float = 0.20
    rescue_min_page_coverage: float = 0.40
    rescue_min_page_margin: float = 0.05
    rescue_min_anchor_coverage: float = 0.90
    rescue_min_anchor_matched_tokens: int = 8
    rescue_min_anchor_margin: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "min_page_coverage",
            "min_page_margin",
            "rescue_min_page_coverage",
            "rescue_min_page_margin",
            "rescue_min_anchor_coverage",
            "rescue_min_anchor_margin",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.min_page_matched_tokens < 1 or self.rescue_min_anchor_matched_tokens < 1:
            raise ValueError("direct-PDF token thresholds must be positive")


@dataclass(frozen=True, slots=True)
class KeyEntry:
    answer: str
    key_page_number: int
    bbox: tuple[float, float, float, float]


def _bbox(raw: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        raise OfficialSourceError(f"{label} must be a four-number bbox")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise OfficialSourceError(f"{label} must be numeric") from exc
    if not all(math.isfinite(value) for value in values):
        raise OfficialSourceError(f"{label} must contain only finite coordinates")
    if not (values[0] < values[2] and values[1] < values[3]):
        raise OfficialSourceError(f"{label} is not ordered")
    return values  # type: ignore[return-value]


def _inside(word: Mapping[str, Any], bbox: tuple[float, float, float, float]) -> bool:
    center_x = (float(word["x0"]) + float(word["x1"])) / 2
    center_y = (float(word["top"]) + float(word["bottom"])) / 2
    return bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]


def _group_lines(words: Sequence[Mapping[str, Any]], tolerance: float = 3.0) -> list[list[Mapping[str, Any]]]:
    lines: list[list[Mapping[str, Any]]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        if not lines:
            lines.append([word])
            continue
        mean_top = sum(float(item["top"]) for item in lines[-1]) / len(lines[-1])
        if abs(float(word["top"]) - mean_top) <= tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
    for line in lines:
        line.sort(key=lambda item: float(item["x0"]))
    return lines


def parse_key_regions(
    pdf_path: Path,
    *,
    key_page_number: int,
    regions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, KeyEntry]]:
    """Parse reviewed coordinate regions while still reading answers from PDF bytes."""

    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise OfficialSourceError("direct official-PDF resolver requires pdfplumber") from exc
    with pdfplumber.open(pdf_path) as pdf:
        if not 1 <= key_page_number <= len(pdf.pages):
            raise OfficialSourceError("configured answer-key page is outside the PDF")
        page = pdf.pages[key_page_number - 1]
        words = page.extract_words() or []
        output: dict[str, dict[int, KeyEntry]] = {}
        for raw_region in regions:
            if not isinstance(raw_region, Mapping):
                raise OfficialSourceError("answer-key region is malformed")
            subject = str(raw_region.get("subject") or "").strip()
            heading = str(raw_region.get("heading") or "").strip()
            expected_count = int(raw_region.get("expected_count", 0))
            region_bbox = _bbox(raw_region.get("answers_bbox"), "answers_bbox")
            heading_bbox = _bbox(raw_region.get("heading_bbox"), "heading_bbox")
            if not subject or not heading or expected_count < 1 or subject in output:
                raise OfficialSourceError("answer-key region identity is invalid")
            for label, configured_bbox in (
                ("answers_bbox", region_bbox),
                ("heading_bbox", heading_bbox),
            ):
                if not (
                    0.0 <= configured_bbox[0] < configured_bbox[2] <= float(page.width)
                    and 0.0 <= configured_bbox[1] < configured_bbox[3] <= float(page.height)
                ):
                    raise OfficialSourceError(f"{label} is outside the answer-key page")
            heading_words = [item for item in words if _inside(item, heading_bbox)]
            observed_heading = " ".join(str(item["text"]) for item in sorted(heading_words, key=lambda item: float(item["x0"])))
            expected_heading_tokens = set(normalize_tokens(heading))
            observed_heading_tokens = set(normalize_tokens(observed_heading))
            if not expected_heading_tokens or not expected_heading_tokens <= observed_heading_tokens:
                raise OfficialSourceError(f"answer-key heading mismatch for {subject}")
            answer_words = [item for item in words if _inside(item, region_bbox)]
            entries: dict[int, KeyEntry] = {}
            for line in _group_lines(answer_words):
                text = " ".join(str(item["text"]) for item in line)
                matches = list(_KEY_PAIR.finditer(text))
                if not matches:
                    continue
                if len(matches) != 1:
                    raise OfficialSourceError(f"multiple key pairs in one {subject} line")
                number = int(matches[0].group(1))
                answer = matches[0].group(2).upper()
                if number in entries:
                    raise OfficialSourceError(f"duplicate key number {number} for {subject}")
                entries[number] = KeyEntry(
                    answer=answer,
                    key_page_number=key_page_number,
                    bbox=(
                        min(float(item["x0"]) for item in line),
                        min(float(item["top"]) for item in line),
                        max(float(item["x1"]) for item in line),
                        max(float(item["bottom"]) for item in line),
                    ),
                )
            if set(entries) != set(range(1, expected_count + 1)):
                raise OfficialSourceError(
                    f"answer-key region for {subject} is incomplete: {sorted(entries)}"
                )
            output[subject] = entries
    return output


def _content_pages(adapter: Mapping[str, Any], page_count: int) -> tuple[int, ...]:
    pages: set[int] = set()
    sections = adapter.get("sections")
    if not isinstance(sections, Sequence):
        raise OfficialSourceError("direct-PDF adapter has no sections")
    for section in sections:
        if not isinstance(section, Mapping):
            raise OfficialSourceError("direct-PDF section is malformed")
        start = int(section.get("start_page", 0))
        end = int(section.get("end_page", 0))
        if not 1 <= start <= end <= page_count:
            raise OfficialSourceError("direct-PDF section range is invalid")
        pages.update(range(start - 1, end))
    return tuple(sorted(pages))


def _subject_for_page(adapter: Mapping[str, Any], page_number: int) -> str | None:
    subjects = {
        str(section["subject"])
        for section in adapter["sections"]
        if int(section["start_page"]) <= page_number <= int(section["end_page"])
    }
    return next(iter(subjects)) if len(subjects) == 1 else None


def _question_marker_present(page_text: str, number: int) -> bool:
    return bool(re.search(rf"(?<!\d){number}\s*[.)](?=\s|$)", page_text))


def resolve_direct_pdf_question(
    observation: OcrObservation,
    source_url: str,
    adapter: Mapping[str, Any],
    matcher: PageMatcher,
    page_texts: Sequence[str],
    answer_key: Mapping[str, Mapping[int, KeyEntry]],
    thresholds: DirectPdfThresholds,
) -> MatchResult:
    if source_url != str(adapter.get("source_url") or ""):
        raise OfficialSourceError("source URL does not exactly match a pinned PDF adapter")
    if matcher.page_count != len(page_texts):
        raise OfficialSourceError("page matcher and PDF text projection do not align")
    problem = problem_for(observation, source_url)
    content_pages = _content_pages(adapter, len(page_texts))
    if len(content_pages) < 2:
        raise OfficialSourceError("direct-PDF adapter has too few content pages")
    scores = {page: matcher.score(observation.statement, page) for page in content_pages}
    order = sorted(content_pages, key=lambda page: (-scores[page][0], page))
    best_page, runner_page = order[:2]
    page_score, page_matched, page_total = scores[best_page]
    page_margin = page_score - scores[runner_page][0]

    anchor_candidates: list[tuple[float, float, int, int, str]] = []
    for block in observation.text_blocks:
        if len(set(normalize_tokens(block))) < 6:
            continue
        block_scores = {page: matcher.score(block, page) for page in content_pages}
        block_order = sorted(content_pages, key=lambda page: (-block_scores[page][0], page))
        if block_order[0] != best_page:
            continue
        score, matched, total = block_scores[best_page]
        margin = score - block_scores[block_order[1]][0]
        anchor_candidates.append((score, margin, matched, total, block))
    anchor_candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[4]))
    if anchor_candidates:
        anchor_score, anchor_margin, anchor_matched, anchor_total, anchor_text = anchor_candidates[0]
    else:
        anchor_score, anchor_margin, anchor_matched, anchor_total, anchor_text = (0.0, 0.0, 0, 0, "")

    strong_page = (
        page_score >= thresholds.min_page_coverage
        and page_matched >= thresholds.min_page_matched_tokens
        and page_margin >= thresholds.min_page_margin
    )
    anchor_rescue = (
        page_score >= thresholds.rescue_min_page_coverage
        and page_matched >= thresholds.min_page_matched_tokens
        and page_margin >= thresholds.rescue_min_page_margin
        and anchor_score >= thresholds.rescue_min_anchor_coverage
        and anchor_matched >= thresholds.rescue_min_anchor_matched_tokens
        and anchor_margin >= thresholds.rescue_min_anchor_margin
    )
    page_number = best_page + 1
    subject = _subject_for_page(adapter, page_number)
    key_entry = answer_key.get(subject or "", {}).get(observation.question_number)
    checks = (
        ("exact_pinned_source_url", source_url == adapter.get("source_url")),
        ("unique_content_page", strong_page or anchor_rescue),
        ("matched_page_in_one_subject_section", subject is not None),
        ("visible_question_number_on_matched_page", _question_marker_present(page_texts[best_page], observation.question_number)),
        ("unique_embedded_key_entry", key_entry is not None),
        ("valid_choice_key", key_entry is not None and key_entry.answer in frozenset("ABCDE")),
    )
    accepted = all(value for _, value in checks)
    answer = key_entry.answer if accepted and key_entry is not None else None
    trace = {
        "schema_version": "official-direct-pdf-source-trace-v2",
        "verifier": VERIFIER,
        "source": {
            "url": source_url,
            "document_name": adapter.get("name"),
            "pdf_sha256": adapter.get("pdf_sha256"),
            "matched_page_number": page_number,
            "runner_up_page_number": runner_page + 1,
            "subject": subject,
            "printed_question_number": observation.question_number,
            "key_page_number": key_entry.key_page_number if key_entry else adapter.get("key_page_number"),
            "key_cell_bbox": list(key_entry.bbox) if key_entry else None,
        },
        "observation": {
            "image_sha256": observation.image_sha256,
            "image_size": [observation.width, observation.height],
            "parser_identity": observation.parser_identity,
        },
        "match": {
            "page_idf_coverage": page_score,
            "page_matched_tokens": page_matched,
            "page_query_tokens": page_total,
            "page_margin": page_margin,
            "anchor_idf_coverage": anchor_score,
            "anchor_matched_tokens": anchor_matched,
            "anchor_query_tokens": anchor_total,
            "anchor_margin": anchor_margin,
            "anchor_excerpt": " ".join(anchor_text.split())[:240],
            "acceptance_path": "strong_page" if strong_page else "anchor_rescue" if anchor_rescue else "none",
        },
        "thresholds": {
            "min_page_coverage": thresholds.min_page_coverage,
            "min_page_matched_tokens": thresholds.min_page_matched_tokens,
            "min_page_margin": thresholds.min_page_margin,
            "rescue_min_page_coverage": thresholds.rescue_min_page_coverage,
            "rescue_min_page_margin": thresholds.rescue_min_page_margin,
            "rescue_min_anchor_coverage": thresholds.rescue_min_anchor_coverage,
            "rescue_min_anchor_matched_tokens": thresholds.rescue_min_anchor_matched_tokens,
            "rescue_min_anchor_margin": thresholds.rescue_min_anchor_margin,
        },
        "checks": {name: value for name, value in checks},
        "accepted": accepted,
    }
    return MatchResult(
        accepted=accepted,
        answer=answer,
        problem=problem,
        checks=checks,
        trace=trace,
    )
