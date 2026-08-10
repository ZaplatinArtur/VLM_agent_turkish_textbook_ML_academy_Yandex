#!/usr/bin/env python3
"""Build the complete source-native Bio9/Physics12 MCQ inventory.

The build reads only two pinned official PDFs.  It covers every address in the
public source-side protocol pool: 147 addresses total, of which the official
book proves 143 as A-E questions and four as unsupported open-response items.
No benchmark row, selected split, opaque input, gold file or prediction is an
input to this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_PACKAGES = REPO_ROOT / "tmp" / "portfolio_official_sources" / "python_pkgs"
for candidate in (PINNED_PACKAGES, REPO_ROOT / "src"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evidence_os.mcq_fullpage_source import (  # noqa: E402
    EXPECTED_CHOICE_KEY_COUNT,
    EXPECTED_CONTENT_PAGE_COUNT,
    EXPECTED_KEY_PAGE_COUNT,
    EXPECTED_PROTOCOL_RECORD_COUNT,
    INVENTORY_SCHEMA,
    KEY_INDEX_SCHEMA,
    McqDocument,
    McqInventory,
    McqKeyCell,
    McqKeyIndex,
    McqQuestionRecord,
    McqSourceError,
    assert_mcq_runtime,
    write_canonical_json,
)
from evidence_os.official_ogm import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)


BIOLOGY_SHA256 = "717548090c5bece21242fab41a3dad26aa43031f5a73d4191538903ab3ec4ea0"
PHYSICS_SHA256 = "0957cb2a74ed46d6b7c3a3165863e03b5a7206cdf444f6ad8ecf6a13179a6307"
BIOLOGY_FAMILY = "biology9_textbook"
PHYSICS_FAMILY = "physics12_textbook"
BIOLOGY_KEY_PAGES = (158, 160, 162)
PHYSICS_KEY_PAGES = (263, 264, 265)

BIO_PAGE_MAP = {
    1: {
        **{question: 63 for question in range(1, 5)},
        **{question: 64 for question in range(5, 9)},
        **{question: 65 for question in range(9, 13)},
    },
    2: {
        **{question: 108 for question in range(1, 4)},
        **{question: 109 for question in range(4, 8)},
        **{question: 110 for question in range(8, 13)},
    },
    3: {question: 154 for question in range(1, 7)},
}
PHYSICS_POOLS = {
    1: range(34, 59),
    2: range(24, 39),
    3: range(28, 39),
    4: range(30, 59),
    5: range(21, 41),
    6: range(30, 47),
}
PHYSICS_PHYSICAL_KEY_QUESTIONS = {
    **{unit: frozenset(questions) for unit, questions in PHYSICS_POOLS.items()},
    2: frozenset(range(28, 39)),
    3: frozenset(range(24, 43)),
}
EXPECTED_PHYSICAL_CHOICE_CELLS = 151
PHYSICS_PAGE_MAP = {
    1: {
        **{question: 74 for question in range(34, 38)},
        **{question: 75 for question in range(38, 42)},
        **{question: 76 for question in range(42, 50)},
        **{question: 77 for question in range(50, 55)},
        **{question: 78 for question in range(55, 59)},
    },
    2: {
        **{question: 100 for question in range(24, 28)},
        **{question: 101 for question in range(28, 35)},
        **{question: 102 for question in range(35, 39)},
    },
    3: {
        **{question: 138 for question in range(28, 31)},
        **{question: 139 for question in range(31, 38)},
        38: 140,
    },
    4: {
        **{question: 180 for question in range(30, 33)},
        **{question: 181 for question in range(33, 37)},
        **{question: 182 for question in range(37, 47)},
        **{question: 183 for question in range(47, 54)},
        **{question: 184 for question in range(54, 59)},
    },
    5: {
        **{question: 224 for question in range(21, 27)},
        **{question: 225 for question in range(27, 34)},
        **{question: 226 for question in range(34, 41)},
    },
    6: {
        **{question: 253 for question in range(30, 35)},
        **{question: 254 for question in range(35, 47)},
    },
}
UNSUPPORTED_OPEN_RESPONSE = frozenset(
    (PHYSICS_FAMILY, 2, question) for question in range(24, 28)
)

# Source-native layout bands for printed question markers.  They cover the
# two Physics body columns and the single Biology body column, while excluding
# page headers, diagrams and off-canvas duplicate text objects.
CONTENT_MARKER_X0_BANDS = {
    BIOLOGY_FAMILY: ((45.0, 80.0),),
    PHYSICS_FAMILY: ((35.0, 80.0), (265.0, 310.0)),
}

# Reviewed physical-PDF regions containing complete official A-E grids.  These
# are source geometry, not selected-task addresses.  Every supported question
# from the full 143-record census is parsed from one of these regions.
PHYSICS_KEY_REGIONS = {
    1: (263, (288.0, 408.0, 512.0, 526.0)),
    2: (264, (40.0, 470.0, 265.0, 502.0)),
    3: (264, (274.0, 132.0, 498.0, 210.0)),
    4: (264, (274.0, 387.0, 498.0, 504.0)),
    5: (265, (54.0, 290.0, 278.0, 367.0)),
    6: (265, (294.0, 613.0, 512.0, 678.0)),
}
BIO_KEY_REGIONS = {
    1: (158, (124.0, 156.0, 434.0, 175.0)),
    2: (160, (138.0, 140.0, 435.0, 159.0)),
    3: (162, (121.0, 106.0, 268.0, 125.0)),
}
BIO_HEADING_REGIONS = {
    1: (158, (88.77, 47.98, 431.86, 119.96)),
    2: (160, (88.77, 47.98, 431.86, 105.56)),
    3: (162, (91.17, 47.98, 441.46, 93.57)),
}
PHYSICS_HEADING_REGIONS = {
    1: (263, (55.18, 62.84, 277.82, 85.87)),
    2: (263, (288.85, 525.76, 511.49, 549.27)),
    3: (264, (40.79, 515.69, 263.42, 539.19)),
    4: (264, (273.50, 221.63, 497.10, 244.65)),
    5: (264, (273.50, 507.05, 497.10, 530.08)),
    6: (265, (54.70, 382.33, 277.82, 405.83)),
}
PHYSICS_GLOBAL_HEADING_REGION = (62.38, 38.38, 491.82, 59.96)
_PLAIN_NUMBER = re.compile(r"^([1-9][0-9]{0,2})$")
_DOTTED_NUMBER = re.compile(r"^([1-9][0-9]{0,2})[.)]$")
_INLINE_PAIR = re.compile(r"^([1-9][0-9]{0,2})[.)]([A-E])$")
_CHOICE = re.compile(r"^[A-E]$")


class SourceBuildError(RuntimeError):
    pass


def _normalize_pdf_text(value: str | None) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).upper()
    return " ".join(text.replace("CEV AP", "CEVAP").split())


def _verify_source_layout_and_headings(
    biology_pdf: Any, physics_pdf: Any
) -> str:
    expected_layouts = (
        (
            biology_pdf,
            173,
            612.283,
            858.898,
            "Biology9",
            {
                *BIOLOGY_KEY_PAGES,
                *(page for pages in BIO_PAGE_MAP.values() for page in pages.values()),
            },
        ),
        (
            physics_pdf,
            275,
            552.756,
            779.528,
            "Physics12",
            {
                *PHYSICS_KEY_PAGES,
                *(page for pages in PHYSICS_PAGE_MAP.values() for page in pages.values()),
            },
        ),
    )
    layout_projection: list[dict[str, Any]] = []
    for pdf, page_count, width, height, label, checked_pages in expected_layouts:
        if len(pdf.pages) != page_count:
            raise SourceBuildError(f"{label} page count changed")
        for page_number in sorted(checked_pages):
            page = pdf.pages[page_number - 1]
            if (
                not math.isclose(float(page.width), width, abs_tol=0.001)
                or not math.isclose(float(page.height), height, abs_tol=0.001)
                or int(getattr(page, "rotation", 0) or 0) != 0
            ):
                raise SourceBuildError(
                    f"{label} page {page_number} MediaBox/rotation changed"
                )
        layout_projection.append(
            {
                "label": label,
                "page_count": page_count,
                "width": width,
                "height": height,
                "rotation": 0,
                "checked_physical_pages": sorted(checked_pages),
            }
        )

    heading_projection: list[dict[str, Any]] = []
    for unit, (page_number, bbox) in BIO_HEADING_REGIONS.items():
        observed = _normalize_pdf_text(
            biology_pdf.pages[page_number - 1].crop(bbox).extract_text()
        )
        expected = (
            f"{unit}. ÜNİTE ÖLÇME VE DEĞERLENDİRME SORULARI CEVAP ANAHTARI"
        )
        if expected not in observed:
            raise SourceBuildError(f"Biology unit {unit} key heading changed")
        heading_projection.append(
            {
                "document": BIOLOGY_FAMILY,
                "unit": unit,
                "page_number": page_number,
                "bbox": list(bbox),
                "normalized_text": observed,
            }
        )
    global_expected = "ÖLÇME VE DEĞERLENDİRME CEVAP ANAHTARLARI"
    for page_number in PHYSICS_KEY_PAGES:
        observed = _normalize_pdf_text(
            physics_pdf.pages[page_number - 1]
            .crop(PHYSICS_GLOBAL_HEADING_REGION)
            .extract_text()
        )
        if observed != global_expected:
            raise SourceBuildError(f"Physics page {page_number} global heading changed")
        heading_projection.append(
            {
                "document": PHYSICS_FAMILY,
                "kind": "global",
                "page_number": page_number,
                "bbox": list(PHYSICS_GLOBAL_HEADING_REGION),
                "normalized_text": observed,
            }
        )
    for unit, (page_number, bbox) in PHYSICS_HEADING_REGIONS.items():
        observed = _normalize_pdf_text(
            physics_pdf.pages[page_number - 1].crop(bbox).extract_text()
        )
        if observed != f"{unit}. ÜNİTE":
            raise SourceBuildError(f"Physics unit {unit} key heading changed")
        heading_projection.append(
            {
                "document": PHYSICS_FAMILY,
                "unit": unit,
                "page_number": page_number,
                "bbox": list(bbox),
                "normalized_text": observed,
            }
        )
    return canonical_json_sha256(
        {
            "schema_version": "mcq-source-layout-heading-projection-v1",
            "layouts": layout_projection,
            "headings": heading_projection,
        }
    )


def _rounded(value: Any) -> float:
    return round(float(value), 4)


def _word_projection(word: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": str(word.get("text") or ""),
        "x0": _rounded(word["x0"]),
        "top": _rounded(word["top"]),
        "x1": _rounded(word["x1"]),
        "bottom": _rounded(word["bottom"]),
    }


def _bbox_union(*words: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        _rounded(min(float(item["x0"]) for item in words)),
        _rounded(min(float(item["top"]) for item in words)),
        _rounded(max(float(item["x1"]) for item in words)),
        _rounded(max(float(item["bottom"]) for item in words)),
    )


def _inside(word: Mapping[str, Any], bbox: Sequence[float]) -> bool:
    center_x = (float(word["x0"]) + float(word["x1"])) / 2.0
    center_y = (float(word["top"]) + float(word["bottom"])) / 2.0
    return bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]


def _record_id(document_id: str, unit: int, page: int, question: int) -> str:
    return f"{document_id}:u{unit}:p{page}:q{question}"


def _find_content_marker(
    page: Any,
    *,
    document_id: str,
    source_family: str,
    pdf_sha256: str,
    unit: int,
    page_number: int,
    question: int,
) -> tuple[str, tuple[float, float, float, float], str]:
    bands = CONTENT_MARKER_X0_BANDS.get(source_family)
    if bands is None:
        raise SourceBuildError("source family has no frozen marker layout")
    matches = [
        word
        for word in (page.extract_words() or [])
        if _DOTTED_NUMBER.fullmatch(str(word.get("text") or "")) is not None
        and int(_DOTTED_NUMBER.fullmatch(str(word["text"])).group(1)) == question
        and any(low <= float(word["x0"]) <= high for low, high in bands)
        and 0.0 <= float(word["x0"]) < float(word["x1"]) <= float(page.width)
        and 0.0 <= float(word["top"]) < float(word["bottom"]) <= float(page.height)
    ]
    if len(matches) != 1:
        raise SourceBuildError(
            f"source page {page_number} has {len(matches)} exact markers for q{question}"
        )
    marker = matches[0]
    marker_text = str(marker["text"])
    bbox = _bbox_union(marker)
    projection = {
        "schema_version": "mcq-content-marker-projection-v1",
        "document_id": document_id,
        "source_family": source_family,
        "pdf_sha256": pdf_sha256,
        "unit_number": unit,
        "content_page_number": page_number,
        "question_number": question,
        "marker": _word_projection(marker),
    }
    return marker_text, bbox, canonical_json_sha256(projection)


def _parse_biology_inline_pairs(
    page: Any,
    *,
    document_id: str,
    pdf_sha256: str,
    unit: int,
    page_number: int,
    region: Sequence[float],
    expected_questions: set[int],
    record_ids: Mapping[int, str],
) -> dict[int, McqKeyCell]:
    words = [word for word in (page.extract_words() or []) if _inside(word, region)]
    words.sort(key=lambda item: (float(item["top"]), float(item["x0"])))
    pairs: dict[int, tuple[str, Mapping[str, Any], Mapping[str, Any]]] = {}
    index = 0
    while index < len(words):
        token = str(words[index].get("text") or "").upper()
        combined = _INLINE_PAIR.fullmatch(token)
        if combined is not None:
            number = int(combined.group(1))
            pairs[number] = (combined.group(2), words[index], words[index])
            index += 1
            continue
        number_match = _DOTTED_NUMBER.fullmatch(token)
        if (
            number_match is not None
            and index + 1 < len(words)
            and _CHOICE.fullmatch(str(words[index + 1].get("text") or "").upper())
            is not None
            and abs(float(words[index]["top"]) - float(words[index + 1]["top"]))
            <= 2.0
        ):
            number = int(number_match.group(1))
            pairs[number] = (
                str(words[index + 1]["text"]).upper(),
                words[index],
                words[index + 1],
            )
            index += 2
            continue
        index += 1
    if set(pairs) != expected_questions:
        raise SourceBuildError(
            f"Biology unit {unit} key coverage changed: {sorted(pairs)}"
        )
    return {
        question: _key_cell(
            record_id=record_ids[question],
            document_id=document_id,
            pdf_sha256=pdf_sha256,
            unit=unit,
            question=question,
            page_number=page_number,
            answer=pairs[question][0],
            number_word=pairs[question][1],
            answer_word=pairs[question][2],
        )
        for question in sorted(pairs)
    }


def _group_word_lines(
    words: Sequence[Mapping[str, Any]], tolerance: float = 2.0
) -> list[list[Mapping[str, Any]]]:
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


def _parse_physics_grid_pairs(
    page: Any,
    *,
    document_id: str,
    pdf_sha256: str,
    unit: int,
    page_number: int,
    region: Sequence[float],
    expected_questions: set[int],
    record_ids: Mapping[int, str],
) -> dict[int, McqKeyCell]:
    if not set(record_ids).issubset(expected_questions):
        raise SourceBuildError("exported Physics keys exceed physical table scope")
    words = [word for word in (page.extract_words() or []) if _inside(word, region)]
    lines = _group_word_lines(words)
    pairs: dict[int, tuple[str, Mapping[str, Any], Mapping[str, Any]]] = {}
    for line_index, number_line in enumerate(lines[:-1]):
        number_words = [
            item
            for item in number_line
            if _PLAIN_NUMBER.fullmatch(str(item.get("text") or "")) is not None
        ]
        if not number_words:
            continue
        answer_line = lines[line_index + 1]
        answer_words = [
            item
            for item in answer_line
            if _CHOICE.fullmatch(str(item.get("text") or "").upper()) is not None
        ]
        if len(answer_words) != len(number_words):
            continue
        for number_word, answer_word in zip(number_words, answer_words):
            number = int(str(number_word["text"]))
            if number not in expected_questions:
                continue
            if number in pairs:
                raise SourceBuildError(
                    f"Physics unit {unit} key number {number} is duplicated"
                )
            if abs(
                (float(number_word["x0"]) + float(number_word["x1"])) / 2.0
                - (float(answer_word["x0"]) + float(answer_word["x1"])) / 2.0
            ) > 6.0:
                raise SourceBuildError(
                    f"Physics unit {unit} key column drifted at q{number}"
                )
            pairs[number] = (
                str(answer_word["text"]).upper(),
                number_word,
                answer_word,
            )
    if set(pairs) != expected_questions:
        raise SourceBuildError(
            f"Physics unit {unit} key coverage changed: "
            f"expected {sorted(expected_questions)}, got {sorted(pairs)}"
        )
    return {
        question: _key_cell(
            record_id=record_ids[question],
            document_id=document_id,
            pdf_sha256=pdf_sha256,
            unit=unit,
            question=question,
            page_number=page_number,
            answer=pairs[question][0],
            number_word=pairs[question][1],
            answer_word=pairs[question][2],
        )
        for question in sorted(record_ids)
    }


def _key_cell(
    *,
    record_id: str,
    document_id: str,
    pdf_sha256: str,
    unit: int,
    question: int,
    page_number: int,
    answer: str,
    number_word: Mapping[str, Any],
    answer_word: Mapping[str, Any],
) -> McqKeyCell:
    bbox = _bbox_union(number_word, answer_word)
    key_text = f"{question} {answer}"
    projection = {
        "schema_version": "mcq-official-key-cell-projection-v1",
        "document_id": document_id,
        "pdf_sha256": pdf_sha256,
        "unit_number": unit,
        "question_number": question,
        "answer": answer,
        "key_page_number": page_number,
        "key_bbox": list(bbox),
        "number_word": _word_projection(number_word),
        "answer_word": _word_projection(answer_word),
    }
    return McqKeyCell(
        record_id=record_id,
        document_id=document_id,
        unit_number=unit,
        question_number=question,
        answer=answer,
        key_page_number=page_number,
        key_bbox=bbox,
        key_text=key_text,
        key_text_sha256=hashlib.sha256(key_text.encode("utf-8")).hexdigest(),
        key_projection_sha256=canonical_json_sha256(projection),
    )


def _document_records(
    pdf: Any,
    *,
    document_id: str,
    source_family: str,
    pdf_sha256: str,
    page_map: Mapping[int, Mapping[int, int]],
    pools: Mapping[int, Iterable[int]] | None = None,
) -> tuple[McqQuestionRecord, ...]:
    records: list[McqQuestionRecord] = []
    for unit in sorted(page_map):
        questions = (
            sorted(pools[unit]) if pools is not None else sorted(page_map[unit])
        )
        for question in questions:
            page_number = int(page_map[unit][question])
            marker, bbox, marker_sha = _find_content_marker(
                pdf.pages[page_number - 1],
                document_id=document_id,
                source_family=source_family,
                pdf_sha256=pdf_sha256,
                unit=unit,
                page_number=page_number,
                question=question,
            )
            records.append(
                McqQuestionRecord(
                    record_id=_record_id(document_id, unit, page_number, question),
                    document_id=document_id,
                    source_family=source_family,
                    unit_number=unit,
                    content_page_number=page_number,
                    question_number=question,
                    source_response_kind=(
                        "unsupported_open_response"
                        if (source_family, unit, question) in UNSUPPORTED_OPEN_RESPONSE
                        else "choice_A-E"
                    ),
                    content_marker=marker,
                    content_marker_bbox=bbox,
                    content_marker_projection_sha256=marker_sha,
                )
            )
    return tuple(
        sorted(
            records,
            key=lambda item: (
                item.document_id,
                item.unit_number,
                item.content_page_number,
                item.question_number,
            ),
        )
    )


def _key_cells(
    biology_pdf: Any,
    physics_pdf: Any,
    *,
    biology_document_id: str,
    physics_document_id: str,
    record_by_source: Mapping[tuple[str, int, int], McqQuestionRecord],
) -> tuple[McqKeyCell, ...]:
    cells: list[McqKeyCell] = []
    for unit, (page_number, region) in BIO_KEY_REGIONS.items():
        parsed = _parse_biology_inline_pairs(
            biology_pdf.pages[page_number - 1],
            document_id=biology_document_id,
            pdf_sha256=BIOLOGY_SHA256,
            unit=unit,
            page_number=page_number,
            region=region,
            expected_questions=set(BIO_PAGE_MAP[unit]),
            record_ids={
                question: record_by_source[
                    (BIOLOGY_FAMILY, unit, question)
                ].record_id
                for question in BIO_PAGE_MAP[unit]
            },
        )
        cells.extend(parsed.values())
    for unit, (page_number, region) in PHYSICS_KEY_REGIONS.items():
        supported = {
            question
            for question in PHYSICS_POOLS[unit]
            if (PHYSICS_FAMILY, unit, question) not in UNSUPPORTED_OPEN_RESPONSE
        }
        parsed = _parse_physics_grid_pairs(
            physics_pdf.pages[page_number - 1],
            document_id=physics_document_id,
            pdf_sha256=PHYSICS_SHA256,
            unit=unit,
            page_number=page_number,
            region=region,
            expected_questions=set(PHYSICS_PHYSICAL_KEY_QUESTIONS[unit]),
            record_ids={
                question: record_by_source[
                    (PHYSICS_FAMILY, unit, question)
                ].record_id
                for question in supported
            },
        )
        cells.extend(parsed.values())
    return tuple(sorted(cells, key=lambda item: item.record_id))


def build_source(
    biology_path: Path,
    physics_path: Path,
) -> tuple[McqInventory, McqKeyIndex, dict[str, Any]]:
    assert_mcq_runtime(require_pdfplumber=True)
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SourceBuildError("pdfplumber is unavailable") from exc
    biology_path = biology_path.resolve()
    physics_path = physics_path.resolve()
    for path, expected, label in (
        (biology_path, BIOLOGY_SHA256, "Biology9"),
        (physics_path, PHYSICS_SHA256, "Physics12"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise SourceBuildError(f"{label} PDF differs from its frozen SHA-256")
    biology_document_id = f"meb_biology9_{BIOLOGY_SHA256[:12]}"
    physics_document_id = f"meb_physics12_{PHYSICS_SHA256[:12]}"
    with pdfplumber.open(biology_path) as biology_pdf, pdfplumber.open(
        physics_path
    ) as physics_pdf:
        source_layout_heading_projection_sha256 = (
            _verify_source_layout_and_headings(biology_pdf, physics_pdf)
        )
        biology_records = _document_records(
            biology_pdf,
            document_id=biology_document_id,
            source_family=BIOLOGY_FAMILY,
            pdf_sha256=BIOLOGY_SHA256,
            page_map=BIO_PAGE_MAP,
        )
        physics_records = _document_records(
            physics_pdf,
            document_id=physics_document_id,
            source_family=PHYSICS_FAMILY,
            pdf_sha256=PHYSICS_SHA256,
            page_map=PHYSICS_PAGE_MAP,
            pools=PHYSICS_POOLS,
        )
        documents = (
            McqDocument(
                document_id=biology_document_id,
                source_family=BIOLOGY_FAMILY,
                pdf_sha256=BIOLOGY_SHA256,
                pdf_size_bytes=biology_path.stat().st_size,
                page_count=len(biology_pdf.pages),
                questions=biology_records,
                key_pages=BIOLOGY_KEY_PAGES,
            ),
            McqDocument(
                document_id=physics_document_id,
                source_family=PHYSICS_FAMILY,
                pdf_sha256=PHYSICS_SHA256,
                pdf_size_bytes=physics_path.stat().st_size,
                page_count=len(physics_pdf.pages),
                questions=physics_records,
                key_pages=PHYSICS_KEY_PAGES,
            ),
        )
        inventory_projection = {
            "schema_version": INVENTORY_SCHEMA,
            "documents": [item.to_mapping() for item in documents],
        }
        inventory = McqInventory(
            documents=documents,
            inventory_projection_sha256=canonical_json_sha256(inventory_projection),
        )
        record_by_source = {
            (item.source_family, item.unit_number, item.question_number): item
            for item in inventory.questions
        }
        cells = _key_cells(
            biology_pdf,
            physics_pdf,
            biology_document_id=biology_document_id,
            physics_document_id=physics_document_id,
            record_by_source=record_by_source,
        )
    key_projection = {
        "schema_version": KEY_INDEX_SCHEMA,
        "inventory_projection_sha256": inventory.inventory_projection_sha256,
        "cells": [item.to_mapping() for item in cells],
    }
    key_index = McqKeyIndex(
        inventory_projection_sha256=inventory.inventory_projection_sha256,
        cells=cells,
        key_index_projection_sha256=canonical_json_sha256(key_projection),
    )
    summary = {
        "schema_version": "mcq-fullpage-source-build-audit-v1",
        "source_only": True,
        "selection_or_gold_access": False,
        "protocol_addresses": len(inventory.questions),
        "official_choice_records": len(key_index.cells),
        "physical_choice_key_cells_verified": (
            30 + sum(len(items) for items in PHYSICS_PHYSICAL_KEY_QUESTIONS.values())
        ),
        "unsupported_open_response_records": sum(
            item.source_response_kind == "unsupported_open_response"
            for item in inventory.questions
        ),
        "candidate_content_pages": len(inventory.candidate_pages),
        "official_key_pages": len(inventory.key_page_addresses),
        "inventory_projection_sha256": inventory.inventory_projection_sha256,
        "key_index_projection_sha256": key_index.key_index_projection_sha256,
        "source_layout_heading_projection_sha256": (
            source_layout_heading_projection_sha256
        ),
        "protocol_defect": {
            "kind": "four_physics_unit2_open_response_items_were_in_public_mcq_pool",
            "source_addresses": [
                {"source_family": PHYSICS_FAMILY, "unit": 2, "question": question}
                for question in range(24, 28)
            ],
            "resolver_policy": "unsupported_open_response_fail_closed",
            "selected_holdout_impact": "not_inspected_before_resolver_freeze",
        },
        "out_of_protocol_official_key_cells": {
            "count": 8,
            "source_family": PHYSICS_FAMILY,
            "unit": 3,
            "questions": [24, 25, 26, 27, 39, 40, 41, 42],
            "policy": "verified_as_table_integrity_controls_but_not_exported",
        },
    }
    if summary != {
        **summary,
        "protocol_addresses": EXPECTED_PROTOCOL_RECORD_COUNT,
        "official_choice_records": EXPECTED_CHOICE_KEY_COUNT,
        "physical_choice_key_cells_verified": EXPECTED_PHYSICAL_CHOICE_CELLS,
        "unsupported_open_response_records": 4,
        "candidate_content_pages": EXPECTED_CONTENT_PAGE_COUNT,
        "official_key_pages": EXPECTED_KEY_PAGE_COUNT,
    }:
        raise SourceBuildError("source census differs from the frozen bounded scope")
    return inventory, key_index, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biology-pdf", type=Path, required=True)
    parser.add_argument("--physics-pdf", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--key-index", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    inventory, key_index, audit = build_source(
        args.biology_pdf, args.physics_pdf
    )
    write_canonical_json(args.inventory, inventory.to_mapping())
    write_canonical_json(args.key_index, key_index.to_mapping())
    write_canonical_json(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
