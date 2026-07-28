from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from schemas.retrieve import RetrievedChunk


class UnitKind(str, Enum):
    THEORY = "theory"
    WORKED_EXAMPLE = "worked_example"
    EXERCISE = "exercise"
    SOLUTION = "solution"
    ANSWER_KEY = "answer_key"
    INSTRUCTION = "instruction"
    OTHER = "other"


class LayoutBlock(BaseModel):
    """A reading-order block. ``bbox`` is normalized to page pixels when known."""

    text: str
    order: int
    line_start: int
    line_end: int
    bbox: tuple[int, int, int, int] | None = None


class EducationalUnit(BaseModel):
    unit_id: str
    parent_chunk_id: str
    kind: UnitKind
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    block_start: int
    block_end: int
    task_number: str | None = None
    section_title: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


_NUMBERED_START_RE = re.compile(r"^\s*(?P<number>\d{1,3})\s*[.)]\s*(?P<body>\S.*)$")
_LETTER_OPTION_RE = re.compile(r"^\s*[\(\[]?[A-Ea-e][\)\].:-]\s*\S")
_ROMAN_OPTION_RE = re.compile(r"^\s*(?:I|II|III|IV|V)\s*[.)-]\s*\S", re.IGNORECASE)
_SHORT_HEADING_RE = re.compile(r"^[A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜ0-9\s:.,'’/\-]{2,80}$")
_INLINE_NUMBERED_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])\s+(?=\d{1,3}[.)]\s+\S)"
)
_ISOLATED_PAGE_NUMBER_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_REFERENCE_ENTRY_RE = re.compile(
    r"^\s*\d{1,3}[.)]\s+.{0,100}\(\d{4}\)[.):,]",
    re.DOTALL,
)

_TASK_MARKERS = (
    "soru",
    "sorular",
    "etkinlik",
    "arastiralim",
    "uygulayalim",
    "degerlendirelim",
    "kendimizi degerlendirelim",
    "calisma",
    "problem",
    "proje",
    "performans gorevi",
    "unite degerlendirmesi",
    "bolum degerlendirmesi",
    "bolum sonu",
    "tema sonu",
    "activity",
    "exercise",
    "questions",
    "question",
    "task",
    "practice",
)
_SOLUTION_MARKERS = (
    "cozum",
    "cozumu",
    "yanit",
    "aciklama",
)
_ANSWER_KEY_MARKERS = (
    "cevap anahtari",
    "yanit anahtari",
)
_EXAMPLE_MARKERS = (
    "ornek",
    "cozumlu ornek",
)
_INSTRUCTION_MARKERS = (
    "yonerge",
    "aciklama",
    "talimat",
    "hazirlanalim",
)
_HEADING_MARKERS = (
    "unite",
    "bolum",
    "tema",
    "konu",
    "ogrenme alani",
)
_QUESTION_SIGNALS = (
    "kac",
    "hangisi",
    "nedir",
    "neden",
    "nasil",
    "ne kadar",
    "bulunuz",
    "bulalim",
    "hesaplayiniz",
    "hesaplayalim",
    "aciklayiniz",
    "yaziniz",
    "isaretleyiniz",
    "eslestiriniz",
    "seciniz",
    "cevaplayiniz",
    "belirleyiniz",
    "karsilastiriniz",
    "tartisiniz",
    "arastiriniz",
    "tamamlayiniz",
    "olusturunuz",
    "gosteriniz",
    "inceleyiniz",
    "inceleyelim",
    "yapiniz",
    "yapalim",
    "okuyunuz",
    "dinleyiniz",
    "doldurunuz",
    "ciziniz",
    "paylasiniz",
    "sununuz",
    "imagine",
    "write",
    "match",
    "complete",
    "answer",
    "calculate",
    "choose",
    "select",
    "find",
    "explain",
    "discuss",
    "compare",
    "fill in",
    "read and",
    "listen and",
)
_STRONG_TASK_SIGNALS = (
    "ask and answer",
    "work in pairs",
    "work in groups",
    "point, ask",
    "say and write",
    "look and",
    "read and",
    "listen and",
    "write",
    "match",
    "complete",
    "answer",
    "calculate",
    "choose",
    "select",
    "fill in",
    "uygun sekilde ciziniz",
    "islemi yaziniz",
    "sorulari cevaplayiniz",
    "cevaplarini yaziniz",
    "defterinize yaziniz",
    "soyleyiniz",
    "goturunuz",
    "isaretleyiniz",
    "eslestiriniz",
    "tamamlayiniz",
    "olusturunuz",
    "gosteriniz",
    "doldurunuz",
    "ciziniz",
    "paylasiniz",
    "sununuz",
)
_BOILERPLATE_MARKERS = (
    "her hakki saklidir",
    "isbn",
    "yayin haklari",
    "milli egitim bakanligi",
    "icindekiler",
    "kaynakca",
)


def _plain(value: str) -> str:
    folded = value.casefold().translate(
        str.maketrans(
            {
                "ı": "i",
                "ğ": "g",
                "ü": "u",
                "ş": "s",
                "ö": "o",
                "ç": "c",
            }
        )
    )
    normalized = unicodedata.normalize("NFKD", folded)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _clean_text(value: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _starts_with_marker(plain: str, markers: Iterable[str]) -> bool:
    stripped = plain.lstrip("0123456789.):- \t")
    for marker in markers:
        if not stripped.startswith(marker):
            continue
        if len(stripped) == len(marker) or not stripped[len(marker)].isalpha():
            return True
    return False


def _is_strong_boundary(line: str) -> bool:
    plain = _plain(line).strip()
    if not plain:
        return False
    if _NUMBERED_START_RE.match(line):
        return True
    if _LETTER_OPTION_RE.match(line) or _ROMAN_OPTION_RE.match(line):
        return False
    all_markers = (
        _TASK_MARKERS
        + _SOLUTION_MARKERS
        + _ANSWER_KEY_MARKERS
        + _EXAMPLE_MARKERS
        + _INSTRUCTION_MARKERS
        + _HEADING_MARKERS
    )
    return _starts_with_marker(plain, all_markers) or bool(_SHORT_HEADING_RE.match(line.strip()))


def split_ocr_blocks(text: str) -> list[LayoutBlock]:
    """Turn flat OCR into stable reading-order blocks without losing line provenance."""

    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[LayoutBlock] = []
    current: list[str] = []
    current_start = 0

    def flush(end_line: int) -> None:
        nonlocal current
        cleaned = _clean_text("\n".join(current))
        if cleaned:
            blocks.append(
                LayoutBlock(
                    text=cleaned,
                    order=len(blocks),
                    line_start=current_start,
                    line_end=end_line,
                )
            )
        current = []

    for line_number, raw_line in enumerate(raw_lines):
        normalized_line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not normalized_line:
            if current:
                flush(line_number - 1)
            continue
        segments = _INLINE_NUMBERED_BOUNDARY_RE.split(normalized_line)
        for line in segments:
            if current and _is_strong_boundary(line):
                flush(line_number - 1)
            if not current:
                current_start = line_number
            current.append(line)
    if current:
        flush(max(0, len(raw_lines) - 1))
    return blocks


class _BlockLabel(BaseModel):
    kind: UnitKind
    confidence: float
    task_number: str | None = None
    heading: bool = False
    option: bool = False


def _classify_block(
    block: LayoutBlock,
    *,
    next_block: LayoutBlock | None = None,
) -> _BlockLabel:
    text = block.text.strip()
    plain = _plain(text)
    first_line = text.splitlines()[0]
    first_plain = _plain(first_line)
    numbered = _NUMBERED_START_RE.match(first_line)
    task_number = numbered.group("number") if numbered else None
    body_plain = _plain(numbered.group("body")) if numbered else first_plain
    option = bool(_LETTER_OPTION_RE.match(first_line) or _ROMAN_OPTION_RE.match(first_line))
    question_signal = any(signal in plain for signal in _QUESTION_SIGNALS)
    has_question_mark = "?" in text
    next_plain = _plain(next_block.text) if next_block is not None else ""
    has_nearby_options = bool(
        next_block
        and (
            _LETTER_OPTION_RE.match(next_block.text.splitlines()[0])
            or _ROMAN_OPTION_RE.match(next_block.text.splitlines()[0])
        )
    )

    if _ISOLATED_PAGE_NUMBER_RE.fullmatch(text):
        return _BlockLabel(kind=UnitKind.OTHER, confidence=0.99)
    if _REFERENCE_ENTRY_RE.match(text) or (
        len(text) >= 120
        and text.count("(") >= 1
        and bool(re.search(r"\(\d{4}\)", text))
        and any(marker in plain for marker in ("dergisi", "yayinlari", "universitesi"))
    ):
        return _BlockLabel(kind=UnitKind.OTHER, confidence=0.93)
    if option:
        return _BlockLabel(kind=UnitKind.OTHER, confidence=0.99, option=True)
    if _starts_with_marker(first_plain, _ANSWER_KEY_MARKERS):
        return _BlockLabel(kind=UnitKind.ANSWER_KEY, confidence=0.99)
    if _starts_with_marker(first_plain, _SOLUTION_MARKERS) or (
        numbered and body_plain.startswith("adim")
    ):
        return _BlockLabel(kind=UnitKind.SOLUTION, confidence=0.94, task_number=task_number)
    if _starts_with_marker(first_plain, _EXAMPLE_MARKERS):
        kind = UnitKind.WORKED_EXAMPLE
        confidence = 0.94
        if "cozum" in first_plain:
            confidence = 0.98
        return _BlockLabel(kind=kind, confidence=confidence, task_number=task_number)
    if _starts_with_marker(first_plain, _TASK_MARKERS):
        return _BlockLabel(kind=UnitKind.EXERCISE, confidence=0.96, task_number=task_number)
    if _starts_with_marker(first_plain, _INSTRUCTION_MARKERS):
        return _BlockLabel(kind=UnitKind.INSTRUCTION, confidence=0.88)

    heading = bool(
        _starts_with_marker(first_plain, _HEADING_MARKERS)
        or (
            len(first_line) <= 90
            and _SHORT_HEADING_RE.match(first_line)
            and not has_question_mark
        )
    )
    if heading:
        return _BlockLabel(kind=UnitKind.THEORY, confidence=0.91, heading=True)

    if numbered and (question_signal or has_question_mark or has_nearby_options):
        return _BlockLabel(
            kind=UnitKind.EXERCISE,
            confidence=0.93 if question_signal or has_question_mark else 0.82,
            task_number=task_number,
        )
    if has_question_mark and (question_signal or has_nearby_options):
        return _BlockLabel(kind=UnitKind.EXERCISE, confidence=0.88)
    if any(signal in plain for signal in _STRONG_TASK_SIGNALS) and len(text) <= 2200:
        return _BlockLabel(
            kind=UnitKind.EXERCISE,
            confidence=0.87,
            task_number=task_number,
        )
    if question_signal and (
        len(text) >= 20
        and (
            first_plain.startswith(("yukaridaki", "asagidaki"))
            or "isten" in plain
            or "verilen" in plain
            or first_plain.startswith(
                (
                    "imagine",
                    "write",
                    "match",
                    "complete",
                    "answer",
                    "calculate",
                    "choose",
                    "select",
                    "find",
                    "explain",
                    "discuss",
                    "compare",
                    "read",
                    "listen",
                )
            )
        )
    ):
        return _BlockLabel(kind=UnitKind.EXERCISE, confidence=0.78, task_number=task_number)

    if any(marker in plain for marker in _BOILERPLATE_MARKERS) and len(text) < 1200:
        return _BlockLabel(kind=UnitKind.OTHER, confidence=0.82)
    if "ornek" in first_plain and len(first_line) < 120:
        return _BlockLabel(kind=UnitKind.WORKED_EXAMPLE, confidence=0.72)
    if "cozum" in next_plain and (has_question_mark or question_signal):
        return _BlockLabel(kind=UnitKind.EXERCISE, confidence=0.76, task_number=task_number)
    ambiguous = bool(
        numbered
        or has_question_mark
        or question_signal
        or first_plain.startswith(("yukaridaki", "asagidaki"))
    )
    return _BlockLabel(
        kind=UnitKind.THEORY,
        confidence=0.56 if ambiguous else 0.84,
    )


def _bbox_union(blocks: Sequence[LayoutBlock]) -> tuple[int, int, int, int] | None:
    boxes = [block.bbox for block in blocks if block.bbox is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _stable_unit_id(parent_chunk_id: str, kind: UnitKind, index: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{parent_chunk_id}\0{kind.value}\0{index}\0{text}".encode("utf-8")
    ).hexdigest()
    return f"edu_{digest[:24]}"


class EducationalChunker:
    """Hybrid deterministic chunker with optional layout blocks.

    It combines textual anchors, local context, page hierarchy and geometric
    blocks when a layout/OCR backend supplies bounding boxes. Ambiguous units
    remain explicitly marked by confidence for a selective VLM refinement pass.
    """

    def __init__(
        self,
        *,
        low_confidence_threshold: float = 0.75,
        max_unit_chars: int = 3000,
    ) -> None:
        if not 0.0 <= low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be between 0 and 1")
        if max_unit_chars < 300:
            raise ValueError("max_unit_chars must be at least 300")
        self.low_confidence_threshold = low_confidence_threshold
        self.max_unit_chars = max_unit_chars

    def segment(
        self,
        page: RetrievedChunk,
        *,
        blocks: Sequence[LayoutBlock] | None = None,
    ) -> list[EducationalUnit]:
        source_blocks = list(blocks) if blocks is not None else split_ocr_blocks(page.text)
        if not source_blocks:
            return []
        labels = [
            _classify_block(
                block,
                next_block=source_blocks[index + 1]
                if index + 1 < len(source_blocks)
                else None,
            )
            for index, block in enumerate(source_blocks)
        ]

        grouped: list[tuple[UnitKind, list[LayoutBlock], list[_BlockLabel]]] = []
        for block, label in zip(source_blocks, labels):
            if label.option and grouped:
                grouped[-1][1].append(block)
                grouped[-1][2].append(label)
                continue

            starts_new = not grouped
            if grouped:
                previous_kind = grouped[-1][0]
                if label.heading:
                    starts_new = True
                elif label.kind in {
                    UnitKind.EXERCISE,
                    UnitKind.WORKED_EXAMPLE,
                    UnitKind.SOLUTION,
                    UnitKind.ANSWER_KEY,
                    UnitKind.INSTRUCTION,
                }:
                    starts_new = True
                elif previous_kind in {
                    UnitKind.EXERCISE,
                    UnitKind.WORKED_EXAMPLE,
                    UnitKind.SOLUTION,
                    UnitKind.ANSWER_KEY,
                    UnitKind.INSTRUCTION,
                }:
                    starts_new = (
                        label.kind is UnitKind.THEORY
                        and label.confidence >= 0.8
                        and len(block.text) >= 260
                    )
                elif previous_kind != label.kind:
                    starts_new = True

            if starts_new:
                grouped.append((label.kind, [block], [label]))
            else:
                grouped[-1][1].append(block)
                grouped[-1][2].append(label)

        units: list[EducationalUnit] = []
        current_section: str | None = None
        previous_exercise_id: str | None = None
        for kind, unit_blocks, unit_labels in grouped:
            text = "\n\n".join(block.text for block in unit_blocks).strip()
            if not text:
                continue
            first_label = unit_labels[0]
            if first_label.heading:
                current_section = unit_blocks[0].text.splitlines()[0][:160]
            confidence = round(
                sum(label.confidence for label in unit_labels) / len(unit_labels),
                4,
            )
            index = len(units)
            unit_id = _stable_unit_id(page.chunk_id, kind, index, text)
            task_number = next(
                (label.task_number for label in unit_labels if label.task_number),
                None,
            )
            relation: dict[str, Any] = {}
            if kind is UnitKind.EXERCISE:
                previous_exercise_id = unit_id
            elif kind in {UnitKind.SOLUTION, UnitKind.ANSWER_KEY} and previous_exercise_id:
                relation["exercise_id"] = previous_exercise_id

            metadata = {
                **page.metadata,
                "unit_kind": kind.value,
                "unit_index": index,
                "parent_chunk_id": page.chunk_id,
                "source_page": page.metadata.get("page"),
                "section_title": current_section,
                "task_number": task_number,
                "segmentation_confidence": confidence,
                "low_confidence": confidence < self.low_confidence_threshold,
                "oversized": len(text) > self.max_unit_chars,
                **relation,
            }
            units.append(
                EducationalUnit(
                    unit_id=unit_id,
                    parent_chunk_id=page.chunk_id,
                    kind=kind,
                    text=text,
                    confidence=confidence,
                    block_start=unit_blocks[0].order,
                    block_end=unit_blocks[-1].order,
                    task_number=task_number,
                    section_title=current_section,
                    bbox=_bbox_union(unit_blocks),
                    metadata=metadata,
                )
            )
        return units

    @staticmethod
    def as_retrieved_chunk(
        unit: EducationalUnit,
        *,
        parent: RetrievedChunk,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=unit.unit_id,
            text=unit.text,
            images=parent.images,
            score=0.0,
            metadata=unit.metadata,
        )

    def chunk_page(
        self,
        page: RetrievedChunk,
        *,
        blocks: Sequence[LayoutBlock] | None = None,
    ) -> list[RetrievedChunk]:
        return [
            self.as_retrieved_chunk(unit, parent=page)
            for unit in self.segment(page, blocks=blocks)
        ]
