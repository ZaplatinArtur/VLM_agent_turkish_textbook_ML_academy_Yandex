"""Task-ID-free exact-source resolver for full-page Turkish MCQ inputs.

The resolver is intentionally narrow.  It recognises one of two pinned MEB
textbooks from image bytes, obtains the requested printed question number from
the observable prompt, and returns an answer only when a source-native key cell
from the same immutable PDF is fully bound to the visual page decision.

Benchmark IDs, selected holdout rows, labels, scores and correctness outcomes
are not accepted by any public API in this module.  A full-page image can occur
in several inputs, so the image selects a page while the prompt selects the
question on that page.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import tempfile
from typing import Any
import unicodedata

from .official_ogm import canonical_json_bytes, canonical_json_sha256, sha256_file
from .visual_coordinate_binding import (
    SiftRuntimeProfile,
    VisualBindingThresholds,
    VisualCoordinateBindingError,
    VisualPageEvidence,
    compute_sift_page_evidence,
    geometry_checks,
    visual_page_evidence_from_mapping,
)


INVENTORY_SCHEMA = "mcq-fullpage-source-inventory-v1"
KEY_INDEX_SCHEMA = "mcq-fullpage-official-key-index-v1"
RENDER_MANIFEST_SCHEMA = "mcq-fullpage-poppler-render-manifest-v1"
CERTIFICATE_SCHEMA = "mcq-fullpage-source-certificate-v1"
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_PDFPLUMBER_VERSION = "0.11.9"
EXPECTED_NUMPY_VERSION = "2.5.1"
EXPECTED_OPENCV_VERSION = "5.0.0"
EXPECTED_POPPLER_VERSION = "26.05.0"
EXPECTED_PDFTOPPM_SHA256 = (
    "742cbbd9a00931ad16c6618410bc40471375d639a45c61c1d86f3dcfc54b6388"
)
EXPECTED_DOCUMENT_COUNT = 2
EXPECTED_PROTOCOL_RECORD_COUNT = 147
EXPECTED_CHOICE_KEY_COUNT = 143
EXPECTED_CONTENT_PAGE_COUNT = 28
EXPECTED_KEY_PAGE_COUNT = 6
FROZEN_BUNDLE_MANIFEST_SCHEMA = "mcq-fullpage-source-adapter-freeze-v1"
EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256 = (
    "5744488edc02e70e921bae9cddbae3d2f60448768a7fd02975a7dc9e5ccb04f7"
)
EXPECTED_FROZEN_BUNDLE_MANIFEST_PROJECTION_SHA256 = (
    "134946a0087cd1f389f2904b187f41708b7bb4ae8899001b5311b668aee14c01"
)
EXPECTED_INVENTORY_FILE_SHA256 = (
    "965e41673aea7df73fb03f98818c3ce3c8a1561873c9deece7e11be9a8b37dec"
)
EXPECTED_KEY_INDEX_FILE_SHA256 = (
    "1ef19fe8e56b0ba97307d8c22abe8a8c8d3a72baf42eb5f7a22eb069b6a5e8e0"
)
EXPECTED_SOURCE_AUDIT_FILE_SHA256 = (
    "e31eef8538ca30eb58111ecbd3ef2485a13e6f4ed75e3c9bcf4ee052bc2939c6"
)
EXPECTED_RENDER_MANIFEST_FILE_SHA256 = (
    "a57c8869ba29a5f9362d9d536b5a78415e810a4bfbdc8a3b2848b19e29cdb458"
)
EXPECTED_FROZEN_REPORT_FILE_SHA256 = (
    "eae63b82a8fe31eee56288f162f68a73a71eb1dcb12eb1dd3d136cfb7c7458a9"
)
EXPECTED_INVENTORY_PROJECTION_SHA256 = (
    "5f9e01678b2a3b7c14600dffadb06e0cce96212712835509d3bcfd1625b4fff3"
)
EXPECTED_KEY_INDEX_PROJECTION_SHA256 = (
    "9ca8672db13c5d6a6b05ee375bc540c4b4e5647f91cb8c54daa4839fdaa317ee"
)
EXPECTED_RENDER_MANIFEST_PROJECTION_SHA256 = (
    "709ecc38a36cfbdad33d8ea8bf80ebf4ad38f00fd650655a5afd518c0d2903aa"
)
EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256 = (
    "85bdb83c27bf4e4cc5e236a8997ecfb6903403cb3f1846685432672fb4c202f9"
)
EXPECTED_FROZEN_ARTIFACTS = {
    "inventory.json": (EXPECTED_INVENTORY_FILE_SHA256, 61_945),
    "official_key_index.json": (EXPECTED_KEY_INDEX_FILE_SHA256, 59_034),
    "source_build_audit.json": (EXPECTED_SOURCE_AUDIT_FILE_SHA256, 1_279),
    "render_manifest.json": (EXPECTED_RENDER_MANIFEST_FILE_SHA256, 7_285),
    "REPORT_RU.md": (EXPECTED_FROZEN_REPORT_FILE_SHA256, 19_474),
}
EXPECTED_DOCUMENT_SOURCES = {
    "biology9_textbook": {
        "pdf_sha256": "717548090c5bece21242fab41a3dad26aa43031f5a73d4191538903ab3ec4ea0",
        "pdf_size_bytes": 26_848_837,
        "page_count": 173,
    },
    "physics12_textbook": {
        "pdf_sha256": "0957cb2a74ed46d6b7c3a3165863e03b5a7206cdf444f6ad8ecf6a13179a6307",
        "pdf_size_bytes": 205_784_209,
        "page_count": 275,
    },
}
EXPECTED_UNSUPPORTED_SOURCE_ADDRESSES = frozenset(
    ("physics12_textbook", 2, question) for question in range(24, 28)
)
EXPECTED_KEY_PAGES = {
    "biology9_textbook": (158, 160, 162),
    "physics12_textbook": (263, 264, 265),
}
EXPECTED_BIOLOGY_PAGE_MAP = {
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
EXPECTED_PHYSICS_PAGE_MAP = {
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


def _expected_protocol_source_records() -> frozenset[tuple[str, int, int, int, str]]:
    records: set[tuple[str, int, int, int, str]] = set()
    for family, page_map in (
        ("biology9_textbook", EXPECTED_BIOLOGY_PAGE_MAP),
        ("physics12_textbook", EXPECTED_PHYSICS_PAGE_MAP),
    ):
        for unit, question_pages in page_map.items():
            for question, page in question_pages.items():
                response_kind = (
                    "unsupported_open_response"
                    if (family, unit, question)
                    in EXPECTED_UNSUPPORTED_SOURCE_ADDRESSES
                    else "choice_A-E"
                )
                records.add((family, unit, page, question, response_kind))
    return frozenset(records)


EXPECTED_PROTOCOL_SOURCE_RECORDS = _expected_protocol_source_records()
FROZEN_VISUAL_THRESHOLDS = VisualBindingThresholds()
FROZEN_SIFT_RUNTIME_PROFILE = SiftRuntimeProfile(
    render_dpi=144,
    nfeatures=12_000,
    contrast_threshold=0.02,
    edge_threshold=12.0,
    ratio_test=0.72,
    ransac_reprojection_px=4.0,
    ransac_max_iters=5_000,
    ransac_confidence=0.999,
    rng_seed=19_870_511,
    expected_opencv_version=EXPECTED_OPENCV_VERSION,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")
_RECORD_ID = re.compile(
    r"^[a-z0-9][a-z0-9._-]{2,80}:u[1-9][0-9]*:p[1-9][0-9]*:q[1-9][0-9]*$"
)
_CHOICES = frozenset("ABCDE")
_PROMPT_PATTERN = re.compile(
    r"Sayfadaki\s+([1-9][0-9]{0,2})\s*\.\s*"
    r"çoktan\s+seçmeli\s+soruyu\s+çözünüz\.\s*"
    r"Yalnızca\s+A\s*,\s*B\s*,\s*C\s*,\s*D\s+veya\s+E\s+yazınız\.",
    re.UNICODE,
)
_FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "accuracy",
        "benchmark",
        "correct",
        "correctness",
        "evaluation",
        "expectedanswer",
        "gold",
        "label",
        "metric",
        "oracle",
        "outcome",
        "prediction",
        "reward",
        "score",
        "selected",
        "taskid",
        "verdict",
    }
)


class McqSourceError(ValueError):
    """The MCQ source or certificate cannot be verified."""


def _compact_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _reject_forbidden_source_keys(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise McqSourceError(f"non-string JSON key at {'.'.join(path)}")
            if _compact_key(raw_key) in _FORBIDDEN_SOURCE_KEYS:
                raise McqSourceError(
                    "benchmark/evaluation field is forbidden in source artifact: "
                    f"{'.'.join(path + (raw_key,))}"
                )
            _reject_forbidden_source_keys(child, path + (raw_key,))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_forbidden_source_keys(child, path + (f"[{index}]",))


def _strict_json_object(path: Path) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise McqSourceError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise McqSourceError(f"non-finite JSON constant in {path}: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise McqSourceError(f"cannot load JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise McqSourceError(f"JSON root is not an object: {path}")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise McqSourceError(f"{label} must be a positive integer")
    return value


def _sha(value: Any, label: str) -> str:
    result = str(value or "")
    if _HEX64.fullmatch(result) is None:
        raise McqSourceError(f"{label} is not a SHA-256 pin")
    return result


def _bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 4
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise McqSourceError(f"{label} must be a finite four-number bbox")
    result = tuple(float(item) for item in value)
    if not (0.0 <= result[0] < result[2] and 0.0 <= result[1] < result[3]):
        raise McqSourceError(f"{label} is not ordered")
    return result  # type: ignore[return-value]


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise McqSourceError(f"{label} must be a finite number")
    return float(value)


def _normal_prompt(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise McqSourceError("observable prompt is empty or malformed")
    normalized = unicodedata.normalize("NFC", value)
    normalized = " ".join(normalized.split())
    return normalized


def parse_observable_mcq_prompt(prompt: str) -> int:
    """Return only the question number explicitly printed in the prompt."""

    normalized = _normal_prompt(prompt)
    match = _PROMPT_PATTERN.fullmatch(normalized)
    if match is None:
        raise McqSourceError("prompt does not match the frozen Turkish MCQ grammar")
    return int(match.group(1))


def assert_mcq_runtime(
    *, require_pdfplumber: bool = False, require_visual: bool = False
) -> dict[str, str]:
    observed = {"python": ".".join(str(item) for item in sys.version_info[:3])}
    if observed["python"] != EXPECTED_PYTHON_VERSION:
        raise McqSourceError(
            f"Python {observed['python']} differs from pinned {EXPECTED_PYTHON_VERSION}"
        )
    if require_pdfplumber:
        try:
            import pdfplumber  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise McqSourceError("pdfplumber is required by the MCQ runtime") from exc
        observed["pdfplumber"] = str(getattr(pdfplumber, "__version__", ""))
        if observed["pdfplumber"] != EXPECTED_PDFPLUMBER_VERSION:
            raise McqSourceError(
                f"pdfplumber {observed['pdfplumber']} differs from pinned "
                f"{EXPECTED_PDFPLUMBER_VERSION}"
            )
    if require_visual:
        try:
            import cv2  # type: ignore
            import numpy  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise McqSourceError("NumPy and OpenCV are required by the MCQ runtime") from exc
        observed["numpy"] = str(getattr(numpy, "__version__", ""))
        observed["opencv"] = str(getattr(cv2, "__version__", ""))
        if observed["numpy"] != EXPECTED_NUMPY_VERSION:
            raise McqSourceError(
                f"NumPy {observed['numpy']} differs from pinned {EXPECTED_NUMPY_VERSION}"
            )
        if observed["opencv"] != EXPECTED_OPENCV_VERSION:
            raise McqSourceError(
                f"OpenCV {observed['opencv']} differs from pinned {EXPECTED_OPENCV_VERSION}"
            )
    return observed


def _require_frozen_profile(
    thresholds: VisualBindingThresholds,
    runtime_profile: SiftRuntimeProfile,
) -> None:
    if thresholds != FROZEN_VISUAL_THRESHOLDS:
        raise VisualCoordinateBindingError(
            "MCQ visual thresholds differ from the exact frozen profile"
        )
    if runtime_profile != FROZEN_SIFT_RUNTIME_PROFILE:
        raise VisualCoordinateBindingError(
            "MCQ SIFT runtime differs from the exact frozen profile"
        )


@dataclass(frozen=True, slots=True)
class McqQuestionRecord:
    record_id: str
    document_id: str
    source_family: str
    unit_number: int
    content_page_number: int
    question_number: int
    source_response_kind: str
    content_marker: str
    content_marker_bbox: tuple[float, float, float, float]
    content_marker_projection_sha256: str

    def __post_init__(self) -> None:
        if _RECORD_ID.fullmatch(self.record_id) is None:
            raise McqSourceError("source question record_id is malformed")
        expected = (
            f"{self.document_id}:u{self.unit_number}:p{self.content_page_number}:"
            f"q{self.question_number}"
        )
        if self.record_id != expected:
            raise McqSourceError("record_id is not its exact source address")
        if _DOCUMENT_ID.fullmatch(self.document_id) is None or not self.source_family:
            raise McqSourceError("source question identity is malformed")
        for name in ("unit_number", "content_page_number", "question_number"):
            _positive_integer(getattr(self, name), name)
        if self.source_response_kind not in {
            "choice_A-E",
            "unsupported_open_response",
        }:
            raise McqSourceError("unknown source response kind")
        if self.content_marker not in {
            f"{self.question_number}.",
            f"{self.question_number})",
        }:
            raise McqSourceError("content marker does not encode the question number")
        _bbox(self.content_marker_bbox, "content_marker_bbox")
        _sha(self.content_marker_projection_sha256, "content marker projection")

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["content_marker_bbox"] = list(self.content_marker_bbox)
        return value


@dataclass(frozen=True, slots=True)
class McqDocument:
    document_id: str
    source_family: str
    pdf_sha256: str
    pdf_size_bytes: int
    page_count: int
    questions: tuple[McqQuestionRecord, ...]
    key_pages: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            _DOCUMENT_ID.fullmatch(self.document_id) is None
            or not self.document_id.endswith(self.pdf_sha256[:12])
        ):
            raise McqSourceError("document_id is not bound to the PDF hash")
        if not self.source_family:
            raise McqSourceError("source family is empty")
        expected_source = EXPECTED_DOCUMENT_SOURCES.get(self.source_family)
        if expected_source is None or {
            "pdf_sha256": self.pdf_sha256,
            "pdf_size_bytes": self.pdf_size_bytes,
            "page_count": self.page_count,
        } != expected_source:
            raise McqSourceError("document identity differs from the frozen official PDF")
        _sha(self.pdf_sha256, "document PDF")
        _positive_integer(self.pdf_size_bytes, "pdf_size_bytes")
        _positive_integer(self.page_count, "page_count")
        if not self.questions or not self.key_pages:
            raise McqSourceError("document has no questions or key pages")
        if any(page > self.page_count for page in self.key_pages):
            raise McqSourceError("key page is outside the PDF")
        if any(
            item.document_id != self.document_id
            or item.source_family != self.source_family
            or item.content_page_number > self.page_count
            for item in self.questions
        ):
            raise McqSourceError("document contains a foreign/out-of-range question")
        addresses = [item.record_id for item in self.questions]
        if len(addresses) != len(set(addresses)):
            raise McqSourceError("document contains duplicate source records")
        expected_question_order = tuple(
            sorted(
                self.questions,
                key=lambda item: (
                    item.unit_number,
                    item.content_page_number,
                    item.question_number,
                ),
            )
        )
        if self.questions != expected_question_order:
            raise McqSourceError("document questions are not in canonical source order")
        if self.key_pages != tuple(sorted(set(self.key_pages))):
            raise McqSourceError("document key pages are not sorted and unique")
        if self.key_pages != EXPECTED_KEY_PAGES[self.source_family]:
            raise McqSourceError("document key pages differ from the frozen source")

    @property
    def content_pages(self) -> tuple[int, ...]:
        return tuple(sorted({item.content_page_number for item in self.questions}))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_family": self.source_family,
            "pdf_sha256": self.pdf_sha256,
            "pdf_size_bytes": self.pdf_size_bytes,
            "page_count": self.page_count,
            "content_pages": list(self.content_pages),
            "key_pages": list(self.key_pages),
            "questions": [item.to_mapping() for item in self.questions],
        }


@dataclass(frozen=True, slots=True)
class McqInventory:
    documents: tuple[McqDocument, ...]
    inventory_projection_sha256: str

    def __post_init__(self) -> None:
        if len(self.documents) != EXPECTED_DOCUMENT_COUNT:
            raise McqSourceError("MCQ inventory must contain exactly two documents")
        if len({item.document_id for item in self.documents}) != len(self.documents):
            raise McqSourceError("MCQ document identity is duplicated")
        if len({item.source_family for item in self.documents}) != len(self.documents):
            raise McqSourceError("MCQ source family is duplicated")
        if self.documents != tuple(
            sorted(self.documents, key=lambda item: item.document_id)
        ) or {item.source_family for item in self.documents} != set(
            EXPECTED_DOCUMENT_SOURCES
        ):
            raise McqSourceError("MCQ documents are not the canonical source pair")
        if len(self.questions) != EXPECTED_PROTOCOL_RECORD_COUNT:
            raise McqSourceError("MCQ inventory does not contain all 147 source records")
        response_counts = Counter(item.source_response_kind for item in self.questions)
        if response_counts != Counter(
            {"choice_A-E": EXPECTED_CHOICE_KEY_COUNT, "unsupported_open_response": 4}
        ):
            raise McqSourceError(
                "MCQ inventory does not preserve the 143-choice/4-open source census"
            )
        unsupported = {
            (item.source_family, item.unit_number, item.question_number)
            for item in self.questions
            if item.source_response_kind == "unsupported_open_response"
        }
        if unsupported != EXPECTED_UNSUPPORTED_SOURCE_ADDRESSES:
            raise McqSourceError("unsupported records are not Physics U2 q24-q27")
        observed_records = {
            (
                item.source_family,
                item.unit_number,
                item.content_page_number,
                item.question_number,
                item.source_response_kind,
            )
            for item in self.questions
        }
        if observed_records != EXPECTED_PROTOCOL_SOURCE_RECORDS:
            raise McqSourceError("MCQ inventory source-address universe changed")
        if len(self.candidate_pages) != EXPECTED_CONTENT_PAGE_COUNT:
            raise McqSourceError("MCQ inventory does not contain all 28 content pages")
        if len(self.key_page_addresses) != EXPECTED_KEY_PAGE_COUNT:
            raise McqSourceError("MCQ inventory does not contain six distinct key pages")
        _sha(self.inventory_projection_sha256, "inventory projection")

    @property
    def questions(self) -> tuple[McqQuestionRecord, ...]:
        return tuple(item for document in self.documents for item in document.questions)

    @property
    def candidate_pages(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (document.document_id, page)
            for document in self.documents
            for page in document.content_pages
        )

    @property
    def key_page_addresses(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (document.document_id, page)
            for document in self.documents
            for page in document.key_pages
        )

    def document(self, document_id: str) -> McqDocument:
        matches = [item for item in self.documents if item.document_id == document_id]
        if len(matches) != 1:
            raise McqSourceError("document id is absent or ambiguous")
        return matches[0]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": INVENTORY_SCHEMA,
            "documents": [item.to_mapping() for item in self.documents],
            "inventory_projection_sha256": self.inventory_projection_sha256,
        }


@dataclass(frozen=True, slots=True)
class McqKeyCell:
    record_id: str
    document_id: str
    unit_number: int
    question_number: int
    answer: str
    key_page_number: int
    key_bbox: tuple[float, float, float, float]
    key_text: str
    key_text_sha256: str
    key_projection_sha256: str

    def __post_init__(self) -> None:
        if _RECORD_ID.fullmatch(self.record_id) is None:
            raise McqSourceError("key cell record id is malformed")
        if _DOCUMENT_ID.fullmatch(self.document_id) is None:
            raise McqSourceError("key cell document id is malformed")
        _positive_integer(self.unit_number, "key unit_number")
        _positive_integer(self.question_number, "key question_number")
        _positive_integer(self.key_page_number, "key_page_number")
        if self.answer not in _CHOICES:
            raise McqSourceError("key answer is not one of A-E")
        _bbox(self.key_bbox, "key_bbox")
        if not self.key_text or hashlib.sha256(
            self.key_text.encode("utf-8")
        ).hexdigest() != _sha(self.key_text_sha256, "key text"):
            raise McqSourceError("key text bytes do not match their pin")
        if self.key_text != f"{self.question_number} {self.answer}":
            raise McqSourceError("key text is not the canonical number/choice pair")
        _sha(self.key_projection_sha256, "key projection")

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["key_bbox"] = list(self.key_bbox)
        return value


@dataclass(frozen=True, slots=True)
class McqKeyIndex:
    inventory_projection_sha256: str
    cells: tuple[McqKeyCell, ...]
    key_index_projection_sha256: str

    def __post_init__(self) -> None:
        _sha(self.inventory_projection_sha256, "key-index inventory projection")
        _sha(self.key_index_projection_sha256, "key-index projection")
        if len(self.cells) != EXPECTED_CHOICE_KEY_COUNT:
            raise McqSourceError("key index does not cover all 143 official A-E records")
        if len({item.record_id for item in self.cells}) != len(self.cells):
            raise McqSourceError("key index contains duplicate records")
        if self.cells != tuple(sorted(self.cells, key=lambda item: item.record_id)):
            raise McqSourceError("key cells are not in canonical source order")

    def cell(self, record_id: str) -> McqKeyCell:
        matches = [item for item in self.cells if item.record_id == record_id]
        if len(matches) != 1:
            raise McqSourceError("source key cell is absent or ambiguous")
        return matches[0]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": KEY_INDEX_SCHEMA,
            "inventory_projection_sha256": self.inventory_projection_sha256,
            "cells": [item.to_mapping() for item in self.cells],
            "key_index_projection_sha256": self.key_index_projection_sha256,
        }


@dataclass(frozen=True, slots=True)
class McqRenderedPage:
    document_id: str
    page_number: int
    relative_path: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    resolved_path: Path | None = None

    def __post_init__(self) -> None:
        if _DOCUMENT_ID.fullmatch(self.document_id) is None:
            raise McqSourceError("rendered page document id is malformed")
        _positive_integer(self.page_number, "rendered page_number")
        _positive_integer(self.size_bytes, "rendered size_bytes")
        _positive_integer(self.width, "rendered width")
        _positive_integer(self.height, "rendered height")
        _sha(self.sha256, "rendered page")
        if (
            not self.relative_path
            or "\\" in self.relative_path
            or not self.relative_path.endswith(".png")
            or PureWindowsPath(self.relative_path).is_absolute()
            or PurePosixPath(self.relative_path).is_absolute()
            or ".." in PureWindowsPath(self.relative_path).parts
            or ".." in PurePosixPath(self.relative_path).parts
        ):
            raise McqSourceError("rendered page path is unsafe")

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("resolved_path")
        return value


@dataclass(frozen=True, slots=True)
class McqRenderManifest:
    inventory_projection_sha256: str
    render_dpi: int
    color_mode: str
    poppler_version: str
    poppler_executable_sha256: str
    pages: tuple[McqRenderedPage, ...]
    render_manifest_projection_sha256: str

    def __post_init__(self) -> None:
        _sha(self.inventory_projection_sha256, "render inventory projection")
        _positive_integer(self.render_dpi, "render_dpi")
        if self.color_mode != "poppler_gray_rgb_png":
            raise McqSourceError(
                "MCQ render color mode must be poppler_gray_rgb_png"
            )
        if self.poppler_version != EXPECTED_POPPLER_VERSION:
            raise McqSourceError("Poppler version differs from the frozen profile")
        if (
            _sha(self.poppler_executable_sha256, "pdftoppm executable")
            != EXPECTED_PDFTOPPM_SHA256
        ):
            raise McqSourceError("pdftoppm executable differs from the frozen binary")
        if len(self.pages) != EXPECTED_CONTENT_PAGE_COUNT:
            raise McqSourceError("render manifest does not contain all 28 pages")
        addresses = [(item.document_id, item.page_number) for item in self.pages]
        if len(addresses) != len(set(addresses)):
            raise McqSourceError("render manifest contains duplicate page addresses")
        if self.pages != tuple(
            sorted(self.pages, key=lambda item: (item.document_id, item.page_number))
        ):
            raise McqSourceError("rendered pages are not in canonical source order")
        _sha(self.render_manifest_projection_sha256, "render manifest projection")

    def page(self, document_id: str, page_number: int) -> McqRenderedPage:
        matches = [
            item
            for item in self.pages
            if item.document_id == document_id and item.page_number == page_number
        ]
        if len(matches) != 1:
            raise McqSourceError("rendered page is absent or ambiguous")
        return matches[0]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": RENDER_MANIFEST_SCHEMA,
            "inventory_projection_sha256": self.inventory_projection_sha256,
            "render_dpi": self.render_dpi,
            "color_mode": self.color_mode,
            "poppler_version": self.poppler_version,
            "poppler_executable_sha256": self.poppler_executable_sha256,
            "pages": [item.to_mapping() for item in self.pages],
            "render_manifest_projection_sha256": (
                self.render_manifest_projection_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class FrozenMcqBundle:
    """Exact public source bundle attested before any opaque input is read."""

    inventory: McqInventory
    key_index: McqKeyIndex
    render_manifest: McqRenderManifest
    freeze_manifest_sha256: str
    freeze_manifest_projection_sha256: str
    page_payloads_projection_sha256: str
    attestation_projection_sha256: str

    def __post_init__(self) -> None:
        if self.freeze_manifest_sha256 != EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256:
            raise McqSourceError("MCQ bundle freeze-manifest SHA changed")
        if (
            self.freeze_manifest_projection_sha256
            != EXPECTED_FROZEN_BUNDLE_MANIFEST_PROJECTION_SHA256
        ):
            raise McqSourceError("MCQ bundle freeze projection changed")
        if (
            self.page_payloads_projection_sha256
            != EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256
        ):
            raise McqSourceError("MCQ bundle page-payload projection changed")
        _sha(self.attestation_projection_sha256, "bundle attestation")


@dataclass(frozen=True, slots=True)
class McqPageDecision:
    accepted: bool
    reason: str
    checks: tuple[tuple[str, bool], ...]
    selected_document_id: str | None = None
    selected_source_family: str | None = None
    selected_page_number: int | None = None
    selected_unit_number: int | None = None
    selected_question_number: int | None = None
    selected_record_id: str | None = None
    best_rank_score: float = 0.0
    runner_rank_score: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool) or not isinstance(self.reason, str) or not self.reason:
            raise McqSourceError("MCQ page decision status is malformed")
        if any(
            not isinstance(name, str) or not name or not isinstance(passed, bool)
            for name, passed in self.checks
        ) or len({name for name, _ in self.checks}) != len(self.checks):
            raise McqSourceError("MCQ page decision checks are malformed or duplicated")
        best = _finite_number(self.best_rank_score, "best_rank_score")
        runner = _finite_number(self.runner_rank_score, "runner_rank_score")
        if best < 0.0 or runner < 0.0:
            raise McqSourceError("MCQ page decision rank scores are negative")
        if self.accepted:
            if self.reason != "accepted" or not self.checks or not all(
                passed for _, passed in self.checks
            ):
                raise McqSourceError("accepted MCQ decision has a failed check")
            if (
                not isinstance(self.selected_document_id, str)
                or not isinstance(self.selected_source_family, str)
                or not isinstance(self.selected_record_id, str)
            ):
                raise McqSourceError("accepted MCQ decision lacks source identity")
            for name in (
                "selected_page_number",
                "selected_unit_number",
                "selected_question_number",
            ):
                _positive_integer(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class McqSourceCertificate:
    task_image_sha256: str
    prompt_sha256: str
    normalized_prompt_sha256: str
    observed_question_number: int
    inventory_projection_sha256: str
    key_index_projection_sha256: str
    render_manifest_projection_sha256: str
    evidences: tuple[VisualPageEvidence, ...]
    decision: McqPageDecision
    answer: str | None
    answer_sha256: str | None
    selected_key_projection_sha256: str | None
    certificate_projection_sha256: str

    def __post_init__(self) -> None:
        _sha(self.task_image_sha256, "certificate image")
        _sha(self.prompt_sha256, "certificate prompt")
        _sha(self.normalized_prompt_sha256, "certificate normalized prompt")
        _positive_integer(self.observed_question_number, "certificate question number")
        _sha(self.inventory_projection_sha256, "certificate inventory")
        _sha(self.key_index_projection_sha256, "certificate key index")
        _sha(self.render_manifest_projection_sha256, "certificate render manifest")
        _sha(self.certificate_projection_sha256, "certificate projection")
        if self.answer is None:
            if self.answer_sha256 is not None or self.selected_key_projection_sha256 is not None:
                raise McqSourceError("abstained certificate carries answer/key pins")
        else:
            if self.answer not in _CHOICES:
                raise McqSourceError("certificate answer is not A-E")
            if self.answer_sha256 != hashlib.sha256(
                self.answer.encode("utf-8")
            ).hexdigest():
                raise McqSourceError("certificate answer hash mismatch")
            _sha(self.selected_key_projection_sha256, "certificate selected key")

    def to_mapping(self) -> dict[str, Any]:
        evidences = []
        for item in self.evidences:
            value = asdict(item)
            value["mapped_polygon"] = (
                [list(point) for point in item.mapped_polygon]
                if item.mapped_polygon is not None
                else None
            )
            evidences.append(value)
        return {
            "schema_version": CERTIFICATE_SCHEMA,
            "task_image_sha256": self.task_image_sha256,
            "prompt_sha256": self.prompt_sha256,
            "normalized_prompt_sha256": self.normalized_prompt_sha256,
            "observed_question_number": self.observed_question_number,
            "inventory_projection_sha256": self.inventory_projection_sha256,
            "key_index_projection_sha256": self.key_index_projection_sha256,
            "render_manifest_projection_sha256": (
                self.render_manifest_projection_sha256
            ),
            "evidences": evidences,
            "decision": asdict(self.decision),
            "answer": self.answer,
            "answer_sha256": self.answer_sha256,
            "selected_key_projection_sha256": self.selected_key_projection_sha256,
            "certificate_projection_sha256": self.certificate_projection_sha256,
        }


_DOCUMENT_FIELDS = frozenset(
    {
        "document_id",
        "source_family",
        "pdf_sha256",
        "pdf_size_bytes",
        "page_count",
        "content_pages",
        "key_pages",
        "questions",
    }
)
_QUESTION_FIELDS = frozenset(
    {
        "record_id",
        "document_id",
        "source_family",
        "unit_number",
        "content_page_number",
        "question_number",
        "source_response_kind",
        "content_marker",
        "content_marker_bbox",
        "content_marker_projection_sha256",
    }
)
_KEY_CELL_FIELDS = frozenset(
    {
        "record_id",
        "document_id",
        "unit_number",
        "question_number",
        "answer",
        "key_page_number",
        "key_bbox",
        "key_text",
        "key_text_sha256",
        "key_projection_sha256",
    }
)
_RENDER_PAGE_FIELDS = frozenset(
    {
        "document_id",
        "page_number",
        "relative_path",
        "sha256",
        "size_bytes",
        "width",
        "height",
    }
)


def load_mcq_inventory(path: Path) -> McqInventory:
    raw = _strict_json_object(path)
    _reject_forbidden_source_keys(raw)
    if set(raw) != {
        "schema_version",
        "documents",
        "inventory_projection_sha256",
    } or raw.get("schema_version") != INVENTORY_SCHEMA:
        raise McqSourceError("unsupported or noncanonical MCQ inventory")
    raw_documents = raw.get("documents")
    if not isinstance(raw_documents, list):
        raise McqSourceError("MCQ inventory documents are missing")
    documents: list[McqDocument] = []
    for raw_document in raw_documents:
        if not isinstance(raw_document, dict) or set(raw_document) != _DOCUMENT_FIELDS:
            raise McqSourceError("MCQ inventory document fields changed")
        raw_questions = raw_document.get("questions")
        if not isinstance(raw_questions, list):
            raise McqSourceError("MCQ inventory questions are missing")
        questions: list[McqQuestionRecord] = []
        for raw_question in raw_questions:
            if not isinstance(raw_question, dict) or set(raw_question) != _QUESTION_FIELDS:
                raise McqSourceError("MCQ inventory question fields changed")
            questions.append(
                McqQuestionRecord(
                    record_id=str(raw_question["record_id"]),
                    document_id=str(raw_question["document_id"]),
                    source_family=str(raw_question["source_family"]),
                    unit_number=_positive_integer(
                        raw_question["unit_number"], "unit_number"
                    ),
                    content_page_number=_positive_integer(
                        raw_question["content_page_number"], "content_page_number"
                    ),
                    question_number=_positive_integer(
                        raw_question["question_number"], "question_number"
                    ),
                    source_response_kind=str(raw_question["source_response_kind"]),
                    content_marker=str(raw_question["content_marker"]),
                    content_marker_bbox=_bbox(
                        raw_question["content_marker_bbox"], "content_marker_bbox"
                    ),
                    content_marker_projection_sha256=_sha(
                        raw_question["content_marker_projection_sha256"],
                        "content marker projection",
                    ),
                )
            )
        document = McqDocument(
            document_id=str(raw_document["document_id"]),
            source_family=str(raw_document["source_family"]),
            pdf_sha256=_sha(raw_document["pdf_sha256"], "document PDF"),
            pdf_size_bytes=_positive_integer(
                raw_document["pdf_size_bytes"], "pdf_size_bytes"
            ),
            page_count=_positive_integer(raw_document["page_count"], "page_count"),
            questions=tuple(questions),
            key_pages=tuple(
                _positive_integer(item, "key page")
                for item in raw_document["key_pages"]
            ),
        )
        if raw_document["content_pages"] != list(document.content_pages):
            raise McqSourceError("declared content pages differ from source records")
        documents.append(document)
    projection = {
        "schema_version": INVENTORY_SCHEMA,
        "documents": [item.to_mapping() for item in documents],
    }
    expected_projection = canonical_json_sha256(projection)
    if raw["inventory_projection_sha256"] != expected_projection:
        raise McqSourceError("MCQ inventory projection hash mismatch")
    inventory = McqInventory(tuple(documents), expected_projection)
    if inventory.to_mapping() != raw:
        raise McqSourceError("MCQ inventory is not in canonical source order")
    return inventory


def load_mcq_key_index(path: Path, inventory: McqInventory) -> McqKeyIndex:
    raw = _strict_json_object(path)
    if set(raw) != {
        "schema_version",
        "inventory_projection_sha256",
        "cells",
        "key_index_projection_sha256",
    } or raw.get("schema_version") != KEY_INDEX_SCHEMA:
        raise McqSourceError("unsupported or noncanonical MCQ key index")
    if raw.get("inventory_projection_sha256") != inventory.inventory_projection_sha256:
        raise McqSourceError("key index is bound to a different inventory")
    raw_cells = raw.get("cells")
    if not isinstance(raw_cells, list):
        raise McqSourceError("MCQ key cells are missing")
    cells: list[McqKeyCell] = []
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, dict) or set(raw_cell) != _KEY_CELL_FIELDS:
            raise McqSourceError("MCQ key-cell fields changed")
        cells.append(
            McqKeyCell(
                record_id=str(raw_cell["record_id"]),
                document_id=str(raw_cell["document_id"]),
                unit_number=_positive_integer(raw_cell["unit_number"], "key unit"),
                question_number=_positive_integer(
                    raw_cell["question_number"], "key question"
                ),
                answer=str(raw_cell["answer"]),
                key_page_number=_positive_integer(
                    raw_cell["key_page_number"], "key page"
                ),
                key_bbox=_bbox(raw_cell["key_bbox"], "key bbox"),
                key_text=str(raw_cell["key_text"]),
                key_text_sha256=_sha(raw_cell["key_text_sha256"], "key text"),
                key_projection_sha256=_sha(
                    raw_cell["key_projection_sha256"], "key projection"
                ),
            )
        )
    projection = {
        "schema_version": KEY_INDEX_SCHEMA,
        "inventory_projection_sha256": inventory.inventory_projection_sha256,
        "cells": [item.to_mapping() for item in cells],
    }
    expected_projection = canonical_json_sha256(projection)
    if raw["key_index_projection_sha256"] != expected_projection:
        raise McqSourceError("MCQ key-index projection hash mismatch")
    index = McqKeyIndex(
        inventory_projection_sha256=inventory.inventory_projection_sha256,
        cells=tuple(cells),
        key_index_projection_sha256=expected_projection,
    )
    if index.to_mapping() != raw:
        raise McqSourceError("MCQ key index is not in canonical source order")
    inventory_records = {item.record_id: item for item in inventory.questions}
    choice_records = {
        record_id: item
        for record_id, item in inventory_records.items()
        if item.source_response_kind == "choice_A-E"
    }
    if set(choice_records) != {item.record_id for item in index.cells}:
        raise McqSourceError(
            "MCQ key index is not a bijection with the 143 A-E source records"
        )
    for cell in index.cells:
        record = choice_records[cell.record_id]
        document = inventory.document(record.document_id)
        if (
            cell.document_id != record.document_id
            or cell.unit_number != record.unit_number
            or cell.question_number != record.question_number
            or cell.key_page_number not in document.key_pages
        ):
            raise McqSourceError("MCQ key cell is cross-bound to a foreign record")
    return index


def load_mcq_render_manifest(
    path: Path,
    inventory: McqInventory,
    *,
    page_root: Path | None = None,
) -> McqRenderManifest:
    raw = _strict_json_object(path)
    if set(raw) != {
        "schema_version",
        "inventory_projection_sha256",
        "render_dpi",
        "color_mode",
        "poppler_version",
        "poppler_executable_sha256",
        "pages",
        "render_manifest_projection_sha256",
    } or raw.get("schema_version") != RENDER_MANIFEST_SCHEMA:
        raise McqSourceError("unsupported or noncanonical MCQ render manifest")
    if raw.get("inventory_projection_sha256") != inventory.inventory_projection_sha256:
        raise McqSourceError("render manifest is bound to a different inventory")
    raw_pages = raw.get("pages")
    if not isinstance(raw_pages, list):
        raise McqSourceError("MCQ rendered pages are missing")
    resolved_root = page_root.resolve() if page_root is not None else None
    pages: list[McqRenderedPage] = []
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict) or set(raw_page) != _RENDER_PAGE_FIELDS:
            raise McqSourceError("MCQ rendered-page fields changed")
        relative_path = str(raw_page["relative_path"])
        resolved_path: Path | None = None
        if resolved_root is not None:
            resolved_path = (resolved_root / Path(relative_path)).resolve()
            try:
                resolved_path.relative_to(resolved_root)
            except ValueError as exc:
                raise McqSourceError("rendered page escapes page_root") from exc
            if not resolved_path.is_file():
                raise McqSourceError("rendered page payload is missing")
            if (
                sha256_file(resolved_path) != raw_page["sha256"]
                or resolved_path.stat().st_size != raw_page["size_bytes"]
            ):
                raise McqSourceError("rendered page payload differs from its pins")
            try:
                header = resolved_path.read_bytes()[:29]
            except OSError as exc:
                raise McqSourceError("rendered page payload cannot be read") from exc
            if (
                len(header) != 29
                or header[:8] != b"\x89PNG\r\n\x1a\n"
                or header[12:16] != b"IHDR"
            ):
                raise McqSourceError("rendered page is not a canonical PNG")
            png_width = int.from_bytes(header[16:20], "big")
            png_height = int.from_bytes(header[20:24], "big")
            bit_depth = header[24]
            color_type = header[25]
            if (
                png_width != raw_page["width"]
                or png_height != raw_page["height"]
                or bit_depth != 8
                or color_type != 2
            ):
                raise McqSourceError("rendered PNG geometry/color differs from manifest")
        pages.append(
            McqRenderedPage(
                document_id=str(raw_page["document_id"]),
                page_number=_positive_integer(raw_page["page_number"], "render page"),
                relative_path=relative_path,
                sha256=_sha(raw_page["sha256"], "rendered page"),
                size_bytes=_positive_integer(raw_page["size_bytes"], "render size"),
                width=_positive_integer(raw_page["width"], "render width"),
                height=_positive_integer(raw_page["height"], "render height"),
                resolved_path=resolved_path,
            )
        )
    projection_pages = []
    for item in pages:
        value = asdict(item)
        value.pop("resolved_path")
        projection_pages.append(value)
    projection = {
        "schema_version": RENDER_MANIFEST_SCHEMA,
        "inventory_projection_sha256": inventory.inventory_projection_sha256,
        "render_dpi": raw["render_dpi"],
        "color_mode": raw["color_mode"],
        "poppler_version": raw["poppler_version"],
        "poppler_executable_sha256": raw["poppler_executable_sha256"],
        "pages": projection_pages,
    }
    expected_projection = canonical_json_sha256(projection)
    if raw["render_manifest_projection_sha256"] != expected_projection:
        raise McqSourceError("MCQ render-manifest projection hash mismatch")
    manifest = McqRenderManifest(
        inventory_projection_sha256=inventory.inventory_projection_sha256,
        render_dpi=_positive_integer(raw["render_dpi"], "render_dpi"),
        color_mode=str(raw["color_mode"]),
        poppler_version=str(raw["poppler_version"]),
        poppler_executable_sha256=_sha(
            raw["poppler_executable_sha256"], "pdftoppm executable"
        ),
        pages=tuple(pages),
        render_manifest_projection_sha256=expected_projection,
    )
    if {(item.document_id, item.page_number) for item in manifest.pages} != set(
        inventory.candidate_pages
    ):
        raise McqSourceError("render manifest does not equal the inventory page set")
    if manifest.to_mapping() != raw:
        raise McqSourceError("MCQ render manifest is not canonical")
    return manifest


def _recompute_object_projection(
    value: Mapping[str, Any], pin_field: str, label: str
) -> str:
    projection = dict(value)
    observed_pin = projection.pop(pin_field, None)
    recomputed = canonical_json_sha256(projection)
    if observed_pin != recomputed:
        raise McqSourceError(f"{label} object projection is not self-consistent")
    return recomputed


def assert_frozen_mcq_objects(
    inventory: McqInventory,
    key_index: McqKeyIndex,
    render_manifest: McqRenderManifest,
) -> None:
    """Reject a self-consistent but non-frozen source/key/render object graph.

    This is deliberately repeated by the resolver, certificate verifier and
    batch executor.  Loading a structurally valid replacement bundle is not a
    trust decision: exact public v1 artifact projections are the trust anchor.
    Rendered page bytes are checked here; SIFT evidence is *not* recomputed by
    this function or by certificate replay.
    """

    inventory_projection = _recompute_object_projection(
        inventory.to_mapping(), "inventory_projection_sha256", "inventory"
    )
    key_projection = _recompute_object_projection(
        key_index.to_mapping(), "key_index_projection_sha256", "key index"
    )
    render_projection = _recompute_object_projection(
        render_manifest.to_mapping(),
        "render_manifest_projection_sha256",
        "render manifest",
    )
    if inventory_projection != EXPECTED_INVENTORY_PROJECTION_SHA256:
        raise McqSourceError("inventory is not the exact frozen source census")
    if key_projection != EXPECTED_KEY_INDEX_PROJECTION_SHA256:
        raise McqSourceError("key index is not the exact frozen official key")
    if render_projection != EXPECTED_RENDER_MANIFEST_PROJECTION_SHA256:
        raise McqSourceError("render manifest is not the exact frozen page set")
    if (
        key_index.inventory_projection_sha256 != inventory_projection
        or render_manifest.inventory_projection_sha256 != inventory_projection
    ):
        raise McqSourceError("frozen MCQ artifacts are cross-bound")
    for page in render_manifest.pages:
        if page.resolved_path is None or not page.resolved_path.is_file():
            raise McqSourceError("frozen rendered page bytes were not loaded")
        if (
            sha256_file(page.resolved_path) != page.sha256
            or page.resolved_path.stat().st_size != page.size_bytes
        ):
            raise McqSourceError("frozen rendered page bytes changed after load")


def _artifact_entry_map(raw_artifacts: Any) -> dict[str, tuple[str, int]]:
    if not isinstance(raw_artifacts, list):
        raise McqSourceError("frozen MCQ artifact list is malformed")
    result: dict[str, tuple[str, int]] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, dict) or set(raw) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise McqSourceError("frozen MCQ artifact entry is malformed")
        path = PurePosixPath(str(raw["path"]))
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise McqSourceError("frozen MCQ artifact path is unsafe")
        if path.name in result:
            raise McqSourceError("frozen MCQ artifact basename is duplicated")
        result[path.name] = (
            _sha(raw["sha256"], "frozen artifact"),
            _positive_integer(raw["size_bytes"], "frozen artifact size"),
        )
    return result


def assert_frozen_mcq_bundle(
    *,
    freeze_manifest_path: Path,
    inventory_path: Path,
    key_index_path: Path,
    render_manifest_path: Path,
    page_root: Path,
) -> FrozenMcqBundle:
    """Attest the exact public source bundle from bytes and frozen projections.

    The SHA of the already-published v1 freeze manifest is embedded in code.
    Therefore an attacker cannot substitute a new internally consistent key,
    inventory, render manifest and freeze file.  Every machine artifact and all
    28 page payloads are re-hashed before the caller may read opaque data.
    """

    freeze_manifest_path = freeze_manifest_path.resolve(strict=False)
    if (
        not freeze_manifest_path.is_file()
        or sha256_file(freeze_manifest_path)
        != EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256
    ):
        raise McqSourceError("MCQ freeze manifest is not the embedded trust anchor")
    raw = _strict_json_object(freeze_manifest_path)
    if raw.get("schema_version") != FROZEN_BUNDLE_MANIFEST_SCHEMA:
        raise McqSourceError("MCQ freeze-manifest schema changed")
    declared_projection = raw.get("manifest_projection_sha256")
    projection = dict(raw)
    projection.pop("manifest_projection_sha256", None)
    recomputed_projection = canonical_json_sha256(projection)
    if (
        declared_projection != recomputed_projection
        or recomputed_projection
        != EXPECTED_FROZEN_BUNDLE_MANIFEST_PROJECTION_SHA256
    ):
        raise McqSourceError("MCQ freeze-manifest projection changed")

    artifact_entries = _artifact_entry_map(raw.get("artifacts"))
    if artifact_entries != EXPECTED_FROZEN_ARTIFACTS:
        raise McqSourceError("MCQ frozen artifact set changed")
    passed_artifacts = {
        "inventory.json": inventory_path.resolve(strict=False),
        "official_key_index.json": key_index_path.resolve(strict=False),
        "render_manifest.json": render_manifest_path.resolve(strict=False),
        "source_build_audit.json": (
            freeze_manifest_path.parent / "source_build_audit.json"
        ),
        "REPORT_RU.md": freeze_manifest_path.parent / "REPORT_RU.md",
    }
    for name, path in passed_artifacts.items():
        expected_sha, expected_size = EXPECTED_FROZEN_ARTIFACTS[name]
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_sha
        ):
            raise McqSourceError(f"frozen MCQ artifact bytes changed: {name}")

    page_payloads = raw.get("page_payloads")
    if not isinstance(page_payloads, dict) or set(page_payloads) != {
        "count",
        "files",
        "combined_projection_sha256",
    }:
        raise McqSourceError("frozen MCQ page-payload declaration is malformed")
    raw_page_files = page_payloads.get("files")
    if (
        page_payloads.get("count") != EXPECTED_CONTENT_PAGE_COUNT
        or not isinstance(raw_page_files, list)
        or len(raw_page_files) != EXPECTED_CONTENT_PAGE_COUNT
        or canonical_json_sha256(raw_page_files)
        != EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256
        or page_payloads.get("combined_projection_sha256")
        != EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256
    ):
        raise McqSourceError("frozen MCQ page-payload projection changed")
    resolved_page_root = page_root.resolve(strict=False)
    if not resolved_page_root.is_dir():
        raise McqSourceError("frozen MCQ page root is missing")
    seen_page_paths: set[str] = set()
    for raw_page in raw_page_files:
        if not isinstance(raw_page, dict) or set(raw_page) != {
            "document_id",
            "height",
            "page_number",
            "path",
            "sha256",
            "size_bytes",
            "width",
        }:
            raise McqSourceError("frozen MCQ page-payload entry changed")
        relative = str(raw_page["path"])
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or "\\" in relative
            or relative in seen_page_paths
        ):
            raise McqSourceError("frozen MCQ page-payload path is unsafe/duplicate")
        seen_page_paths.add(relative)
        payload = (resolved_page_root / Path(relative)).resolve(strict=False)
        try:
            payload.relative_to(resolved_page_root)
        except ValueError as exc:
            raise McqSourceError("frozen MCQ page payload escapes page_root") from exc
        if (
            not payload.is_file()
            or payload.stat().st_size != raw_page["size_bytes"]
            or sha256_file(payload) != raw_page["sha256"]
        ):
            raise McqSourceError("frozen MCQ page payload bytes changed")

    inventory = load_mcq_inventory(passed_artifacts["inventory.json"])
    key_index = load_mcq_key_index(
        passed_artifacts["official_key_index.json"], inventory
    )
    render_manifest = load_mcq_render_manifest(
        passed_artifacts["render_manifest.json"],
        inventory,
        page_root=resolved_page_root,
    )
    assert_frozen_mcq_objects(inventory, key_index, render_manifest)
    frozen_pages = {
        (item.document_id, item.page_number): (
            item.relative_path,
            item.sha256,
            item.size_bytes,
            item.width,
            item.height,
        )
        for item in render_manifest.pages
    }
    declared_pages = {
        (str(item["document_id"]), int(item["page_number"])): (
            str(item["path"]),
            str(item["sha256"]),
            int(item["size_bytes"]),
            int(item["width"]),
            int(item["height"]),
        )
        for item in raw_page_files
    }
    if frozen_pages != declared_pages:
        raise McqSourceError("render manifest and freeze page payloads differ")
    source_census = raw.get("source_census")
    if not isinstance(source_census, dict) or (
        source_census.get("inventory_projection_sha256")
        != EXPECTED_INVENTORY_PROJECTION_SHA256
        or source_census.get("key_index_projection_sha256")
        != EXPECTED_KEY_INDEX_PROJECTION_SHA256
        or source_census.get("render_manifest_projection_sha256")
        != EXPECTED_RENDER_MANIFEST_PROJECTION_SHA256
    ):
        raise McqSourceError("freeze source-census projections changed")
    attestation = {
        "freeze_manifest_sha256": EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256,
        "freeze_manifest_projection_sha256": (
            EXPECTED_FROZEN_BUNDLE_MANIFEST_PROJECTION_SHA256
        ),
        "inventory_file_sha256": EXPECTED_INVENTORY_FILE_SHA256,
        "inventory_projection_sha256": EXPECTED_INVENTORY_PROJECTION_SHA256,
        "key_index_file_sha256": EXPECTED_KEY_INDEX_FILE_SHA256,
        "key_index_projection_sha256": EXPECTED_KEY_INDEX_PROJECTION_SHA256,
        "render_manifest_file_sha256": EXPECTED_RENDER_MANIFEST_FILE_SHA256,
        "render_manifest_projection_sha256": (
            EXPECTED_RENDER_MANIFEST_PROJECTION_SHA256
        ),
        "page_payloads_projection_sha256": (
            EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256
        ),
    }
    return FrozenMcqBundle(
        inventory=inventory,
        key_index=key_index,
        render_manifest=render_manifest,
        freeze_manifest_sha256=EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256,
        freeze_manifest_projection_sha256=(
            EXPECTED_FROZEN_BUNDLE_MANIFEST_PROJECTION_SHA256
        ),
        page_payloads_projection_sha256=(
            EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256
        ),
        attestation_projection_sha256=canonical_json_sha256(attestation),
    )


def _evidence_mapping(item: VisualPageEvidence) -> dict[str, Any]:
    value = asdict(item)
    value["mapped_polygon"] = (
        [list(point) for point in item.mapped_polygon]
        if item.mapped_polygon is not None
        else None
    )
    return value


def decide_mcq_page_binding(
    evidences: Sequence[VisualPageEvidence],
    inventory: McqInventory,
    render_manifest: McqRenderManifest,
    key_index: McqKeyIndex,
    *,
    expected_task_image_sha256: str,
    observed_question_number: int,
    thresholds: VisualBindingThresholds = FROZEN_VISUAL_THRESHOLDS,
) -> McqPageDecision:
    """Select one book/page, then bind the prompt number on that page."""

    if thresholds != FROZEN_VISUAL_THRESHOLDS:
        raise VisualCoordinateBindingError(
            "MCQ decision thresholds differ from the exact frozen profile"
        )
    _sha(expected_task_image_sha256, "task image")
    _positive_integer(observed_question_number, "observed question number")
    if not evidences:
        return McqPageDecision(False, "no_page_evidence", ())
    addresses = [(item.document_id, item.page_number) for item in evidences]
    expected_addresses = set(inventory.candidate_pages)
    complete = (
        len(addresses) == len(expected_addresses)
        and len(addresses) == len(set(addresses))
        and set(addresses) == expected_addresses
    )
    identity_ok = complete
    if complete:
        for evidence in evidences:
            document = inventory.document(evidence.document_id)
            rendered = render_manifest.page(evidence.document_id, evidence.page_number)
            if (
                evidence.task_image_sha256 != expected_task_image_sha256
                or evidence.pdf_sha256 != document.pdf_sha256
                or evidence.rendered_page_sha256 != rendered.sha256
            ):
                identity_ok = False
                break
    if not complete or not identity_ok:
        return McqPageDecision(
            False,
            "incomplete_or_foreign_page_evidence",
            (("complete_candidate_page_sweep", complete), ("source_identity", identity_ok)),
        )
    ordered = sorted(
        evidences,
        key=lambda item: (
            -item.rank_score,
            -item.inliers,
            item.document_id,
            item.page_number,
            item.rendered_page_sha256,
        ),
    )
    best = ordered[0]
    runner = ordered[1]
    score_margin = best.rank_score - runner.rank_score
    score_ratio = best.rank_score / max(runner.rank_score, 1e-9)
    checks = list(geometry_checks(best, thresholds))
    checks.extend(
        (
            ("complete_candidate_page_sweep", complete),
            ("source_identity", identity_ok),
            ("page_rank_margin", score_margin >= thresholds.min_rank_score_margin),
            ("page_rank_ratio", score_ratio >= thresholds.min_rank_score_ratio),
        )
    )
    document = inventory.document(best.document_id)
    matches = [
        item
        for item in document.questions
        if item.content_page_number == best.page_number
        and item.question_number == observed_question_number
    ]
    unique_prompt_record = len(matches) == 1
    checks.append(("unique_prompt_number_on_selected_page", unique_prompt_record))
    selected = matches[0] if unique_prompt_record else None
    key_cell: McqKeyCell | None = None
    if selected is not None:
        try:
            key_cell = key_index.cell(selected.record_id)
        except McqSourceError:
            key_cell = None
    source_key_bound = (
        selected is not None
        and selected.source_response_kind == "choice_A-E"
        and key_cell is not None
        and key_cell.document_id == selected.document_id
        and key_cell.unit_number == selected.unit_number
        and key_cell.question_number == selected.question_number
        and key_cell.answer in _CHOICES
    )
    checks.append(("source_key_cell_bound", source_key_bound))
    accepted = all(passed for _, passed in checks)
    return McqPageDecision(
        accepted=accepted,
        reason=("accepted" if accepted else "page_prompt_or_key_binding_failed"),
        checks=tuple(checks),
        selected_document_id=best.document_id,
        selected_source_family=document.source_family,
        selected_page_number=best.page_number,
        selected_unit_number=selected.unit_number if accepted and selected else None,
        selected_question_number=(
            selected.question_number if accepted and selected else None
        ),
        selected_record_id=selected.record_id if accepted and selected else None,
        best_rank_score=best.rank_score,
        runner_rank_score=runner.rank_score,
    )


def _certificate_projection(
    *,
    task_image_sha256: str,
    prompt_sha256: str,
    normalized_prompt_sha256: str,
    observed_question_number: int,
    inventory_projection_sha256: str,
    key_index_projection_sha256: str,
    render_manifest_projection_sha256: str,
    evidences: Sequence[VisualPageEvidence],
    decision: McqPageDecision,
    answer: str | None,
    answer_sha256: str | None,
    selected_key_projection_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": CERTIFICATE_SCHEMA,
        "task_image_sha256": task_image_sha256,
        "prompt_sha256": prompt_sha256,
        "normalized_prompt_sha256": normalized_prompt_sha256,
        "observed_question_number": observed_question_number,
        "inventory_projection_sha256": inventory_projection_sha256,
        "key_index_projection_sha256": key_index_projection_sha256,
        "render_manifest_projection_sha256": render_manifest_projection_sha256,
        "evidences": [_evidence_mapping(item) for item in evidences],
        "decision": asdict(decision),
        "answer": answer,
        "answer_sha256": answer_sha256,
        "selected_key_projection_sha256": selected_key_projection_sha256,
    }


def issue_mcq_source_certificate(
    prompt: str,
    task_image_sha256: str,
    evidences: Sequence[VisualPageEvidence],
    inventory: McqInventory,
    render_manifest: McqRenderManifest,
    key_index: McqKeyIndex,
) -> McqSourceCertificate:
    question_number = parse_observable_mcq_prompt(prompt)
    decision = decide_mcq_page_binding(
        evidences,
        inventory,
        render_manifest,
        key_index,
        expected_task_image_sha256=task_image_sha256,
        observed_question_number=question_number,
    )
    cell = (
        key_index.cell(decision.selected_record_id)
        if decision.accepted and decision.selected_record_id is not None
        else None
    )
    answer = cell.answer if cell is not None else None
    answer_sha = (
        hashlib.sha256(answer.encode("utf-8")).hexdigest()
        if answer is not None
        else None
    )
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    normalized_prompt_sha = hashlib.sha256(
        _normal_prompt(prompt).encode("utf-8")
    ).hexdigest()
    ordered = tuple(
        sorted(
            evidences,
            key=lambda item: (item.document_id, item.page_number),
        )
    )
    projection = _certificate_projection(
        task_image_sha256=task_image_sha256,
        prompt_sha256=prompt_sha,
        normalized_prompt_sha256=normalized_prompt_sha,
        observed_question_number=question_number,
        inventory_projection_sha256=inventory.inventory_projection_sha256,
        key_index_projection_sha256=key_index.key_index_projection_sha256,
        render_manifest_projection_sha256=(
            render_manifest.render_manifest_projection_sha256
        ),
        evidences=ordered,
        decision=decision,
        answer=answer,
        answer_sha256=answer_sha,
        selected_key_projection_sha256=(
            cell.key_projection_sha256 if cell is not None else None
        ),
    )
    return McqSourceCertificate(
        task_image_sha256=task_image_sha256,
        prompt_sha256=prompt_sha,
        normalized_prompt_sha256=normalized_prompt_sha,
        observed_question_number=question_number,
        inventory_projection_sha256=inventory.inventory_projection_sha256,
        key_index_projection_sha256=key_index.key_index_projection_sha256,
        render_manifest_projection_sha256=(
            render_manifest.render_manifest_projection_sha256
        ),
        evidences=ordered,
        decision=decision,
        answer=answer,
        answer_sha256=answer_sha,
        selected_key_projection_sha256=(
            cell.key_projection_sha256 if cell is not None else None
        ),
        certificate_projection_sha256=canonical_json_sha256(projection),
    )


def verify_mcq_source_certificate(
    prompt: str,
    inventory: McqInventory,
    render_manifest: McqRenderManifest,
    key_index: McqKeyIndex,
    certificate: McqSourceCertificate,
    *,
    expected_task_image_bytes: bytes,
) -> McqPageDecision:
    """Replay a certificate against exact source objects and expected image bytes.

    Replay validates recorded evidence and bindings; it does not recompute SIFT.
    A caller that needs fresh visual evidence must call ``resolve_mcq_image_bytes``.
    """

    assert_frozen_mcq_objects(inventory, key_index, render_manifest)
    if not isinstance(expected_task_image_bytes, bytes) or not expected_task_image_bytes:
        raise McqSourceError("expected MCQ task image bytes are empty")
    if (
        hashlib.sha256(expected_task_image_bytes).hexdigest()
        != certificate.task_image_sha256
    ):
        raise McqSourceError("MCQ certificate is bound to different expected image bytes")
    question_number = parse_observable_mcq_prompt(prompt)
    if (
        certificate.prompt_sha256 != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        or certificate.normalized_prompt_sha256
        != hashlib.sha256(_normal_prompt(prompt).encode("utf-8")).hexdigest()
        or certificate.observed_question_number != question_number
        or certificate.inventory_projection_sha256
        != inventory.inventory_projection_sha256
        or certificate.key_index_projection_sha256
        != key_index.key_index_projection_sha256
        or certificate.render_manifest_projection_sha256
        != render_manifest.render_manifest_projection_sha256
    ):
        raise McqSourceError("MCQ certificate input/source pins changed")
    decision = decide_mcq_page_binding(
        certificate.evidences,
        inventory,
        render_manifest,
        key_index,
        expected_task_image_sha256=certificate.task_image_sha256,
        observed_question_number=question_number,
    )
    if decision != certificate.decision:
        raise McqSourceError("MCQ certificate decision does not replay")
    cell = (
        key_index.cell(decision.selected_record_id)
        if decision.accepted and decision.selected_record_id is not None
        else None
    )
    expected_answer = cell.answer if cell is not None else None
    expected_answer_sha = (
        hashlib.sha256(expected_answer.encode("utf-8")).hexdigest()
        if expected_answer is not None
        else None
    )
    expected_key_projection = cell.key_projection_sha256 if cell is not None else None
    if (
        certificate.answer != expected_answer
        or certificate.answer_sha256 != expected_answer_sha
        or certificate.selected_key_projection_sha256 != expected_key_projection
    ):
        raise McqSourceError("MCQ certificate answer/key binding changed")
    projection = _certificate_projection(
        task_image_sha256=certificate.task_image_sha256,
        prompt_sha256=certificate.prompt_sha256,
        normalized_prompt_sha256=certificate.normalized_prompt_sha256,
        observed_question_number=certificate.observed_question_number,
        inventory_projection_sha256=certificate.inventory_projection_sha256,
        key_index_projection_sha256=certificate.key_index_projection_sha256,
        render_manifest_projection_sha256=(
            certificate.render_manifest_projection_sha256
        ),
        evidences=certificate.evidences,
        decision=certificate.decision,
        answer=certificate.answer,
        answer_sha256=certificate.answer_sha256,
        selected_key_projection_sha256=certificate.selected_key_projection_sha256,
    )
    if canonical_json_sha256(projection) != certificate.certificate_projection_sha256:
        raise McqSourceError("MCQ certificate projection hash mismatch")
    return decision


def load_mcq_source_certificate(path: Path) -> McqSourceCertificate:
    raw = _strict_json_object(path)
    expected_fields = {
        "schema_version",
        "task_image_sha256",
        "prompt_sha256",
        "normalized_prompt_sha256",
        "observed_question_number",
        "inventory_projection_sha256",
        "key_index_projection_sha256",
        "render_manifest_projection_sha256",
        "evidences",
        "decision",
        "answer",
        "answer_sha256",
        "selected_key_projection_sha256",
        "certificate_projection_sha256",
    }
    if set(raw) != expected_fields or raw.get("schema_version") != CERTIFICATE_SCHEMA:
        raise McqSourceError("unsupported MCQ certificate schema")
    raw_evidences = raw.get("evidences")
    raw_decision = raw.get("decision")
    if not isinstance(raw_evidences, list) or not isinstance(raw_decision, dict):
        raise McqSourceError("MCQ certificate evidence/decision is malformed")
    expected_decision_fields = set(McqPageDecision.__dataclass_fields__)
    if set(raw_decision) != expected_decision_fields:
        raise McqSourceError("MCQ certificate decision fields changed")
    raw_checks = raw_decision.get("checks")
    if not isinstance(raw_checks, list):
        raise McqSourceError("MCQ certificate checks are malformed")
    checks: list[tuple[str, bool]] = []
    for item in raw_checks:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], bool)
        ):
            raise McqSourceError("MCQ certificate check is malformed")
        checks.append((item[0], item[1]))
    if not isinstance(raw_decision["accepted"], bool) or not isinstance(
        raw_decision["reason"], str
    ):
        raise McqSourceError("MCQ certificate decision status is malformed")
    decision = McqPageDecision(
        accepted=raw_decision["accepted"],
        reason=raw_decision["reason"],
        checks=tuple(checks),
        selected_document_id=raw_decision["selected_document_id"],
        selected_source_family=raw_decision["selected_source_family"],
        selected_page_number=raw_decision["selected_page_number"],
        selected_unit_number=raw_decision["selected_unit_number"],
        selected_question_number=raw_decision["selected_question_number"],
        selected_record_id=raw_decision["selected_record_id"],
        best_rank_score=_finite_number(
            raw_decision["best_rank_score"], "certificate best rank score"
        ),
        runner_rank_score=_finite_number(
            raw_decision["runner_rank_score"], "certificate runner rank score"
        ),
    )
    certificate = McqSourceCertificate(
        task_image_sha256=_sha(raw["task_image_sha256"], "certificate image"),
        prompt_sha256=_sha(raw["prompt_sha256"], "certificate prompt"),
        normalized_prompt_sha256=_sha(
            raw["normalized_prompt_sha256"], "certificate normalized prompt"
        ),
        observed_question_number=_positive_integer(
            raw["observed_question_number"], "certificate question number"
        ),
        inventory_projection_sha256=_sha(
            raw["inventory_projection_sha256"], "certificate inventory"
        ),
        key_index_projection_sha256=_sha(
            raw["key_index_projection_sha256"], "certificate key index"
        ),
        render_manifest_projection_sha256=_sha(
            raw["render_manifest_projection_sha256"], "certificate render manifest"
        ),
        evidences=tuple(
            visual_page_evidence_from_mapping(item) for item in raw_evidences
        ),
        decision=decision,
        answer=raw["answer"],
        answer_sha256=raw["answer_sha256"],
        selected_key_projection_sha256=raw["selected_key_projection_sha256"],
        certificate_projection_sha256=_sha(
            raw["certificate_projection_sha256"], "certificate projection"
        ),
    )
    if certificate.to_mapping() != raw:
        raise McqSourceError("MCQ certificate fields/order changed")
    return certificate


def resolve_mcq_image_bytes(
    prompt: str,
    image_bytes: bytes,
    inventory: McqInventory,
    render_manifest: McqRenderManifest,
    key_index: McqKeyIndex,
    *,
    thresholds: VisualBindingThresholds = FROZEN_VISUAL_THRESHOLDS,
    runtime_profile: SiftRuntimeProfile = FROZEN_SIFT_RUNTIME_PROFILE,
) -> McqSourceCertificate:
    """Sweep all 28 pinned pages and issue an answer-bound certificate."""

    assert_frozen_mcq_objects(inventory, key_index, render_manifest)
    _require_frozen_profile(thresholds, runtime_profile)
    assert_mcq_runtime(require_visual=True)
    parse_observable_mcq_prompt(prompt)
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise McqSourceError("MCQ task image bytes are empty")
    if any(item.resolved_path is None for item in render_manifest.pages):
        raise McqSourceError("render payload paths were not verified at load time")
    task_sha = hashlib.sha256(image_bytes).hexdigest()
    evidences: list[VisualPageEvidence] = []
    with tempfile.TemporaryDirectory(prefix="mcq_fullpage_source_") as raw_temp:
        task_path = Path(raw_temp) / "task.png"
        task_path.write_bytes(image_bytes)
        for rendered in render_manifest.pages:
            document = inventory.document(rendered.document_id)
            assert rendered.resolved_path is not None
            evidences.append(
                compute_sift_page_evidence(
                    task_path,
                    rendered.resolved_path,
                    task_image_sha256=task_sha,
                    document_id=document.document_id,
                    pdf_sha256=document.pdf_sha256,
                    page_number=rendered.page_number,
                    profile=runtime_profile,
                )
            )
    certificate = issue_mcq_source_certificate(
        prompt,
        task_sha,
        evidences,
        inventory,
        render_manifest,
        key_index,
    )
    verify_mcq_source_certificate(
        prompt,
        inventory,
        render_manifest,
        key_index,
        certificate,
        expected_task_image_bytes=image_bytes,
    )
    return certificate


def write_canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
