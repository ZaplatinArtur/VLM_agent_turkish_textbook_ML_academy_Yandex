"""Normalize retrieval metadata found in textbook filenames.

The parser corpus predates the shared retrieval contract: most JSONL chunks
contain only ``textbook`` and ``page``.  Agent tool calls, however, can filter
by an English or Turkish subject and by grade.  This module derives those
fields from slugs such as ``7-sinif-matematik-...`` and provides one canonical
comparison boundary.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from schemas.retrieve import RetrievedChunk


_SUBJECT_BY_SLUG = {
    "din-kulturu-ve-ahlak-bilgisi": "religious culture and ethics",
    "turk-dili-ve-edebiyati": "turkish language and literature",
    "fen-bilimleri": "science",
    "hayat-bilgisi": "life sciences",
    "sosyal-bilgiler": "social studies",
    "inkilap-tarihi": "history",
    "matematik": "math",
    "ingilizce": "english",
    "turkce": "turkish language and literature",
    "fizik": "physics",
    "kimya": "chemistry",
    "biyoloji": "biology",
    "cografya": "geography",
    "sosyoloji": "sociology",
    "felsefe": "philosophy",
    "tarih": "history",
}

_ALIASES = {
    "math": "math",
    "mathematics": "math",
    "matematik": "math",
    "science": "science",
    "fen bilimleri": "science",
    "english": "english",
    "ingilizce": "english",
    "turkish": "turkish language and literature",
    "turkce": "turkish language and literature",
    "turkish language and literature": "turkish language and literature",
    "turk dili ve edebiyati": "turkish language and literature",
    "life science": "life sciences",
    "life sciences": "life sciences",
    "hayat bilgisi": "life sciences",
    "social studies": "social studies",
    "sosyal bilgiler": "social studies",
    "religious culture and ethics": "religious culture and ethics",
    "din kulturu ve ahlak bilgisi": "religious culture and ethics",
    "history": "history",
    "tarih": "history",
    "inkilap tarihi": "history",
    "ataturkculuk": "history",
    "physics": "physics",
    "fizik": "physics",
    "chemistry": "chemistry",
    "kimya": "chemistry",
    "biology": "biology",
    "biyoloji": "biology",
    "geography": "geography",
    "cografya": "geography",
    "sociology": "sociology",
    "sosyoloji": "sociology",
    "philosophy": "philosophy",
    "felsefe": "philosophy",
}

_GRADE_PREFIX = re.compile(r"^(?P<grade>\d+)-sinif-")


def _plain(value: str) -> str:
    translated = value.casefold().translate(
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
    normalized = unicodedata.normalize("NFKD", translated)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()


def canonical_subject(value: Any) -> str | None:
    """Return a stable English subject label for Turkish or English input."""

    if value is None:
        return None
    plain = _plain(str(value))
    return _ALIASES.get(plain, plain or None)


def subjects_match(left: Any, right: Any) -> bool:
    left_subject = canonical_subject(left)
    right_subject = canonical_subject(right)
    return left_subject is not None and left_subject == right_subject


def infer_textbook_metadata(textbook: str) -> dict[str, Any]:
    """Derive grade and subject from a normalized textbook slug."""

    slug = textbook.casefold()
    inferred: dict[str, Any] = {}
    grade_match = _GRADE_PREFIX.match(slug)
    if grade_match:
        inferred["grade"] = int(grade_match.group("grade"))
    for subject_slug, canonical in _SUBJECT_BY_SLUG.items():
        if f"-{subject_slug}-" in f"-{slug}-":
            inferred["subject"] = canonical
            break
    return inferred


def enrich_chunk_metadata(chunk: RetrievedChunk) -> RetrievedChunk:
    """Fill contract fields without overwriting explicit parser metadata."""

    metadata = dict(chunk.metadata)
    textbook = str(metadata.get("textbook") or chunk.chunk_id.split(":", 1)[0])
    metadata.setdefault("textbook", textbook)
    inferred = infer_textbook_metadata(textbook)
    if metadata.get("subject") in (None, "") and inferred.get("subject") is not None:
        metadata["subject"] = inferred["subject"]
    elif metadata.get("subject") not in (None, ""):
        metadata["subject"] = canonical_subject(metadata["subject"])
    if metadata.get("grade") in (None, "") and inferred.get("grade") is not None:
        metadata["grade"] = inferred["grade"]
    return chunk if metadata == chunk.metadata else chunk.model_copy(update={"metadata": metadata})
