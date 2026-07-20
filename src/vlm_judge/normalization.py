from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction


_FINAL_ANSWER_PATTERNS = (
    re.compile(
        r"(?is)(?:^|\n)\s*#{0,6}\s*(?:final\s+answer|answer|ответ|cevap)"
        r"\s*(?::|\-)?\s*(.+)$"
    ),
    re.compile(
        r"(?is)(?:the\s+)?(?:final\s+answer|answer|ответ|cevap)\s+is\s*[:\-]?\s*(.+)$"
    ),
    re.compile(r"(?is)\\boxed\s*\{([^{}]+)\}"),
)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.casefold().strip()
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"[`*_#]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .;,:!?")


def extract_final_answer(value: str) -> str:
    for pattern in _FINAL_ANSWER_PATTERNS:
        matches = list(pattern.finditer(value))
        if matches:
            return matches[-1].group(1).strip()
    return value.strip()


def normalize_multiple_choice(value: str) -> str | None:
    final = extract_final_answer(value)
    patterns = (
        r"(?i)^\s*\(?([A-E])\)?(?:\s*[).:\-]|\s*$)",
        r"(?i)(?:the\s+)?(?:correct\s+|final\s+)?answer\s+is\s*[:\-]?\s*\(?([A-E])\)?",
        r"(?i)(?:answer|ответ|cevap)\s*[:\-]?\s*\(?([A-E])\)?",
        r"(?i)\boption\s+([A-E])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, final)
        if match:
            return match.group(1).upper()
    return None


def parse_numeric(value: str) -> Fraction | None:
    final = normalize_text(extract_final_answer(value))
    final = final.replace(" ", "")
    fraction_match = re.fullmatch(r"([+-]?\d+)\/([+-]?\d+)(?:[a-zа-яçğıöşü²³]+)?", final)
    if fraction_match:
        denominator = int(fraction_match.group(2))
        if denominator == 0:
            return None
        return Fraction(int(fraction_match.group(1)), denominator)
    number_match = re.fullmatch(r"([+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+))(?:[a-zа-яçğıöşü²³]+)?", final)
    if not number_match:
        return None
    try:
        number = Decimal(number_match.group(1).replace(",", "."))
    except InvalidOperation:
        return None
    return Fraction(number)
