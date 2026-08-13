"""Content-only medium-reasoning candidate for the public 185-row DEV set.

The runtime receives only subject/question/choices/has_figure.  Opaque task
identifiers, prior predictions, answer keys, retrieval hits, and benchmark
outcomes are outside this module's contract.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Mapping


MODEL_ID = "qwen/qwen3.5-9b"
PROVIDER_NAME = "SiliconFlow"
PROVIDER_SLUG = "siliconflow"
PROVIDER_QUANTIZATION = "fp8"
PROVIDER_ROUTED_MODEL_ID = "qwen/qwen3.5-9b-20260310"
MAX_TOKENS = 32768
REASONING = {"effort": "medium", "exclude": True}
CHOICE_LABELS = tuple("ABCDE")


class CandidateError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    if type(row) is not dict:
        raise CandidateError("public row must be an object")
    if type(row.get("subject")) is not str or not row["subject"].strip():
        raise CandidateError("subject contract mismatch")
    if type(row.get("question")) is not str or not row["question"].strip():
        raise CandidateError("question contract mismatch")
    if type(row.get("has_figure")) is not bool or row["has_figure"] is not False:
        raise CandidateError("candidate is text-only and requires has_figure=false")
    choices = row.get("choices")
    if (
        type(choices) is not dict
        or set(choices) != set(CHOICE_LABELS)
        or any(type(choices[label]) is not str or not choices[label].strip() for label in CHOICE_LABELS)
    ):
        raise CandidateError("five-choice contract mismatch")
    return {
        "subject": row["subject"],
        "question": row["question"],
        "choices": {label: choices[label] for label in CHOICE_LABELS},
        "has_figure": False,
    }


def content_digest(row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(content_projection(row)))


def theory_projection(row: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = row.get("theory")
    if type(raw) is not list or len(raw) > 2:
        raise CandidateError("theory support contract mismatch")
    result: list[dict[str, str]] = []
    for item in raw:
        if type(item) is not dict or set(item) != {"text", "text_sha256", "title"}:
            raise CandidateError("theory item schema mismatch")
        title, text, expected = item["title"], item["text"], item["text_sha256"]
        if type(title) is not str or not title.strip() or type(text) is not str or not text.strip():
            raise CandidateError("empty theory support")
        if type(expected) is not str or sha256_bytes(text.encode("utf-8")) != expected:
            raise CandidateError("theory text hash mismatch")
        result.append({"title": title, "text": text, "text_sha256": expected})
    expected_projection = row.get("theory_sha256")
    actual_projection = sha256_bytes(canonical_json_bytes(result))
    if expected_projection != actual_projection:
        raise CandidateError("theory projection hash mismatch")
    return result


def theory_support_text(row: Mapping[str, Any]) -> str:
    theory = theory_projection(row)
    if not theory:
        return "(yok)"
    return "\n\n".join(
        f"[TEORİ {index}; {item['title']}]\n{item['text']}"
        for index, item in enumerate(theory, 1)
    )


def content_seed(row: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(
        b"generic-medium-sse-alt-v1\0" + bytes.fromhex(content_digest(row))
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def option_aliases(row: Mapping[str, Any]) -> dict[str, str]:
    originals = list(CHOICE_LABELS)
    random.Random(content_seed(row)).shuffle(originals)
    return dict(zip(CHOICE_LABELS, originals))


_IDENTITY_MARKER = re.compile(
    r"(?:\b(?:benchmark|task|queue|source)[_-]?id\b|"
    r"\b(?:yh|val)_[0-9a-z_-]{4,}\b)",
    re.IGNORECASE,
)


def assert_request_blind(request: Mapping[str, Any]) -> None:
    if type(request) is not dict:
        raise CandidateError("request must be an object")

    def walk(value: Any) -> None:
        if type(value) is dict:
            for key, child in value.items():
                if _IDENTITY_MARKER.search(str(key)):
                    raise CandidateError("opaque identity marker leaked into request key")
                walk(child)
        elif type(value) is list:
            for child in value:
                walk(child)
        elif type(value) is str and _IDENTITY_MARKER.search(value):
            raise CandidateError("opaque identity marker leaked into request text")

    walk(request)


def _answer_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": list(CHOICE_LABELS)}},
        "required": ["answer"],
        "additionalProperties": False,
    }


def build_request(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    public = content_projection(row)
    support = theory_support_text(row)
    aliases = option_aliases(public)
    rendered_choices = "\n".join(
        f"{alias}) {public['choices'][original]}"
        for alias, original in aliases.items()
    )
    request = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Türkçe lise düzeyindeki çoktan seçmeli soruyu yalnız verilen içerikten çöz. "
                    "Yalnız soruyu ve istek içinde verilen genel teori alıntılarını kullan; "
                    "görev kimliği, cevap anahtarı, önceki tahmin, çözülmüş örnek veya soru "
                    "veritabanı kullanma. İç muhakemede önce soru kutbunu (doğru/yanlış/değildir) "
                    "belirle; sonra beş seçeneğin her birini bağımsız denetle. Nicel sorularda "
                    "denklem, tanım alanı, işaret ve birimleri; fen sorularında mekanizma ve sınır "
                    "durumlarını; sözel sorularda kapsam, metin kanıtı ve çeldiricileri kontrol et. "
                    "Son cevabında açıklama yazma; yalnız şemaya uyan JSON döndür."
                ),
            },
            {
                "role": "user",
                "content": (
                    "GENEL TEORİ DESTEĞİ (çözülmüş örnek değildir; ilgisizse yok say):\n"
                    f"{support}\n\n---\n"
                    f"Ders: {public['subject']}\n\n"
                    f"Soru:\n{public['question']}\n\n"
                    f"Seçenekler (bu istek için yeniden harflendirildi):\n{rendered_choices}\n\n"
                    "Tüm seçenekleri içten karşılaştır ve yalnız {\"answer\":\"A\"} biçiminde "
                    "tek nihai seçenek döndür."
                ),
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": MAX_TOKENS,
        "reasoning": dict(REASONING),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "generic_medium_answer_alt_v1",
                "strict": True,
                "schema": _answer_schema(),
            },
        },
        "provider": {
            "only": [PROVIDER_SLUG],
            "allow_fallbacks": False,
            "require_parameters": True,
            "quantizations": [PROVIDER_QUANTIZATION],
            "data_collection": "deny",
            "zdr": True,
        },
        "stream": False,
    }
    assert_request_blind(request)
    return request, aliases


def validate_model_content(content: str) -> str:
    if type(content) is not str or not content.strip():
        raise CandidateError("model content is empty")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise CandidateError("model content is not JSON") from exc
    if type(value) is not dict or set(value) != {"answer"}:
        raise CandidateError("model answer schema mismatch")
    answer = value.get("answer")
    if answer not in CHOICE_LABELS:
        raise CandidateError("model answer is outside A-E")
    return answer


def map_model_answer(content: str, aliases: Mapping[str, str]) -> str:
    if type(aliases) is not dict or set(aliases) != set(CHOICE_LABELS):
        raise CandidateError("alias map schema mismatch")
    if set(aliases.values()) != set(CHOICE_LABELS):
        raise CandidateError("alias map is not a permutation")
    return aliases[validate_model_content(content)]


def fixed_smoke_row() -> dict[str, Any]:
    """Exact fixed non-benchmark row shared by freeze and live smoke."""

    theory: list[dict[str, str]] = []
    return {
        "subject": "Genel",
        "question": "İki ile ikinin toplamı kaçtır?",
        "choices": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        "has_figure": False,
        "theory": theory,
        "theory_sha256": sha256_bytes(canonical_json_bytes(theory)),
    }
