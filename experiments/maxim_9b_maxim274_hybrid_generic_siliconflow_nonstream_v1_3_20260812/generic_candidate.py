"""Single-policy, ID-free Maxim274 fallback candidate for Hybrid V3.1."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

MODEL_ID = "qwen/qwen3.5-9b"
PROVIDER_NAME = "SiliconFlow"
PROVIDER_SLUG = "siliconflow"
PROVIDER_QUANTIZATION = "fp8"
PROVIDER_ROUTED_MODEL_ID = "qwen/qwen3.5-9b-20260310"
MAX_TOKENS = 32768
REASONING = {"effort": "medium", "exclude": True}
ANSWER_TYPES = {"choice", "numeric", "short_text", "free_form"}


class CandidateError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "subject", "answer_type", "ocr_text", "source_input_mode"}
    if type(row) is not dict or set(row) != required or row.get("schema_version") != "maxim256-idfree-ocr-row-v1":
        raise CandidateError("content-only queue row schema mismatch")
    if type(row.get("subject")) is not str or not row["subject"].strip():
        raise CandidateError("subject mismatch")
    if row.get("answer_type") not in ANSWER_TYPES:
        raise CandidateError("answer type mismatch")
    if type(row.get("ocr_text")) is not str or not row["ocr_text"].strip():
        raise CandidateError("OCR text is empty")
    if row.get("source_input_mode") not in {"text_only", "multimodal_degraded_to_ocr_only"}:
        raise CandidateError("source input mode mismatch")
    return {key: row[key] for key in ("subject", "answer_type", "ocr_text", "source_input_mode")}


_IDENTITY = re.compile(r"(?:\b(?:task|benchmark|queue|controller|source)[_-]?id\b|\b(?:val|yh)_[0-9a-z_-]{4,}\b)", re.I)
_FORBIDDEN_KEYS = {"task_id", "benchmark_id", "queue_id", "controller_id", "content_sha256", "image_sha256", "gold", "reference_answer", "answer_key"}


def assert_request_blind(request: Mapping[str, Any]) -> None:
    def walk(value: Any) -> None:
        if type(value) is dict:
            for key, child in value.items():
                if str(key).casefold() in _FORBIDDEN_KEYS or _IDENTITY.search(str(key)):
                    raise CandidateError("identity/reference key leaked to wire")
                walk(child)
        elif type(value) is list:
            for child in value:
                walk(child)
        elif type(value) is str and _IDENTITY.search(value):
            raise CandidateError("identity marker leaked to wire")
    walk(request)


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "minLength": 1, "maxLength": 1000},
            "option_label": {"type": "string", "enum": ["A", "B", "C", "D", "E", "NA"]},
        },
        "required": ["answer", "option_label"],
        "additionalProperties": False,
    }


def _answer_rule(answer_type: str) -> str:
    if answer_type == "choice":
        return "Görünen A-E seçeneklerinden birini option_label alanında seç. answer alanına seçeneğin kısa metnini yaz."
    if answer_type == "numeric":
        return "option_label alanı NA olsun; answer alanına yalnız istenen sayıyı, gerekiyorsa birimiyle yaz."
    if answer_type == "short_text":
        return "option_label alanı NA olsun; answer alanına boşluğu doğrudan dolduran en kısa cevabı yaz."
    return "option_label alanı NA olsun; answer alanına sorunun istediği özlü nihai cevabı yaz."


def build_request(row: Mapping[str, Any]) -> tuple[dict[str, Any], None]:
    public = content_projection(row)
    request = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Türkçe lise düzeyindeki soruyu yalnız görünür içerikten bağımsız çöz. Görev kimliği, cevap anahtarı, "
                    "soru veritabanı, önceki tahmin veya çözülmüş örnek kullanma. Önce soru kutbunu ve istenen cevap "
                    "biçimini belirle; sonra verilen ifadeleri ve varsa A-E seçeneklerini ayrı ayrı denetle. Nicel sorularda "
                    "denklem, tanım alanı, işaret ve birimleri; fen sorularında mekanizma ve sınır durumlarını; sözel "
                    "sorularda metin kanıtı, kapsam ve çeldiricileri kontrol et. Sonunda yalnız şemaya uyan JSON döndür."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Ders: {public['subject']}\nCevap türü: {public['answer_type']}\n\n"
                    "Dondurulmuş OCR (hata içerebilir; yalnız görünür metne dayan):\n"
                    f"{public['ocr_text']}\n\nYanıt kuralı: {_answer_rule(public['answer_type'])}"
                ),
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": MAX_TOKENS,
        "reasoning": dict(REASONING),
        "response_format": {"type": "json_schema", "json_schema": {"name": "maxim256_direct_answer_v1", "strict": True, "schema": _schema()}},
        "provider": {"only": [PROVIDER_SLUG], "allow_fallbacks": False, "require_parameters": True, "quantizations": [PROVIDER_QUANTIZATION], "data_collection": "deny", "zdr": True},
        "stream": False,
    }
    assert_request_blind(request)
    return request, None


def validate_model_content(content: str, answer_type: str = "choice") -> dict[str, str]:
    if answer_type not in ANSWER_TYPES or type(content) is not str:
        raise CandidateError("model content contract mismatch")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise CandidateError("model content is not JSON") from exc
    if type(value) is not dict or set(value) != {"answer", "option_label"}:
        raise CandidateError("model answer schema mismatch")
    answer, label = value["answer"], value["option_label"]
    if type(answer) is not str or not answer.strip() or len(answer) > 1000 or label not in {"A", "B", "C", "D", "E", "NA"}:
        raise CandidateError("model answer value mismatch")
    if (answer_type == "choice") is not (label in "ABCDE"):
        raise CandidateError("answer type/option label mismatch")
    return {"answer": answer.strip(), "option_label": label}


def fixed_smoke_row() -> dict[str, Any]:
    return {"schema_version": "maxim256-idfree-ocr-row-v1", "subject": "Genel", "answer_type": "choice", "ocr_text": "İki ile ikinin toplamı kaçtır? A) 1 B) 2 C) 3 D) 4 E) 5", "source_input_mode": "text_only"}
