"""Generate short Turkish retrieval queries for page-bundle fine-tuning."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from .manifest import clean_answer_text

QUERY_SYSTEM_PROMPT = """\
Sen Türkçe ders kitabı arama asistanısın.
Görevin: verilen sayfa çözüm metninden kısa arama sorguları üretmek.
Kurallar:
- Sorgular Türkçe olsun.
- Her sorgu kısa olsun (3-12 kelime): konu + işlem/kavram/ayırt edici terim.
- Sorunun veya çözümün tamamını kopyalama.
- Nihai cevabı (şık harfi / tek sayı) sorguya yazma.
- Öğrenci bir ders kitabı arama motoruna yazacakmış gibi yaz.
- JSON dışında hiçbir şey yazma.
"""

QUERY_USER_TEMPLATE = """\
Sınıf: {grade}
Ders: {subject}
Sayfa metni / çözüm:
---
{answer_text}
---
Bu sayfa için {n_queries} kısa arama sorgusu üret.
JSON formatı:
{{"queries":["...","..."]}}
"""


def heuristic_queries(bundle: dict[str, Any], *, n_queries: int = 3) -> list[str]:
    """Cheap offline queries from slug + answer keywords (no LLM)."""

    text = clean_answer_text(str(bundle.get("answer_text") or ""), max_chars=1_500)
    grade = bundle.get("grade")
    subject = bundle.get("subject") or ""
    book = str(bundle.get("book_slug") or "")

    subject_tr = {
        "math": "matematik",
        "science": "fen bilimleri",
        "english": "ingilizce",
        "turkish language and literature": "türkçe",
        "life sciences": "hayat bilgisi",
        "social studies": "sosyal bilgiler",
        "religious culture and ethics": "din kültürü",
        "history": "inkılap tarihi",
    }.get(str(subject), str(subject).replace("_", " "))

    seeds: list[str] = []
    if grade and subject_tr:
        seeds.append(f"{grade}. sınıf {subject_tr}")

    # Pull distinctive content-ish tokens from the answer.
    lowered = text.casefold()
    keyword_patterns = [
        r"\b(kesir|payda|pay|çarpma|bölme|toplama|çıkarma|alan|çevre|"
        r"üçgen|dikdörtgen|fotosentez|hücre|kuvvet|enerji|osmanlı|"
        r"atatürk|cumhuriyet|lozan|denklem|yüzde|açı|hacim)\b",
    ]
    found: list[str] = []
    for pattern in keyword_patterns:
        found.extend(re.findall(pattern, lowered))
    for token in dict.fromkeys(found):
        if grade and subject_tr:
            seeds.append(f"{grade}. sınıf {subject_tr} {token}")
        else:
            seeds.append(token)

    # First informative line / "Soru" snippet.
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 20:
            continue
        if line.casefold().startswith(("merhaba", "sevgili", "harika")):
            continue
        clipped = re.sub(r"\s+", " ", line)
        if len(clipped) > 80:
            clipped = clipped[:80].rsplit(" ", 1)[0]
        seeds.append(clipped)
        break

    if not seeds and book:
        seeds.append(book.replace("-", " ")[:80])

    # Dedup preserve order.
    unique: list[str] = []
    seen: set[str] = set()
    for query in seeds:
        key = query.casefold().strip()
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        unique.append(query.strip())
        if len(unique) >= n_queries:
            break
    return unique


class OpenAICompatibleQueryGenerator:
    """Generate queries via vLLM / OpenRouter / any OpenAI-compatible chat API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 120.0,
        temperature: float = 0.3,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("MLA_VLLM_BASE_URL")
            or "http://localhost:8000/v1"
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("MLA_VLLM_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or "EMPTY"
        )
        self.model = model or os.environ.get("MLA_MODEL_NAME") or "Qwen/Qwen3.5-9B"
        self.timeout_s = timeout_s
        self.temperature = temperature

    def generate(
        self,
        bundle: dict[str, Any],
        *,
        n_queries: int = 3,
    ) -> list[str]:
        answer_text = clean_answer_text(str(bundle.get("answer_text") or ""), max_chars=2_500)
        if len(answer_text) < 40:
            return heuristic_queries(bundle, n_queries=n_queries)

        user = QUERY_USER_TEMPLATE.format(
            grade=bundle.get("grade") or "?",
            subject=bundle.get("subject") or "?",
            answer_text=answer_text,
            n_queries=n_queries,
        )
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_queries_json(content, n_queries=n_queries) or heuristic_queries(
            bundle, n_queries=n_queries
        )


def _parse_queries_json(content: str, *, n_queries: int) -> list[str]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    raw = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    queries: list[str] = []
    seen: set[str] = set()
    for item in raw:
        query = str(item or "").strip()
        key = query.casefold()
        if len(query) < 3 or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= n_queries:
            break
    return queries


def neighbor_hard_negatives(
    bundle: dict[str, Any],
    bundles_by_book: dict[str, list[dict[str, Any]]],
    *,
    k: int = 4,
) -> list[str]:
    book = str(bundle["book_slug"])
    page = int(bundle["page_number"])
    siblings = bundles_by_book.get(book) or []
    ranked = sorted(
        (
            sibling
            for sibling in siblings
            if sibling["page_id"] != bundle["page_id"] and sibling.get("page_image")
        ),
        key=lambda row: abs(int(row["page_number"]) - page),
    )
    return [str(row["page_id"]) for row in ranked[:k]]


def generate_training_rows(
    bundles: list[dict[str, Any]],
    *,
    mode: str = "heuristic",
    n_queries: int = 3,
    hard_negatives_k: int = 4,
    workers: int = 4,
    require_page_image: bool = True,
    llm: OpenAICompatibleQueryGenerator | None = None,
) -> list[dict[str, Any]]:
    usable = [
        bundle
        for bundle in bundles
        if bundle.get("has_solution")
        and bundle.get("answer_text")
        and (bundle.get("page_image") or not require_page_image)
    ]
    by_book: dict[str, list[dict[str, Any]]] = {}
    for bundle in bundles:
        by_book.setdefault(str(bundle["book_slug"]), []).append(bundle)

    generator = llm or OpenAICompatibleQueryGenerator()

    def _one(bundle: dict[str, Any]) -> list[dict[str, Any]]:
        if mode == "llm":
            queries = generator.generate(bundle, n_queries=n_queries)
        else:
            queries = heuristic_queries(bundle, n_queries=n_queries)
        negatives = neighbor_hard_negatives(
            bundle, by_book, k=hard_negatives_k
        )
        rows: list[dict[str, Any]] = []
        for query in queries:
            rows.append(
                {
                    "query": query,
                    "positive_page_id": bundle["page_id"],
                    "positive_image": bundle.get("page_image"),
                    "positive_answer_text": bundle.get("answer_text"),
                    "hard_negative_page_ids": negatives,
                    "subject": bundle.get("subject"),
                    "grade": bundle.get("grade"),
                    "book_slug": bundle.get("book_slug"),
                    "source": f"query_gen:{mode}",
                }
            )
        return rows

    results: list[dict[str, Any]] = []
    if mode == "llm" and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_one, bundle) for bundle in usable]
            for future in as_completed(futures):
                results.extend(future.result())
    else:
        for bundle in usable:
            results.extend(_one(bundle))
    return results
