"""Семантический гейт: LLM отбирает из выдачи фрагменты, где есть запрошенное.

Ходит в тот же vLLM, что обслуживает агента, одним запросом на весь список.
Порог из confidence.py ему не нужен, а кросс-энкодер перед ним обязателен: гейт
видит только top-k. Fail-closed — ошибка сети, нечитаемый ответ или пустой keep
скрывают выдачу целиком.

Без этих переменных гейт выключен и оценкой занимается confidence.py:
    RETRIEVE_GATE_URL=http://127.0.0.1:8000/v1
    RETRIEVE_GATE_MODEL=Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from schemas.retrieve import RetrievedChunk

from .confidence import Relevance, RelevanceVerdict

DEFAULT_GATE_MODEL = "Qwen/Qwen3.5-9B"
GATE_URL_ENV = "RETRIEVE_GATE_URL"
GATE_MODEL_ENV = "RETRIEVE_GATE_MODEL"
MAX_FRAGMENT_CHARS = 1200

SYSTEM_PROMPT = (
    "You filter search results for a school-task solver. You are given what the "
    "solver asked for and the fragments a textbook search returned. Keep only "
    "fragments that contain what was asked for. Answer with JSON only."
)

USER_TEMPLATE = """The solver asked for: {query}

Fragments:
{fragments}

Keep a fragment only if it actually contains the requested rule, formula,
definition, worked example, or exercise. Drop fragments that merely touch the
same subject, mention the topic in passing, or are contents pages, prefaces,
exercise lists without solutions.

Reply with JSON: {{"keep": [numbers of kept fragments], "reason": "one short sentence"}}
An empty list is a valid and expected answer when nothing fits."""


class ChatBackend(Protocol):
    """Минимум от LLM: два промпта на входе, текст на выходе."""

    def ask(self, system_prompt: str, user_prompt: str) -> str: ...


class VLLMBackend:
    """Тот же сервер, что у агента; HTTP-часть переиспользуем из vlm_judge."""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        from vlm_judge.backends import OpenAICompatibleBackend

        self.model = model
        self._backend = OpenAICompatibleBackend(
            base_url,
            model,
            timeout=timeout,
            max_tokens=200,
            enable_thinking=False,  # гейт на горячем пути, рассуждения тут только замедляют
        )

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        from vlm_judge.prompts import JudgeRequest

        request = JudgeRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_urls=(),
        )
        return self._backend.complete(request).text


def _render(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{number}] {chunk.text.strip()[:MAX_FRAGMENT_CHARS]}"
        for number, chunk in enumerate(chunks, start=1)
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Модель любит обернуть JSON в ```json или в пояснение — берём объект целиком."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("в ответе нет JSON-объекта")
    payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("ожидался JSON-объект")
    return payload


def _kept_numbers(payload: dict[str, Any], total: int) -> list[int]:
    raw = payload.get("keep")
    if not isinstance(raw, list):
        raise ValueError("нет поля keep со списком номеров")
    numbers: list[int] = []
    for value in raw:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue  # «фрагмент 2» вместо 2 — пункт теряем, это в сторону fail-closed
        if 1 <= number <= total and number not in numbers:
            numbers.append(number)
    return numbers


def _reason(payload: dict[str, Any]) -> str:
    return str(payload.get("reason") or "без пояснения")[:200]


class SemanticGate:
    def __init__(self, backend: ChatBackend) -> None:
        self.backend = backend

    def judge(
            self,
            query: str,
            chunks: list[RetrievedChunk],
    ) -> tuple[list[RetrievedChunk], RelevanceVerdict]:
        """Возвращает отобранные фрагменты и вердикт в том же виде, что порог."""
        if not chunks:
            return [], RelevanceVerdict(Relevance.EMPTY, None, "выдача пуста")
        prompt = USER_TEMPLATE.format(query=query.strip(), fragments=_render(chunks))
        try:
            payload = _extract_json(self.backend.ask(SYSTEM_PROMPT, prompt))
            numbers = _kept_numbers(payload, len(chunks))
        except Exception as exc:  # сеть, таймаут, мусор в ответе — всё в fail-closed
            return [], RelevanceVerdict(Relevance.ERROR, None, f"гейт недоступен: {exc}")
        if not numbers:
            return [], RelevanceVerdict(
                Relevance.WEAK,
                chunks[0].score,
                f"гейт отклонил всю выдачу: {_reason(payload)}",
            )
        kept = [chunks[number - 1] for number in numbers]
        return kept, RelevanceVerdict(
            Relevance.CONFIDENT,
            kept[0].score,
            f"гейт оставил {len(kept)} из {len(chunks)}: {_reason(payload)}",
        )


class GateRanker:
    """Гейт как ступень пайплайна — чтобы мерить его наравне с ранкерами.
    В отличие от прод-обвязки в service.py вердикт не возвращает, только выдачу.

    Одобренные идут наверх, отвергнутые следом в прежнем порядке: состав выдачи
    не меняется, только порядок, иначе проседают метрики на глубине. top_n
    ограничивает промпт, пайплайн передаёт ранкерам всю выдачу (fetch_k=200).
    """

    def __init__(self, gate: SemanticGate | None = None, top_n: int = 10) -> None:
        self.gate = gate or gate_from_env()
        self.top_n = top_n

    def rank(
            self,
            query: str,
            chunks: list[RetrievedChunk] | None = None,
            subject: str | None = None,
            grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        del subject, grade
        if not chunks:
            return []
        if self.gate is None:
            raise RuntimeError(f"гейт не настроен: выставьте {GATE_URL_ENV}")
        head = chunks[: self.top_n]
        kept, _ = self.gate.judge(query, head)
        approved = {chunk.chunk_id for chunk in kept}
        return kept + [chunk for chunk in chunks if chunk.chunk_id not in approved]


def gate_from_env() -> SemanticGate | None:
    """None — значит гейт не настроен и оценкой занимается confidence.assess_relevance."""
    base_url = os.environ.get(GATE_URL_ENV, "").strip()
    if not base_url:
        return None
    return SemanticGate(VLLMBackend(base_url, os.environ.get(GATE_MODEL_ENV, DEFAULT_GATE_MODEL)))
