from __future__ import annotations

import json
import re
from typing import Any

import requests
from pydantic import BaseModel, Field

from schemas.retrieve import RetrievedChunk

from .educational import UnitKind


class RefinementDecision(BaseModel):
    index: int = Field(ge=0)
    kind: UnitKind
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = Field(default="not provided", max_length=300)


class RefinementResult(BaseModel):
    page_id: str
    decisions: list[RefinementDecision]
    raw_response: str = ""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> Any:
    candidates = [text.strip()]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Qwen refinement response contains no valid JSON object")


class QwenEducationalRefiner:
    """Selective second-pass classifier for ambiguous educational units."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen/Qwen3.5-9B",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    @staticmethod
    def _prompt(page_id: str, units: list[RetrievedChunk]) -> str:
        payload = [
            {
                "index": index,
                "rule_kind": unit.metadata.get("unit_kind"),
                "text": " ".join(unit.text.split())[:1200],
            }
            for index, unit in enumerate(units)
        ]
        kinds = ", ".join(kind.value for kind in UnitKind)
        return (
            "You classify OCR blocks from Turkish school textbooks. "
            "For every block choose exactly one semantic kind. "
            "Do not judge whether the content is factually correct.\n\n"
            "Kinds:\n"
            "- theory: explanatory instructional content or ordinary headings\n"
            "- worked_example: an example problem together with demonstrated steps\n"
            "- exercise: a question, activity, assignment, prompt, or answer options\n"
            "- solution: reasoning or a worked answer to a preceding exercise\n"
            "- answer_key: compact list/table of final answers\n"
            "- instruction: navigation or general directions not themselves an assignment\n"
            "- other: copyright, contents, index, noise, isolated page number\n\n"
            f"Allowed values: {kinds}.\n"
            "Return only one JSON object with this schema:\n"
            '{"decisions":[{"index":0,"kind":"exercise",'
            '"confidence":0.95,"reason":"short reason"}]}\n'
            "Return one decision for every supplied index, in the same order. "
            "Use the actual text, not rule_kind, as the authority.\n\n"
            f"page_id={page_id}\nblocks={json.dumps(payload, ensure_ascii=False)}"
        )

    def _request(
        self,
        page_id: str,
        units: list[RetrievedChunk],
        *,
        image_url: str | None = None,
    ) -> tuple[list[RefinementDecision], str]:
        content: str | list[dict[str, Any]]
        prompt = self._prompt(page_id, units)
        if image_url:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            content = prompt

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": 1400,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
        raw = str(message.get("content") or "").strip()
        parsed = _extract_json(raw)
        decisions: list[RefinementDecision] = []
        for item in parsed.get("decisions", []):
            try:
                decisions.append(RefinementDecision.model_validate(item))
            except (TypeError, ValueError):
                continue
        return decisions, raw

    def refine(
        self,
        page_id: str,
        units: list[RetrievedChunk],
        *,
        image_url: str | None = None,
    ) -> RefinementResult:
        if not units:
            return RefinementResult(page_id=page_id, decisions=[])

        raw_responses: list[str] = []
        try:
            initial, raw = self._request(page_id, units, image_url=image_url)
            raw_responses.append(raw)
        except ValueError:
            initial = []
        by_index = {
            decision.index: decision
            for decision in initial
            if decision.index < len(units)
        }
        for index in range(len(units)):
            if index in by_index:
                continue
            repaired, raw = self._request(
                f"{page_id}#unit-{index}",
                [units[index]],
                image_url=image_url,
            )
            raw_responses.append(raw)
            if len(repaired) != 1 or repaired[0].index != 0:
                raise ValueError(
                    f"Qwen could not repair missing index {index} for {page_id}"
                )
            by_index[index] = repaired[0].model_copy(update={"index": index})

        decisions = [by_index[index] for index in range(len(units))]
        expected = list(range(len(units)))
        actual = [decision.index for decision in decisions]
        if actual != expected:
            raise ValueError(
                f"Qwen returned indices {actual}, expected {expected} for {page_id}"
            )
        return RefinementResult(
            page_id=page_id,
            decisions=decisions,
            raw_response="\n\n".join(raw_responses),
        )
