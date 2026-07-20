"""Робастный разбор ответа модели в SolveOutput.

9B-модель может обернуть JSON в <think>, код-блоки или добавить текст вокруг —
парсер снимает всё это слоями и валидирует результат.
"""

import json
import re

from .schemas import SolveOutput

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_solve_output(raw: str) -> SolveOutput | None:
    text = _THINK_RE.sub("", raw).strip()

    candidates = [text]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return SolveOutput.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValueError):
            continue
    return None
