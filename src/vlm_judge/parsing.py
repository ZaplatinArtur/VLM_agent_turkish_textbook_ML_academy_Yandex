from __future__ import annotations

import json
import re
from typing import Any

from .schema import JudgeVerdict


_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def parse_judge_verdict(raw: str) -> JudgeVerdict:
    value = raw.strip()
    fenced = _FENCED_JSON.search(value)
    if fenced:
        value = fenced.group(1)
    try:
        payload: Any = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("judge did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")
    return JudgeVerdict.from_dict(payload)

