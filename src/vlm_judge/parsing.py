from __future__ import annotations

import json
import re
from typing import Any

from .schema import JudgeVerdict


_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_MAX_RATIONALE_CHARS = 1200


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
    rationale = payload.get("rationale")
    if isinstance(rationale, str) and len(rationale) > _MAX_RATIONALE_CHARS:
        # Rationale is audit metadata and must not invalidate an otherwise
        # well-formed score. The unabridged model response remains in cache.
        payload = dict(payload)
        payload["rationale"] = rationale[: _MAX_RATIONALE_CHARS - 3] + "..."
    return JudgeVerdict.from_dict(payload)

