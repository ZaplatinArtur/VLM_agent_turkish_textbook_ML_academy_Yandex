"""Compact retry profile for rare v2 parallel8 structured-output failures."""

from __future__ import annotations

import run_maxim_agent_ideas as core
import run_maxim_agent_ideas_reasoning_first_v2 as v2


v2.SYSTEM_PROMPT += (
    " JSON çıktısını tek satır ve kompakt yaz; gereksiz boşluk veya satır sonunu "
    "asla tekrarlama. Her reasoning en fazla iki kısa cümle olsun."
)

v2.CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 350},
        "final_answer": {"type": "string", "maxLength": 100},
    },
    "required": ["reasoning", "final_answer"],
    "additionalProperties": False,
}

v2.SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 400},
        "selected_index": {"type": "integer", "minimum": 1, "maximum": 8},
        "final_answer": {"type": "string", "maxLength": 100},
    },
    "required": ["reasoning", "selected_index", "final_answer"],
    "additionalProperties": False,
}

_complete = core.EndpointPool.complete


def _compact_complete(self: core.EndpointPool, **kwargs):
    schema_name = str(kwargs.get("schema_name") or "")
    if schema_name.startswith("candidate_") or schema_name == "parallel_selector":
        kwargs["max_tokens"] = min(int(kwargs["max_tokens"]), 900)
    return _complete(self, **kwargs)


core.EndpointPool.complete = _compact_complete


if __name__ == "__main__":
    raise SystemExit(v2.main())
