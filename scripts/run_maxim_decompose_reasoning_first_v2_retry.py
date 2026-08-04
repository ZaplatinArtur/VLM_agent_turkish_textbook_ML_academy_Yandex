"""Fail-closed retry profile for rare v2 decomposition schema truncations.

Use only with ``--mode decompose --retry-errors`` on a completed v2 output.
Successful rows are preserved; failed rows are regenerated with the same model,
prompt and seeds but a smaller plan/solution JSON surface.
"""

from __future__ import annotations

import run_maxim_agent_ideas as core
import run_maxim_agent_ideas_reasoning_first_v2 as v2


v2.SYSTEM_PROMPT += (
    " JSON çıktısını tek satır ve kompakt yaz; gereksiz boşluk veya satır sonunu "
    "asla tekrarlama. reasoning en fazla iki kısa cümle olsun. Sayısal bir "
    "final_answer yalnızca sayı ve gerekiyorsa birim içersin; Markdown, yıldız, "
    "parantez veya süslü ayraç kullanma."
)


core.PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_decomposition": {"type": "boolean"},
        "task_type": {"type": "string", "maxLength": 40},
        "critical_evidence": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "maxItems": 3,
        },
        "subtasks": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "minItems": 1,
            "maxItems": 3,
        },
    },
    "required": [
        "needs_decomposition",
        "task_type",
        "critical_evidence",
        "subtasks",
    ],
    "additionalProperties": False,
}

v2.SOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 250},
        "final_answer": {
            "type": "string",
            "maxLength": 100,
            "pattern": "^-?[0-9]+(?:[.,][0-9]+)?(?:/[0-9]+)?$",
        },
    },
    "required": ["reasoning", "final_answer"],
    "additionalProperties": False,
}


_complete = core.EndpointPool.complete


def _compact_complete(self: core.EndpointPool, **kwargs):
    schema_name = str(kwargs.get("schema_name") or "")
    if schema_name == "decomposition_plan":
        kwargs["max_tokens"] = min(int(kwargs["max_tokens"]), 600)
    elif schema_name == "decomposition_solution":
        kwargs["max_tokens"] = min(int(kwargs["max_tokens"]), 900)
    return _complete(self, **kwargs)


core.EndpointPool.complete = _compact_complete


if __name__ == "__main__":
    raise SystemExit(v2.main())
