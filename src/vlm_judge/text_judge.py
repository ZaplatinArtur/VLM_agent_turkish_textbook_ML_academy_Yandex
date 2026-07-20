"""Text-only binary judge contract for the first evaluation milestone."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import JudgeBackend


TEXT_BINARY_PROMPT_VERSION = "text-binary-v1"

TEXT_BINARY_SYSTEM_PROMPT = """You are a strict binary evaluator of a candidate answer to a text-only school task.

Inputs:
- QUESTION: the task the candidate had to solve
- REFERENCE: the trusted correct answer
- CANDIDATE: the answer being evaluated

Return score 1 only if the candidate is correct and complete for the question.
Return score 0 if it is wrong, partially correct, incomplete, contradictory,
irrelevant, empty, or unsupported.

Rules:
1. Judge substance, not style, verbosity, confidence, spelling, or formatting.
2. Accept equivalent wording, notation, units, languages, and mathematically equivalent forms.
3. For a multi-part question, all required parts must be correct.
4. Do not award 1 for a lucky final token when the explanation contains a material contradiction.
5. Do not infer information absent from the candidate or reference.
6. If the reference is missing, ambiguous, or obviously invalid, use score 0 and mention reference_issue.

Return exactly one JSON object with exactly these keys:
{"score": 0 or 1, "rationale": "brief reason"}
"""


@dataclass(frozen=True, slots=True)
class TextBinaryJudgeRequest:
    system_prompt: str
    user_prompt: str
    image_urls: tuple[str, ...] = ()
    image_labels: tuple[str, ...] = ()


def build_text_binary_request(
    question_text: str,
    reference_answer: str,
    candidate_answer: str,
) -> TextBinaryJudgeRequest:
    """Build a text-only request; image attachments are impossible by contract."""
    fields = {
        "QUESTION": question_text,
        "REFERENCE": reference_answer,
        "CANDIDATE": candidate_answer,
    }
    if not all(isinstance(value, str) for value in fields.values()):
        raise TypeError("question_text, reference_answer, and candidate_answer must be strings")
    if not question_text.strip() or not reference_answer.strip():
        raise ValueError("question_text and reference_answer must not be empty")
    user_prompt = "\n\n".join(f"{key}:\n{value}" for key, value in fields.items())
    return TextBinaryJudgeRequest(TEXT_BINARY_SYSTEM_PROMPT, user_prompt)


def parse_text_binary_verdict(raw: str) -> dict[str, Any]:
    """Parse and validate the strict Qwen response for the binary judge."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("text binary judge did not return valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"score", "rationale"}:
        raise ValueError("binary verdict must contain exactly score and rationale")
    score = value["score"]
    if isinstance(score, bool) or not isinstance(score, int) or score not in {0, 1}:
        raise ValueError("score must be integer 0 or 1")
    if not isinstance(value["rationale"], str) or len(value["rationale"]) > 1200:
        raise ValueError("rationale must be a string of at most 1200 characters")
    return {"score": score, "rationale": value["rationale"]}


def evaluate_text_records(
    records: list[dict[str, Any]],
    backend: JudgeBackend,
    output_path: Path,
    *,
    max_attempts: int = 2,
    retry_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    """Run text records through an OpenAI-compatible backend and save JSONL results."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    failed = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for index, record in enumerate(records, start=1):
            task_id = str(record.get("task_id") or f"row-{index}")
            request = build_text_binary_request(
                record.get("question_text"),
                record.get("reference_answer"),
                record.get("candidate_answer"),
            )
            verdict = None
            raw_response = None
            metadata: dict[str, Any] = {}
            error = None
            attempts = 0
            for attempt in range(1, max_attempts + 1):
                attempts = attempt
                if attempt > 1 and retry_delay_seconds:
                    time.sleep(retry_delay_seconds * (attempt - 1))
                try:
                    response = backend.complete(request)
                    raw_response = response.text
                    metadata = response.metadata
                    verdict = parse_text_binary_verdict(response.text)
                    error = None
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            if verdict is None:
                failed += 1
            else:
                succeeded += 1
            result = {
                "task_id": task_id,
                "prompt_version": TEXT_BINARY_PROMPT_VERSION,
                "verdict": verdict,
                "judge": {
                    "backend": backend.name,
                    "model": backend.model,
                    "attempts": attempts,
                    "error": error,
                    "metadata": metadata,
                },
            }
            if "manual_score" in record:
                result["manual_score"] = record["manual_score"]
                result["agreement"] = (
                    verdict is not None and verdict["score"] == record["manual_score"]
                )
            if error:
                result["raw_response"] = raw_response
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
    return {
        "records": len(records),
        "succeeded": succeeded,
        "failed": failed,
        "output": str(output_path),
    }
